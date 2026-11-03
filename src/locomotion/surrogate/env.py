"""Run the interface of ``locomotion.omniwalk.adapter.OmniEnv`` on CPU MuJoCo physics."""

from __future__ import annotations

import ast
import copy
import hashlib
import json
import math
from dataclasses import asdict, dataclass, fields, replace
from pathlib import Path

import mujoco
import numpy as np
import torch
from mujoco import rollout as mj_rollout
from scipy.spatial import ConvexHull

from locomotion.omniwalk.mdp.actions import emitted_target, executed_action_feature
from locomotion.omniwalk.mdp.frames import inverse_rotate, navigation, rotate
from locomotion.robots.hexapod import BODY_NAMES, JOINT_NAMES, KD, LEGS, MASS_KG, MODEL_SHA256
from locomotion.robots.rs05 import motor_force

# The package resolves repository data from its own location.
ROOT = Path(__file__).resolve().parents[3]
ROBOT = ROOT / "robot/hexapod_mkii_updated_v1"
DEFAULT_MODEL = ROBOT / "model_rs05_mass_corrected.json"
DEFAULT_GEOMETRY = ROBOT / "geometry/geometry.json"
DEFAULT_STANCE = ROBOT / "stance.json"
# Isaac's articulation view lists links breadth first (native_readback.json, session.json).
NATIVE_BODY_NAMES = ["body"] + [
    f"{leg}_{part}" for part in ("coxa", "femur", "tibia") for leg in LEGS
]
NATIVE_JOINT_NAMES = [
    f"{leg}_{joint}" for joint in ("coxa_yaw", "femur_pitch", "tibia_pitch") for leg in LEGS
]
MAX_JOINT_SPEED = 50.26548245743669
TOE_CAP_LOWER_X = 0.115
# One floor-contact sensor per collision geom: 13 single-geom links, then each tibia's cap and shaft.
CONTACT_GEOMS = (
    [name + "_hull" for name in NATIVE_BODY_NAMES[:13]]
    + [name + "_toe" for name in NATIVE_BODY_NAMES[13:]]
    + [name + "_shaft" for name in NATIVE_BODY_NAMES[13:]]
)


@dataclass(frozen=True)
class ContactModel:
    """MuJoCo contact, solver and limit parameters. README.md records the calibration."""

    solref: tuple = (0.005, 1.0)
    solimp: tuple = (0.9, 0.95, 0.001, 0.5, 2.0)
    friction: tuple = (1.0, 0.005, 0.0001)
    condim: int = 3
    cone: str = "elliptic"
    impratio: float = 1.0
    noslip_iterations: int = 4
    noslip_tolerance: float = 1e-6
    refsafe: bool = True
    # Position update q += dt (v_old + blend (v_new - v_old)). 1.0 is MuJoCo's semi-implicit Euler.
    # 33/64 reproduces the native solver: 32 TGS iterations, each adding 1/32 of the external
    # impulse and then integrating position for dt/32 (measured on native free-flight substeps).
    position_blend: float = 33.0 / 64.0
    # A replica whose geom makes new floor contact in a substep keeps MuJoCo's Euler position
    # update for that substep: the native solver resolves an impact in its first iteration.
    impact_euler: bool = True
    solver: str = "Newton"
    iterations: int = 100
    tolerance: float = 1e-8
    integrator: str = "Euler"
    # No margin: the soft limit engages at the limit, so a joint that a load holds on its stop rests
    # beyond it and the 2e-6 rad termination fires, as the native hard limit does.
    limit_margin: float = 0.0
    limit_solref: tuple = (0.005, 1.0)
    limit_solimp: tuple = (0.99, 0.999, 0.001, 0.5, 2.0)
    # Toe touchdown (README.md, "Touchdown model"). The native solver settles friction over the
    # iterations that remain after a toe reaches the floor; a toe that arrives late in a substep
    # leaves it with part of a full Coulomb impulse against its slide. overshoot_phase holds the
    # arrival phases where that part starts and where it reaches one. False keeps MuJoCo's soft
    # contact alone.
    touchdown_overshoot: bool = True
    overshoot_phase: tuple = (0.75, 0.9)
    # The reversed slide never exceeds this multiple of the incoming slide (native touchdowns: 3 to 18).
    overshoot_gain: float = 15.0
    # The step clips joint speeds to the URDF limit after each MuJoCo substep, as PhysX enforces
    # its maximum joint velocity. The motor ceiling reaches zero at this speed. The touchdown
    # impulse acts after the clip, so it alone can exceed the limit for one substep.
    joint_speed_clamp: bool = True

    def __post_init__(self):
        if self.position_blend != 1.0 and self.integrator != "Euler":
            raise ValueError(
                "The position blend corrects MuJoCo's Euler step; use the Euler integrator"
            )
        if not 0.0 <= self.overshoot_phase[0] < self.overshoot_phase[1]:
            raise ValueError("The overshoot phases must increase")


def contact_overrides(items, base=None):
    """A ``ContactModel`` with ``FIELD=VALUE`` overrides; each value is a Python literal."""
    names = {field.name for field in fields(ContactModel)}
    options = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or key not in names:
            raise ValueError("Expected a ContactModel FIELD=VALUE, got " + item)
        options[key] = ast.literal_eval(value)
    return replace(ContactModel() if base is None else base, **options)


@dataclass(frozen=True)
class DynamicsRandomization:
    """Per-episode dynamics draws for sim2real robustness training (design evidence only).

    Each scale field is a closed ``(low, high)`` range on the canonical value, drawn uniformly
    per episode and replica from a dedicated per-row stream, so draws are independent of batch
    width, threading and the global torch/numpy generators. With ``randomization=None`` the
    environment skips every branch below — no draw, no model copy, no extra tensor op — and the
    trajectory stays bitwise identical to today's. ``action_delay_controls`` is the largest
    whole-control transport delay a row may draw; ``sag_per_minute`` drains the torque ceiling
    within an episode toward ``sag_floor``, keeping the drifting term separate from the slow
    per-episode scales so a later hardware calibration can freeze the slow ones and keep
    re-estimating the drift. The canonical motor, model and contact files stay untouched: the
    scales act on this environment's per-replica copies alone.
    """

    friction_scale: tuple = (1.0, 1.0)
    mass_scale: tuple = (1.0, 1.0)
    motor_scale: tuple = (1.0, 1.0)
    kd_scale: tuple = (1.0, 1.0)
    action_delay_controls: int = 0
    sag_per_minute: tuple = (0.0, 0.0)
    sag_floor: float = 0.8

    def __post_init__(self):
        bounds = {
            "friction_scale": (0.2, 3.0),
            "mass_scale": (0.5, 1.5),
            "motor_scale": (0.5, 1.5),
            "kd_scale": (0.5, 1.5),
            "sag_per_minute": (0.0, 0.5),
        }
        for name, (least, most) in bounds.items():
            low, high = (float(value) for value in getattr(self, name))
            if not (math.isfinite(low) and math.isfinite(high) and low <= high):
                raise ValueError(name + " needs a finite low <= high range")
            if low < least or high > most:
                raise ValueError(f"{name} must stay within ({least}, {most})")
        low, high = self.delay_range()
        if not (type(low) is int and type(high) is int and 0 <= low <= high <= 2):
            raise ValueError(
                "action_delay_controls is a whole count or (low, high) range of 0 to 2 controls"
            )
        if not 0.5 <= self.sag_floor <= 1.0:
            raise ValueError("sag_floor lies in [0.5, 1]")

    def delay_range(self):
        """The delay draw as a closed (low, high) range; the whole-count form means (0, count)."""
        value = self.action_delay_controls
        if type(value) is int:
            return (0, value)
        if isinstance(value, (tuple, list)) and len(value) == 2:
            return (value[0], value[1])
        return (value, value)

    def field_range(self, name):
        """The closed (low, high) range of a drawn field, for target scaling and calibration."""
        if name == "action_delay_controls":
            low, high = self.delay_range()
            return (float(low), float(high))
        low, high = getattr(self, name)
        return (float(low), float(high))


def randomization_overrides(items, base=None):
    """A ``DynamicsRandomization`` with ``FIELD=VALUE`` overrides; each value is a Python literal."""
    names = {field.name for field in fields(DynamicsRandomization)}
    options = {}
    for item in items:
        key, separator, value = item.partition("=")
        if not separator or key not in names:
            raise ValueError("Expected a DynamicsRandomization FIELD=VALUE, got " + item)
        options[key] = ast.literal_eval(value)
    return replace(DynamicsRandomization() if base is None else base, **options)


@dataclass(frozen=True)
class SurrogateConfig:
    """Same fields as ``omniwalk.env_cfg.EnvConfig``; that class accepts cuda:0 alone.

    The floor fields name the native floor this surrogate stands in for; MuJoCo uses its own flat plane.
    """

    num_envs: int = 128
    spacing_m: float = 2.0
    physics_dt: float = 0.0025
    decimation: int = 8
    episode_seconds: float = 20.0
    action_scale_rad: float = 0.35
    target_slew_rad: float = 0.040
    reset_height_m: float = 0.08161108940839767
    command_forward_mps: float = 0.10
    command_left_mps: float = 0.0
    command_yaw_rad_s: float = 0.0
    seed: int = 20260914
    record_motion_features: bool = False
    render: bool = False
    device: str = "cpu"
    floor: str = "isaaclab_mesh"
    floor_contact_offset_m: float | None = None
    floor_rest_offset_m: float | None = None
    # Declared for field parity with EnvConfig; the surrogate takes its terrain as the
    # SurrogateEnv ``terrain`` argument and these fields stay at their defaults here.
    terrain_spec_path: str | None = None
    terrain_mesh_resolution_m: float = 0.08
    terrain_spawn: bool = False

    def __post_init__(self):
        if type(self.num_envs) is not int or self.num_envs < 1:
            raise ValueError("The surrogate needs at least one replica")
        if self.terrain_spec_path is not None or self.terrain_spawn:
            raise ValueError("The surrogate takes its terrain as the SurrogateEnv argument")
        if self.physics_dt != 0.0025 or self.decimation != 8 or self.target_slew_rad != 0.040:
            raise ValueError("Canonical 400/50Hz servo and target slew are fixed")
        if self.spacing_m < 2 or self.device != "cpu":
            raise ValueError("The surrogate runs on cpu with at least 2m spacing")
        if not all(
            math.isfinite(x)
            for x in (
                self.spacing_m,
                self.episode_seconds,
                self.action_scale_rad,
                self.reset_height_m,
                self.command_forward_mps,
                self.command_left_mps,
                self.command_yaw_rad_s,
            )
        ):
            raise ValueError("Configuration contains a nonfinite value")
        if not 0 < self.action_scale_rad <= 0.5 or not 0 < self.episode_seconds <= 120:
            raise ValueError("Invalid bounded action scale or episode length")

    @property
    def control_dt(self):
        return self.physics_dt * self.decimation

    def declaration(self):
        return {
            **asdict(self),
            "schema": "canonical_paper_walk_experiment_v1",
            "actor_width": 231,
            "critic_width": 234,
            "amp_width": 61,
            "joint_names": list(JOINT_NAMES),
            "legs": list(LEGS),
            "surrogate": True,
        }


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def repository_state():
    """Revision and uncommitted paths of the checkout; each value is None where Git metadata is absent."""
    import subprocess

    def git(*arguments):
        try:
            result = subprocess.run(
                ["git", "-C", str(ROOT), *arguments], capture_output=True, text=True
            )
        except OSError:
            return None
        return result.stdout.rstrip("\n") if result.returncode == 0 else None

    # Porcelain lines are "XY path": two status columns, one space, the path.
    status = git("status", "--porcelain")
    return {
        "revision": git("rev-parse", "HEAD"),
        "branch": git("rev-parse", "--abbrev-ref", "HEAD"),
        "dirty": None if status is None else bool(status),
        "uncommitted_paths": None if status is None else [line[3:] for line in status.splitlines()],
    }


def _numbers(values):
    return " ".join(repr(float(x)) for x in np.ravel(values))


def _hull_vertices(points):
    points = np.unique(np.round(np.asarray(points, dtype=float), 7), axis=0)
    return points[ConvexHull(points).vertices]


def build_mjcf(model, clouds, contact=ContactModel(), *, reset_height=0.1028, terrain=None):
    """MJCF with the exact ledger masses, inertias, joint frames and limits.

    ``terrain`` (a ``terrain.Terrain``) makes the floor a height field over its grid; SurrogateEnv writes the
    heights after compiling. The floor keeps its name, so its contact sensors read the terrain."""
    links = {link["name"]: link for link in model["links"]}
    children = {}
    for joint in model["joints"]:
        if not np.allclose(joint["axis"], [0.0, 0.0, 1.0]):
            raise ValueError("The surrogate kinematics expect joint axes along local Z")
        children.setdefault(joint["parent"], []).append(joint)
    meshes, bodies = [], []

    def mesh(name, points):
        meshes.append(f'<mesh name="{name}" vertex="{_numbers(_hull_vertices(points))}"/>')

    def emit(name, joint, depth):
        link, pad = links[name], "  " * depth
        tensor = np.asarray(link["inertia"], dtype=float)
        if joint is None:
            bodies.append(f'{pad}<body name="{name}" pos="0 0 {reset_height!r}">')
            bodies.append(f'{pad} <freejoint name="root"/>')
        else:
            x, y, z, w = joint["quaternion_xyzw"]
            bodies.append(
                f'{pad}<body name="{name}" pos="{_numbers(joint["xyz"])}" quat="{_numbers([w, x, y, z])}">'
            )
            bodies.append(
                f'{pad} <joint name="{joint["name"]}" type="hinge" axis="0 0 1" limited="true" '
                f'range="{_numbers([joint["lower"], joint["upper"]])}"/>'
            )
        bodies.append(
            f'{pad} <inertial pos="{_numbers(link["com"])}" mass="{float(link["mass"])!r}" fullinertia="'
            f'{_numbers([tensor[0, 0], tensor[1, 1], tensor[2, 2], tensor[0, 1], tensor[0, 2], tensor[1, 2]])}"/>'
        )
        if name.endswith("_tibia"):
            points = np.asarray(clouds["all__" + name])
            mesh(name + "_toe", points[points[:, 0] >= TOE_CAP_LOWER_X - 1e-9])
            # The shaft hull stops 0.1 mm short of the cap plane, so each floor contact has one class.
            shaft = np.array(clouds["non_toe__" + name], dtype=float)
            shaft[:, 0] = np.minimum(shaft[:, 0], TOE_CAP_LOWER_X - 1e-4)
            mesh(name + "_shaft", shaft)
            bodies.append(
                f'{pad} <geom name="{name}_toe" type="mesh" mesh="{name}_toe" class="toe"/>'
            )
            bodies.append(
                f'{pad} <geom name="{name}_shaft" type="mesh" mesh="{name}_shaft" class="link"/>'
            )
        else:
            mesh(name + "_hull", clouds["all__" + name])
            bodies.append(
                f'{pad} <geom name="{name}_hull" type="mesh" mesh="{name}_hull" class="{"chassis" if joint is None else "link"}"/>'
            )
        for child in children.get(name, []):
            emit(child["child"], child, depth + 1)
        bodies.append(f"{pad}</body>")

    emit("body", None, 2)
    sensors = [
        f'<contact name="floor_{geom}" geom1="floor" geom2="{geom}" data="force pos" reduce="netforce"/>'
        for geom in CONTACT_GEOMS
    ]
    motors = "".join(
        f'<motor name="motor_{name}" joint="{name}" gear="1"/>' for name in JOINT_NAMES
    )
    c = contact
    floor, field = 'type="plane" size="0 0 0.5"', ""
    if terrain is not None:
        low, top = float(terrain.heights.min()), max(1e-6, float(np.ptp(terrain.heights)))
        field = (
            f'<hfield name="terrain" nrow="{terrain.ny}" ncol="{terrain.nx}" '
            f'size="{terrain.half[0]!r} {terrain.half[1]!r} {top!r} 0.1"/>'
        )
        floor = (
            f'type="hfield" hfield="terrain" '
            f'pos="{terrain.center_xy[0]!r} {terrain.center_xy[1]!r} {low!r}"'
        )
    # Robot geoms collide with the floor alone: contype 2 meets the floor's conaffinity 2.
    return f"""<mujoco model="hexapod_surrogate">
 <compiler angle="radian" autolimits="false"/>
 <option timestep="0.0025" gravity="0 0 -9.81" integrator="{c.integrator}" cone="{c.cone}" impratio="{c.impratio!r}"
         noslip_iterations="{c.noslip_iterations}" noslip_tolerance="{c.noslip_tolerance!r}" solver="{c.solver}" iterations="{c.iterations}" tolerance="{c.tolerance!r}"/>
 <option><flag refsafe="{"enable" if c.refsafe else "disable"}"/></option>
 <visual><global offwidth="1280" offheight="720"/><quality shadowsize="2048"/><map znear="0.02"/></visual>
 <default>
  <geom contype="2" conaffinity="0" condim="{c.condim}" friction="{_numbers(c.friction)}"
        solref="{_numbers(c.solref)}" solimp="{_numbers(c.solimp)}"/>
  <joint damping="0" armature="0" frictionloss="0" stiffness="0" margin="{c.limit_margin!r}"
         solreflimit="{_numbers(c.limit_solref)}" solimplimit="{_numbers(c.limit_solimp)}"/>
  <default class="toe"><geom rgba="0.85 0.25 0.2 1"/></default>
  <default class="link"><geom rgba="0.55 0.6 0.7 1"/></default>
  <default class="chassis"><geom rgba="0.3 0.32 0.38 1"/></default>
 </default>
 <asset>
  <texture name="grid" type="2d" builtin="checker" rgb1="0.82 0.82 0.8" rgb2="0.7 0.7 0.68" width="256" height="256"/>
  <material name="grid" texture="grid" texrepeat="10 10" texuniform="true" reflectance="0"/>
  {"".join(meshes)}
  {field}
 </asset>
 <worldbody>
  <light pos="0 0 3" dir="0 0 -1" directional="true" diffuse="0.8 0.8 0.8"/>
  <geom name="floor" {floor} contype="1" conaffinity="2" material="grid" condim="{c.condim}"
        friction="{_numbers(c.friction)}" solref="{_numbers(c.solref)}" solimp="{_numbers(c.solimp)}"/>
{chr(10).join(bodies)}
 </worldbody>
 <actuator>{motors}</actuator>
 <sensor>{"".join(sensors)}</sensor>
</mujoco>"""


def quaternion_multiply(a, b):
    """Hamilton product of XYZW quaternions."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        dim=-1,
    )


class _StepCounter:
    """The part of Isaac's SimulationContext that repository wrappers read."""

    def __init__(self, env):
        self._env = env

    def get_physics_step_count(self):
        return self._env.physics_steps


class _Buffer:
    """Stands in for a warp array: the repository reads contact buffers with ``.numpy()``."""

    def __init__(self, array):
        self._array = array

    def numpy(self):
        return self._array


class _ContactView:
    """The last substep's floor contacts in the layout of Isaac's rigid contact view."""

    def __init__(self, env):
        self.env = env
        self.filter_count = 1
        # Geoms of each native body, as indices into CONTACT_GEOMS.
        self.geoms = [[i] for i in range(13)] + [[13 + k, 19 + k] for k in range(6)]

    def get_contact_data(self, dt):
        env = self.env
        data = env._sensor[:, 0].reshape(env.num_envs, 25, 6)
        origins = env._origins64.numpy()
        force, point, normal, separation = [], [], [], []
        counts = np.zeros((19 * env.num_envs, 1), dtype=np.int64)
        starts = np.zeros((19 * env.num_envs, 1), dtype=np.int64)
        for e in range(env.num_envs):
            for b, geoms in enumerate(self.geoms):
                starts[e * 19 + b, 0] = len(force)
                for g in geoms:
                    if data[e, g, 2] > 0.0:
                        force.append([data[e, g, 2]])
                        point.append(data[e, g, 3:] + origins[e])
                        normal.append([0.0, 0.0, 1.0])
                        separation.append([0.0])
                        counts[e * 19 + b, 0] += 1
        # One spare row: the repository reads a full buffer as a sign of truncation.
        force.append([0.0])
        point.append(np.zeros(3))
        normal.append([0.0, 0.0, 0.0])
        separation.append([0.0])
        return tuple(
            _Buffer(np.asarray(x, dtype=np.float64)) for x in (force, point, normal, separation)
        ) + (_Buffer(counts), _Buffer(starts))

    def get_contact_force_matrix(self, dt):
        return _Buffer(self.env._forces.reshape(-1, 1, 3).numpy())


class SurrogateEnv:
    num_actions = 18
    observation_width = 231
    critic_width = 234
    amp_width = 61

    def __init__(
        self,
        cfg=None,
        asset=None,
        model_path=DEFAULT_MODEL,
        geometry_path=DEFAULT_GEOMETRY,
        output=None,
        *,
        reference_metadata=None,
        threads=2,
        contact=None,
        substep_state=True,
        terrain=None,
        start=None,
        spawn=None,
        randomization=None,
    ):
        """The constructor ignores ``asset`` (the native USD). ``reference_metadata``
        defaults to the repository stance.json, which native training passes.

        ``terrain`` (a ``terrain.Terrain``) replaces the flat floor with its height field, and ``start``
        (x, y, ground height, yaw) places every reset there instead of at the origin. Both serve scenes
        (``surrogate/scene.py``): the touchdown model assumes the flat floor at z = 0, so a terrain needs
        ``touchdown_overshoot=False``, and the fall check still reads the root height above z = 0. ``spawn``
        (rows -> [len(rows), 4] of x, y, ground height, yaw) instead places each reset row, for training on a
        terrain from random starts."""
        self.cfg = SurrogateConfig() if cfg is None else cfg
        cfg = self.cfg
        self.device, self.num_envs = cfg.device, cfg.num_envs
        self.output = None if output is None else Path(output)
        if self.output is not None:
            self.output.mkdir(parents=True, exist_ok=True)
        model_path = Path(model_path)
        if sha(model_path) != MODEL_SHA256:
            raise ValueError("Canonical model identity differs")
        self.model = json.loads(model_path.read_text())
        if {x["name"] for x in self.model["links"]} != set(BODY_NAMES) or [
            x["name"] for x in self.model["joints"]
        ] != list(JOINT_NAMES):
            raise ValueError("Canonical 19-body/18-joint topology differs")
        if abs(sum(x["mass"] for x in self.model["links"]) - MASS_KG) > 1e-9:
            raise ValueError("Canonical mass ledger differs")
        self.geometry_path = Path(geometry_path)
        self.geometry_meta = json.loads(self.geometry_path.read_text())
        self.geometry_extrema_path = self.geometry_path.parent / "geometry_extrema.npz"
        if reference_metadata is None:
            reference_metadata = json.loads(DEFAULT_STANCE.read_text())
        self.reference_metadata = reference_metadata
        self.contact_model = ContactModel() if contact is None else contact
        self.spawn = spawn
        # Kept for terrain-aware observations (mdp/footfall.py reads it duck-typed off the env).
        self.terrain = terrain
        if terrain is not None and self.contact_model.touchdown_overshoot:
            raise ValueError(
                "A terrain needs touchdown_overshoot=False: the model assumes a flat floor"
            )
        # Per-episode dynamics draws (DynamicsRandomization); None keeps the canonical dynamics
        # bitwise and skips every randomization branch below.
        self.randomization = randomization
        if (
            randomization is not None
            and randomization.friction_scale != (1.0, 1.0)
            and self.contact_model.touchdown_overshoot
        ):
            raise ValueError(
                "Friction randomization needs touchdown_overshoot=False: the touchdown "
                "correction reads the one shared contact friction"
            )
        self.torch = lambda x: torch.as_tensor(x, dtype=torch.float32, device=self.device)
        self.joint_names = list(JOINT_NAMES)
        self.body_names = list(BODY_NAMES)
        self.native_body_names = list(NATIVE_BODY_NAMES)
        self.native_joint_names = list(NATIVE_JOINT_NAMES)
        self.neutral = self.torch(
            self.reference_metadata.get("nominal_joint_position_rad", [0.0] * 18)
        )
        self.reset_height = float(
            self.reference_metadata.get("reset_root_height_m", cfg.reset_height_m)
        )
        if self.neutral.shape != (18,) or not torch.isfinite(self.neutral).all():
            raise ValueError("Invalid named neutral reference")
        with np.load(self.geometry_extrema_path, allow_pickle=False) as clouds:
            self.mjcf = build_mjcf(
                self.model,
                clouds,
                self.contact_model,
                reset_height=self.reset_height,
                terrain=terrain,
            )
        self.mj_model = mujoco.MjModel.from_xml_string(self.mjcf)
        m = self.mj_model
        if terrain is not None:
            low = terrain.heights.min()
            m.hfield_data[:] = (
                (terrain.heights - low) / max(1e-6, np.ptp(terrain.heights))
            ).ravel()
        if m.nq != 25 or m.nv != 24 or m.nu != 18 or abs(float(m.body_mass.sum()) - MASS_KG) > 1e-9:
            raise ValueError("MuJoCo model differs from the canonical ledger")
        if (
            np.any(m.dof_damping != 0)
            or np.any(m.dof_armature != 0)
            or np.any(m.jnt_stiffness != 0)
        ):
            raise ValueError("Unexpected implicit drive, damping or armature")
        joint_ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name) for name in JOINT_NAMES]
        self._qpos_columns = np.array([m.jnt_qposadr[i] for i in joint_ids])
        self._qvel_columns = np.array([m.jnt_dofadr[i] for i in joint_ids])
        actuator_joints = [
            mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, m.actuator_trnid[i, 0])
            for i in range(m.nu)
        ]
        if actuator_joints != list(JOINT_NAMES):
            raise ValueError("Actuator order differs from JOINT_NAMES")
        named = {j["name"]: j for j in self.model["joints"]}
        limits = self.torch([[named[n]["lower"], named[n]["upper"]] for n in JOINT_NAMES])
        self.lower, self.upper = limits[:, 0], limits[:, 1]
        if torch.any(self.neutral < self.lower) or torch.any(self.neutral > self.upper):
            raise ValueError("Reference neutral violates named canonical limits")
        self.kd = self.torch(KD)
        side = math.ceil(math.sqrt(cfg.num_envs))
        origins = [
            (i % side * cfg.spacing_m, i // side * cfg.spacing_m, 0.0) for i in range(cfg.num_envs)
        ]
        self.origins = self.torch(origins)
        self._origins64 = torch.as_tensor(origins, dtype=torch.float64)
        self.roots = [f"/Robot_{i:03d}" for i in range(cfg.num_envs)]
        self.body_index = 0
        self.toe_index = torch.tensor(
            [self.native_body_names.index(leg + "_tibia") for leg in LEGS]
        )
        self._other_index = [
            i for i, name in enumerate(self.native_body_names) if not name.endswith("_tibia")
        ]
        links = {x["name"]: x for x in self.model["links"]}
        self.root_com_local = self.torch(links["body"]["com"]).expand(cfg.num_envs, -1)
        self._root_com64 = torch.as_tensor(links["body"]["com"], dtype=torch.float64)
        # Fixed joint frames for batched forward kinematics: [level, leg, ...].
        self._joint_xyz = torch.tensor(
            [
                [named[f"{leg}_{joint}"]["xyz"] for leg in LEGS]
                for joint in ("coxa_yaw", "femur_pitch", "tibia_pitch")
            ],
            dtype=torch.float64,
        )
        self._joint_quat = torch.tensor(
            [
                [named[f"{leg}_{joint}"]["quaternion_xyzw"] for leg in LEGS]
                for joint in ("coxa_yaw", "femur_pitch", "tibia_pitch")
            ],
            dtype=torch.float64,
        )
        default_toes = []
        for leg in LEGS:
            shape = next(x for x in self.geometry_meta["shapes"] if x["body"] == leg + "_tibia")
            transform = np.asarray(shape["shape_to_link"])
            bounds = np.asarray(shape["cap_bounds_m"])
            default_toes.append((transform @ np.array([bounds[1, 0], 0.0, 0.0, 1.0]))[:3])
        self.toe_local = self.torch(
            np.asarray(self.reference_metadata.get("toe_local_points_m", default_toes))
        )
        if self.toe_local.shape != (6, 3) or not torch.isfinite(self.toe_local).all():
            raise ValueError("Invalid reference toe points")
        self._toe_local64 = self.toe_local.to(torch.float64)
        # Batched stepping: one MjData per worker thread, one state row per replica.
        self.threads = max(0, int(threads))
        self._pool = mj_rollout.Rollout(nthread=self.threads)
        self._datas = [mujoco.MjData(m) for _ in range(max(1, self.threads))]
        if self.randomization is None:
            self._models = [m] * cfg.num_envs
        else:
            # Each replica owns a model copy so a reset can redraw its masses and friction alone;
            # the canonical checks above ran on the shared base model first.
            self._models = [copy.copy(m) for _ in range(cfg.num_envs)]
            self._base_body_mass = m.body_mass.copy()
            self._base_body_inertia = m.body_inertia.copy()
            self._base_geom_friction = m.geom_friction.copy()
        self._chunk = max(1, math.ceil(cfg.num_envs / max(1, self.threads)))
        self._nstate = mujoco.mj_stateSize(m, mujoco.mjtState.mjSTATE_FULLPHYSICS)
        self._state = np.zeros((cfg.num_envs, self._nstate))
        self._next = np.zeros((cfg.num_envs, 1, self._nstate))
        self._sensor = np.zeros((cfg.num_envs, 1, m.nsensordata))
        self._warmstart = np.zeros((cfg.num_envs, m.nv))
        # A world-frame force (N) and torque (N m) on each replica's body at its centre of mass, [n, 6];
        # None applies none. Scenes push robots with it; keep zeros rather than None once one acted, as
        # MuJoCo keeps the last applied force in its data.
        self.applied_wrench = None
        self._control = np.zeros((cfg.num_envs, 1, m.nu))
        self._qpos0 = np.zeros(25)
        self._qpos0[2] = self.reset_height
        self._qpos0[3] = 1.0
        if start is not None:
            x, y, ground, heading = (float(value) for value in start)
            self._qpos0[:3] = (x, y, ground + self.reset_height)
            self._qpos0[3:7] = (math.cos(heading / 2), 0.0, 0.0, math.sin(heading / 2))
        self._qpos0[self._qpos_columns] = self.neutral.numpy().astype(np.float64)
        self.physics_steps = 0
        self.speed_clamp_events = 0
        self.peak_joint_speed = 0.0
        self.touchdown_events = 0
        self.overshoot_events = 0
        # Touchdown model: one scratch MjData, and each toe cap's hull vertices in its tibia frame.
        self._impact_data = mujoco.MjData(m)
        self._tibia_body, self._toe_hull = [], []
        for leg in LEGS:
            geom = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, leg + "_tibia_toe")
            mesh = m.geom_dataid[geom]
            vertices = m.mesh_vert[
                m.mesh_vertadr[mesh] : m.mesh_vertadr[mesh] + m.mesh_vertnum[mesh]
            ].astype(np.float64)
            rotation = np.zeros(9)
            mujoco.mju_quat2Mat(rotation, m.geom_quat[geom])
            self._tibia_body.append(int(m.geom_bodyid[geom]))
            self._toe_hull.append(m.geom_pos[geom] + vertices @ rotation.reshape(3, 3).T)
        self.sim = _StepCounter(self)
        self.native_errors = []
        self.substep_state = substep_state
        self.history = torch.zeros((cfg.num_envs, 5, 42))
        self.previous_action = torch.zeros((cfg.num_envs, 18))
        self.held = self.neutral.expand(cfg.num_envs, -1).clone()
        if self.randomization is None:
            self._kd_rows = self._motor_scale = self._sag_scale = self._sag_rate = None
            self._delay_queue = self._delay_fill = self._delay_steps = None
            self.randomization_draws = None
        else:
            self._episode_counter = np.zeros(cfg.num_envs, dtype=np.int64)
            self._kd_rows = self.kd.expand(cfg.num_envs, -1).clone()
            self._motor_scale = torch.ones((cfg.num_envs, 1))
            self._sag_scale = torch.ones((cfg.num_envs, 1))
            self._sag_rate = torch.zeros((cfg.num_envs, 1))
            # A row that drew delay d reads the action from d controls back, so the queue keeps
            # the largest delay plus the current control.
            depth = self.randomization.delay_range()[1] + 1
            queued = self.randomization.delay_range()[1] > 0
            self._delay_queue = torch.zeros((cfg.num_envs, depth, 18)) if queued else None
            self._delay_fill = torch.zeros(cfg.num_envs, dtype=torch.long) if queued else None
            self._delay_steps = torch.zeros(cfg.num_envs, dtype=torch.long) if queued else None
            self.randomization_draws = {
                "friction_scale": np.ones(cfg.num_envs),
                "mass_scale": np.ones(cfg.num_envs),
                "motor_scale": np.ones(cfg.num_envs),
                "kd_scale": np.ones(cfg.num_envs),
                "action_delay_controls": np.zeros(cfg.num_envs),
                "sag_per_minute": np.zeros(cfg.num_envs),
            }
        self.commands = (
            self.torch([cfg.command_forward_mps, cfg.command_left_mps, cfg.command_yaw_rad_s])
            .expand(cfg.num_envs, -1)
            .clone()
        )
        self.episode_steps = torch.zeros(cfg.num_envs, dtype=torch.long)
        self.total_controls = 0
        self.telemetry = {}
        self.capture = None
        self.contact_detail = {}
        self._forces = torch.zeros((cfg.num_envs, 19, 3))
        self._toe_cap = torch.zeros((cfg.num_envs, 6))
        self._shaft = torch.zeros((cfg.num_envs, 6))
        self._touching = np.zeros((cfg.num_envs, 25), dtype=bool)
        self.contact = _ContactView(self)
        self.sensor_map = [
            (e, name) for e in range(cfg.num_envs) for name in self.native_body_names
        ]
        self.native_readback = {
            "native_joint_names": self.native_joint_names,
            "native_body_names": self.native_body_names,
            "canonical_joint_names": self.joint_names,
            "root_paths": self.roots,
            "mass_per_replica_kg": [float(m.body_mass.sum())] * cfg.num_envs,
            "limits": [limits.tolist()] * cfg.num_envs,
            "native_max_velocity": [[MAX_JOINT_SPEED] * 18] * cfg.num_envs,
            "implicit_drive_and_armature_zero": True,
            "asset_mass_target_kg": MASS_KG,
            "surrogate": {
                "engine": "mujoco " + mujoco.__version__,
                "contact_model": asdict(self.contact_model),
                "randomization": None if self.randomization is None else asdict(self.randomization),
            },
            "config": cfg.declaration(),
        }
        if self.output is not None:
            self.save("native_readback.json", self.native_readback)
        self.reset()

    # ----- helpers -------------------------------------------------------------------------
    def save(self, name, value):
        (self.output / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")

    def verify_native_recipe(self, phase):
        """Native reads PhysX attributes here. The surrogate checks its own fixed recipe."""
        m = self.mj_model
        if m.opt.timestep != 0.0025 or np.any(m.dof_damping != 0) or np.any(m.dof_armature != 0):
            raise ValueError("Surrogate timing or joint recipe differs")
        self.native_readback.setdefault("recipe_readbacks", {})[phase] = {
            "timestep": m.opt.timestep,
            "surrogate": True,
        }

    def _joint_state(self):
        q = torch.from_numpy(self._state[:, 1 + self._qpos_columns].astype(np.float32))
        dq = torch.from_numpy(self._state[:, 26 + self._qvel_columns].astype(np.float32))
        return q, dq

    def _advance(self, applied):
        """One 400 Hz step of every replica with the given float32 joint torques."""
        m = self.mj_model
        self._control[:, 0] = applied.numpy()
        control, spec = self._control, int(mujoco.mjtState.mjSTATE_CTRL)
        if self.applied_wrench is not None:
            wrench = np.zeros((self.num_envs, 1, m.nbody, 6))
            wrench[:, 0, 1] = np.asarray(self.applied_wrench, dtype=float)
            control = np.concatenate((control, wrench.reshape(self.num_envs, 1, -1)), -1)
            spec |= int(mujoco.mjtState.mjSTATE_XFRC_APPLIED)
        before = sum(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number for d in self._datas)
        self._pool.rollout(
            self._models,
            self._datas,
            self._state,
            control,
            control_spec=spec,
            nstep=1,
            initial_warmstart=self._warmstart,
            state=self._next,
            sensordata=self._sensor,
            chunk_size=self._chunk,
            skip_checks=True,
        )
        if sum(d.warning[mujoco.mjtWarning.mjWARN_BADQACC].number for d in self._datas) != before:
            self.native_errors.append(
                {"type": "mujoco_bad_qacc", "physics_steps": self.physics_steps}
            )
            raise FloatingPointError("MuJoCo reported an unstable replica")
        new = self._next[:, 0]
        # Euler: qvel' = qvel + dt qacc, so this is the solver's qacc for the next warm start.
        self._warmstart[:] = (new[:, 26:] - self._state[:, 26:]) / m.opt.timestep
        self._clamp_joint_speed(new)
        blend = self.contact_model.position_blend
        touching = self._sensor[:, 0, 2::6] > 0.0
        if blend != 1.0:
            old = self._state
            euler = new[:, 1:26].copy()
            impact = (
                (touching & ~self._touching).any(-1) if self.contact_model.impact_euler else None
            )
            velocity = old[:, 26:] + blend * (new[:, 26:] - old[:, 26:])
            step = m.opt.timestep
            new[:, 1:4] = old[:, 1:4] + step * velocity[:, :3]
            new[:, 8:26] = old[:, 8:26] + step * velocity[:, 6:]
            # Free-joint orientation: body-frame angular velocity, right multiplication (WXYZ).
            rotation = step * velocity[:, 3:6]
            angle = np.linalg.norm(rotation, axis=-1, keepdims=True)
            scale = np.where(angle > 1e-12, np.sin(angle / 2) / np.maximum(angle, 1e-12), 0.5)
            dw, dv = np.cos(angle / 2)[:, 0], scale * rotation
            w, v = old[:, 4], old[:, 5:8]
            new[:, 4] = w * dw - np.einsum("ij,ij->i", v, dv)
            new[:, 5:8] = w[:, None] * dv + dw[:, None] * v + np.cross(v, dv)
            new[:, 4:8] /= np.linalg.norm(new[:, 4:8], axis=-1, keepdims=True)
            if impact is not None and impact.any():
                new[impact, 1:26] = euler[impact]
        if self.contact_model.touchdown_overshoot:
            # After the position update: the native impact substep moves with its pre-impact velocity.
            events = np.argwhere(touching[:, 13:19] & ~self._touching[:, 13:19])
            for replica, leg in events:
                self._touchdown(int(replica), int(leg), new)
            if len(events):
                # No clamp here: a native contact impulse also leaves a joint above its speed limit
                # for one substep (the native probe records 56.7 rad/s against the 50.27 rad/s limit).
                self.peak_joint_speed = max(
                    self.peak_joint_speed,
                    float(np.abs(new[events[:, 0]][:, 26 + self._qvel_columns]).max()),
                )
        self._touching = touching
        self._state[:] = new
        self.physics_steps += 1
        normal = torch.from_numpy(
            self._sensor[:, 0].reshape(self.num_envs, 25, 6)[:, :, 2].astype(np.float32)
        )
        forces = torch.zeros((self.num_envs, 19, 3))
        forces[:, :13, 2] = normal[:, :13]
        forces[:, 13:, 2] = normal[:, 13:19] + normal[:, 19:]
        self._forces = forces
        self._toe_cap = normal[:, 13:19].clone()
        self._shaft = normal[:, 19:].clone()
        return forces

    def _clamp_joint_speed(self, new):
        speed = new[:, 26 + self._qvel_columns]
        self.peak_joint_speed = max(self.peak_joint_speed, float(np.abs(speed).max()))
        if self.contact_model.joint_speed_clamp:
            clipped = np.clip(speed, -MAX_JOINT_SPEED, MAX_JOINT_SPEED)
            changed = int((clipped != speed).sum())
            if changed:
                new[:, 26 + self._qvel_columns] = clipped
                self.speed_clamp_events += changed

    def _touchdown(self, replica, leg, new):
        """Apply the calibrated friction correction at toe touchdown."""
        m, d, old = self.mj_model, self._impact_data, self._state[replica]
        step, friction = m.opt.timestep, self.contact_model.friction[0]
        d.qpos[:] = old[1:26]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        body = self._tibia_body[leg]
        points = d.xpos[body] + self._toe_hull[leg] @ d.xmat[body].reshape(3, 3).T
        low = points[np.argmin(points[:, 2])]
        jacobian = np.zeros((3, m.nv))
        mujoco.mj_jac(m, d, jacobian, None, low, body)
        before = jacobian @ old[26:]
        self.touchdown_events += 1
        slide = float(np.linalg.norm(before[:2]))
        if before[2] >= 0.0 or slide < 1e-6:
            return
        phase = 1.0 - max(0.0, -low[2]) / (-before[2] * step)
        start, full = self.contact_model.overshoot_phase
        share = min(1.0, (phase - start) / (full - start))
        if share <= 0.0:
            return
        mujoco.mj_makeM(m, d)
        mujoco.mj_factorM(m, d)
        response = np.zeros((3, m.nv))
        mujoco.mj_solveM(m, d, response, jacobian)
        mobility = jacobian @ response.T
        after = jacobian @ new[replica, 26:]
        slot = (13 + leg) * 6 + 2
        normal = self._sensor[replica, 0, slot] * step
        # Slide that a full Coulomb impulse against the incoming slide leaves, and the share of it.
        wanted = share * (before[:2] - mobility[:2, :2] @ (friction * normal * before[:2] / slide))
        reverse = -(wanted @ before[:2]) / slide
        limit = self.contact_model.overshoot_gain * slide
        if reverse > limit:
            wanted *= limit / reverse
        impulse = np.zeros(3)
        impulse[:2] = np.linalg.solve(mobility[:2, :2], wanted - after[:2])
        # The floor keeps the toe from digging in under the added impulse.
        impulse[2] = max(0.0, -(mobility[2, :2] @ impulse[:2]) / mobility[2, 2])
        new[replica, 26:] += response.T @ impulse
        self.overshoot_events += 1

    def _read(self, full=True):
        state = torch.from_numpy(self._state)
        if not bool(torch.isfinite(state).all()):
            raise FloatingPointError("Nonfinite surrogate state")
        q64 = state[:, 1 + self._qpos_columns]
        position = state[:, 1:4] + self._origins64
        quaternion = state[:, [5, 6, 7, 4]]
        quaternion = quaternion / torch.linalg.vector_norm(quaternion, dim=-1, keepdim=True)
        origin_velocity = state[:, 26:29]
        angular = state[:, 29:32]
        angular_world = rotate(quaternion, angular)
        com_velocity = origin_velocity + torch.cross(
            angular_world, rotate(quaternion, self._root_com64.expand_as(position)), dim=-1
        )
        linear = inverse_rotate(quaternion, origin_velocity)
        gravity = inverse_rotate(
            quaternion, torch.tensor([0.0, 0.0, -1.0], dtype=torch.float64).expand_as(position)
        )

        def f32(x):
            return x.to(torch.float32)

        result = {
            "q": f32(q64),
            "dq": f32(state[:, 26 + self._qvel_columns]),
            "root": f32(torch.cat((position, quaternion), dim=-1)),
            "root_velocity": f32(torch.cat((com_velocity, angular_world), dim=-1)),
            "linear": f32(linear),
            "angular": f32(angular),
            "gravity": f32(gravity),
            "toe_cap_force_n": self._toe_cap,
            "shaft_force_n": self._shaft,
        }
        if full:
            angles = q64.reshape(self.num_envs, 6, 3)
            parent_position = position[:, None].expand(-1, 6, -1)
            parent_quaternion = quaternion[:, None].expand(-1, 6, -1)
            poses = [torch.cat((position, quaternion), dim=-1)[:, None]]
            for level in range(3):
                half = angles[:, :, level] / 2
                spin = torch.stack(
                    (
                        torch.zeros_like(half),
                        torch.zeros_like(half),
                        torch.sin(half),
                        torch.cos(half),
                    ),
                    dim=-1,
                )
                parent_position = parent_position + rotate(
                    parent_quaternion, self._joint_xyz[level].expand(self.num_envs, -1, -1)
                )
                parent_quaternion = quaternion_multiply(
                    quaternion_multiply(
                        parent_quaternion, self._joint_quat[level].expand(self.num_envs, -1, -1)
                    ),
                    spin,
                )
                poses.append(torch.cat((parent_position, parent_quaternion), dim=-1))
            toe_world = parent_position + rotate(
                parent_quaternion, self._toe_local64.expand(self.num_envs, -1, -1)
            )
            toe_body = inverse_rotate(
                quaternion[:, None].expand(-1, 6, -1), toe_world - position[:, None]
            )
            result.update(
                link=f32(torch.cat(poses, dim=1)), toe_world=f32(toe_world), toe_body=f32(toe_body)
            )
        return result

    def _proprio(self, state):
        return torch.cat(
            (
                navigation(state["angular"]),
                navigation(state["gravity"]),
                state["q"] - self.neutral,
                state["dq"],
            ),
            dim=-1,
        )

    def observations(self, state):
        obs = torch.cat((self.history.flatten(1), self.commands, self.previous_action), dim=-1)
        critic = torch.cat((obs, navigation(state["linear"])), dim=-1)
        return {"obs": obs, "critic": critic}

    def _randomize(self, rows):
        """Redraw the selected rows' dynamics from their dedicated streams (reset path only).

        Each draw seeds its own generator from (seed, tag, row, episode), so a row's sequence is
        reproducible from the run seed and independent of batch width, threading and the global
        torch and numpy streams.
        """
        r = self.randomization
        for row in rows:
            row = int(row)
            rng = np.random.default_rng(
                (self.cfg.seed, 20811, row, int(self._episode_counter[row]))
            )
            self._episode_counter[row] += 1
            friction, mass, motor, kd, sag = (
                float(rng.uniform(low, high))
                for low, high in (
                    r.friction_scale,
                    r.mass_scale,
                    r.motor_scale,
                    r.kd_scale,
                    r.sag_per_minute,
                )
            )
            low_delay, high_delay = r.delay_range()
            delay = int(rng.integers(low_delay, high_delay + 1)) if high_delay else 0
            model = self._models[row]
            model.body_mass[:] = self._base_body_mass * mass
            model.body_inertia[:] = self._base_body_inertia * mass
            model.geom_friction[:, 0] = self._base_geom_friction[:, 0] * friction
            self._kd_rows[row] = self.kd * kd
            self._motor_scale[row] = motor
            self._sag_scale[row] = 1.0
            self._sag_rate[row] = sag * self.cfg.control_dt / 60.0
            if self._delay_queue is not None:
                self._delay_queue[row] = 0.0
                self._delay_fill[row] = 0
                self._delay_steps[row] = delay
            draws = self.randomization_draws
            draws["friction_scale"][row] = friction
            draws["mass_scale"][row] = mass
            draws["motor_scale"][row] = motor
            draws["kd_scale"][row] = kd
            draws["action_delay_controls"][row] = delay
            draws["sag_per_minute"][row] = sag

    def _randomized_action(self, action):
        """The action the servo sees under the drawn transport delay, advancing the sag drift."""
        self._sag_scale = torch.clamp(
            self._sag_scale - self._sag_rate, min=self.randomization.sag_floor
        )
        if self._delay_queue is None:
            return action
        depth = self._delay_queue.shape[1]
        self._delay_queue = torch.roll(self._delay_queue, -1, dims=1)
        self._delay_queue[:, -1] = action
        # A row younger than its delay reads the newest slot, so an episode starts undelayed.
        take = torch.minimum(self._delay_steps, self._delay_fill)
        self._delay_fill = torch.clamp(self._delay_fill + 1, max=depth)
        return self._delay_queue[torch.arange(self.num_envs), depth - 1 - take]

    # ----- OmniEnv interface ---------------------------------------------------------------
    def reset(self, indices=None):
        if self.native_errors:
            raise RuntimeError("Cannot reset across a physics error")
        indices = (
            torch.arange(self.num_envs)
            if indices is None
            else torch.as_tensor(indices, dtype=torch.long)
        )
        if (
            indices.ndim != 1
            or bool(torch.any(indices < 0))
            or bool(torch.any(indices >= self.num_envs))
            or len(indices.unique()) != len(indices)
        ):
            raise ValueError("Invalid selected reset rows")
        if len(indices):
            rows = indices.numpy()
            self._state[rows, 1:26] = self._qpos0
            if self.spawn is not None:
                x, y, ground, heading = np.asarray(self.spawn(rows), dtype=float).T
                self._state[rows, 1:4] = np.stack((x, y, ground + self.reset_height), -1)
                self._state[rows, 4:8] = np.stack(
                    (np.cos(heading / 2), 0 * heading, 0 * heading, np.sin(heading / 2)), -1
                )
            self._state[rows, 26:] = 0.0
            self._warmstart[rows] = 0.0
            self._forces[indices] = 0.0
            self._toe_cap[indices] = 0.0
            self._shaft[indices] = 0.0
            self._sensor[rows] = 0.0
            self._touching[rows] = False
            if self.randomization is not None:
                self._randomize(rows)
            self.held[indices] = self.neutral
            self.previous_action[indices] = 0
            self.episode_steps[indices] = 0
            state = self._read()
            self.history[indices] = self._proprio(state)[indices, None].expand(-1, 5, -1)
        else:
            state = self._read()
        self.current = state
        return self.observations(state)

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.shape != (self.num_envs, 18) or not bool(torch.isfinite(action).all()):
            raise ValueError("Policy action shape/finite check failed")
        if self.randomization is not None:
            action = self._randomized_action(action)
        target = emitted_target(
            action,
            self.held,
            self.neutral,
            self.lower,
            self.upper,
            self.cfg.action_scale_rad,
            self.cfg.target_slew_rad,
        )
        saturation = torch.zeros_like(target, dtype=torch.long)
        torque_square = torch.zeros_like(target)
        requested_max = torch.zeros_like(target)
        applied_max = torch.zeros_like(target)
        other_force_max = torch.zeros(self.num_envs)
        tibia_min_force = torch.full((self.num_envs, 6), torch.inf)
        toe_cap_min = torch.full((self.num_envs, 6), torch.inf)
        shaft_max = torch.zeros((self.num_envs, 6))
        last = self.cfg.decimation - 1
        for substep in range(self.cfg.decimation):
            q, dq = self._joint_state()
            if not bool(torch.isfinite(q).all() and torch.isfinite(dq).all()):
                raise FloatingPointError("Nonfinite pre-servo state")
            requested, applied, ceiling = motor_force(
                q, dq, target, self.kd if self._kd_rows is None else self._kd_rows
            )
            if self._motor_scale is not None:
                # The drawn motor strength and the in-episode sag drain the ceiling together,
                # reproducing the canonical clip (robots/rs05.py) against the scaled curve.
                ceiling = ceiling * (self._motor_scale * self._sag_scale)
                applied = torch.maximum(torch.minimum(requested, ceiling), -ceiling)
            forces = self._advance(applied)
            # The control endpoint needs link poses. Earlier substeps build a state for a capture
            # hook alone: full with substep_state, otherwise without link and toe poses.
            # A hook that declares uses_state = False (TrainingLoads reads none) gets None.
            if substep == last:
                state = self._read(full=True)
            elif self.capture is not None and getattr(self.capture, "uses_state", True):
                state = self._read(full=self.substep_state)
            else:
                state = None
            norms = torch.linalg.vector_norm(forces, dim=-1)
            other_force_max = torch.maximum(other_force_max, norms[:, self._other_index].amax(-1))
            tibia_min_force = torch.minimum(tibia_min_force, norms[:, self.toe_index])
            toe_cap_min = torch.minimum(toe_cap_min, self._toe_cap)
            shaft_max = torch.maximum(shaft_max, self._shaft)
            saturation += requested.abs() > 1.6
            torque_square += applied.square()
            requested_max = torch.maximum(requested_max, requested.abs())
            applied_max = torch.maximum(applied_max, applied.abs())
            if self.capture is not None:
                self.capture(
                    self,
                    state,
                    q,
                    dq,
                    requested,
                    applied,
                    ceiling,
                    target,
                    forces,
                    substep,
                    applied,
                )
        self.held = target
        self.previous_action = executed_action_feature(
            target, self.neutral, self.cfg.action_scale_rad
        ).clone()
        self.history = torch.cat((self.history[:, 1:], self._proprio(state)[:, None]), dim=1)
        self.episode_steps += 1
        self.total_controls += 1
        velocity_nav = navigation(state["linear"])
        joint_violation = ((state["q"] < self.lower - 2e-6) | (state["q"] > self.upper + 2e-6)).any(
            -1
        )
        terminated = (
            (state["root"][:, 2] < 0.045)
            | (state["gravity"][:, 2] > -math.cos(0.85))
            | joint_violation
        )
        truncated = self.episode_steps >= round(self.cfg.episode_seconds / self.cfg.control_dt)
        self.current = state
        self.contact_detail = {
            "toe_cap_force_n": state["toe_cap_force_n"].clone(),
            "shaft_force_n": state["shaft_force_n"].clone(),
        }
        self.telemetry = {
            "root_pose_xyzw": state["root"].clone(),
            "linear_velocity_body": state["linear"].clone(),
            "linear_velocity_nav": velocity_nav.clone(),
            "angular_velocity_body": state["angular"].clone(),
            "joint_position_rad": state["q"].clone(),
            "joint_velocity_rad_s": state["dq"].clone(),
            "joint_target_rad": target.clone(),
            "computed_torque_nm": requested.clone(),
            "applied_torque_nm": applied.clone(),
            "saturation_count_400hz": saturation,
            "torque_square_sum_400hz": torque_square,
            "requested_torque_abs_max_400hz": requested_max,
            "applied_torque_abs_max_400hz": applied_max,
            "tibia_floor_force_world_n": forces[:, self.toe_index].clone(),
            "tibia_floor_force_min_norm_400hz": tibia_min_force,
            "other_body_force_max_400hz": other_force_max,
            "toe_xyz_world": state["toe_world"].clone(),
            "toe_xyz_body": state["toe_body"].clone(),
            "command": self.commands.clone(),
            "action": action.clone(),
            "terminated": terminated.clone(),
            "truncated": truncated.clone(),
            # Keys that the surrogate adds: the native compact telemetry does not separate toe cap from shaft.
            "surrogate_toe_cap_force_min_400hz": toe_cap_min,
            "surrogate_shaft_force_max_400hz": shaft_max,
        }
        result = self.observations(state)
        result.update(terminated=terminated, truncated=truncated)
        return result

    # ----- surrogate extras ----------------------------------------------------------------
    def local_qpos(self, index=0):
        """MuJoCo qpos of one replica in its own world (no allocation offset)."""
        return self._state[index, 1:26].copy()

    def close(self):
        self._pool.close()


def make_env(
    num_envs=128,
    *,
    episode_seconds=20.0,
    threads=2,
    contact=None,
    record_motion_features=False,
    output=None,
    substep_state=True,
    seed=20260914,
):
    cfg = SurrogateConfig(
        num_envs=num_envs,
        episode_seconds=episode_seconds,
        seed=seed,
        record_motion_features=record_motion_features,
    )
    return SurrogateEnv(
        cfg,
        None,
        DEFAULT_MODEL,
        DEFAULT_GEOMETRY,
        output,
        threads=threads,
        contact=contact,
        substep_state=substep_state,
    )


class Renderer:
    """Offscreen MuJoCo view of one replica (three-quarter or side), tracking the body."""

    def __init__(self, env, *, width=960, height=540, view="three_quarter"):
        self.env = env
        self.data = mujoco.MjData(env.mj_model)
        self.renderer = mujoco.Renderer(env.mj_model, height=height, width=width)
        self.camera = mujoco.MjvCamera()
        self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
        self.camera.distance = 1.05
        self.camera.elevation = -22.0 if view == "three_quarter" else -8.0
        self.camera.azimuth = 135.0 if view == "three_quarter" else 180.0

    def frame(self, index=0):
        self.data.qpos[:] = self.env.local_qpos(index)
        mujoco.mj_forward(self.env.mj_model, self.data)
        self.camera.lookat[:] = (self.data.qpos[0], self.data.qpos[1], 0.05)
        self.renderer.update_scene(self.data, self.camera)
        return self.renderer.render()

    def close(self):
        self.renderer.close()
