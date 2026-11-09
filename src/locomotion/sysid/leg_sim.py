"""Replay a stand recording on a one-leg MuJoCo model built from the approved robot assets.

The leg is the lf chain of ``robot/active_model.json``'s model (masses, centres of mass, inertias and
joint frames of lf_coxa -> lf_femur -> lf_tibia), mounted level as on the robot body, gravity along -z.
Two fixtures (docs/LEG_STAND_HARDWARE.md, after the 2026-09-04 stand specification):

- ``clamped``: the mount is fixed, the leg hangs free (the carriage resting on a support);
- ``rail``: the mount rides a carriage on a passive vertical rail (a slide joint), and the toe, a
  sphere at the model's toe point, touches a floor plane. The carriage mass and rail friction are
  parameters, because the stand CAD (HEXAPOD_MKII_STAND_ASSY) is not in the repository.

Each joint is driven by the recorded motor commands through the policy's motor law,
torque = kp (target - q) - kd dq, at 400 Hz like training's servo, with these parameters
(``LegParams``):

- ``sign``, ``offset``: the joint mapping, q_urdf = sign (q_motor - offset);
- ``delay_s``: end-to-end actuation delay; the motor acts on the command sent ``delay_s`` earlier;
- ``kp_scale``, ``kd_scale``: the motor's realised stiffness and damping over the commanded ones;
- ``tau_max``: torque ceiling (Nm); ``frictionloss`` (Nm) and ``damping`` (Nm s/rad): joint friction;
- ``armature``: reflected rotor inertia (kg m^2), which the training model leaves at zero;
- ``carriage_mass`` (kg), ``rail_friction`` (N), ``toe_friction``: the rail fixture.
"""

import json
from dataclasses import dataclass, field, replace
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
LEG = ("lf_coxa_yaw", "lf_femur_pitch", "lf_tibia_pitch")
LINKS = ("lf_coxa", "lf_femur", "lf_tibia")
MOTORS = (11, 12, 13)
PHYSICS_DT = 0.0025
TOE_RADIUS_M = 0.012
SETTLE_S = 0.6  # rail fixture: hold the first command this long so the carriage settles on the toe


def active_model_path():
    active = json.loads((ROOT / "robot/active_model.json").read_text())
    return ROOT / active["model"]["path"]


def stance_path():
    return active_model_path().parent / "stance.json"


def _vec(values):
    return " ".join(f"{float(v):.10g}" for v in values)


def build_xml(model=None, fixture="clamped"):
    """MJCF for the lf leg on the stand from the approved model json."""
    if fixture not in ("clamped", "rail", "floor"):
        raise ValueError("fixture is clamped, rail or floor")
    model = json.loads(active_model_path().read_text()) if model is None else model
    toe = json.loads(stance_path().read_text())["toe_local_points_m"][0]
    joints = {j["name"]: j for j in model["joints"]}
    links = {x["name"]: x for x in model["links"]}
    for name in LEG:
        if list(joints[name]["axis"]) != [0.0, 0.0, 1.0]:
            raise ValueError(f"{name}: the leg model assumes a +Z joint axis")

    def inertial(link):
        x = links[link]
        i = np.asarray(x["inertia"], dtype=float)
        full = (i[0, 0], i[1, 1], i[2, 2], i[0, 1], i[0, 2], i[1, 2])
        return (
            f'<inertial pos="{_vec(x["com"])}" mass="{x["mass"]:.10g}" fullinertia="{_vec(full)}"/>'
        )

    toe_geom = (
        f'<geom name="toe" type="sphere" size="{TOE_RADIUS_M}" pos="{_vec(toe)}" '
        f'contype="{0 if fixture == "clamped" else 1}" conaffinity="0" friction="1 0.005 0.0001"/>'
    )
    chain = ""
    for name, link in zip(reversed(LEG), reversed(LINKS)):
        j = joints[name]
        x, y, z, w = j["quaternion_xyzw"]
        tip = toe_geom if link == "lf_tibia" else ""
        chain = (
            f'<body name="{link}" pos="{_vec(j["xyz"])}" quat="{_vec((w, x, y, z))}">'
            f'<joint name="{name}" type="hinge" axis="0 0 1" limited="false"/>'
            f"{inertial(link)}{tip}{chain}</body>"
        )
    if fixture == "rail":
        mount = (
            '<body name="carriage" pos="0 0 0.3">'
            '<joint name="rail" type="slide" axis="0 0 1" limited="false"/>'
            '<inertial pos="0 0 0" mass="1" diaginertia="1e-3 1e-3 1e-3"/>'
            # The mount plate collides with the floor, so a carriage the leg cannot hold comes to rest
            # on the floor instead of passing through it.
            '<geom name="mount" type="box" size="0.06 0.06 0.01" contype="1" conaffinity="0"/>'
            f"{chain}</body>"
        )
        floor = '<geom name="floor" type="plane" size="1 1 0.1" contype="0" conaffinity="1"/>'
    elif fixture == "floor":
        # The mount is held still (the carriage on its stop) while the toe presses on a floor whose
        # height below the mount is ``LegParams.floor_height`` (set per replay).
        mount = f'<body name="carriage">{chain}</body>'
        floor = '<geom name="floor" type="plane" size="1 1 0.1" pos="0 0 -0.1" contype="0" conaffinity="1"/>'
    else:
        mount = f'<body name="carriage">{chain}</body>'
        floor = ""
    return (
        '<mujoco model="lf_leg_stand"><compiler angle="radian"/>'
        f'<option timestep="{PHYSICS_DT}" gravity="0 0 -9.81" integrator="implicitfast"/>'
        f"<worldbody>{floor}{mount}</worldbody></mujoco>"
    )


def _three(value):
    return np.full(3, float(value))


@dataclass
class LegParams:
    delay_s: float = 0.0
    sign: np.ndarray = field(default_factory=lambda: np.ones(3))
    offset: np.ndarray = field(default_factory=lambda: np.zeros(3))
    kp_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    kd_scale: np.ndarray = field(default_factory=lambda: np.ones(3))
    tau_max: np.ndarray = field(default_factory=lambda: _three(1.6))
    frictionloss: np.ndarray = field(default_factory=lambda: np.zeros(3))
    damping: np.ndarray = field(default_factory=lambda: np.zeros(3))
    armature: np.ndarray = field(default_factory=lambda: np.zeros(3))
    carriage_mass: float = 1.0
    rail_friction: float = 0.0
    toe_friction: float = 1.0
    floor_height: float = -0.1

    SCALARS = ("delay_s", "carriage_mass", "rail_friction", "toe_friction", "floor_height")

    def copy(self, **changes):
        fresh = {
            k: (v if k in self.SCALARS else np.array(v, dtype=float))
            for k, v in self.__dict__.items()
        }
        return replace(self, **{**fresh, **changes})

    def to_json(self):
        return {
            k: (float(v) if np.isscalar(v) else [float(x) for x in v])
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_json(cls, d):
        return cls(
            **{
                k: (float(v) if k in cls.SCALARS else np.asarray(v, dtype=float))
                for k, v in d.items()
            }
        )


def load_recording(stem):
    """Per motor: command times, targets, gains; feedback times, positions, velocities, torques."""
    stem = str(stem).removesuffix(".npz").removesuffix(".json")
    data = dict(np.load(stem + ".npz"))
    meta = json.loads(Path(stem + ".json").read_text())
    rec = {"meta": meta, "motors": {}, "stem": stem}
    for m in MOTORS:
        rows = data["motor"] == m
        order = np.argsort(data["t_send"][rows])
        rec["motors"][m] = {k: data[k][rows][order] for k in data if k != "motor"}
    return rec


class LegSim:
    def __init__(self, model=None, fixture="clamped"):
        import mujoco

        self.mujoco, self.fixture = mujoco, fixture
        self.model = mujoco.MjModel.from_xml_string(build_xml(model, fixture))
        self.data = mujoco.MjData(self.model)
        m = self.model
        ids = [mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in LEG]
        self.qi = np.array([m.jnt_qposadr[i] for i in ids])
        self.vi = np.array([m.jnt_dofadr[i] for i in ids])
        self.rail = fixture == "rail"
        self.carriage = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, "carriage")
        if self.rail:
            rail = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, "rail")
            self.zq, self.zv = m.jnt_qposadr[rail], m.jnt_dofadr[rail]
        self.toe = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "toe")
        self.floor = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor")

    def gravity_torque(self, q_urdf):
        """Joint torques (URDF frame) that hold the leg static at ``q_urdf`` (mount fixed)."""
        d = self.data
        d.qpos[:] = 0
        d.qpos[self.qi] = q_urdf
        d.qvel[:] = 0
        self.mujoco.mj_forward(self.model, d)
        return d.qfrc_bias[self.vi].copy()

    def toe_height_below_mount(self, q_urdf):
        """Height of the toe sphere's bottom relative to the mount origin (negative: below)."""
        d = self.data
        d.qpos[:] = 0
        d.qpos[self.qi] = q_urdf
        self.mujoco.mj_kinematics(self.model, d)
        mount_z = d.xpos[self.carriage][2]
        return float(d.geom_xpos[self.toe][2] - mount_z - TOE_RADIUS_M)

    def _set_params(self, params):
        m = self.model
        m.dof_armature[self.vi] = params.armature
        m.dof_frictionloss[self.vi] = params.frictionloss
        m.dof_damping[self.vi] = params.damping
        if self.rail:
            m.body_mass[self.carriage] = params.carriage_mass
            m.dof_frictionloss[self.zv] = params.rail_friction
        if self.fixture == "floor":
            m.geom_pos[self.floor, 2] = params.floor_height
        m.geom_friction[self.toe, 0] = params.toe_friction

    def replay(self, rec, params, t0=None, t1=None, q0_motor=None):
        """Simulate the recorded commands over [t0, t1] (recording seconds). Returns, per motor, the
        simulated position (motor frame) and motor torque at that motor's feedback times, and under
        ``"stand"`` the 400 Hz grid, the mount height above the floor and the toe contact."""
        mj, model, d = self.mujoco, self.model, self.data
        motors = rec["motors"]
        t0 = min(motors[m]["t_send"][0] for m in MOTORS) if t0 is None else t0
        t1 = max(motors[m]["t_read"][-1] for m in MOTORS) if t1 is None else t1
        self._set_params(params)
        steps = int(np.ceil((t1 - t0) / PHYSICS_DT)) + 1
        grid = t0 + PHYSICS_DT * np.arange(steps)
        target = np.zeros((steps, 3))
        kp = np.zeros((steps, 3))
        kd = np.zeros((steps, 3))
        for i, m in enumerate(MOTORS):
            c = motors[m]
            idx = np.searchsorted(c["t_send"], grid - params.delay_s, side="right") - 1
            live = idx >= 0
            safe = np.clip(idx, 0, None)
            target[:, i] = np.where(live, c["target"][safe], np.nan)
            kp[:, i] = np.where(live, c["kp"][safe], 0.0) * params.kp_scale[i]
            kd[:, i] = np.where(live, c["kd"][safe], 0.0) * params.kd_scale[i]
        if q0_motor is None:
            q0_motor = np.array(
                [np.interp(t0, motors[m]["t_read"], motors[m]["pos"]) for m in MOTORS]
            )
        target = np.where(np.isnan(target), q0_motor, target)
        sign, offset, tau_max = params.sign, params.offset, params.tau_max
        q0 = sign * (np.asarray(q0_motor) - offset)
        mj.mj_resetData(model, d)
        d.qpos[self.qi] = q0
        if self.rail:
            # Start with the toe just touching the floor, then let the carriage settle on it under the
            # first command before the replay clock starts.
            d.qpos[self.zq] = -self.toe_height_below_mount(q0)
            d.qpos[self.qi] = q0
            d.qvel[:] = 0
            live = np.flatnonzero(kp.any(1))
            k0 = live[0] if len(live) else 0
            hold_target, hold_kp, hold_kd = target[k0], kp[k0], kd[k0]
            for _ in range(int(SETTLE_S / PHYSICS_DT)):
                qm = sign * d.qpos[self.qi] + offset
                dqm = sign * d.qvel[self.vi]
                tau = np.clip(hold_kp * (hold_target - qm) - hold_kd * dqm, -tau_max, tau_max)
                d.qfrc_applied[self.vi] = sign * tau
                mj.mj_step(model, d)
            d.qpos[self.qi] = (
                q0  # joints restart at the measured pose; the carriage keeps its settle
            )
            d.qvel[self.vi] = 0
        q_motor = np.empty((steps, 3))
        tau_motor = np.empty((steps, 3))
        height = np.zeros(steps)
        contact = np.zeros(steps, dtype=bool)
        for k in range(steps):
            qm = sign * d.qpos[self.qi] + offset
            dqm = sign * d.qvel[self.vi]
            tau = np.clip(kp[k] * (target[k] - qm) - kd[k] * dqm, -tau_max, tau_max)
            q_motor[k], tau_motor[k] = qm, tau
            if self.rail:
                height[k] = d.qpos[self.zq]
            if self.fixture != "clamped":
                contact[k] = any(
                    d.contact[c].geom1 == self.toe or d.contact[c].geom2 == self.toe
                    for c in range(d.ncon)
                )
            d.qfrc_applied[self.vi] = sign * tau
            mj.mj_step(model, d)
        out = {}
        for i, m in enumerate(MOTORS):
            t = motors[m]["t_read"]
            keep = (t >= t0) & (t <= t1)
            out[m] = {
                "t": t[keep],
                "pos": np.interp(t[keep], grid, q_motor[:, i]),
                "torque": np.interp(t[keep], grid, tau_motor[:, i]),
            }
        out["stand"] = {"t": grid, "height": height, "contact": contact, "q_motor": q_motor}
        return out
