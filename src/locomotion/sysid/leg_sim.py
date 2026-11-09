"""Replay a stand recording on a one-leg MuJoCo model built from the approved robot assets.

The model is the lf chain of ``robot/active_model.json``'s model (masses, centres of mass, inertias and
joint frames of body -> lf_coxa -> lf_femur -> lf_tibia), with the body fixed and level as the stand
holds it and gravity along -z. Each joint is driven by the recorded motor commands through the
policy's motor law, torque = kp (target - q) - kd dq, evaluated at 400 Hz like training's servo, with
these parameters per joint (``LegParams``):

- ``sign``, ``offset``: the joint mapping, q_urdf = sign (q_motor - offset);
- ``delay_s``: end-to-end actuation delay; the motor acts on the command sent ``delay_s`` earlier;
- ``kp_scale``, ``kd_scale``: the motor's realised stiffness and damping over the commanded ones;
- ``tau_max``: torque ceiling (Nm); ``frictionloss`` (Nm) and ``damping`` (Nm s/rad): joint friction;
- ``armature``: reflected rotor inertia (kg m^2), which the training model leaves at zero.
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


def active_model_path():
    active = json.loads((ROOT / "robot/active_model.json").read_text())
    return ROOT / active["model"]["path"]


def _vec(values):
    return " ".join(f"{float(v):.10g}" for v in values)


def build_xml(model=None):
    """MJCF for the fixed-base lf leg from the approved model json."""
    model = json.loads(active_model_path().read_text()) if model is None else model
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

    body = inertial("body")
    chain = ""
    for name, link in zip(reversed(LEG), reversed(LINKS)):
        j = joints[name]
        x, y, z, w = j["quaternion_xyzw"]
        chain = (
            f'<body name="{link}" pos="{_vec(j["xyz"])}" quat="{_vec((w, x, y, z))}">'
            f'<joint name="{name}" type="hinge" axis="0 0 1" limited="false"/>{inertial(link)}{chain}</body>'
        )
    return (
        '<mujoco model="lf_leg_stand"><compiler angle="radian"/>'
        f'<option timestep="{PHYSICS_DT}" gravity="0 0 -9.81" integrator="implicitfast"/>'
        f'<worldbody><body name="body">{body}{chain}</body></worldbody></mujoco>'
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

    def copy(self, **changes):
        fresh = {k: np.array(v, dtype=float) for k, v in self.__dict__.items() if k != "delay_s"}
        return replace(self, **{"delay_s": self.delay_s, **fresh, **changes})

    def to_json(self):
        return {
            k: (float(v) if np.isscalar(v) else [float(x) for x in v])
            for k, v in self.__dict__.items()
        }

    @classmethod
    def from_json(cls, d):
        return cls(
            **{
                k: (float(v) if k == "delay_s" else np.asarray(v, dtype=float))
                for k, v in d.items()
            }
        )


def load_recording(stem):
    """Per motor: command times, targets, gains; feedback times, positions, velocities, torques."""
    stem = str(stem).removesuffix(".npz").removesuffix(".json")
    data = dict(np.load(stem + ".npz"))
    meta = json.loads(Path(stem + ".json").read_text())
    rec = {"meta": meta, "motors": {}}
    for m in MOTORS:
        rows = data["motor"] == m
        order = np.argsort(data["t_send"][rows])
        rec["motors"][m] = {k: data[k][rows][order] for k in data if k != "motor"}
    return rec


class LegSim:
    def __init__(self, model=None):
        import mujoco

        self.mujoco = mujoco
        self.model = mujoco.MjModel.from_xml_string(build_xml(model))
        self.data = mujoco.MjData(self.model)

    def gravity_torque(self, q_urdf):
        """Joint torques (URDF frame) that hold the leg static at ``q_urdf``."""
        d = self.data
        d.qpos[:] = q_urdf
        d.qvel[:] = 0
        self.mujoco.mj_forward(self.model, d)
        return d.qfrc_bias.copy()

    def replay(self, rec, params, t0=None, t1=None, q0_motor=None):
        """Simulate the recorded commands over [t0, t1] (recording seconds); returns, per motor, the
        simulated position (motor frame) and motor torque at that motor's feedback times."""
        mj, model, d = self.mujoco, self.model, self.data
        motors = rec["motors"]
        t0 = min(motors[m]["t_send"][0] for m in MOTORS) if t0 is None else t0
        t1 = max(motors[m]["t_read"][-1] for m in MOTORS) if t1 is None else t1
        model.dof_armature[:] = params.armature
        model.dof_frictionloss[:] = params.frictionloss
        model.dof_damping[:] = params.damping
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
        mj.mj_resetData(model, d)
        d.qpos[:] = sign * (np.asarray(q0_motor) - offset)
        d.qvel[:] = 0
        q_motor = np.empty((steps, 3))
        tau_motor = np.empty((steps, 3))
        for k in range(steps):
            qm = sign * d.qpos + offset
            dqm = sign * d.qvel
            tau = np.clip(kp[k] * (target[k] - qm) - kd[k] * dqm, -tau_max, tau_max)
            q_motor[k], tau_motor[k] = qm, tau
            d.qfrc_applied[:] = sign * tau
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
        return out
