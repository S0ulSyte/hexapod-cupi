"""The 50 Hz control loop around ``PolicyLoop``: modes, command validity and sensing checks.

Modes: DISABLED -> enable() -> STANDING_UP (targets ramp from the measured pose to neutral) ->
READY (hold neutral) -> start() -> WALKING (the policy) -> stand() -> STANDING_UP. Any failed check
latches FAULT until disable(). DISABLED and FAULT output damping only (zero stiffness), which
lets the robot settle rather than fight; it is not a substitute for the independent hardware
emergency stop that ARCHITECTURE requires.

Each control returns a ``MotorCommand``: position targets, stiffness, damping and torque limit
per joint in JOINT_NAMES order. Training ran this PD at 400 Hz with zero target velocity:
torque = kp (target - q) - kd dq, clipped at the limit. The motor drivers must run that law.

Commands follow ARCHITECTURE: each carries an envelope check (``contracts.command``) and a time;
an expired command reads as zero (the trained stop), and a non-finite one is refused.
"""

import math
from dataclasses import dataclass
from enum import Enum

import numpy as np

from contracts.command import CommandEnvelope
from locomotion.runtime.inputs import PolicyLoop


class Mode(Enum):
    DISABLED = "disabled"
    STANDING_UP = "standing_up"
    READY = "ready"
    WALKING = "walking"
    FAULT = "fault"


@dataclass(frozen=True)
class SafetyLimits:
    """Thresholds of the per-control checks; the defaults follow the simulator's terminations."""

    command_timeout_s: float = 0.25
    max_control_gap_s: float = 0.06
    joint_limit_margin_rad: float = 0.05
    tilt_limit_rad: float | None = None  # None reads the bundle's (the training termination)
    stand_rate_rad: float = 0.01  # per control: 0.5 rad/s
    stand_tolerance_rad: float = 0.05
    stand_settle_controls: int = 25


@dataclass(frozen=True)
class MotorCommand:
    position: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    torque_limit: np.ndarray
    mode: Mode


class Controller:
    def __init__(self, bundle, envelope: CommandEnvelope, limits=SafetyLimits(), ghost="odometry"):
        self.bundle, self.envelope, self.limits = bundle, envelope, limits
        self.loop = PolicyLoop(bundle, ghost=ghost)
        tilt = limits.tilt_limit_rad
        self.tilt = bundle.inputs["tilt_limit_rad"] if tilt is None else tilt
        self.mode = Mode.DISABLED
        self.fault = None
        self.target = bundle.neutral.copy()
        self.command, self.command_time = np.zeros(3), -math.inf
        self.last_time = None
        self.settled = 0

    # ----- operator transitions ---------------------------------------------------------
    def enable(self):
        if self.mode is not Mode.DISABLED:
            raise RuntimeError(f"enable() needs DISABLED, not {self.mode.value}")
        self.mode, self.target, self.settled = Mode.STANDING_UP, None, 0

    def start(self):
        if self.mode is not Mode.READY:
            raise RuntimeError(f"start() needs READY, not {self.mode.value}")
        self.loop.restart()
        self.mode = Mode.WALKING

    def stand(self):
        """Leave the policy and ramp back to neutral (send a zero command first to stop), so the
        next start() begins where training episodes begin."""
        if self.mode is Mode.WALKING:
            self.mode, self.settled = Mode.STANDING_UP, 0

    def disable(self):
        self.mode, self.fault = Mode.DISABLED, None

    def set_command(self, forward_mps, left_mps, yaw_rate_rad_s, time_s):
        """Accept a command at ``time_s``, clamped into the envelope; a non-finite one is refused."""
        values = (float(forward_mps), float(left_mps), float(yaw_rate_rad_s))
        if not all(math.isfinite(v) for v in values):
            raise ValueError("Non-finite command refused")
        bounds = (
            self.envelope.forward_mps,
            self.envelope.lateral_mps,
            self.envelope.yaw_rate_rad_s,
        )
        stand = values == (0.0, 0.0, 0.0) and self.envelope.allow_stand
        self.command = np.array(
            values if stand else [min(max(v, lo), hi) for v, (lo, hi) in zip(values, bounds)]
        )
        self.command_time = float(time_s)

    # ----- the control ------------------------------------------------------------------
    def _check(self, reading, time_s):
        b = self.bundle
        arrays = (
            reading.gyro_body,
            reading.gravity_body,
            reading.joint_position,
            reading.joint_velocity,
        )
        shapes = ((3,), (3,), (18,), (18,))
        for value, shape in zip(arrays, shapes):
            value = np.asarray(value, dtype=np.float64)
            if value.shape != shape or not np.all(np.isfinite(value)):
                return "sensor reading missing, misshaped or non-finite"
        if (
            self.last_time is not None
            and not 0 < time_s - self.last_time <= self.limits.max_control_gap_s
        ):
            return f"control gap {time_s - self.last_time:.3f} s outside (0, {self.limits.max_control_gap_s}]"
        gravity = np.asarray(reading.gravity_body, dtype=np.float64)
        if not 0.5 < np.linalg.norm(gravity) < 1.5:
            return "gravity direction is not a unit vector"
        if self.mode in (Mode.READY, Mode.WALKING) and gravity[2] / np.linalg.norm(
            gravity
        ) > -math.cos(self.tilt):
            return "body tilt beyond the training termination"
        q = np.asarray(reading.joint_position, dtype=np.float64)
        margin = self.limits.joint_limit_margin_rad
        if self.mode in (Mode.READY, Mode.WALKING) and (
            np.any(q < b.lower - margin) or np.any(q > b.upper + margin)
        ):
            return "joint position beyond its limit"
        if np.any(np.abs(reading.joint_velocity) > b.robot["max_joint_speed_rad_s"]):
            return "joint speed beyond the motor limit"
        return None

    def _damping(self, reading):
        q = np.asarray(reading.joint_position, dtype=np.float64)
        position = np.where(np.isfinite(q), q, self.bundle.neutral)
        b = self.bundle
        return MotorCommand(position, np.zeros(18), b.kd.copy(), b.torque_limit.copy(), self.mode)

    def step(self, reading, time_s):
        """One control at ``time_s`` (seconds, monotonic)."""
        b = self.bundle
        problem = self._check(reading, time_s)
        self.last_time = time_s
        if problem is not None and self.mode is not Mode.DISABLED:
            self.mode, self.fault = Mode.FAULT, problem
        if self.mode in (Mode.DISABLED, Mode.FAULT):
            return self._damping(reading)
        q = np.asarray(reading.joint_position, dtype=np.float64)
        if self.mode is Mode.STANDING_UP:
            start = np.clip(q, b.lower, b.upper) if self.target is None else self.target
            rate = self.limits.stand_rate_rad
            self.target = start + np.clip(b.neutral - start, -rate, rate)
            close = np.all(np.abs(self.target - b.neutral) < 1e-12) and np.all(
                np.abs(q - b.neutral) < self.limits.stand_tolerance_rad
            )
            self.settled = self.settled + 1 if close else 0
            if self.settled >= self.limits.stand_settle_controls:
                self.mode = Mode.READY
        elif self.mode is Mode.WALKING:
            expired = time_s - self.command_time > self.limits.command_timeout_s
            command = np.zeros(3) if expired else self.command
            self.target = self.loop.step(reading, command)
        return MotorCommand(
            self.target.copy(), b.kp.copy(), b.kd.copy(), b.torque_limit.copy(), self.mode
        )
