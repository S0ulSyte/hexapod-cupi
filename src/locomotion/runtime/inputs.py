"""Rebuild the 278 ``terrain_v2_odom`` actor inputs from real sensors and turn actions into targets.

This is the evaluator's policy path (``surrogate/evaluate.py`` ``DeterministicPolicy`` with
``OdomTerrainTracker``) for one robot, written against sensor readings instead of simulator
state. Per control, before the action:

- five 42-value proprioception frames (body rates and projected gravity in navigation axes,
  joint offsets from neutral, joint rates), oldest first, filled with the first reading;
- the command (forward m/s, left m/s, yaw rate rad/s) and the executed-target feature;
- the speed-scaled gait clock (``mdp/gait.py``), zero under a zero command;
- the nine ``ghost_v11`` values from leg odometry (``mdp/odometry.py``): stance toes from joint
  forward kinematics and foot contact, heading from the IMU. The ghost restarts at the odometry
  pose on every command change, so only heading and position CHANGES matter: the odometry
  frame starts at zero and needs no GPS or absolute heading;
- 36 terrain values, zero on flat ground (the native and flat suites run them zeroed).

Then the actor, the two-control action mean and the 0.040 rad slew-limited held target
(``mdp/actions.py``). The replay test checks this module against the evaluator.
"""

from dataclasses import dataclass, field

import numpy as np

from locomotion.runtime.bundle import actor


def navigation(vector):
    """Body (x, y, z) to navigation axes (forward = -body y, left = body x, up = body z)."""
    return np.array([-vector[1], vector[0], vector[2]])


def _quaternion_multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _rotate(quaternion, vector):
    axis, w = quaternion[:3], quaternion[3]
    t = 2 * np.cross(axis, vector)
    return vector + w * t + np.cross(axis, t)


def toe_positions_body(bundle, joint_position):
    """[6, 3] toe positions in the body frame by joint forward kinematics (the simulator's chain)."""
    angles = np.asarray(joint_position, dtype=np.float64).reshape(6, 3)
    toes = np.empty((6, 3))
    for leg in range(6):
        position, quaternion = np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0])
        for level in range(3):
            position = position + _rotate(quaternion, bundle.joint_xyz[level, leg])
            half = angles[leg, level] / 2
            spin = np.array([0.0, 0.0, np.sin(half), np.cos(half)])
            quaternion = _quaternion_multiply(
                _quaternion_multiply(quaternion, bundle.joint_quaternion[level, leg]), spin
            )
        toes[leg] = position + _rotate(quaternion, bundle.toe_local[leg])
    return toes


def _rotation(theta, points):
    c, s = np.cos(theta), np.sin(theta)
    return np.stack(
        (c * points[..., 0] - s * points[..., 1], s * points[..., 0] + c * points[..., 1]), -1
    )


def advance_destination(destination, command, dt):
    """One control of the command from a planar pose (x, y, heading); navigation is body (-y, x)."""
    body = np.array([command[1], -command[0]])
    middle = destination[2] + command[2] * dt / 2
    step = _rotation(middle, body) * dt
    return np.array(
        [destination[0] + step[0], destination[1] + step[1], destination[2] + command[2] * dt]
    )


def ghost_features(constants, pose, previous, destination, elapsed, command):
    """The nine ``ghost_v11`` values for planar poses (x, y, heading) now and one control earlier."""
    c, s = np.cos(pose[2]), np.sin(pose[2])

    def nav(vector):
        return np.array([-(-s * vector[0] + c * vector[1]), c * vector[0] + s * vector[1]])

    gap = nav(destination[:2] - pose[:2])
    turn = destination[2] - pose[2]
    ghost = _rotation(turn, np.asarray(command[:2], dtype=np.float64))
    speed = float(np.hypot(*ghost))
    along = (gap @ (ghost / max(speed, 1e-9))) * ghost / max(speed, 1e-9) if speed > 0 else 0 * gap
    side = gap - along
    velocity = nav((pose[:2] - previous[:2]) / constants["control_dt_s"])
    distance, speed_scale = constants["distance_scale_m"], constants["velocity_scale_mps"]
    return np.concatenate(
        (
            side / distance,
            [np.sin(turn), np.cos(turn)],
            [elapsed / constants["longest_command_controls"]],
            (velocity - ghost) / speed_scale,
            along / distance,
        )
    )


class GaitClock:
    """The speed-scaled gait phase in cycles: zero at the first control, then one control over the
    cycle of the command each control ran under (``GaitPhase.follow``)."""

    def __init__(self, constants):
        self.speeds = np.asarray(constants["gait_speeds_mps"], dtype=np.float64)
        self.periods = np.asarray(constants["gait_periods_controls"], dtype=np.float64)
        self.swings = np.asarray(constants["gait_swing_share"], dtype=np.float64)
        self.phase, self.command = 0.0, np.zeros(3)

    def cycle(self, command):
        speed = float(np.hypot(command[0], command[1]))
        speed = float(np.clip(np.round(speed * 1e6) / 1e6, self.speeds[0], self.speeds[-1]))
        upper = int(np.clip(np.searchsorted(self.speeds, speed), 1, len(self.speeds) - 1))
        weight = (speed - self.speeds[upper - 1]) / (self.speeds[upper] - self.speeds[upper - 1])

        def between(values):
            return values[upper - 1] + weight * (values[upper] - values[upper - 1])

        return between(self.periods), between(self.swings)

    def follow(self, command, first):
        period, _ = self.cycle(self.command)
        self.phase = 0.0 if first else float(np.remainder(self.phase + 1 / period, 1.0))
        self.command = np.array(command, dtype=np.float64)
        return self.phase


class LegOdometry:
    """Planar dead reckoning from stance toes plus the command-integrated ghost (``OdometryGhost``)."""

    def __init__(self, constants):
        self.constants = constants
        self.xy, self.theta = np.zeros(2), 0.0
        self.previous_xy, self.previous_theta = np.zeros(2), 0.0
        self.destination = np.zeros(3)
        self.prev_toes, self.prev_contact = np.zeros((6, 2)), np.ones(6, dtype=bool)
        self.prev_heading = 0.0

    def reset(self, toes, contact, heading):
        """Start a frame at the body: only changes in position and heading reach the features."""
        self.xy, self.theta = np.zeros(2), 0.0
        self.previous_xy, self.previous_theta = np.zeros(2), 0.0
        self.destination = np.zeros(3)
        self.prev_toes = np.array(toes, dtype=np.float64)
        self.prev_contact = np.ones(6, dtype=bool) if contact is None else np.array(contact, bool)
        self.prev_heading = float(heading)

    def advance(self, toes, contact, heading, command, fresh):
        self.previous_xy, self.previous_theta = self.xy.copy(), self.theta
        delta = float(np.remainder(heading - self.prev_heading + np.pi, 2 * np.pi) - np.pi)
        self.prev_heading = float(heading)
        theta_prev = self.theta
        self.theta = self.theta + delta
        contact = self.prev_contact if contact is None else np.array(contact, dtype=bool)
        stance = contact & self.prev_contact
        steps = _rotation(theta_prev, self.prev_toes) - _rotation(self.theta, toes)
        self.xy = self.xy + steps[stance].sum(0) / max(float(stance.sum()), 1.0)
        self.prev_toes, self.prev_contact = np.array(toes, dtype=np.float64), contact
        if fresh:
            self.destination = np.array([self.xy[0], self.xy[1], self.theta])
        self.destination = advance_destination(
            self.destination, command, self.constants["control_dt_s"]
        )

    def features(self, fresh, elapsed, command):
        here = np.array([self.xy[0], self.xy[1], self.theta])
        before = np.array([self.previous_xy[0], self.previous_xy[1], self.previous_theta])
        destination = here if fresh else self.destination
        return ghost_features(
            self.constants, here, before, destination, 0 if fresh else elapsed, command
        )


@dataclass
class Reading:
    """One control's sensor reading in the canonical body frame and JOINT_NAMES order.

    ``gravity_body`` is the unit gravity direction in the body frame (from the IMU attitude;
    level standing reads (0, 0, -1)). ``contact`` is six booleans, or None for a missing read
    (the previous stance is kept); legs with ``contact_valid`` False use the gait schedule.
    ``yaw_rad`` is the IMU heading with any fixed offset; None integrates the gyro instead.
    """

    gyro_body: np.ndarray
    gravity_body: np.ndarray
    joint_position: np.ndarray
    joint_velocity: np.ndarray
    contact: np.ndarray | None = None
    contact_valid: np.ndarray = field(default_factory=lambda: np.ones(6, dtype=bool))
    yaw_rad: float | None = None


class PolicyLoop:
    """The actor's inputs, action and held target, one control at a time, for one robot.

    ``ghost`` is "odometry" (the trained input) or "zero" (no trustworthy odometry: the measured
    floor is 39/96 against 53/96, and a wrong ghost scores worse than none)."""

    def __init__(self, bundle, ghost="odometry"):
        if ghost not in ("odometry", "zero"):
            raise ValueError("ghost must be 'odometry' or 'zero'")
        self.bundle, self.constants, self.ghost = bundle, bundle.inputs, ghost
        c = self.constants
        self.scales = np.asarray(c["frame_scales"] * c["frames"], dtype=np.float64)
        self.command_scales = np.asarray(c["command_scales"], dtype=np.float64)
        self.tripod_a = np.asarray(c["tripod_a"], dtype=bool)
        self.restart()

    def restart(self):
        """The next control is a first control: the robot stands at the neutral pose."""
        self.first = True
        self.history = None
        self.held = self.bundle.neutral.copy()
        self.previous_action = np.zeros(18)
        self.clock = GaitClock(self.constants)
        self.odometry = LegOdometry(self.constants)
        self.command = np.zeros(3)
        self.elapsed = 0
        self.yaw = 0.0
        self.last_rate = None
        self.trace = {}

    def heading(self, reading):
        if reading.yaw_rad is not None:
            return float(reading.yaw_rad)
        # Yaw rate about world up: the body rate projected on -gravity, trapezoid over the control.
        up = -np.asarray(reading.gravity_body, dtype=np.float64)
        rate = float(np.asarray(reading.gyro_body, dtype=np.float64) @ (up / np.linalg.norm(up)))
        if self.last_rate is not None and not self.first:
            self.yaw += 0.5 * (rate + self.last_rate) * self.constants["control_dt_s"]
        self.last_rate = rate
        return self.yaw

    def schedule_contact(self, phase, command):
        """Stance unless the tripod's swing window says the foot is up while a command is active."""
        share = self.constants["schedule_contact_swing_share"]
        fraction = float(np.remainder(np.float32(phase), 1.0))
        swing_a, swing_b = fraction < share, 0.5 <= fraction < 0.5 + share
        swing = np.where(self.tripod_a, swing_a, swing_b)
        return ~(swing & bool(np.any(command != 0)))

    def inputs(self, reading, command):
        """The 278 actor inputs for this control (advances the clock, ghost and odometry)."""
        c, bundle = self.constants, self.bundle
        command = np.asarray(command, dtype=np.float64)
        q = np.asarray(reading.joint_position, dtype=np.float64)
        proprio = np.concatenate(
            (
                navigation(np.asarray(reading.gyro_body, dtype=np.float64)),
                navigation(np.asarray(reading.gravity_body, dtype=np.float64)),
                q - bundle.neutral,
                np.asarray(reading.joint_velocity, dtype=np.float64),
            )
        )
        if self.history is None:
            self.history = np.tile(proprio, (c["frames"], 1))
        else:
            self.history = np.vstack((self.history[1:], proprio))
        executed = (self.held - bundle.neutral) / bundle.action["scale_rad"]
        base = np.concatenate(
            (self.history.ravel() * self.scales, command * self.command_scales, executed)
        )
        phase = self.clock.follow(command, self.first)
        moving = float(np.any(command != 0))
        angle = 2 * np.pi * float(np.float32(phase))
        clock = np.array([np.sin(angle), np.cos(angle)]) * moving
        fresh = self.first or bool(np.any(command != self.command))
        self.elapsed = 0 if fresh else self.elapsed + 1
        self.command = command.copy()
        toes = toe_positions_body(bundle, q)[:, :2]
        heading = self.heading(reading)
        contact = reading.contact
        valid = np.asarray(reading.contact_valid, dtype=bool)
        if contact is None and not valid.all():
            contact = self.odometry.prev_contact.copy()
        if contact is not None and not valid.all():
            contact = np.where(valid, contact, self.schedule_contact(phase, command))
        if self.first:
            self.odometry.reset(toes, contact, heading)
        self.odometry.advance(toes, contact, heading, command, fresh)
        ghost = self.odometry.features(fresh, self.elapsed, command)
        if self.ghost == "zero":
            ghost = np.zeros_like(ghost)
        terrain = np.zeros(c["terrain_features"])
        self.trace = {"phase": phase, "contact": contact, "toes": toes, "heading": heading}
        return np.concatenate((base, clock, ghost, terrain))

    def step(self, reading, command):
        """This control's slew-limited joint targets (rad, JOINT_NAMES order)."""
        bundle = self.bundle
        inputs = self.inputs(reading, command)
        raw = actor(bundle.layers, inputs)
        action = raw
        if bundle.action["smoothing"] == "mean2":
            action = 0.5 * (raw + (np.zeros(18) if self.first else self.previous_action))
        self.previous_action = raw
        scale, slew = bundle.action["scale_rad"], bundle.action["slew_rad"]
        requested = np.clip(
            bundle.neutral + scale * np.clip(action, -1, 1), bundle.lower, bundle.upper
        )
        low = np.maximum(bundle.lower, self.held - slew)
        high = np.minimum(bundle.upper, self.held + slew)
        self.held = np.clip(requested, low, high)
        self.first = False
        self.trace.update(inputs=inputs, raw_action=raw, action=action, target=self.held.copy())
        return self.held.copy()
