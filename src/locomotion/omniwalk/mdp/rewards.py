"""Compute the yaw-gated omnidirectional walking reward from one control's telemetry."""

import math
from dataclasses import asdict, dataclass

import torch

from locomotion.omniwalk.mdp.commands import GAIT_CLOCK_CONTROLS
from locomotion.omniwalk.mdp.frames import inverse_rotate, navigation
from locomotion.robots.rs05 import RATED_TORQUE_NM

SCHEMA = "canonical_paper_walking_task_v1"
REWARD_VERSION = "noise_sized_tripod_schedule_v4_yaw_gate_experimental"
CONTROL_DT_S = 0.02
SUBSTEPS = 8
TRIPOD_A = (True, False, True, False, True, False)


@dataclass(frozen=True)
class RewardConfig:
    tracking_window_controls: int = 10
    fixed_speed_scale_mps: float = 0.05
    fixed_yaw_scale_rad_s: float = 0.2
    tracking_floor: float = 1.0
    linear_tracking_weight: float = 1.0
    yaw_tracking_weight: float = 0.5
    schedule_weight: float = 2.0
    schedule_period_controls: int = GAIT_CLOCK_CONTROLS
    schedule_swing_fraction: float = 0.4
    # None keeps the historical fixed tripod order; True mirrors the schedule (and the footfall
    # predictions that read it) on rows whose commanded yaw is negative, removing the gait's
    # handedness that no checkpoint ever out-trained on turn_-0.40.
    schedule_mirror_turns: bool | None = None
    schedule_load_n: float = 12.2
    swing_travel_weight: float = 1.0
    swing_travel_clip: float = 6.0
    swing_travel_full: float = 6.0
    contact_force_n: float = 1.0
    vertical_velocity_weight: float = 0.3
    vertical_velocity_scale_mps: float = 0.04
    roll_pitch_weight: float = 0.05
    roll_pitch_scale_rad_s: float = 0.3
    yaw_rate_weight: float = 0.1
    yaw_rate_scale_rad_s: float = 0.1
    planar_rate_weight: float = 0.3
    tilt_weight: float = 0.5
    tilt_scale_rad: float = 0.0873
    torque_weight: float = 0.05
    acceleration_weight: float = 0.05
    acceleration_scale_rad_s2: float = 100.0
    target_rate_weight: float = 0.05
    target_rate_scale_rad: float = 0.04
    height_weight: float = 0.2
    height_scale_m: float = 0.02
    collision_weight: float = 1.0
    torque_limit_weight: float = 0.5
    over_rating_weight: float = 10.0
    over_rating_onset_nm: float = 1.4
    joint_margin_weight: float = 1.0
    joint_margin_rad: float = 0.1
    action_limit_weight: float = 6.0
    action_limit_onset: float = 0.5
    quiet_joint_rate_weight: float = 0.5
    quiet_joint_rate_scale_rad_s: float = 0.5
    quiet_target_weight: float = 0.5
    quiet_target_scale_rad: float = 0.02
    quiet_contact_weight: float = 3.0
    quiet_load_n: float = 12.2
    quiet_strain_weight: float = 10.0
    quiet_grace_controls: int = 50
    termination_weight: float = 22.0

    yaw_gate_scale_rad_s: float = 0.1
    gate_window_controls: int = 120
    turn_draw_fraction: float = 0.3
    yaw_gate_min_command_rad_s: float = 0.18
    yaw_gate_bonus: float = 0.25
    yaw_gate_floor: float = 0.5
    stride_tracking_weight: float = 0.3
    stride_window_controls: int = 60
    stride_length_weight: float = 2.0
    # Swing clearance over rough ground (0 keeps the term, its telemetry and the old reward
    # byte-identical): scheduled-swing unloaded toes earn up to the weight, walking only, in
    # proportion to their height above the local ground up to the target — and only where the
    # ground's 5-point spread exceeds the roughness threshold, so flat ground (its 5 mm training
    # noise included) pays nothing and cannot teach high-stepping.
    swing_clearance_weight: float = 0.0
    swing_clearance_target_m: float = 0.035
    swing_clearance_rough_m: float = 0.008
    # Landing-conditioned lift (0 keeps the term, its telemetry and the old reward
    # byte-identical): when a swinging toe's predicted touchdown ground sits at least the rise
    # above the ground under the toe now, the toe earns up to the weight, walking only, in
    # proportion to its clearance over the current ground, saturating a target margin above the
    # landing ground itself — the ceiling scales with the obstacle. Flat ground and descents
    # never open the rise gate, so they pay nothing by construction.
    landing_lift_weight: float = 0.0
    landing_lift_target_m: float = 0.02
    landing_lift_rise_m: float = 0.01
    # Situation-priced tracking (0 keeps the reward byte-identical): each tracking term's weight
    # grows with how hard its own axis is commanded — up to (1 + the factor) times the registered
    # weight at the reference command — so the price lands on the rows that exercise the axis and
    # every other situation keeps the base price. Both validators count the ceiling at the full
    # factor, and the stride-tracking term prices its inner weights the same way.
    situation_tracking: float = 0.0
    situation_speed_ref_mps: float = 0.2
    situation_yaw_ref_rad_s: float = 0.4
    # Quiet pose hold (0 keeps the reward byte-identical): while still, each joint pays for its
    # squared deviation from the pose it settled into — the wrapper's quiet_anchor_joint_rad,
    # captured on the first still control, so the hold never fights the settle. The quiet gate
    # bounds the window RANGE of joint positions, and the rate and step prices are
    # quadratically blind to the slow wander that moves it.
    quiet_hold_weight: float = 0.0
    quiet_hold_scale_rad: float = 0.01

    def validate(self):
        for key, low in (
            ("tracking_window_controls", 1),
            ("schedule_period_controls", 10),
            ("quiet_grace_controls", 0),
        ):
            if type(getattr(self, key)) is not int or not low <= getattr(self, key) <= 250:
                raise ValueError(f"{key} needs {low} to 250 controls")
        for key, value in asdict(self).items():
            if isinstance(value, float) and (not math.isfinite(value) or value < 0):
                raise ValueError("Nonnegative finite reward v4 coefficient required: " + key)
        positive = [key for key in asdict(self) if "_scale_" in key] + [
            "schedule_load_n",
            "swing_travel_full",
            "swing_travel_clip",
            "contact_force_n",
            "joint_margin_rad",
            "quiet_load_n",
            "swing_clearance_target_m",
            "landing_lift_target_m",
            "situation_speed_ref_mps",
            "situation_yaw_ref_rad_s",
        ]
        if any(getattr(self, key) <= 0 for key in positive):
            raise ValueError("Positive reward v4 scales required")
        if not 0 < self.schedule_swing_fraction <= 0.5:
            raise ValueError("Each tripod swings for at most half the schedule period")
        if self.schedule_mirror_turns not in (None, True):
            raise ValueError("schedule_mirror_turns is None or True")
        if not 0 < self.action_limit_onset < 1:
            raise ValueError("The action-limit onset lies inside the action range")
        if not 0 < self.over_rating_onset_nm < RATED_TORQUE_NM:
            raise ValueError("The over-rating onset lies below 1.6 N.m")
        # A fall must cost more than the discounted loss of the largest per-control tracking reward.
        if self.termination_weight < 10 * (
            self.linear_tracking_weight + self.yaw_tracking_weight
        ) * (1 + self.situation_tracking):
            raise ValueError("The termination weight must reach ten times the tracking ceiling")
        if not math.isfinite(self.yaw_gate_scale_rad_s) or self.yaw_gate_scale_rad_s <= 0:
            raise ValueError("The yaw gate needs a positive finite scale")
        if type(self.gate_window_controls) is not int or not 1 <= self.gate_window_controls <= 250:
            raise ValueError("gate_window_controls needs 1 to 250 controls")
        if not 0 <= self.turn_draw_fraction <= 1:
            raise ValueError("turn_draw_fraction is a share from 0 to 1")
        for key in ("yaw_gate_min_command_rad_s", "yaw_gate_bonus"):
            if not math.isfinite(getattr(self, key)) or getattr(self, key) < 0:
                raise ValueError(key + " must be finite and nonnegative")
        if not 0 <= self.yaw_gate_floor <= 1:
            raise ValueError("yaw_gate_floor is a share of the schedule pay from 0 to 1")
        if not math.isfinite(self.stride_tracking_weight) or self.stride_tracking_weight < 0:
            raise ValueError("stride_tracking_weight must be finite and nonnegative")
        if (
            type(self.stride_window_controls) is not int
            or not 1 <= self.stride_window_controls <= 250
        ):
            raise ValueError("stride_window_controls needs 1 to 250 controls")
        if not math.isfinite(self.stride_length_weight) or self.stride_length_weight < 0:
            raise ValueError("stride_length_weight must be finite and nonnegative")
        ceiling = (self.linear_tracking_weight + self.yaw_tracking_weight) * (
            1 + self.situation_tracking
        ) * (1 + self.stride_tracking_weight) + 6 * self.stride_length_weight / (
            self.schedule_period_controls
        )
        if self.termination_weight < 10 * ceiling:
            raise ValueError(
                "The termination weight must reach ten times the tracking ceiling, stride term included"
            )


REWARD_CONFIG = RewardConfig()


class StrideWindow:
    """Per-replica causal mean of the last ``window`` values since that replica's latest restart."""

    def __init__(self, replicas, window, width, device="cpu"):
        self.window = window
        self.values = torch.zeros(replicas, window, width, device=device)
        self.count = torch.zeros(replicas, dtype=torch.long, device=device)
        self.command = torch.zeros(replicas, 3, device=device)

    def restart(self, indices, command):
        self.values[indices] = 0
        self.count[indices] = 0
        self.command[indices] = command

    def update(self, values, command):
        changed = (command != self.command).any(-1)
        self.values[changed] = 0
        self.count[changed] = 0
        self.command = command.clone()
        self.values = torch.cat((self.values[:, 1:], values[:, None].to(self.values)), dim=1)
        self.count = (self.count + 1).clamp_max(self.window)
        recent = (
            torch.arange(self.window, device=self.count.device) >= self.window - self.count[:, None]
        )
        return (self.values * recent[..., None]).sum(1) / self.count[:, None]


def heading(quaternion_xyzw):
    """World yaw of the body, from a unit XYZW quaternion."""
    x, y, z, w = quaternion_xyzw.unbind(-1)
    return torch.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))


def displacement_velocity(pose, previous_pose):
    """Navigation-frame planar velocity and yaw rate from two consecutive root poses."""
    planar = navigation(
        inverse_rotate(pose[:, 3:], (pose[:, :3] - previous_pose[:, :3]) / CONTROL_DT_S)
    )[:, :2]
    turn = heading(pose[:, 3:]) - heading(previous_pose[:, 3:])
    turn = torch.atan2(torch.sin(turn), torch.cos(turn))
    return torch.cat((planar, (turn / CONTROL_DT_S)[:, None]), -1)


def scheduled_swing(episode_steps, config=REWARD_CONFIG):
    """Six booleans per replica: the feet that the contact schedule wants off the floor at this control."""
    fraction = (
        episode_steps.to(torch.float32) % config.schedule_period_controls
    ) / config.schedule_period_controls
    first = torch.tensor(TRIPOD_A, device=episode_steps.device)
    swing_a = fraction < config.schedule_swing_fraction
    swing_b = (fraction >= 0.5) & (fraction < 0.5 + config.schedule_swing_fraction)
    return torch.where(first, swing_a[:, None], swing_b[:, None])


def situation_weights(commands, config=REWARD_CONFIG):
    """The tracking weights, each grown with how hard its axis is commanded when situation
    pricing is on; zero situation_tracking returns the registered scalars unchanged."""
    if not config.situation_tracking:
        return config.linear_tracking_weight, config.yaw_tracking_weight
    speed_share = (
        torch.linalg.vector_norm(commands[:, :2], dim=-1) / config.situation_speed_ref_mps
    ).clamp(0, 1)
    yaw_share = (commands[:, 2].abs() / config.situation_yaw_ref_rad_s).clamp(0, 1)
    return (
        config.linear_tracking_weight * (1 + config.situation_tracking * speed_share),
        config.yaw_tracking_weight * (1 + config.situation_tracking * yaw_share),
    )


def tracking_errors(tracked, commands, config=REWARD_CONFIG):
    """Dimensionless planar and yaw errors of already selected velocities."""
    speed, turn = torch.linalg.vector_norm(commands[:, :2], dim=-1), commands[:, 2].abs()
    linear = torch.where(speed > 0, speed, torch.full_like(speed, config.fixed_speed_scale_mps))
    yaw = torch.where(turn > 0, turn, torch.full_like(turn, config.fixed_yaw_scale_rad_s))
    return (
        torch.linalg.vector_norm(tracked[:, :2] - commands[:, :2], dim=-1) / linear,
        (tracked[:, 2] - commands[:, 2]).abs() / yaw,
    )


def kernel(error, config=REWARD_CONFIG):
    return (1 - error.square()).clamp_min(-config.tracking_floor)


def base_reward(telemetry, commands, terminated, config=REWARD_CONFIG, nominal_height=None):
    """Version 4 terms from one control's telemetry plus the caller's memory.

    Memory fields: ``tracked_velocity_nav`` (planar velocity and yaw rate, already averaged),
    ``previous_joint_velocity_rad_s``, ``previous_joint_target_rad``, ``scheduled_swing`` (six
    booleans), ``toe_velocity_nav`` (planar toe velocity over the floor, in body axes),
    ``toe_xyz_nav`` (planar toe position in the body frame) and ``joint_limit_margin_rad``.
    """
    n, device = commands.shape[0], commands.device

    def field(name, shape, dtype=torch.float32):
        value = telemetry.get(name)
        if (
            not torch.is_tensor(value)
            or tuple(value.shape) != (n, *shape)
            or not bool(torch.isfinite(value.to(torch.float32)).all())
        ):
            raise ValueError("Missing/nonfinite reward v4 telemetry: " + name)
        return value.to(device, dtype)

    velocity = field("linear_velocity_nav", (3,))
    gyro = field("angular_velocity_body", (3,))
    pose = field("root_pose_xyzw", (7,))
    rate = field("joint_velocity_rad_s", (18,))
    previous_rate = field("previous_joint_velocity_rad_s", (18,))
    target = field("joint_target_rad", (18,))
    previous_target = field("previous_joint_target_rad", (18,))
    torque_square = field("torque_square_sum_400hz", (18,))
    requested = field("requested_torque_abs_max_400hz", (18,))
    demand = field("computed_torque_nm", (18,)).abs()
    other_force = field("other_body_force_max_400hz", ())
    toe_force = torch.linalg.vector_norm(field("tibia_floor_force_world_n", (6, 3)), dim=-1)
    tracked = field("tracked_velocity_nav", (3,))
    wanted_swing = field("scheduled_swing", (6,), torch.bool).to(torch.float32)
    toe_velocity = field("toe_velocity_nav", (6, 2))
    toe = field("toe_xyz_nav", (6, 2))
    margin = field("joint_limit_margin_rad", (18,))
    executed = field("executed_action", (18,)).abs()
    # A record without the command age scores every control as settled.
    age = telemetry.get("command_age_controls")
    if not torch.equal(field("command", (3,)), commands):
        raise ValueError("Reward command differs from the native completed hold")
    terminated = torch.as_tensor(terminated, device=device)
    if tuple(terminated.shape) != (n,) or terminated.dtype != torch.bool:
        raise ValueError("Reward v4 needs one boolean termination flag per row")
    if nominal_height is None:
        raise ValueError("Reward v4 needs the nominal plate height")
    c = config
    quiet = (commands == 0).all(-1).to(torch.float32)
    moving = 1 - quiet
    still = (
        quiet
        if age is None
        else quiet * (age.to(device) > c.quiet_grace_controls).to(torch.float32)
    )
    linear_error, yaw_error = tracking_errors(tracked, commands, c)
    target_step = target - previous_target
    airborne = (toe_force <= c.contact_force_n).to(torch.float32)
    # A foot that carries less than its even share of the weight counts as partly lifted.
    unloaded = (1 - toe_force / c.schedule_load_n).clamp(0, 1)
    # Commanded body velocity at each toe: translation plus yaw rate times the lever arm.
    wanted = commands[:, None, :2] + commands[:, None, 2:] * torch.stack(
        (-toe[..., 1], toe[..., 0]), -1
    )
    ratio = (toe_velocity * wanted).sum(-1) / wanted.square().sum(-1).clamp_min(1e-8)
    travel = (ratio.clamp(-c.swing_travel_clip, c.swing_travel_clip) * airborne).sum(
        -1
    ) / c.swing_travel_full
    x, y, z, w = (pose[:, 3:] / torch.linalg.vector_norm(pose[:, 3:], dim=-1, keepdim=True)).unbind(
        -1
    )
    gravity_xy_square = 4 * ((x * z - w * y).square() + (y * z + w * x).square())
    speed = torch.linalg.vector_norm(commands[:, :2], dim=-1)
    planar = (
        torch.linalg.vector_norm(velocity[:, :2] - commands[:, :2], dim=-1)
        / speed.clamp_min(1e-8)
        * (speed > 0)
    )
    over = (
        ((demand - c.over_rating_onset_nm) / (RATED_TORQUE_NM - c.over_rating_onset_nm))
        .clamp(0, 1)
        .mean(-1)
    )
    linear_weight, yaw_weight = situation_weights(commands, c)
    components = {
        "linear_tracking": linear_weight * kernel(linear_error, c),
        "yaw_tracking": yaw_weight * kernel(yaw_error, c),
        "gait_schedule": c.schedule_weight
        * moving
        * (unloaded * (2 * wanted_swing - 1)).sum(-1)
        / 3,
        "swing_travel": c.swing_travel_weight * moving * travel.clamp_max(1.0),
        "vertical_velocity": -c.vertical_velocity_weight
        * (velocity[:, 2] / c.vertical_velocity_scale_mps).square(),
        "roll_pitch_rate": -c.roll_pitch_weight
        * (gyro[:, :2] / c.roll_pitch_scale_rad_s).square().mean(-1),
        "yaw_rate": -c.yaw_rate_weight
        * ((gyro[:, 2] - commands[:, 2]) / c.yaw_rate_scale_rad_s).square(),
        "planar_rate": -c.planar_rate_weight * planar.square(),
        "tilt": -c.tilt_weight * gravity_xy_square / c.tilt_scale_rad**2,
        "joint_torque": -c.torque_weight * (torque_square / SUBSTEPS).mean(-1) / RATED_TORQUE_NM**2,
        "joint_acceleration": -c.acceleration_weight
        * ((rate - previous_rate) / CONTROL_DT_S / c.acceleration_scale_rad_s2).square().mean(-1),
        "target_rate": -c.target_rate_weight
        * (target_step / c.target_rate_scale_rad).square().mean(-1),
        "height": -c.height_weight * ((pose[:, 2] - nominal_height) / c.height_scale_m).square(),
        "collisions": -c.collision_weight * (other_force > c.contact_force_n).to(torch.float32),
        "torque_limit": -c.torque_limit_weight
        * ((requested - RATED_TORQUE_NM).clamp_min(0) / RATED_TORQUE_NM).mean(-1),
        "over_rating": -c.over_rating_weight * over,
        "joint_margin": -c.joint_margin_weight
        * ((c.joint_margin_rad - margin) / c.joint_margin_rad).clamp(0, 1).square().sum(-1),
        "action_limit": -c.action_limit_weight
        * ((executed - c.action_limit_onset).clamp_min(0) / (1 - c.action_limit_onset))
        .square()
        .mean(-1),
        "quiet_joint_rate": -c.quiet_joint_rate_weight
        * still
        * (rate / c.quiet_joint_rate_scale_rad_s).square().mean(-1),
        "quiet_target_motion": -c.quiet_target_weight
        * still
        * (target_step / c.quiet_target_scale_rad).square().mean(-1),
        "quiet_contact": -c.quiet_contact_weight
        * quiet
        * (1 - toe_force / c.quiet_load_n).clamp(0, 1).mean(-1),
        "quiet_strain": -c.quiet_strain_weight
        * quiet
        * (torque_square[:, 0::3] / SUBSTEPS).mean(-1)
        / RATED_TORQUE_NM**2,
        "termination": -c.termination_weight * terminated.to(torch.float32),
    }
    reward = sum(components.values())
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError("Nonfinite reward v4")
    return reward, components


def yaw_gate(tracked_yaw_rate, commands, config=REWARD_CONFIG):
    """One gate per row in [0, 1]: 1 when the windowed yaw rate matches the command."""
    scale = commands[:, 2].abs().clamp_min(config.yaw_gate_scale_rad_s)
    error = (tracked_yaw_rate - commands[:, 2]).abs() / scale
    return (1 - error.square()).clamp(0, 1)


def measured_reward(telemetry, commands, terminated, config=REWARD_CONFIG, nominal_height=None):
    """Reward v4 plus ``schedule_yaw_gate``; the v4 components are returned unchanged.

    ``gate_yaw_rate_rad_s`` is the caller's stride-mean heading rate, one value per row.
    ``stride_velocity_nav`` is the caller's ``stride_window_controls`` mean of planar velocity and
    yaw rate; it is read only when ``stride_tracking_weight`` is positive.
    """
    reward, components = base_reward(telemetry, commands, terminated, config, nominal_height)
    device = commands.device
    stride_yaw = telemetry.get("gate_yaw_rate_rad_s")
    if (
        not torch.is_tensor(stride_yaw)
        or tuple(stride_yaw.shape) != (len(commands),)
        or not bool(torch.isfinite(stride_yaw).all())
    ):
        raise ValueError("Missing/nonfinite yaw-gate telemetry: gate_yaw_rate_rad_s")
    toe_force = torch.linalg.vector_norm(
        telemetry["tibia_floor_force_world_n"].to(device, torch.float32), dim=-1
    )
    unloaded = (1 - toe_force / config.schedule_load_n).clamp(0, 1)
    swing = telemetry["scheduled_swing"].to(device, torch.bool).to(torch.float32)
    turning = (commands[:, 2] != 0) & (commands[:, 2].abs() >= config.yaw_gate_min_command_rad_s)
    active = turning
    earned = config.schedule_weight * active.to(torch.float32) * (unloaded * swing).sum(-1) / 3
    gate = yaw_gate(stride_yaw.to(device, torch.float32), commands, config)
    pay = config.yaw_gate_floor + (1 + config.yaw_gate_bonus - config.yaw_gate_floor) * gate
    components["schedule_yaw_gate"] = (pay - 1) * earned
    reward = reward + components["schedule_yaw_gate"]
    if config.stride_tracking_weight > 0:
        stride = telemetry.get("stride_velocity_nav")
        if (
            not torch.is_tensor(stride)
            or tuple(stride.shape) != (len(commands), 3)
            or not bool(torch.isfinite(stride).all())
        ):
            raise ValueError("Missing/nonfinite stride-tracking telemetry: stride_velocity_nav")
        linear_error, yaw_error = tracking_errors(
            stride.to(device, torch.float32), commands, config
        )
        moving = (commands != 0).any(-1).to(torch.float32)
        linear_weight, yaw_weight = situation_weights(commands, config)
        components["stride_tracking"] = (
            config.stride_tracking_weight
            * moving
            * (
                linear_weight * kernel(linear_error, config)
                + yaw_weight * kernel(yaw_error, config)
            )
        )
        reward = reward + components["stride_tracking"]
    if config.stride_length_weight > 0:
        landed = telemetry.get("stride_touchdown")
        travel = telemetry.get("stride_displacement_nav")
        if (
            not torch.is_tensor(landed)
            or tuple(landed.shape) != (len(commands), 6)
            or not torch.is_tensor(travel)
            or tuple(travel.shape) != (len(commands), 6, 2)
            or not bool(torch.isfinite(travel).all())
        ):
            raise ValueError(
                "Missing/nonfinite stride-length telemetry: stride_touchdown, stride_displacement_nav"
            )
        toe = telemetry["toe_xyz_nav"].to(device, torch.float32)
        wanted = commands[:, None, :2] + commands[:, None, 2:] * torch.stack(
            (-toe[..., 1], toe[..., 0]), -1
        )
        # A speed-scaled gait clock gives each row its own cycle (``schedule_period_s``); each stride then
        # pays in proportion to its cycle, so the stride pay per second stays that of the fixed clock.
        fixed = config.schedule_period_controls * CONTROL_DT_S
        period = telemetry.get("schedule_period_s")
        period = fixed if period is None else period.to(device, torch.float32)[:, None]
        ratio = (travel.to(device, torch.float32) * wanted).sum(-1) / (
            wanted.square().sum(-1).clamp_min(1e-8) * period
        )
        moving = (commands != 0).any(-1).to(torch.float32)
        paid = landed.to(device, torch.bool).to(torch.float32) * ratio.clamp(0, 1).square()
        paid = paid * (period / fixed)
        components["stride_length"] = config.stride_length_weight * moving * paid.sum(-1)
        reward = reward + components["stride_length"]
    if config.swing_clearance_weight:
        clearance = telemetry.get("toe_clearance_m")
        rough = telemetry.get("toe_rough_m")
        if (
            not torch.is_tensor(clearance)
            or tuple(clearance.shape) != (len(commands), 6)
            or not torch.is_tensor(rough)
            or tuple(rough.shape) != (len(commands), 6)
            or not bool(torch.isfinite(clearance).all())
            or not bool(torch.isfinite(rough).all())
        ):
            raise ValueError(
                "Missing/nonfinite swing-clearance telemetry: toe_clearance_m, toe_rough_m"
            )
        airborne = (toe_force <= config.contact_force_n).to(torch.float32)
        rough_gate = (rough.to(device, torch.float32) > config.swing_clearance_rough_m).to(
            torch.float32
        )
        walking = (commands[:, :2] != 0).any(-1).to(torch.float32)
        height = (clearance.to(device, torch.float32) / config.swing_clearance_target_m).clamp(0, 1)
        components["swing_clearance"] = (
            config.swing_clearance_weight
            * walking
            * (rough_gate * swing * airborne * height).sum(-1)
            / 3
        )
        reward = reward + components["swing_clearance"]
    if config.landing_lift_weight:
        rise = telemetry.get("toe_landing_rise_m")
        clearance = telemetry.get("toe_clearance_m")
        if (
            not torch.is_tensor(rise)
            or tuple(rise.shape) != (len(commands), 6)
            or not torch.is_tensor(clearance)
            or tuple(clearance.shape) != (len(commands), 6)
            or not bool(torch.isfinite(rise).all())
            or not bool(torch.isfinite(clearance).all())
        ):
            raise ValueError(
                "Missing/nonfinite landing-lift telemetry: toe_landing_rise_m, toe_clearance_m"
            )
        rise = rise.to(device, torch.float32)
        airborne = (toe_force <= config.contact_force_n).to(torch.float32)
        rising = (rise > config.landing_lift_rise_m).to(torch.float32)
        walking = (commands[:, :2] != 0).any(-1).to(torch.float32)
        height = (
            clearance.to(device, torch.float32) / (rise.clamp_min(0) + config.landing_lift_target_m)
        ).clamp(0, 1)
        components["landing_lift"] = (
            config.landing_lift_weight * walking * (rising * swing * airborne * height).sum(-1) / 3
        )
        reward = reward + components["landing_lift"]
    if config.quiet_hold_weight:
        anchor = telemetry.get("quiet_anchor_joint_rad")
        position = telemetry.get("joint_position_rad")
        if (
            not torch.is_tensor(anchor)
            or tuple(anchor.shape) != (len(commands), 18)
            or not bool(torch.isfinite(anchor.to(torch.float32)).all())
            or not torch.is_tensor(position)
            or tuple(position.shape) != (len(commands), 18)
            or not bool(torch.isfinite(position.to(torch.float32)).all())
        ):
            raise ValueError(
                "Missing/nonfinite quiet-hold telemetry: quiet_anchor_joint_rad, joint_position_rad"
            )
        quiet = (commands == 0).all(-1).to(torch.float32)
        age = telemetry.get("command_age_controls")
        still = (
            quiet
            if age is None
            else quiet * (age.to(device) > config.quiet_grace_controls).to(torch.float32)
        )
        deviation = (
            position.to(device, torch.float32) - anchor.to(device, torch.float32)
        ) / config.quiet_hold_scale_rad
        components["quiet_hold"] = -config.quiet_hold_weight * still * deviation.square().mean(-1)
        reward = reward + components["quiet_hold"]
    if not bool(torch.isfinite(reward).all()):
        raise FloatingPointError("Nonfinite yaw-gated reward v4")
    return reward, components
