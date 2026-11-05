"""Experiment ``destination_v1``: pay every control for staying near the pose the command would have reached.

The canonical task and reward stay unchanged. ``DestinationRewardConfig`` adds its fields to
``RewardConfig`` and ``DestinationTrainingTask`` follows each command's destination
(``mdp/destination.py``):

    destination = w * exp(-d / destination_distance_m)        exponential
                  w * 1 / (1 + (d / scale) ** power)         rational
                  w * max(0, 1 - d / scale)                  linear

every control, for the planar distance ``d`` from the destination; a fall forfeits it, so the fall-cost
ceiling counts the weight. ``command_controls`` holds every command that many controls instead of the
draw of 100 to 250. The experiment also gives the actor and critic the gap to the destination
(``locomotion.rl.destination_wrapper``).

``destination_power`` sets the rational kernel's exponent and stays None for the other kernels.

Two zero-command fields leave moving commands alone. ``destination_quiet="flat"`` pays the full weight
while the command is zero, so standing still never trades joint motion for millimetres of ghost
distance; ``destination_quiet="off"`` pays nothing while the command is zero. ``hold_weight`` follows each joint's running mean over ``hold_smoothing_controls`` (an
exponential mean restarted when a command begins), which removes the zero-mean exploration jitter of
training but keeps slow drift. The mean is anchored ``hold_grace_controls`` into a zero command and
from then on the task charges

    hold = -hold_weight * mean_j clamp((|mean_j - anchor_j| - hold_deadband_rad) / hold_span_rad, 0, 1)

linear in the drift, like the evaluator's joint range check that opens 2 s into a zero command. A
fall forfeits the destination pay and the hold charge. None leaves each field off.

``EXPERIMENTS`` names the reviewed settings. ``destination_v1`` is the surrogate result of
2026-10-17: started from the PR model, every command 20 s, the exponential kernel at 3 cm and weight
1, termination 32 (ten times the 3.15 tracking ceiling). ``destination_v2`` keeps all of it but
halves the pay every centimetre, to train on from the native ``destination_v1`` checkpoint.
``destination_v3`` keeps v2 but concentrates the pay at the destination with the rational kernel,
to train on from the native ``destination_v2`` checkpoint. ``destination_v4`` keeps v3's kernel
without the hold charge and pays nothing for the destination under a zero command, also from the
native ``destination_v2`` checkpoint. ``destination_v5`` is v4 that pays the destination under a zero
command too: v3's kernel without either zero-command change. ``destination_v6`` is v5 that also
trains stopping: ``stop_fraction`` of the moving commands drawn last ``stop_after_min_controls`` to
``stop_after_max_controls`` controls and give way to a zero command for the rest of the
``command_controls``, so a walk is followed by a stop as in the evaluator's walk-then-stop probe.
``destination_v7`` is v5 with a heading term of the same form as the rational destination pay,

    heading = heading_weight / (1 + (|wrap(heading_d - heading)| / heading_width_rad) ** heading_power)

every control for the gap to the destination's heading, which starts at the body's heading and turns
only with the commanded yaw rate; a fall forfeits it and the fall-cost ceiling counts its weight.
``destination_v8`` is v7 with every joint penalty of the canonical reward doubled (``JOINT_PENALTIES``).
``destination_v9`` is v8 with v6's stops after a walk, for a short fine-tune from the native v8
checkpoint at v8's learning rate and final action deviation.

``destination_v10`` is v9 with a walking-speed curriculum. Every fresh walk (a planar command without
yaw) gets a uniform direction and, with ``speed_curriculum_new_share``, the current top speed, otherwise
one of the slower levels (``speed_curriculum_step_mps`` apart, from one step up); turns, arcs, zero
commands and the stops after a walk are v9's. The top speed starts one step above
``speed_curriculum_start_mps`` and rises by one step whenever the destination pay, averaged over all
rows for the last ``speed_curriculum_window_controls`` controls at that speed, exceeds
``speed_curriculum_threshold``, up to ``speed_curriculum_final_mps``; the task's status reports the level.
``destination_v11`` is v10 with a quarter of the walks at the top speed and a back-off: a full window
below ``speed_curriculum_backoff_threshold`` lowers the top speed one step (not below the first level),
so training does not stay at a speed whose pay keeps falling. ``destination_v12`` is v10 with v11's
back-off and v10's half of the walks at the top speed. ``destination_v13`` is v12 where every fresh
command is a walk (``speed_curriculum_walks_only``: no zero commands, turns or arcs; a fifth still stop
after 4 s to 12 s), the destination pay is four times wider (0.8 at 5 mm instead of at 1.2 mm) and the
level gate averages the last 10 updates instead of 50. ``destination_v14`` is v13 whose back-off judges
the last ``speed_curriculum_backoff_window_controls`` (50 updates) at the level, so a level gets 50
updates before it can be lowered while it still rises after 10. ``destination_v15`` is v14 that raises
the level only when no robot fell in the window it judges (``speed_curriculum_advance_without_falls``).
"""

import hashlib
import math
from dataclasses import dataclass, replace
from pathlib import Path

import torch

from locomotion.omniwalk.mdp.commands import GAIT_CLOCK_SCALINGS
from locomotion.omniwalk.mdp.destination import (
    advance_destination,
    destination_distance,
    planar_pose,
)
from locomotion.omniwalk.mdp.footfall import (
    RING_XY,
    TerrainGrid,
    touchdown_points,
    touchdown_seconds,
)
from locomotion.omniwalk.mdp.frames import inverse_rotate, navigation
from locomotion.omniwalk.mdp.gait import GaitPhase
from locomotion.omniwalk.mdp.rewards import RewardConfig
from locomotion.omniwalk.task import TrainingTask

REWARD_VERSION = "noise_sized_tripod_schedule_v4_yaw_gate_destination_experimental"


@dataclass(frozen=True)
class DestinationRewardConfig(RewardConfig):
    destination_weight: float = 0.0
    destination_distance_m: float = 0.1
    destination_kernel: str = "linear"
    destination_power: float | None = None
    destination_quiet: str | None = None
    hold_weight: float | None = None
    hold_grace_controls: int | None = None
    hold_deadband_rad: float | None = None
    hold_span_rad: float | None = None
    hold_smoothing_controls: int | None = None
    stop_fraction: float | None = None
    stop_after_min_controls: int | None = None
    stop_after_max_controls: int | None = None
    heading_weight: float | None = None
    heading_width_rad: float | None = None
    heading_power: float | None = None
    speed_curriculum_start_mps: float | None = None
    speed_curriculum_step_mps: float | None = None
    speed_curriculum_final_mps: float | None = None
    speed_curriculum_threshold: float | None = None
    speed_curriculum_window_controls: int | None = None
    speed_curriculum_new_share: float | None = None
    speed_curriculum_backoff_threshold: float | None = None
    speed_curriculum_walks_only: bool | None = None
    speed_curriculum_backoff_window_controls: int | None = None
    speed_curriculum_advance_without_falls: bool | None = None
    speed_curriculum_backoff_distance_m: float | None = None
    speed_curriculum_pass_checks: bool | None = None
    speed_curriculum_planar_bound_scale: float | None = None
    speed_curriculum_measure: str | None = None
    planar_error_weight: float | None = None
    planar_error_cap: float | None = None
    walk_speed_min_mps: float | None = None
    walk_speed_max_mps: float | None = None
    destination_linear_from_m: float | None = None
    destination_penalty_cap: float | None = None
    gait_clock_scaling: str | None = None
    end_destination_weight: float | None = None
    end_destination_distance_m: float | None = None
    end_destination_power: float | None = None
    end_destination_kernel: str | None = None
    end_destination_floor: float | None = None
    run_mix: tuple | None = None
    long_stand_controls: int | None = None
    walk_forward_only: bool | None = None
    walk_forward_bias: float | None = None
    turn_rate_choices_rad_s: tuple | None = None
    arc_fraction: float | None = None
    arc_yaw_choices_rad_s: tuple | None = None
    slope_inputs: str | None = None
    command_controls: int = 0

    def validate(self):
        super().validate()
        if self.destination_kernel not in ("linear", "exponential", "rational"):
            raise ValueError("destination_kernel is linear, exponential or rational")
        if (self.destination_kernel == "rational") != (self.destination_power is not None) or (
            self.destination_power is not None
            and not (math.isfinite(self.destination_power) and self.destination_power > 0)
        ):
            raise ValueError("The rational kernel, and only it, needs a positive finite power")
        # A run mix draws its own share of walks that stop, but still needs the stop's bounds.
        fraction = 1.0 if self.run_mix is not None else self.stop_fraction
        stop = (fraction, self.stop_after_min_controls, self.stop_after_max_controls)
        if any(value is not None for value in stop) and (
            any(value is None for value in stop)
            or not math.isfinite(fraction)
            or not 0 < fraction <= 1
            or type(self.stop_after_min_controls) is not int
            or type(self.stop_after_max_controls) is not int
            or not 10 <= self.stop_after_min_controls <= self.stop_after_max_controls
            or self.stop_after_max_controls >= self.command_controls
        ):
            raise ValueError(
                "Stops need a fraction in (0, 1] and 10 to fewer than command_controls controls "
                "of walking first"
            )
        heading = (self.heading_weight, self.heading_width_rad, self.heading_power)
        if any(value is not None for value in heading) and (
            any(value is None for value in heading)
            or not all(math.isfinite(value) and value > 0 for value in heading)
        ):
            raise ValueError("The heading pay needs a positive finite weight, width and power")
        curriculum = (
            self.speed_curriculum_start_mps,
            self.speed_curriculum_step_mps,
            self.speed_curriculum_final_mps,
            self.speed_curriculum_threshold,
            self.speed_curriculum_window_controls,
            self.speed_curriculum_new_share,
        )
        if any(value is not None for value in curriculum) and (
            any(value is None for value in curriculum)
            or not all(math.isfinite(value) for value in curriculum)
            or not 0 < self.speed_curriculum_step_mps <= self.speed_curriculum_start_mps
            or self.speed_curriculum_final_mps <= self.speed_curriculum_start_mps
            or abs(
                (self.speed_curriculum_final_mps - self.speed_curriculum_start_mps)
                / self.speed_curriculum_step_mps
                - round(
                    (self.speed_curriculum_final_mps - self.speed_curriculum_start_mps)
                    / self.speed_curriculum_step_mps
                )
            )
            > 1e-6
            or not 0 < self.speed_curriculum_threshold <= self.destination_weight
            or type(self.speed_curriculum_window_controls) is not int
            or self.speed_curriculum_window_controls < 1
            or not 0 <= self.speed_curriculum_new_share <= 1
        ):
            raise ValueError(
                "The speed curriculum needs a step up to its start, a final speed a whole number "
                "of steps above it, a threshold within the destination weight, a positive window "
                "and a share from 0 to 1"
            )
        backoff = self.speed_curriculum_backoff_threshold
        if backoff is not None and (
            self.speed_curriculum_threshold is None
            or not math.isfinite(backoff)
            or not 0 < backoff < self.speed_curriculum_threshold
        ):
            raise ValueError(
                "The back-off threshold needs a curriculum and lies below its threshold"
            )
        if self.speed_curriculum_walks_only is not None and (
            self.speed_curriculum_walks_only is not True or self.speed_curriculum_threshold is None
        ):
            raise ValueError("Walks only is True or None and needs a speed curriculum")
        span = self.speed_curriculum_backoff_window_controls
        if span is not None and (
            self.speed_curriculum_backoff_threshold is None
            or type(span) is not int
            or span < self.speed_curriculum_window_controls
        ):
            raise ValueError(
                "A back-off window needs a back-off and spans at least the level window"
            )
        if self.speed_curriculum_advance_without_falls is not None and (
            self.speed_curriculum_advance_without_falls is not True
            or self.speed_curriculum_threshold is None
        ):
            raise ValueError("Advancing without falls is True or None and needs a speed curriculum")
        distance = self.speed_curriculum_backoff_distance_m
        if distance is not None and (
            self.speed_curriculum_threshold is None or not math.isfinite(distance) or distance <= 0
        ):
            raise ValueError(
                "A back-off distance needs a speed curriculum and is positive and finite"
            )
        if (
            self.gait_clock_scaling is not None
            and self.gait_clock_scaling not in GAIT_CLOCK_SCALINGS
        ):
            raise ValueError("gait_clock_scaling is None or a name in mdp/commands.py")
        walk = (self.walk_speed_min_mps, self.walk_speed_max_mps)
        if any(value is not None for value in walk) and (
            any(value is None for value in walk)
            or not all(math.isfinite(value) for value in walk)
            or not 0 < self.walk_speed_min_mps < self.walk_speed_max_mps
            or self.speed_curriculum_start_mps is not None
        ):
            raise ValueError(
                "Uniform walk speeds need a positive minimum below the maximum and no speed curriculum"
            )
        if self.planar_error_cap is not None and (
            self.planar_error_weight is None
            or not math.isfinite(self.planar_error_cap)
            or self.planar_error_cap <= 0
        ):
            raise ValueError("A planar error cap needs the planar error charge and is positive")
        if self.speed_curriculum_measure not in (None, "drift") or (
            self.speed_curriculum_measure is not None and self.speed_curriculum_threshold is None
        ):
            raise ValueError("The curriculum measure is None (the destination pay) or drift")
        scale = self.speed_curriculum_planar_bound_scale
        if scale is not None and (
            not self.speed_curriculum_pass_checks or not math.isfinite(scale) or scale <= 0
        ):
            raise ValueError("A planar bound scale needs the walking checks and is positive")
        linear = (self.destination_linear_from_m, self.destination_penalty_cap)
        if any(value is not None for value in linear) and (
            any(value is None for value in linear)
            or self.destination_kernel != "rational"
            or not all(math.isfinite(value) for value in linear)
            or self.destination_linear_from_m <= 0
            or self.destination_penalty_cap <= 0
        ):
            raise ValueError(
                "A linear tail needs the rational kernel, a positive start and a positive penalty cap"
            )
        if self.speed_curriculum_pass_checks is not None and (
            self.speed_curriculum_pass_checks is not True or self.speed_curriculum_threshold is None
        ):
            raise ValueError("Passing the checks is True or None and needs a speed curriculum")
        if self.destination_quiet not in (None, "flat", "off"):
            raise ValueError("destination_quiet is None (track the ghost), flat or off")
        hold = (
            self.hold_weight,
            self.hold_grace_controls,
            self.hold_deadband_rad,
            self.hold_span_rad,
            self.hold_smoothing_controls,
        )
        if any(value is not None for value in hold) and (
            any(value is None for value in hold)
            or not all(math.isfinite(value) for value in hold)
            or self.hold_weight <= 0
            or type(self.hold_grace_controls) is not int
            or not 1 <= self.hold_grace_controls <= 1000
            or self.hold_deadband_rad < 0
            or self.hold_span_rad <= 0
            or type(self.hold_smoothing_controls) is not int
            or not 1 <= self.hold_smoothing_controls <= self.hold_grace_controls
        ):
            raise ValueError(
                "The hold charge needs a positive weight, 1 to 1000 grace controls, "
                "a nonnegative deadband and a positive span and 1 to grace smoothing controls"
            )
        if not math.isfinite(self.destination_distance_m) or self.destination_distance_m <= 0:
            raise ValueError("destination_distance_m must be positive and finite")
        if type(self.command_controls) is not int or not (
            self.command_controls == 0 or 10 <= self.command_controls <= 5000
        ):
            raise ValueError("command_controls is 0 (the drawn durations) or 10 to 5000 controls")
        # The run-end pay is rational (a power, no floor) or linear (a floor, no power).
        linear = self.end_destination_kernel == "linear"
        shape = self.end_destination_floor if linear else self.end_destination_power
        unused = self.end_destination_power if linear else self.end_destination_floor
        end = (self.end_destination_weight, self.end_destination_distance_m, shape)
        if (any(value is not None for value in end + (self.end_destination_kernel, unused))) and (
            any(value is None for value in end)
            or unused is not None
            or self.end_destination_kernel not in (None, "linear")
            or not all(math.isfinite(value) and value > 0 for value in end)
            or self.destination_weight <= 0
        ):
            raise ValueError(
                "The run-end destination pay needs a positive finite weight and width beside the "
                "destination pay, and a power (rational) or a floor (linear)"
            )
        mix = self.run_mix
        if mix is not None and (
            type(mix) is not tuple
            or len(mix) != 4
            or not all(math.isfinite(share) and share >= 0 for share in mix)
            or abs(sum(mix) - 1) > 1e-9
            or self.stop_fraction is not None
            or self.walk_speed_min_mps is None
            or self.command_controls == 0
            or type(self.long_stand_controls) is not int
            or not self.command_controls <= self.long_stand_controls <= 5000
        ):
            raise ValueError(
                "A run mix gives four shares (turns, stands, walks that stop, walks) summing to 1, "
                "with uniform walk speeds, no stop fraction, fixed command controls and long stands "
                "from command_controls to 5000 controls"
            )
        if self.long_stand_controls is not None and mix is None:
            raise ValueError("Long stands need a run mix")
        if self.walk_forward_only is not None and (
            self.walk_forward_only is not True or self.walk_speed_min_mps is None
        ):
            raise ValueError("Forward-only walks are True or None and need uniform walk speeds")
        if self.walk_forward_bias is not None and (
            self.walk_forward_only is not None
            or self.walk_speed_min_mps is None
            or not 0 < self.walk_forward_bias < 1
        ):
            raise ValueError(
                "A forward bias is the share of walks facing forward, between 0 and 1, "
                "needs uniform walk speeds and excludes forward-only walks"
            )
        if self.slope_inputs not in (None, "gravity", "femurs", "joints"):
            raise ValueError("slope_inputs is None, gravity, femurs or joints (mdp/slope.py)")
        choices = self.turn_rate_choices_rad_s
        if choices is not None and (
            self.run_mix is None
            or type(choices) is not tuple
            or not choices
            or not all(
                isinstance(rate, float) and math.isfinite(rate) and rate > 0 for rate in choices
            )
        ):
            raise ValueError(
                "Turn-rate choices are a nonempty tuple of positive finite rad/s rates and "
                "need a run mix"
            )
        arc = (self.arc_fraction, self.arc_yaw_choices_rad_s)
        if any(value is not None for value in arc) and (
            any(value is None for value in arc)
            or self.run_mix is None
            or not (math.isfinite(self.arc_fraction) and 0 < self.arc_fraction <= 1)
            or type(self.arc_yaw_choices_rad_s) is not tuple
            or not self.arc_yaw_choices_rad_s
            or not all(
                isinstance(rate, float) and math.isfinite(rate) and rate > 0
                for rate in self.arc_yaw_choices_rad_s
            )
        ):
            raise ValueError(
                "Arcing walks need a fraction in (0, 1] of the drawn walks, a nonempty tuple "
                "of positive finite rad/s yaw rates, and a run mix"
            )
        ceiling = (
            (self.linear_tracking_weight + self.yaw_tracking_weight)
            * (1 + self.situation_tracking)
            * (1 + self.stride_tracking_weight)
            + 6 * self.stride_length_weight / self.schedule_period_controls
            + self.destination_weight
            + (self.heading_weight or 0.0)
            + (self.end_destination_weight or 0.0)
        )
        if self.termination_weight < 10 * ceiling:
            raise ValueError(
                "The termination weight must reach ten times the tracking ceiling, "
                "destination term included"
            )


EXPERIMENTS = {
    "destination_v1": {
        "reward": DestinationRewardConfig(
            termination_weight=32.0,
            destination_weight=1.0,
            destination_distance_m=0.03,
            destination_kernel="exponential",
            command_controls=1000,
        ),
        "episode_seconds": 20.0,
    },
}
# destination_v2 trains on from destination_v1 with the pay concentrated near the destination: it
# halves every centimetre (1 at 0, 0.71 at 0.5 cm, 0.5 at 1 cm, 0.25 at 2 cm, 0.03 at 5 cm).
EXPERIMENTS["destination_v2"] = {
    **EXPERIMENTS["destination_v1"],
    "reward": replace(
        EXPERIMENTS["destination_v1"]["reward"], destination_distance_m=0.01 / math.log(2)
    ),
}
# destination_v3 trains on from destination_v2 with the pay concentrated at the destination: the
# rational kernel at 3.7 mm and power 1.2 (1 at 0, 0.83 at 1 mm, 0.62 at 2.5 mm, 0.41 at 5 mm,
# 0.23 at 1 cm, 0.12 at 2 cm, 0.04 at 5 cm). A zero command pays the full destination weight and,
# from 2 s in, charges up to 2 per control for joints whose 1 s running mean drifts 5 mrad to
# 50 mrad from where it stood (exploration noise alone moves standing joints about 20 mrad).
EXPERIMENTS["destination_v3"] = {
    **EXPERIMENTS["destination_v2"],
    "reward": replace(
        EXPERIMENTS["destination_v2"]["reward"],
        destination_kernel="rational",
        destination_distance_m=0.0037,
        destination_power=1.2,
        destination_quiet="flat",
        hold_weight=2.0,
        hold_grace_controls=100,
        hold_deadband_rad=0.005,
        hold_span_rad=0.045,
        hold_smoothing_controls=50,
    ),
}
# destination_v4 keeps destination_v3's rational kernel without the hold charge and pays nothing for
# the destination while the command is zero; it also trains on from destination_v2.
EXPERIMENTS["destination_v4"] = {
    **EXPERIMENTS["destination_v3"],
    "reward": replace(
        EXPERIMENTS["destination_v3"]["reward"],
        destination_quiet="off",
        hold_weight=None,
        hold_grace_controls=None,
        hold_deadband_rad=None,
        hold_span_rad=None,
        hold_smoothing_controls=None,
    ),
}
# destination_v5 is destination_v4 that pays the destination under a zero command as well: v3's
# rational kernel with neither zero-command change, also from destination_v2.
EXPERIMENTS["destination_v5"] = {
    **EXPERIMENTS["destination_v4"],
    "reward": replace(EXPERIMENTS["destination_v4"]["reward"], destination_quiet=None),
}
# destination_v6 is destination_v5 that also trains stopping: a fifth of the moving commands last
# 4 s to 12 s and then give way to a zero command for the rest of the 20 s.
EXPERIMENTS["destination_v6"] = {
    **EXPERIMENTS["destination_v5"],
    "reward": replace(
        EXPERIMENTS["destination_v5"]["reward"],
        stop_fraction=0.2,
        stop_after_min_controls=200,
        stop_after_max_controls=600,
    ),
}
# destination_v7 is destination_v5 with a heading pay shaped like its destination pay: 1 at the
# destination's heading, 0.5 at 1 deg, 0.3 at 2 deg, 0.13 at 5 deg. Termination 42 keeps ten times
# the 4.15 tracking ceiling.
EXPERIMENTS["destination_v7"] = {
    **EXPERIMENTS["destination_v5"],
    "reward": replace(
        EXPERIMENTS["destination_v5"]["reward"],
        heading_weight=1.0,
        heading_width_rad=math.radians(1.0),
        heading_power=1.2,
        termination_weight=42.0,
    ),
}
# The canonical reward's joint penalties: torque, acceleration, target steps, torque requests over
# the rating, demand near the rating, joint-limit margin, action range, and the zero-command joint
# rate, target motion and coxa strain.
JOINT_PENALTIES = (
    "torque_weight",
    "acceleration_weight",
    "target_rate_weight",
    "torque_limit_weight",
    "over_rating_weight",
    "joint_margin_weight",
    "action_limit_weight",
    "quiet_joint_rate_weight",
    "quiet_target_weight",
    "quiet_strain_weight",
)
# destination_v8 is destination_v7 with every joint penalty doubled, also from destination_v2.
EXPERIMENTS["destination_v8"] = {
    **EXPERIMENTS["destination_v7"],
    "reward": replace(
        EXPERIMENTS["destination_v7"]["reward"],
        **{
            name: 2 * getattr(EXPERIMENTS["destination_v7"]["reward"], name)
            for name in JOINT_PENALTIES
        },
    ),
}
# destination_v9 is destination_v8 with destination_v6's stops: a fifth of the moving commands last
# 4 s to 12 s and then give way to a zero command for the rest of the 20 s.
EXPERIMENTS["destination_v9"] = {
    **EXPERIMENTS["destination_v8"],
    "reward": replace(
        EXPERIMENTS["destination_v8"]["reward"],
        stop_fraction=0.2,
        stop_after_min_controls=200,
        stop_after_max_controls=600,
    ),
}
# destination_v10 is destination_v9 with a walking-speed curriculum from the native v9 checkpoint: the
# top walking speed starts at 0.075 m/s and rises by 0.025 m/s to 0.15 m/s each time the destination pay
# averages above 0.8 over the last 1800 controls (50 updates of 36 controls); half the walks go at the
# top speed and half at the slower levels, in uniform directions.
EXPERIMENTS["destination_v10"] = {
    **EXPERIMENTS["destination_v9"],
    "reward": replace(
        EXPERIMENTS["destination_v9"]["reward"],
        speed_curriculum_start_mps=0.05,
        speed_curriculum_step_mps=0.025,
        speed_curriculum_final_mps=0.15,
        speed_curriculum_threshold=0.8,
        speed_curriculum_window_controls=1800,
        speed_curriculum_new_share=0.5,
    ),
}
# destination_v11 is destination_v10 with three quarters of the walks at the slower levels and a back-off:
# training at 0.125 m/s let the destination pay fall from 0.73 to 0.42 and cost the 0.05 m/s walks their
# millimetre end gaps, so a full window below 0.65 lowers the top speed one step.
EXPERIMENTS["destination_v11"] = {
    **EXPERIMENTS["destination_v10"],
    "reward": replace(
        EXPERIMENTS["destination_v10"]["reward"],
        speed_curriculum_new_share=0.25,
        speed_curriculum_backoff_threshold=0.65,
    ),
}
# destination_v12 is destination_v10 with destination_v11's back-off: a quarter of the walks at the top
# speed left the 0.1 m/s pay flat for 4000 updates, so half return to the top speed.
EXPERIMENTS["destination_v12"] = {
    **EXPERIMENTS["destination_v10"],
    "reward": replace(
        EXPERIMENTS["destination_v10"]["reward"], speed_curriculum_backoff_threshold=0.65
    ),
}
# destination_v13 is destination_v12 with every fresh command a walk in a uniform direction (a fifth
# still stop after 4 s to 12 s) and the rational destination pay widened so 5 mm pays 0.8: the width
# solves 1 / (1 + (5 mm / w) ** 1.2) = 0.8, w = 15.9 mm (v3 to v12: 3.7 mm, which pays 0.8 at 1.2 mm).
# The level gate averages the last 360 controls (10 updates of 36) instead of 50 updates.
EXPERIMENTS["destination_v13"] = {
    **EXPERIMENTS["destination_v12"],
    "reward": replace(
        EXPERIMENTS["destination_v12"]["reward"],
        destination_distance_m=0.005 / 0.25 ** (1 / 1.2),
        speed_curriculum_walks_only=True,
        speed_curriculum_window_controls=360,
    ),
}
# destination_v14 is destination_v13 whose back-off averages the last 1800 controls (50 updates) at the
# level: on v13's 10-update window, 10 updates at 0.15 m/s dropped the level and the next 10 at 0.125 m/s
# usually dropped it again, so 0.15 m/s never trained for more than about 10 updates at a time.
EXPERIMENTS["destination_v14"] = {
    **EXPERIMENTS["destination_v13"],
    "reward": replace(
        EXPERIMENTS["destination_v13"]["reward"], speed_curriculum_backoff_window_controls=1800
    ),
}
# destination_v15 is destination_v14 that raises the level only when no robot fell during the 10 updates
# it judges: the v14 checkpoints at 0.125 and 0.15 m/s fell within 2 s in most directions.
EXPERIMENTS["destination_v15"] = {
    **EXPERIMENTS["destination_v14"],
    "reward": replace(
        EXPERIMENTS["destination_v14"]["reward"], speed_curriculum_advance_without_falls=True
    ),
}
# destination_v16 is destination_v15 with a gait clock that follows the walking speed and a back-off
# whenever a robot ends up more than 5 cm from its ghost. On the fixed 1.2 s clock 0.125 m/s needs 90 mm
# strides, beyond the 66-80 mm the +-0.35 rad action range reaches; speed_v1 shortens the cycle from 60
# controls at 0.075 m/s to 52, 44 and 40 at 0.1, 0.125 and 0.15 m/s, each tripod swinging 0.4 to 0.5 of it.
EXPERIMENTS["destination_v16"] = {
    **EXPERIMENTS["destination_v15"],
    "reward": replace(
        EXPERIMENTS["destination_v15"]["reward"],
        gait_clock_scaling="speed_v1",
        speed_curriculum_backoff_distance_m=0.05,
    ),
}
# destination_v17 is destination_v16 without any back-off and with a 30-update window: attempt 8 tracked
# 0.125 m/s at 0.80-0.92 of the destination pay, but the 5 cm rule lowered the level within about 10
# updates every time. The level now rises only after 30 updates (1080 controls) above 0.8 without a fall
# and never falls.
EXPERIMENTS["destination_v17"] = {
    **EXPERIMENTS["destination_v16"],
    "reward": replace(
        EXPERIMENTS["destination_v16"]["reward"],
        speed_curriculum_window_controls=1080,
        speed_curriculum_backoff_threshold=None,
        speed_curriculum_backoff_window_controls=None,
        speed_curriculum_backoff_distance_m=None,
    ),
}
# destination_v18 is destination_v17 that also needs every walk finished in the 30 updates to pass the
# evaluator's moving checks: at update 500 of destination_v17 seven directions held 0.125 m/s within 6 mm of
# the ghost, but every case failed the planar-error check and three asked for torque above 1.6 Nm too often.
EXPERIMENTS["destination_v18"] = {
    **EXPERIMENTS["destination_v17"],
    "reward": replace(EXPERIMENTS["destination_v17"]["reward"], speed_curriculum_pass_checks=True),
}

# destination_v19 is destination_v18 whose destination pay falls in a straight line beyond 20 mm, at the
# rational kernel's slope there (0.431 at 20 mm, 0 at 49 mm, -1 from 117 mm), so a growing lag behind the
# ghost keeps costing; and whose walking checks allow twice the evaluator's planar error. From update 1450
# to 4400 of destination_v18 at 0.15 m/s the destination pay fell from 0.76 to 0.34 per control while the
# speed held: the robots let the gap grow into the flat tail of the kernel.
EXPERIMENTS["destination_v19"] = {
    **EXPERIMENTS["destination_v18"],
    "reward": replace(
        EXPERIMENTS["destination_v18"]["reward"],
        destination_linear_from_m=0.02,
        destination_penalty_cap=1.0,
        speed_curriculum_planar_bound_scale=2.0,
    ),
}
# destination_v20 trains the v10 model: destination_v19's reward and curriculum with the evaluator's own
# planar bound in the walking checks, and the ghost_v10 inputs (the sideways distance to the ghost and the
# body's velocity relative to it) in place of the full gap. Attempts 10 and 11 fell behind the ghost in the
# first seconds at 0.15 m/s and, at their top speed, could not close the gap again.
EXPERIMENTS["destination_v20"] = {
    **EXPERIMENTS["destination_v19"],
    "reward": replace(
        EXPERIMENTS["destination_v19"]["reward"], speed_curriculum_planar_bound_scale=None
    ),
}
# destination_v21 is destination_v20 with the ghost_v11 inputs: the along-track gap returns beside the
# sideways gap and the relative velocity. At update 600 of destination_v20 the v10 model held 0.075 m/s
# 4-10% off the command in most directions and ended 7-126 mm from the ghost; its window pay sat near 0.47.
EXPERIMENTS["destination_v21"] = {**EXPERIMENTS["destination_v20"]}
# destination_v22 is destination_v21 whose curriculum judges each walk by its drift instead of the destination
# pay: how far the gap to the ghost moved between the walk's halfway control and its last, scored on the
# destination kernel (0.8 at 5 mm; a fall scores 0). The window mean over walks finished in the 30 updates
# must pass 0.8. A lag left by the ghost's instant start no longer counts; holding the ghost's speed and
# direction does. Attempts 10, 11 and 13 lost window pay at 0.15 m/s mostly by starting more slowly.
EXPERIMENTS["destination_v22"] = {
    **EXPERIMENTS["destination_v21"],
    "reward": replace(EXPERIMENTS["destination_v21"]["reward"], speed_curriculum_measure="drift"),
}
# destination_v23 is destination_v22 with a per-control charge on the evaluator's planar speed error,
# -0.5 * (|velocity - command| / max(0.025, a quarter of the speed))^2 while walking. At 0.15 m/s the
# destination_v21 robots averaged the command within about 1 mm/s over each step but swung 25-40 mm/s
# within it, failing the evaluator's planar check; no reward term saw that swing.
EXPERIMENTS["destination_v23"] = {
    **EXPERIMENTS["destination_v22"],
    "reward": replace(EXPERIMENTS["destination_v22"]["reward"], planar_error_weight=0.5),
}
# destination_v24 is destination_v23 whose speed curriculum continues to 0.175 m/s. Attempt 15 of
# destination_v23 mastered 0.15 m/s at update 747. The speed_v1 gait clock holds its 40-control cycle above
# 0.15 m/s, so 0.175 m/s needs 70 mm strides, beyond the 66 mm sideways reach of the action range.
EXPERIMENTS["destination_v24"] = {
    **EXPERIMENTS["destination_v23"],
    "reward": replace(EXPERIMENTS["destination_v23"]["reward"], speed_curriculum_final_mps=0.175),
}
# destination_v25 is destination_v23 whose planar error charge is linear and capped:
# -0.5 * min(|velocity - command| / bound, 2), at most -1 per control. Natively (Isaac) the squared charge
# of destination_v23 met brief velocity spikes: from update 140 its mean rose from -0.15 to -1.25 while the
# mean error only doubled, the value loss climbed from about 25 to 11000 within each episode and the
# reward fell from 5 to 1.3.
EXPERIMENTS["destination_v25"] = {
    **EXPERIMENTS["destination_v23"],
    "reward": replace(EXPERIMENTS["destination_v23"]["reward"], planar_error_cap=2.0),
}
# destination_v26 fine-tunes destination_v25's walking for stopping, as destination_v9 did for v8: every
# command is a walk in a uniform direction at a uniform speed from 0.05 to 0.15 m/s, half of them stop
# after 4 to 12 s for the rest of the 20 s, and there is no speed curriculum. The native destination_v25
# update-1350 checkpoint walks 8/8 at 0.075 to 0.15 m/s but fidgets after a stop at 0.05 to 0.1 m/s
# (joint motion 0.04-0.37 rad/s RMS against 0.03, target steps to 0.04 rad against 0.002).
EXPERIMENTS["destination_v26"] = {
    **EXPERIMENTS["destination_v25"],
    "reward": replace(
        EXPERIMENTS["destination_v25"]["reward"],
        speed_curriculum_start_mps=None,
        speed_curriculum_step_mps=None,
        speed_curriculum_final_mps=None,
        speed_curriculum_threshold=None,
        speed_curriculum_window_controls=None,
        speed_curriculum_new_share=None,
        speed_curriculum_walks_only=None,
        speed_curriculum_advance_without_falls=None,
        speed_curriculum_pass_checks=None,
        speed_curriculum_measure=None,
        walk_speed_min_mps=0.05,
        walk_speed_max_mps=0.15,
        stop_fraction=0.5,
    ),
}
# destination_v27 is destination_v26 with a quarter of the walks stopping. Fine-tuned 600 updates from the
# native destination_v25 update-1350 checkpoint, v26 stood still (stops 8/8 at 0.05 to 0.1 m/s, quiet probes
# pass by update 300) but fell behind the ghost the longer it trained: the median end gap of the 0.05 m/s
# walk grew from 2.46 mm to 2.62, 3.32 and 3.96 mm at updates 100, 300 and 600.
EXPERIMENTS["destination_v27"] = {
    **EXPERIMENTS["destination_v26"],
    "reward": replace(EXPERIMENTS["destination_v26"]["reward"], stop_fraction=0.25),
}
# destination_v28 is destination_v27 with destination_v9's destination pay added back (the rational
# kernel at 3.7 mm, power 1.2: 0.62 at 2.5 mm, 0.23 at 1 cm) and scaled by how far the run is through,
# from 0 at its start to 1 at its last control, so the end of each run pays for millimetre gaps to the
# ghost. Every run draws one of four kinds: 10% turn in place at 0.2 rad/s for 20 s, 10% stand still for
# 20 s or 32 s (half each, as the evaluator's quiet probes), 50% walk 4 to 12 s and then stop for the rest
# of the 20 s, and 30% walk the whole 20 s; walks go in a uniform direction at 0.05 to 0.15 m/s.
# Episodes last up to 32 s and every other run ends at 20 s. Termination 52 keeps ten times the 5.15
# tracking ceiling. The native destination_v27 update-900 checkpoint stops 8/8 at every speed but its
# 0.05 m/s walk ends 3.34 mm from the ghost (v9 0.82 mm).
EXPERIMENTS["destination_v28"] = {
    **EXPERIMENTS["destination_v27"],
    "reward": replace(
        EXPERIMENTS["destination_v27"]["reward"],
        end_destination_weight=1.0,
        end_destination_distance_m=0.0037,
        end_destination_power=1.2,
        run_mix=(0.1, 0.1, 0.5, 0.3),
        long_stand_controls=1600,
        stop_fraction=None,
        termination_weight=52.0,
    ),
    "episode_seconds": 32.0,
}
# destination_v29 is destination_v28 whose run-end pay is linear in the gap to the ghost,
# share * max(1 - d / 25 mm, -1): 1 at the ghost, 0 at 25 mm and -1 from 50 mm, so each millimetre
# closed pays the same at any lag up to 50 mm. destination_v28's narrow kernel paid about 0.07 at 30 mm
# and barely changed there; its update-600 checkpoint falls 31 mm behind the ghost in the first second
# of a 0.15 m/s walk and holds the lag (36 mm at 20 s; 0.05 m/s walk 5.08 mm against 2.82).
EXPERIMENTS["destination_v29"] = {
    **EXPERIMENTS["destination_v28"],
    "reward": replace(
        EXPERIMENTS["destination_v28"]["reward"],
        end_destination_kernel="linear",
        end_destination_distance_m=0.025,
        end_destination_power=None,
        end_destination_floor=1.0,
    ),
}
# destination_v32 fine-tunes destination_v29 to walk forward over terrain: every run is a 20 s walk straight
# ahead at 0.05 to 0.15 m/s (no turns, stands or stops), and the policy's femur and tibia inputs gain the tilt
# that its front and rear toes measure (slope_inputs "joints", mdp/slope.py). The surrogate trains it on a
# terrain of hills, steps and bumps (locomotion.surrogate.train --terrain), each run from a random place and
# heading. destination_v29 with these inputs added at run time stalls at the foot of a 9.5 degree ramp in 6
# starts of 6, where without them it reaches the top in 3.
EXPERIMENTS["destination_v32"] = {
    **EXPERIMENTS["destination_v29"],
    "reward": replace(
        EXPERIMENTS["destination_v29"]["reward"],
        run_mix=(0.0, 0.0, 0.0, 1.0),
        walk_forward_only=True,
        slope_inputs="joints",
    ),
    "episode_seconds": 20.0,
}
# destination_v30 is destination_v25's speed curriculum (walks only, a fifth stopping, judged by drift and
# the evaluator's checks, to 0.15 m/s) with destination_v29's linear run-end pay, share * max(1 - d / 25 mm,
# -1), over each 20 s run. Fine-tuned from v9, destination_v29 tracked the ghost to 1-2 mm at 0.05 to
# 0.1 m/s and closed its starting lag, where the destination_v25 curriculum lineage held a steady lag.
# Termination 52 keeps ten times the 5.15 tracking ceiling.
EXPERIMENTS["destination_v30"] = {
    **EXPERIMENTS["destination_v25"],
    "reward": replace(
        EXPERIMENTS["destination_v25"]["reward"],
        end_destination_weight=1.0,
        end_destination_distance_m=0.025,
        end_destination_kernel="linear",
        end_destination_floor=1.0,
        termination_weight=52.0,
    ),
}
# destination_v33 is destination_v29 for the 2026-10-30 terrain campaign (runs/terrain_campaign):
# the v29 reward and run mix, with half the walks facing forward (walk_forward_bias 0.5, a chosen
# split) instead of all bearings uniform - destination_v32's forward-only walks regressed on flat
# ground and stand as the recorded cautionary control. Trained on terrain from random starts; the
# campaign arms vary only the recorded destination_input (terrain_v1, the 30 footfall-lookahead
# inputs of mdp/footfall.py, against ghost_v11), so arm A (terrain_v1) and arm C (ghost_v11) both
# train this task at equal experience, seed 20260917, 1000 updates.
EXPERIMENTS["destination_v33"] = {
    **EXPERIMENTS["destination_v29"],
    "reward": replace(
        EXPERIMENTS["destination_v29"]["reward"],
        walk_forward_bias=0.5,
    ),
}
# destination_v34 is destination_v33 plus the run-time slope inputs destination_v32 used
# (slope_inputs "joints", mdp/slope.py): campaign arm B, the honest rerun of v32's idea under the
# campaign mix, terrain, updates and seed, so the terrain_v1 comparison holds the slope-input
# alternative as a control.
EXPERIMENTS["destination_v34"] = {
    **EXPERIMENTS["destination_v33"],
    "reward": replace(EXPERIMENTS["destination_v33"]["reward"], slope_inputs="joints"),
}
# destination_v35 is destination_v33's reward unchanged with the terrain_v2 input (campaign round 3):
# terrain_v1's footfall lookahead plus a 6-point forward height profile along the command bearing
# (mdp/footfall.py PROFILE_DISTANCES_M, 5-40 cm). Rounds 1 and 2 (decisions.md 2026-10-30): every
# terrain-trained policy climbs the full 9.5-degree ramp and stalls at the crest, which one cycle of
# footfall lookahead (~10 cm at 0.1 m/s) cannot yet see; the profile puts the crest in view a body
# length ahead. Same experience as tc2_a (mixed_v3, seed 20260917, 1000 updates) isolates the
# profile's addition.
EXPERIMENTS["destination_v35"] = {**EXPERIMENTS["destination_v33"]}
# destination_v36 is destination_v35 with the run-end pay's linear ramp extended, 25 -> 75 mm
# (share * max(1 - d / 75 mm, -1): 0 at 75 mm, -1 from 150 mm). Campaign rounds 3-6 (decisions.md
# 2026-10-30/31): at a 2.5 cm step base the robot falls more than 50 mm behind the ghost, where the
# 25 mm ramp sits clamped at -1 - no gradient reaches the climb that would close the gap, and step
# crossings flicker (2/6 at one rung, 0/6 at the next). The wider ramp keeps each millimetre closed
# worth the same out to 150 mm of lag; the cost is a softer precision signal near the ghost, judged
# by the unchanged flat gate.
EXPERIMENTS["destination_v36"] = {
    **EXPERIMENTS["destination_v35"],
    "reward": replace(
        EXPERIMENTS["destination_v35"]["reward"],
        end_destination_distance_m=0.075,
    ),
}
# destination_v37 is destination_v35 with the terrain blocks on the critic alone
# (destination_input "terrain_v2_critic"): the required ablation for the native proposal.
# docs/TRAINING.md keeps privileged simulation inputs in the critic; this arm measures what that
# costs against tc3_a (terrain_v2 on actor and critic, same reward, terrain, start function, seed
# and updates). The actor keeps the nine ghost_v11 inputs it could run on the robot unchanged.
EXPERIMENTS["destination_v37"] = {**EXPERIMENTS["destination_v35"]}
# destination_v38 is destination_v35 retargeted at the 2026-11-01 campaign scope (decisions.md
# "NEW CAMPAIGN TARGET"): the four left/rear translate_0.10 failures trace to walk_forward_bias
# 0.5 starving side and rear bearings, turn_+/-0.40 to the single trained rate (the task's
# yaw_rate_rad_s, 0.2 — 0.4 rad/s was never commanded), and formal__0.16 plus the omni-0.15
# requirement sit at or past the trained 0.15 m/s top speed. Walks rebalance to bias 0.25 (kept
# nonzero: v32's forward-only walks regressed flat and v33 chose the bias to protect forward
# quality), each turn run draws 0.2 or 0.4 rad/s, the turn share rises 0.10 -> 0.15 paid by
# walks-that-stop (all in-scope stops pass), and walk speeds extend to 0.17 m/s — below the
# 0.20 tier the record excludes as torque-infeasible. The reward terms are v35's unchanged.
EXPERIMENTS["destination_v38"] = {
    **EXPERIMENTS["destination_v35"],
    "reward": replace(
        EXPERIMENTS["destination_v35"]["reward"],
        walk_forward_bias=0.25,
        run_mix=(0.15, 0.10, 0.45, 0.30),
        walk_speed_max_mps=0.17,
        turn_rate_choices_rad_s=(0.2, 0.4),
    ),
}
# destination_v39 is destination_v38 with the over-rating torque charge tripled (10 -> 30):
# the omni-0.15 judge fails ten of twelve bearings on computed demand over the 1.6 Nm rating
# (0.0056-0.0278 vs the 0.005 bound) and the continuation control tc9_ctrl shows the demand
# habit worsening with training — at weight 10 a ~1% violation rate costs ~0.1/control against
# ~1.5 of tracking pay, so violating stays profitable. One change against v38; the 9.5-degree
# crest median is the recorded canary for pricing torque the climb may need.
EXPERIMENTS["destination_v39"] = {
    **EXPERIMENTS["destination_v38"],
    "reward": replace(EXPERIMENTS["destination_v38"]["reward"], over_rating_weight=30.0),
}
# destination_v40 is destination_v38 with a turn-heavy mix as the 0.40 rad/s stepping stone:
# after 200 v38 updates both turn_+/-0.40 still fail (yaw error 0.26/0.36 vs 0.10, demand
# 0.043-0.056) — a 7.5% share per drawn rate is too little practice for a never-trained
# capability. Turns take 0.35 of runs and draw 0.3 or 0.4 rad/s (0.3 bridges from the mastered
# 0.2), paid by walks; walks keep the v38 bias and speeds. One change against v38, meant for
# short dedicated legs whose walking quality is re-anchored afterwards, not for long runs.
EXPERIMENTS["destination_v40"] = {
    **EXPERIMENTS["destination_v38"],
    "reward": replace(
        EXPERIMENTS["destination_v38"]["reward"],
        run_mix=(0.35, 0.10, 0.35, 0.20),
        turn_rate_choices_rad_s=(0.3, 0.4),
    ),
}
# destination_v41 is destination_v38 with the swing-clearance pay on (weight 0.5, subordinate to
# the schedule's 2.0): the 2.5 cm step wall held at ~0.9 m through five input/reward experiments
# and the rebalanced v38 distribution alike — the recorded diagnosis is a gait barrier at the
# riser (feet never clear it), and no clearance term existed. Scheduled-swing unloaded toes earn
# in proportion to height above the local ground up to 3.5 cm (inside the 6.15 cm kinematic
# swing ceiling), only where the ground's 5-point spread exceeds 8 mm — flat ground, its 5 mm
# training noise included, pays nothing by construction, answering the recorded high-stepping
# objection. One change against v38.
EXPERIMENTS["destination_v41"] = {
    **EXPERIMENTS["destination_v38"],
    "reward": replace(EXPERIMENTS["destination_v38"]["reward"], swing_clearance_weight=0.5),
}
# destination_v42 is the consolidation recipe: destination_v38's bearings and speeds with the
# round-B turn lever at consolidation dose — turns take 0.20 of runs and draw 0.3 or 0.4 rad/s
# (arm b3 halved turn_+0.40's yaw error with ~70 such runs; 1000 updates at 0.20 give ~200,
# and b3 held the ±0.20 cases without practising 0.2). Clearance stays off (tc9_c1). Meant for
# the NATIVE retrain, where fresh runs showed none of the surrogate legs' drift tax.
EXPERIMENTS["destination_v42"] = {
    **EXPERIMENTS["destination_v38"],
    "reward": replace(
        EXPERIMENTS["destination_v38"]["reward"],
        run_mix=(0.20, 0.10, 0.40, 0.30),
        turn_rate_choices_rad_s=(0.3, 0.4),
    ),
}
# destination_v43 is destination_v38 with the landing-conditioned lift on (weight 0.5,
# subordinate to the schedule's 2.0): tc9_c1 showed apex pay over the current cell makes the
# 2.5 cm steps WORSE — the stalls read as horizontal toe-riser contact, which height over the
# ground behind the riser cannot prevent. This successor pays a swinging unloaded toe only when
# its predicted touchdown (the footfall schedule-and-command advance, ring maximum) lands at
# least 1 cm above the ground under the toe now — the 2.5 cm risers, the 2-4 cm bumps and the
# upper 9.5-degree ramp open the gate; flat ground, descents and the held 4/7-degree ramps
# never do — in proportion to its clearance over the current ground, saturating 2 cm above the
# landing ground itself, so a taller obstacle demands a proportionally higher swing. One change
# against v38.
EXPERIMENTS["destination_v43"] = {
    **EXPERIMENTS["destination_v38"],
    "reward": replace(EXPERIMENTS["destination_v38"]["reward"], landing_lift_weight=0.5),
}
# destination_v44 is destination_v42 (the native consolidation recipe: v38 bearings and
# speeds, turns at 0.20 share drawing 0.3/0.4 rad/s) with the landing-conditioned lift on at
# v43's proven dose. The tc9_d2 leg (v43, 1000 surrogate updates from tc9_a u200) crossed the
# 9.5-degree hill 6/6 where its matched-experience control (tc9_a u1000, v38 alone) managed
# 2/6 bimodal, held every held course and scored the best in-scope flat of any leg (31/36)
# at a horizon that collapsed the control — the lift is proven and consolidates natively,
# where fresh runs showed none of the surrogate legs' drift tax.
EXPERIMENTS["destination_v44"] = {
    **EXPERIMENTS["destination_v42"],
    "reward": replace(EXPERIMENTS["destination_v42"]["reward"], landing_lift_weight=0.5),
}
# destination_v45 is destination_v44 with arcing walks: the run mix has only ever drawn pure
# turns, stands and straight walks, so the scored turn-while-walk families were never in the
# command distribution — v29's near-threshold misses and u600_002's demand-heavy arc failures
# both say exposure is the binding lever (decisions.md, the 2026-11-03 arc scope ruling). A
# quarter of the drawn walks now carry a signed yaw from the two scored arc tiers (0.05 rad/s
# under gentle_arc, 0.2 rad/s under combined and the transition's diagonal turns); the stopping
# walks arc too, so stops from an arc are practised. One change against v44.
EXPERIMENTS["destination_v45"] = {
    **EXPERIMENTS["destination_v44"],
    "reward": replace(
        EXPERIMENTS["destination_v44"]["reward"],
        arc_fraction=0.25,
        arc_yaw_choices_rad_s=(0.05, 0.2),
    ),
}
# destination_v46 is destination_v43 (v38 + landing lift, the 31/36 flat champion at the
# 1000-update surrogate horizon) with the arcing walks alone — no v42 turn-mix change, which
# the tw round showed costs flat precision at long horizons (tw_ctrl2 25/36 vs tc9_d2 31/36).
# The arc share is the reproduced steps-2.5 lever (tw_a2/tw_a3, 2/6 each vs controls 0/6);
# tc9_d2 is this experiment's matched-experience control. One change against v43.
EXPERIMENTS["destination_v46"] = {
    **EXPERIMENTS["destination_v43"],
    "reward": replace(
        EXPERIMENTS["destination_v43"]["reward"],
        arc_fraction=0.25,
        arc_yaw_choices_rad_s=(0.05, 0.2),
    ),
}
# destination_v48 is destination_v45 with the turn share halved (0.20 -> 0.10, returned to the
# walks): the tw round showed the turn-heavy mix costs flat precision at long horizons
# (tc9_d2 31/36 vs tw_ctrl2 25/36) while v46 showed the arc share NEEDS the turn mix for the
# steps and crest synergies (tw_b lost both without it). This arm asks whether a lighter turn
# share keeps the synergies at a smaller flat cost. One change against v45.
EXPERIMENTS["destination_v48"] = {
    **EXPERIMENTS["destination_v45"],
    "reward": replace(
        EXPERIMENTS["destination_v45"]["reward"],
        run_mix=(0.10, 0.10, 0.45, 0.35),
    ),
}
# destination_v47 is destination_v45 with the mirrored tripod schedule on clockwise turns:
# the chirality probe measured counter-clockwise 0.40 rad/s tracking at 98 percent while
# clockwise stalls at 43 percent with tilt spikes, on the native baseline too, and round B
# showed practice cannot fix it — the fixed first-tripod convention hands the gait a
# preferred spin. Mirroring swaps the tripods (schedule reward and footfall predictions
# together) on rows whose commanded yaw is negative; straight walks, stands and
# counter-clockwise turns are byte-identical. One change against v45.
EXPERIMENTS["destination_v47"] = {
    **EXPERIMENTS["destination_v45"],
    "reward": replace(
        EXPERIMENTS["destination_v45"]["reward"],
        schedule_mirror_turns=True,
    ),
}
# destination_v49 raises the roll-pitch rate penalty on the arcing recipe for the anchored
# flat polish (the two-phase consolidation's second phase): the formal cases fail on a
# ~0.73-0.95 degree tilt rms — the recorded standing lean — and tw_g's levelest-ever turn
# (0.53 degrees) came from flat polish alone. One change against v45.
EXPERIMENTS["destination_v49"] = {
    **EXPERIMENTS["destination_v45"],
    "reward": replace(EXPERIMENTS["destination_v45"]["reward"], roll_pitch_weight=0.3),
}
# destination_v50 drops the forward walk bias (uniform bearings) for the anchored flat
# polish: the two surviving translate misses are the rear bearings the 0.25 forward bias
# under-practises, and the omni demand failures sit at the side and rear too. One change
# against v45.
EXPERIMENTS["destination_v50"] = {
    **EXPERIMENTS["destination_v45"],
    "reward": replace(EXPERIMENTS["destination_v45"]["reward"], walk_forward_bias=None),
}
# destination_v51 doubles the arc share for the anchored flat polish: tw_g's gentle and
# combined misses are near-threshold (planar 0.024-0.044 vs 0.03, yaw 0.053-0.061, demand
# 0.006-0.009 vs 0.005) with the demand bound co-failing on all sixteen gentle cases, and
# the anchor holds displacement only where gradients are strong and persistent — so the
# polish mix itself must carry the arc gradient. Half the drawn walks now arc. One change
# against v45.
EXPERIMENTS["destination_v51"] = {
    **EXPERIMENTS["destination_v45"],
    "reward": replace(EXPERIMENTS["destination_v45"]["reward"], arc_fraction=0.5),
}
# destination_v52 composes the two proven single-change polish levers: v51's doubled arc
# share (tw_l: the first fully consistent crest, 6/6 at full distance) and v50's uniform
# bearings (tw_i: the first rear-translate pass, translate_0.10_180). Each bought its case
# and paid terrain or turn elsewhere; this asks whether the gains compose. One change
# against v51, judged against tw_l and tw_i, its single-change parents.
EXPERIMENTS["destination_v52"] = {
    **EXPERIMENTS["destination_v51"],
    "reward": replace(EXPERIMENTS["destination_v51"]["reward"], walk_forward_bias=None),
}
# destination_v53 puts real weight on standing tilt. The stage2c formal screen bounds
# tilt RMS at 0.50 deg (speed 0.00) and 0.90 deg (0.16); the whole v29_terrain_v2 lineage
# sits at 0.84-0.88 deg regardless of length (u200 through u1200) or terrain, while the
# terrain-master lineage holds 0.33-0.44 — under v45's tilt term (weight 0.5 at a 5 deg
# scale) that wobble costs ~0.014 reward units, so nothing ever ground it down. Weight 15
# prices 0.84 deg at ~0.42 units and leaves the bound itself at ~0.15. One field against
# v45, judged against tw_o/tw_p, its single-change parents at u200 and u1200.
EXPERIMENTS["destination_v53"] = {
    **EXPERIMENTS["destination_v45"],
    "reward": replace(EXPERIMENTS["destination_v45"]["reward"], tilt_weight=15.0),
}
# v54/v55 each price one residual whisker class of the v53 plateau (round 13): every
# in-goal miss at the plateau's best points is a near-threshold yaw-tracking error
# (turn_-0.40 +4.8%, gentle_arc_315 pair +1-3%, combined yaw +11%) or an instantaneous
# torque spike past the 1.6 Nm rating (gentle_135 pair, omni15 135/315 deg, +4-20%).
# One field each against v53, judged against tw_u_u1200 at matched length and seed.
EXPERIMENTS["destination_v54"] = {
    **EXPERIMENTS["destination_v53"],
    # One lever, two coupled fields: the validator requires termination to stay ten times
    # the tracking ceiling, which the yaw raise lifts from 5.15 to 5.8.
    "reward": replace(
        EXPERIMENTS["destination_v53"]["reward"], yaw_tracking_weight=1.0, termination_weight=58.5
    ),
}
EXPERIMENTS["destination_v55"] = {
    **EXPERIMENTS["destination_v53"],
    "reward": replace(EXPERIMENTS["destination_v53"]["reward"], acceleration_weight=0.2),
}
# destination_v56 prices each tracking axis by how hard the row's command exercises it
# (rewards.situation_weights): up to twice the registered weight at the manifest's maxima
# (0.20 m/s, 0.40 rad/s), unchanged where the axis is idle. The global raises moved whiskers
# instead of removing them (v54 round 14: new yaw misses at 225/270 and omni -3; v55: demand
# misses traded 135 for 292.5) because a flat price also buys stillness where nothing is
# commanded; a situation price pays turn_-0.40 at x2 yaw, combined yaw at x1.5 and a quiet
# stand at x1. One field against v53.
EXPERIMENTS["destination_v56"] = {
    **EXPERIMENTS["destination_v53"],
    # One lever, two coupled fields: the full factor lifts the validator's tracking ceiling
    # from 5.15 to 7.1, so termination must reach 71.
    "reward": replace(
        EXPERIMENTS["destination_v53"]["reward"], situation_tracking=1.0, termination_weight=71.5
    ),
}
# destination_v57 keeps v56's situation-priced tracking and prices the stand fidget it pays
# for: tw_z (round 17) moved every alignment axis forward but stage2_long's quiet joint
# range (0.0217 at u1200, 0.0309 at u1400 against the 0.02 quiet gate). quiet_target_weight
# is the term that prices target wander while still, so it carries the raise - and being
# gated by stillness it cannot touch the moving families v56 just won.
EXPERIMENTS["destination_v57"] = {
    **EXPERIMENTS["destination_v56"],
    "reward": replace(EXPERIMENTS["destination_v56"]["reward"], quiet_target_weight=2.0),
}
# destination_v58 adds the quiet pose hold to v56. Every v56 point carries stage2_long's
# joint range at 0.0211-0.0309 against the 0.02 quiet gate while quiet_target x4 (v57) and
# anchor x2/x4 (rounds 19-20) never moved it: the gate bounds the window RANGE of positions,
# and the rate and step prices are quadratically blind to slow wander (~0.002 rad/s costs
# ~1e-5 units). quiet_hold prices deviation from the settled pose, captured on the first
# still control (anchoring at quiet onset fought the settle and cost formal: tw_q, round 21);
# at the observed
# half-ranges (0.011-0.016 rad over the 0.01 scale) weight 3 costs 3.6-7.7 units per still
# control, the scale of price that cracked formal.
EXPERIMENTS["destination_v58"] = {
    **EXPERIMENTS["destination_v56"],
    "reward": replace(EXPERIMENTS["destination_v56"]["reward"], quiet_hold_weight=3.0),
}
# destination_v59 gives the hold its own three-second settle. The settled-pose hold (tw_q2,
# round 22) cured the quiet range at every length and unlocked gentle 16/16 and omni 16/16,
# but the pose frozen after one second still leans: formal 0.00 tilt 0.57-0.71 deg against
# the 0.50 bound, and once held, a 0.02 rad correction costs ~12 hold units against ~0.14
# tilt units saved, so the lean is locked in. Capturing at 150 controls hands the hold the
# level pose the tilt price has had three seconds to find.
EXPERIMENTS["destination_v59"] = {
    **EXPERIMENTS["destination_v58"],
    "reward": replace(EXPERIMENTS["destination_v58"]["reward"], quiet_hold_grace_controls=150),
}

# The evaluator's moving checks (``evaluation/scoring.py``, profile omni_static) and the controls it skips
# at the start of each case; the surrogate clamps joints at the native speed limit.
WALK_SKIPPED_CONTROLS = 100
WALK_CHECKS = (
    "planar_error_mps",
    "yaw_error_rad_s",
    "tilt_rms_degrees",
    "vertical_velocity_rms_mps",
    "computed_demand_over_rating_fraction",
    "joint_speed_limit",
    "terminated",
)
JOINT_SPEED_LIMIT_RAD_S = 50.26548245743669


def walk_failures(sums, counted, speed, fast, fell, planar_scale=1.0):
    """Each row's failed walking checks, as booleans in ``WALK_CHECKS`` order, from the per-control sums of
    planar error, yaw-rate error, squared tilt, squared vertical velocity and share of joints above 1.6 Nm.
    ``planar_scale`` multiplies the evaluator's planar-error bound."""
    mean = sums / counted.clamp_min(1)[:, None]
    return torch.stack(
        (
            mean[:, 0] > planar_scale * torch.clamp(0.25 * speed, min=0.025),
            mean[:, 1] > 0.06,
            torch.rad2deg(mean[:, 2].sqrt()) > 5.0,
            mean[:, 3].sqrt() > 0.04,
            mean[:, 4] > 0.005,
            fast,
            fell,
        ),
        -1,
    )


def linear_tail(distance, closeness, config):
    """Continue the rational kernel past ``destination_linear_from_m`` along its tangent there, down to a
    penalty of ``destination_penalty_cap``."""
    start, power = config.destination_linear_from_m, config.destination_power
    ratio = start / config.destination_distance_m
    value = 1 / (1 + ratio**power)
    slope = -power * ratio ** (power - 1) / config.destination_distance_m * value**2
    tail = (value + slope * (distance - start)).clamp_min(-config.destination_penalty_cap)
    return torch.where(distance > start, tail.to(closeness), closeness)


class DestinationTrainingTask(TrainingTask):
    """The canonical task plus each command's destination and the per-control destination pay."""

    def __init__(self, env, config=None, output_dir=None, reward_config=None):
        chosen = EXPERIMENTS["destination_v1"]["reward"] if reward_config is None else reward_config
        if not isinstance(chosen, DestinationRewardConfig):
            raise ValueError("The destination task needs a DestinationRewardConfig")
        self.destination = torch.zeros(env.num_envs, 3, device=env.device)
        self.command_started = torch.ones(env.num_envs, dtype=torch.bool, device=env.device)
        self.command_elapsed = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.hold_anchor = torch.zeros(env.num_envs, 18, device=env.device)
        self.hold_mean = torch.zeros(env.num_envs, 18, device=env.device)
        self.stop_pending = torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        self.stop_rest = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        # Each run's planned length and the controls it has run, for the run-end pay and the run mix.
        self.run_length = torch.full(
            (env.num_envs,), chosen.command_controls, dtype=torch.long, device=env.device
        )
        self.run_elapsed = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        # The tilt from the front and rear toes, and the inputs it corrects (mdp/slope.py).
        self.slope, self.slope_resets = None, None
        if chosen.slope_inputs is not None:
            from locomotion.omniwalk.mdp.slope import SlopeTracker

            self.slope = SlopeTracker(env.num_envs, env.model["joints"], chosen.slope_inputs)
        # The root pose before the latest control, for the ghost_v10 velocity input.
        self.control_start_pose = env.current["root"].detach().clone()
        # The terrain grid for the swing-clearance telemetry, resolved on first use ("unset"
        # because the stock flat floor is a valid None).
        self._clearance_grid = "unset"
        self.gait = (
            None
            if chosen.gait_clock_scaling is None
            else GaitPhase(env.num_envs, chosen.gait_clock_scaling, env.device)
        )
        if chosen.speed_curriculum_start_mps is not None:
            self.curriculum_level = round(
                chosen.speed_curriculum_start_mps + chosen.speed_curriculum_step_mps, 6
            )
            self.curriculum_window = torch.zeros(
                chosen.speed_curriculum_window_controls, dtype=torch.float64, device=env.device
            )
            # Falls per control over the same window, for advancing only after a fall-free window.
            self.curriculum_falls = torch.zeros_like(self.curriculum_window)
            # Robots further than the back-off distance from their ghost, per control over the window.
            self.curriculum_lost = torch.zeros_like(self.curriculum_window)
            self.curriculum_filled = 0
            span = chosen.speed_curriculum_backoff_window_controls
            self.curriculum_backoff_window = torch.zeros(
                span or 1, dtype=torch.float64, device=env.device
            )
            self.curriculum_backoff_filled = 0
            self.curriculum_complete = False
            self.curriculum_advances = []
            # Walking at the top speed since the level began: controls and their summed speed along the command.
            self.top_speed_controls = torch.zeros((), dtype=torch.float64, device=env.device)
            self.top_speed_sum = torch.zeros((), dtype=torch.float64, device=env.device)
            # Walks finished, and those that failed a check, per control over the window.
            self.curriculum_walks = torch.zeros_like(self.curriculum_window)
            self.curriculum_failed = torch.zeros_like(self.curriculum_window)
            self.pending_walks = self.pending_failed = 0
            self.check_failures = dict.fromkeys(WALK_CHECKS, 0)
            n, device = env.num_envs, env.device
            self.walk_counted = torch.zeros(n, dtype=torch.float64, device=device)
            self.walk_sums = torch.zeros(n, 5, dtype=torch.float64, device=device)
            self.walk_fast = torch.zeros(n, dtype=torch.bool, device=device)
            self.walk_fell = torch.zeros(n, dtype=torch.bool, device=device)
            # Each walk's planned length, its gap to the ghost at halfway and now, and whether it fell.
            self.drift_plan = torch.zeros(n, dtype=torch.long, device=device)
            self.drift_walking = torch.zeros(n, dtype=torch.bool, device=device)
            self.drift_middle = torch.zeros(n, 2, dtype=torch.float64, device=device)
            self.drift_halfway = torch.zeros(n, dtype=torch.bool, device=device)
            self.drift_end = torch.zeros(n, 2, dtype=torch.float64, device=device)
            self.drift_fell = torch.zeros(n, dtype=torch.bool, device=device)
            # Walks finished per control over the window, the sum of their scores and of their drift.
            self.curriculum_scored = torch.zeros_like(self.curriculum_window)
            self.curriculum_score = torch.zeros_like(self.curriculum_window)
            self.curriculum_drift = torch.zeros_like(self.curriculum_window)
            self.pending_scored = self.pending_score = self.pending_drift = 0.0
        super().__init__(env, config, output_dir, chosen)

    @property
    def gait_clock_phase(self):
        """The speed-scaled gait phase in cycles that the wrapper shows the policy, or None."""
        return None if self.gait is None else self.gait.phase

    def reset(self, indices=None):
        # A reset draws a fresh command; a stop scheduled for the ended episode lapses.
        self.stop_pending[self._indices(indices)] = False
        self.run_elapsed[self._indices(indices)] = 0
        if self.slope is not None:
            self.slope_resets = self._indices(indices).cpu()
            self.slope.reset(self.slope_resets)
        if self.gait is not None:
            self.gait.reset(self._indices(indices))
        output = super().reset(indices)
        selected = self._indices(indices)
        self.control_start_pose[selected] = (
            self.env.current["root"][selected].detach().to(self.control_start_pose)
        )
        return output

    def reward_telemetry(self, command):
        telemetry = super().reward_telemetry(command)
        if self.gait is not None:
            # The schedule at the phase this control ran under, with its command's cycle.
            swing, period = self.gait.swing(command, bool(self.reward_config.schedule_mirror_turns))
            device = telemetry["scheduled_swing"].device
            telemetry["scheduled_swing"] = swing.to(device)
            telemetry["schedule_period_s"] = period.to(device, torch.float32)
        if self.reward_config.swing_clearance_weight or self.reward_config.landing_lift_weight:
            telemetry.update(self._swing_clearance_telemetry(telemetry))
        if self.reward_config.landing_lift_weight:
            telemetry.update(self._landing_lift_telemetry(telemetry, command))
        return telemetry

    def _swing_clearance_telemetry(self, telemetry):
        """Toe height above the local ground and the ground's 5-point spread, [n, 6] each,
        sampled on detached cpu float64 copies of the state so native and surrogate values
        stay bitwise equal (the footfall grid discipline). The stock flat floor reports the
        toe's world height over zero roughness, which the rough gate prices at nothing."""
        world = telemetry["toe_xyz_world"].detach().cpu()
        if self._clearance_grid == "unset":
            terrain = getattr(self.env, "terrain", None)
            self._clearance_grid = None if terrain is None else TerrainGrid(terrain)
        if self._clearance_grid is None:
            return {
                "toe_clearance_m": world[..., 2].float().clone(),
                "toe_rough_m": torch.zeros(world.shape[0], 6),
            }
        x, y = world[..., 0], world[..., 1]
        samples = torch.stack([self._clearance_grid.sample(x + dx, y + dy) for dx, dy in RING_XY])
        return {
            "toe_clearance_m": world[..., 2].float() - samples[0],
            "toe_rough_m": samples.amax(0) - samples.amin(0),
        }

    def _landing_lift_telemetry(self, telemetry, command):
        """Each toe's predicted-touchdown ground rise over the ground under it now, [n, 6]: the
        footfall module's schedule-and-command touchdown prediction, sampled at the five ring
        points and taken at their maximum so a prediction landing just short of a riser still
        reads the riser top, on the same detached cpu float64 copies as the footfall inputs.
        The stock flat floor reports zero rise, which the reward's rise gate prices at nothing."""
        world = telemetry["toe_xyz_world"].detach().cpu()
        if self._clearance_grid == "unset":
            terrain = getattr(self.env, "terrain", None)
            self._clearance_grid = None if terrain is None else TerrainGrid(terrain)
        if self._clearance_grid is None:
            return {"toe_landing_rise_m": torch.zeros(world.shape[0], 6)}
        if self.gait is None:
            raise ValueError("The landing lift needs the task's speed-scaled gait clock")
        state = self.env.current
        cpu_command = command.detach().cpu()
        seconds = touchdown_seconds(
            self.gait.phase.detach().cpu(),
            cpu_command,
            self.reward_config.gait_clock_scaling,
            bool(self.reward_config.schedule_mirror_turns),
        )
        points = touchdown_points(
            state["root"].detach().cpu(), cpu_command, state["toe_body"].detach().cpu(), seconds
        )
        ring = torch.tensor(RING_XY, dtype=torch.float64)
        landing = self._clearance_grid.sample(
            points[:, :, None, 0] + ring[None, None, :, 0],
            points[:, :, None, 1] + ring[None, None, :, 1],
        ).amax(-1)
        ground = self._clearance_grid.sample(world[..., 0], world[..., 1])
        return {"toe_landing_rise_m": landing - ground}

    def _next_command_observation(self, output):
        output = super()._next_command_observation(output)
        if self.slope is None:
            return output
        if self.slope_resets is not None:
            current = self.env.current
            self.slope.push(
                current["root"][:, 3:7].cpu(), current["q"].cpu(), rows=self.slope_resets
            )
            self.slope_resets = None
        obs = self.slope.apply(output["obs"])
        critic = output["critic"].clone()
        critic[:, : obs.shape[1]] = obs
        return {**output, "obs": obs, "critic": critic}

    def step(self, action):
        result = super().step(action)
        if self.reward_config.run_mix is not None:
            # A run shorter than the episode times out at its planned length.
            ended = (self.run_elapsed >= self.run_length) & ~result["terminated"]
            result["truncated"] = result["truncated"] | ended.to(result["truncated"].device)
        return result

    def _resample(self, indices):
        if self.reward_config.run_mix is not None:
            # A run at its planned length keeps its command until the reset that ends it.
            indices = indices[self.run_elapsed[indices] < self.run_length[indices]]
        if self.reward_config.speed_curriculum_pass_checks and len(indices):
            self._finish_walks(indices)
        drift = self.reward_config.speed_curriculum_measure == "drift"
        if drift and len(indices):
            self._finish_drift(indices)
        super()._resample(indices)
        if len(indices):
            if self.reward_config.speed_curriculum_start_mps is not None:
                self._curriculum_walks(indices)
            elif self.reward_config.run_mix is not None:
                self._mixed_runs(indices)
            elif self.reward_config.walk_speed_min_mps is not None:
                self._uniform_walks(indices)
            self.command_started[indices] = True
            if self.reward_config.command_controls and self.reward_config.run_mix is None:
                self.remaining_controls[indices] = self.reward_config.command_controls
            if self.reward_config.stop_fraction is not None:
                self._schedule_stops(indices)
            if drift:
                moving = (self.env.commands[indices] != 0).any(-1).to(self.drift_walking.device)
                self.drift_walking[indices] = moving
                self.drift_plan[indices] = self.remaining_controls[indices].to(self.drift_plan)
                self.drift_halfway[indices] = False
                self.drift_fell[indices] = False

    def _uniform_walks(self, indices):
        """Make every fresh command a walk in a uniform direction at a uniform speed between the bounds."""
        config = self.reward_config
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.env.commands.device)
        draws = torch.rand(len(indices), 3, generator=self.rng).to(self.env.commands)
        speed = (
            config.walk_speed_min_mps
            + (config.walk_speed_max_mps - config.walk_speed_min_mps) * draws[:, 0]
        )
        angle = draws[:, 1] * 2 * math.pi
        if config.walk_forward_only:
            angle = torch.zeros_like(angle)
        elif config.walk_forward_bias is not None:
            # The biased share faces forward; the rest keep their uniform bearing.
            angle = torch.where(
                draws[:, 2] < config.walk_forward_bias, torch.zeros_like(angle), angle
            )
        self.env.commands[indices, 0] = speed * torch.cos(angle)
        self.env.commands[indices, 1] = speed * torch.sin(angle)
        self.env.commands[indices, 2] = 0

    def _mixed_runs(self, indices):
        """Stop the walks whose walking part ended; draw each fresh run a turn, stand, walk that stops or walk."""
        config = self.reward_config
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.stop_pending.device)
        stopping = indices[self.stop_pending[indices]]
        self.env.commands[stopping] = 0
        self.remaining_controls[stopping] = self.stop_rest[stopping].to(self.remaining_controls)
        self.stop_pending[stopping] = False
        fresh = indices[~torch.isin(indices, stopping)]
        if not len(fresh):
            return
        # Extra draw columns exist only when their knobs are set, so the RNG stream of every
        # experiment without them stays byte-identical.
        columns = 2 if config.turn_rate_choices_rad_s is None else 3
        arc_columns = None
        if config.arc_fraction is not None:
            arc_columns = (columns, columns + 1)
            columns += 2
        draws = torch.rand(len(fresh), columns, generator=self.rng).to(fresh.device)
        edges = torch.tensor(config.run_mix, dtype=torch.float64).cumsum(0).to(fresh.device)
        kind = torch.bucketize(draws[:, 0].double(), edges[:-1], right=True)  # 0 turn .. 3 walk
        self.run_length[fresh] = config.command_controls
        self.remaining_controls[fresh] = config.command_controls
        self._uniform_walks(fresh[kind >= 2])
        if arc_columns is not None:
            # A drawn share of the walks, stopping walks included, carries a signed yaw rate:
            # one draw picks the arcing walks, the other their rate and sign together.
            walking = kind >= 2
            member = draws[walking, arc_columns[0]]
            arcs = fresh[walking][member < config.arc_fraction]
            choices = torch.tensor(config.arc_yaw_choices_rad_s).to(self.env.commands)
            signed = (
                draws[walking, arc_columns[1]][member < config.arc_fraction] * 2 * len(choices)
            ).long()
            signed = signed.clamp_(max=2 * len(choices) - 1)
            rate = choices[signed % len(choices)]
            sign = torch.where(signed >= len(choices), 1.0, -1.0).to(rate)
            self.env.commands[arcs, 2] = sign * rate
        turns = fresh[kind == 0]
        sign = torch.where(draws[kind == 0, 1] < 0.5, -1.0, 1.0).to(self.env.commands)
        rate = self.config.yaw_rate_rad_s
        if config.turn_rate_choices_rad_s is not None:
            choices = torch.tensor(config.turn_rate_choices_rad_s).to(self.env.commands)
            pick = (draws[kind == 0, 2] * len(choices)).long().clamp_(max=len(choices) - 1)
            rate = choices[pick]
        self.env.commands[turns] = 0
        self.env.commands[turns, 2] = sign * rate
        stands = fresh[kind == 1]
        self.env.commands[stands] = 0
        long = stands[draws[kind == 1, 1] < 0.5]
        self.run_length[long] = config.long_stand_controls
        self.remaining_controls[long] = config.long_stand_controls
        stops = fresh[kind == 2]
        walking = torch.randint(
            config.stop_after_min_controls,
            config.stop_after_max_controls + 1,
            (len(stops),),
            generator=self.rng,
        ).to(stops.device)
        self.remaining_controls[stops] = walking.to(self.remaining_controls)
        self.stop_rest[stops] = config.command_controls - walking
        self.stop_pending[stops] = True

    def _curriculum_walks(self, indices):
        """Give each fresh walk a uniform direction and the top or a slower curriculum speed."""
        config = self.reward_config
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.env.commands.device)
        command = self.env.commands[indices]
        if config.speed_curriculum_walks_only:
            # Every fresh command becomes a walk; the stop schedule then zeroes the rows it ends.
            self.env.commands[indices, 2] = 0
            walks = indices
        else:
            walks = indices[(command[:, :2] != 0).any(-1) & (command[:, 2] == 0)]
        if not len(walks):
            return
        step = config.speed_curriculum_step_mps
        top = round(self.curriculum_level / step)
        draws = torch.rand(len(walks), 3, generator=self.rng)
        level = torch.full((len(walks),), top, dtype=torch.long)
        if top > 1:
            # The slower levels are 1 to top - 1 steps.
            slower = 1 + (draws[:, 1] * (top - 1)).long().clamp(max=top - 2)
            level = torch.where(draws[:, 0] < config.speed_curriculum_new_share, level, slower)
        speed = (level * step).to(self.env.commands)
        angle = (draws[:, 2] * 2 * math.pi).to(self.env.commands)
        self.env.commands[walks, 0] = speed * torch.cos(angle)
        self.env.commands[walks, 1] = speed * torch.sin(angle)

    def _record_walks(self, command, telemetry, terminated):
        """Add this control to each walk's sums for the evaluator's moving checks after its first 100 controls."""
        t = telemetry
        command = command.to(self.walk_sums.device)
        moving = (command != 0).any(-1)
        counted = moving & (self.command_elapsed.to(moving.device) > WALK_SKIPPED_CONTROLS)
        pose = t["root_pose_xyzw"].to(self.walk_sums)
        x, y, z, w = pose[:, 3:].unbind(-1)
        up = 1 - 2 * (x * x + y * y)
        # The evaluator reads the root's centre-of-mass velocity, as the environment's state holds it.
        world = self.env.current["root_velocity"][:, :3].to(self.walk_sums)
        planar = navigation(inverse_rotate(pose[:, 3:], world))[:, :2] - command[:, :2].to(
            self.walk_sums
        )
        vertical = world[:, 2]
        values = torch.stack(
            (
                torch.linalg.vector_norm(planar, dim=-1),
                (t["angular_velocity_body"].to(self.walk_sums)[:, 2] - command[:, 2]).abs(),
                torch.arccos(up.clamp(-1, 1)).square(),
                vertical.square(),
                (t["computed_torque_nm"].to(self.walk_sums).abs() > 1.6)
                .to(self.walk_sums)
                .mean(-1),
            ),
            -1,
        )
        self.walk_sums += values * counted[:, None]
        self.walk_counted += counted.to(self.walk_counted)
        fast = (t["joint_velocity_rad_s"].abs() > JOINT_SPEED_LIMIT_RAD_S).any(-1)
        self.walk_fast |= fast.to(counted.device) & counted
        self.walk_fell |= terminated.to(moving.device) & moving

    def _record_drift(self, telemetry, terminated):
        """Keep each walk's gap to its ghost at its halfway control and at its latest control."""
        actual = planar_pose(telemetry["root_pose_xyzw"]).to(self.drift_end)
        gap = self.destination[:, :2].to(self.drift_end) - actual[:, :2]
        halfway = self.command_elapsed.to(self.drift_plan.device) == (
            self.drift_plan // 2
        ).clamp_min(1)
        self.drift_middle = torch.where(halfway[:, None], gap, self.drift_middle)
        self.drift_halfway |= halfway
        self.drift_end = gap
        self.drift_fell |= terminated.to(self.drift_fell.device)

    def _finish_drift(self, indices):
        """Score the walks that end at these rows: the destination kernel of the gap's movement from halfway
        to the end, or 0 after a fall. Walks that neither reached halfway nor fell are not scored."""
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.drift_walking.device)
        rows = indices[
            self.drift_walking[indices] & (self.drift_halfway[indices] | self.drift_fell[indices])
        ]
        if len(rows):
            config = self.reward_config
            drift = torch.linalg.vector_norm(self.drift_end[rows] - self.drift_middle[rows], dim=-1)
            score = 1 / (1 + (drift / config.destination_distance_m).pow(config.destination_power))
            score = torch.where(self.drift_fell[rows], torch.zeros_like(score), score)
            self.pending_scored += len(rows)
            self.pending_score += float(score.sum())
            self.pending_drift += float(drift.sum())
        self.drift_walking[indices] = False

    def _record_top_speed(self, command, telemetry, terminated):
        """Add the speed along the command of each standing robot commanded at the top level (a report only)."""
        command = command.to(self.top_speed_sum)
        speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
        top = (speed - self.curriculum_level).abs() < 1e-4
        top &= ~terminated.to(top.device)
        current = getattr(self.env, "current", {})
        if "root_velocity" in current:
            pose = telemetry["root_pose_xyzw"].to(self.top_speed_sum)
            world = current["root_velocity"][:, :3].to(self.top_speed_sum)
            planar = navigation(inverse_rotate(pose[:, 3:], world))[:, :2]
        else:
            planar = telemetry["linear_velocity_nav"].to(self.top_speed_sum)[:, :2]
        along = (planar * command[:, :2]).sum(-1) / speed.clamp_min(1e-9)
        self.top_speed_controls += top.sum()
        self.top_speed_sum += (along * top).sum()

    def _finish_walks(self, indices):
        """Score the walks that end at these rows (their command is about to change) and clear their sums."""
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.walk_sums.device)
        ended = (self.walk_counted[indices] > 0) | self.walk_fell[indices]
        rows = indices[ended]
        if len(rows):
            speed = torch.linalg.vector_norm(self.env.commands[rows, :2], dim=-1).to(self.walk_sums)
            failed = walk_failures(
                self.walk_sums[rows],
                self.walk_counted[rows],
                speed,
                self.walk_fast[rows],
                self.walk_fell[rows],
                self.reward_config.speed_curriculum_planar_bound_scale or 1.0,
            )
            for name, count in zip(WALK_CHECKS, failed.sum(0).tolist()):
                self.check_failures[name] += int(count)
            self.pending_walks += len(rows)
            self.pending_failed += int(failed.any(-1).sum())
        self.walk_sums[indices] = 0
        self.walk_counted[indices] = 0
        self.walk_fast[indices] = False
        self.walk_fell[indices] = False

    def _curriculum_step(self, pay, terminated=None, distance=None):
        """Record the mean destination pay of this control; raise the top speed past the threshold and
        lower it below the back-off, each judged on its own window of controls at the level. With a
        back-off distance, any standing robot further than it from its ghost in the window lowers it too."""
        config = self.reward_config
        value = pay.detach().mean()
        window = config.speed_curriculum_window_controls
        self.curriculum_window[self.curriculum_filled % window] = value.to(self.curriculum_window)
        falls = 0 if terminated is None else terminated.detach().sum()
        self.curriculum_falls[self.curriculum_filled % window] = falls
        limit = config.speed_curriculum_backoff_distance_m
        lost = 0
        if limit is not None and distance is not None:
            standing = torch.ones_like(distance, dtype=torch.bool)
            if terminated is not None:
                standing = ~terminated.to(distance.device)
            lost = ((distance.detach() > limit) & standing).sum()
        self.curriculum_lost[self.curriculum_filled % window] = lost
        if config.speed_curriculum_measure == "drift":
            slot = self.curriculum_filled % window
            self.curriculum_scored[slot] = self.pending_scored
            self.curriculum_score[slot] = self.pending_score
            self.curriculum_drift[slot] = self.pending_drift
            self.pending_scored = self.pending_score = self.pending_drift = 0.0
        if config.speed_curriculum_pass_checks:
            self.curriculum_walks[self.curriculum_filled % window] = self.pending_walks
            self.curriculum_failed[self.curriculum_filled % window] = self.pending_failed
            self.pending_walks = self.pending_failed = 0
        self.curriculum_filled += 1
        span = config.speed_curriculum_backoff_window_controls
        if span is not None:
            self.curriculum_backoff_window[self.curriculum_backoff_filled % span] = value.to(
                self.curriculum_backoff_window
            )
            self.curriculum_backoff_filled += 1
        # Read the window about once per update (50 updates per window) to avoid a host sync each control.
        if self.curriculum_filled < window or self.curriculum_filled % max(1, window // 50):
            return
        mean = float(self.curriculum_window.mean())
        if config.speed_curriculum_measure == "drift":
            scored = float(self.curriculum_scored.sum())
            mean = float(self.curriculum_score.sum()) / scored if scored else 0.0
        first = round(config.speed_curriculum_start_mps + config.speed_curriculum_step_mps, 6)
        backoff = config.speed_curriculum_backoff_threshold
        judged = mean
        if span is not None:
            judged = (
                float(self.curriculum_backoff_window.mean())
                if self.curriculum_backoff_filled >= span
                else None
            )
        low = backoff is not None and judged is not None and judged < backoff
        far = limit is not None and float(self.curriculum_lost.sum()) > 0
        if (low or far) and self.curriculum_level > first + 1e-9:
            self.curriculum_advances.append(
                {
                    "controls_completed": self.controls_completed,
                    "level_mps": round(self.curriculum_level, 6),
                    "window_mean": judged if low else mean,
                    "back_off": True,
                    **({"reason": "pay" if low else "distance"} if limit is not None else {}),
                }
            )
            self.curriculum_level = round(
                self.curriculum_level - config.speed_curriculum_step_mps, 6
            )
            self.curriculum_complete = False
            self.curriculum_filled = self.curriculum_backoff_filled = 0
            self.top_speed_controls.zero_()
            self.top_speed_sum.zero_()
            return
        # A robot past the back-off distance also holds the first level.
        if far or mean <= config.speed_curriculum_threshold:
            return
        if config.speed_curriculum_advance_without_falls and float(self.curriculum_falls.sum()) > 0:
            return
        if config.speed_curriculum_pass_checks and (
            float(self.curriculum_failed.sum()) > 0 or float(self.curriculum_walks.sum()) == 0
        ):
            return
        record = {
            "controls_completed": self.controls_completed,
            "level_mps": round(self.curriculum_level, 6),
            "window_mean": mean,
        }
        if self.curriculum_level + 1e-9 >= config.speed_curriculum_final_mps:
            if not self.curriculum_complete:
                self.curriculum_complete = True
                self.curriculum_advances.append({**record, "complete": True})
            return
        self.curriculum_advances.append(record)
        self.curriculum_level = round(self.curriculum_level + config.speed_curriculum_step_mps, 6)
        self.curriculum_filled = self.curriculum_backoff_filled = 0
        self.top_speed_controls.zero_()
        self.top_speed_sum.zero_()
        if config.speed_curriculum_pass_checks:
            self.check_failures = dict.fromkeys(WALK_CHECKS, 0)

    def _drift_status(self, filled):
        """Under the drift measure ``window_mean`` is the window's drift score; the pay moves to
        ``window_pay``."""
        scored = float(self.curriculum_scored[:filled].sum())
        return {
            "window_measure": "drift",
            "window_mean": float(self.curriculum_score[:filled].sum()) / scored if scored else None,
            "window_pay": float(self.curriculum_window[:filled].mean()) if filled else None,
            "window_drift_walks": int(scored),
            "window_drift_mm": 1000 * float(self.curriculum_drift[:filled].sum()) / scored
            if scored
            else None,
        }

    def curriculum_status(self):
        if self.reward_config.speed_curriculum_start_mps is None:
            return None
        window = self.reward_config.speed_curriculum_window_controls
        filled = min(self.curriculum_filled, window)
        return {
            "level_mps": round(self.curriculum_level, 6),
            "window_controls": filled,
            "window_mean": float(self.curriculum_window[:filled].mean()) if filled else None,
            "window_falls": int(self.curriculum_falls[:filled].sum()),
            "window_lost": int(self.curriculum_lost[:filled].sum()),
            **(self._drift_status(filled) if self.reward_config.speed_curriculum_measure else {}),
            "top_speed_controls": int(self.top_speed_controls),
            "top_speed_along_command_sum_mps": float(self.top_speed_sum),
            **(
                {
                    "window_walks": int(self.curriculum_walks[:filled].sum()),
                    "window_walks_failed": int(self.curriculum_failed[:filled].sum()),
                    "level_check_failures": dict(self.check_failures),
                }
                if self.reward_config.speed_curriculum_pass_checks
                else {}
            ),
            "complete": self.curriculum_complete,
            "advances": self.curriculum_advances,
        }

    def _schedule_stops(self, indices):
        """End a scheduled walk with a zero command; cut a share of fresh moving commands short."""
        config = self.reward_config
        indices = torch.as_tensor(indices, dtype=torch.long, device=self.stop_pending.device)
        stopping = indices[self.stop_pending[indices]]
        self.env.commands[stopping] = 0
        self.remaining_controls[stopping] = self.stop_rest[stopping].to(self.remaining_controls)
        self.stop_pending[stopping] = False
        fresh = indices[~torch.isin(indices, stopping)]
        moving = (self.env.commands[fresh] != 0).any(-1).to(fresh.device)
        draw = torch.rand(len(fresh), generator=self.rng).to(fresh.device)
        walking = torch.randint(
            config.stop_after_min_controls,
            config.stop_after_max_controls + 1,
            (len(fresh),),
            generator=self.rng,
        ).to(fresh.device)
        chosen = fresh[moving & (draw < config.stop_fraction)]
        walking = walking[moving & (draw < config.stop_fraction)]
        self.remaining_controls[chosen] = walking.to(self.remaining_controls)
        self.stop_rest[chosen] = config.command_controls - walking
        self.stop_pending[chosen] = True

    def _advance_destination(self, command):
        """Start a new command's destination at the pose before its first control, then move it one control."""
        start = planar_pose(self.previous_pose).to(self.destination)
        self.destination = torch.where(self.command_started[:, None], start, self.destination)
        self.command_elapsed = (
            torch.where(
                self.command_started, torch.zeros_like(self.command_elapsed), self.command_elapsed
            )
            + 1
        )
        self.command_started[:] = False
        self.destination = advance_destination(self.destination, command)

    def _reward(self, command, terminated):
        self.control_start_pose = self.previous_pose.detach().clone().to(self.control_start_pose)
        self._advance_destination(command)
        self.run_elapsed += 1
        if self.slope is not None:
            t, current = self.env.telemetry, self.env.current
            force = torch.linalg.vector_norm(t["tibia_floor_force_world_n"], dim=-1)
            self.slope.update(t["toe_xyz_world"].cpu(), (force > 1.0).cpu())
            self.slope.push(current["root"][:, 3:7].cpu(), current["q"].cpu())
        reward, components, telemetry = super()._reward(command, terminated)
        if self.gait is not None:
            self.gait.advance(command)
        if self.reward_config.speed_curriculum_pass_checks:
            self._record_walks(command, telemetry, terminated)
        if self.reward_config.speed_curriculum_start_mps is not None:
            self._record_top_speed(command, telemetry, terminated)
        if self.reward_config.speed_curriculum_measure == "drift":
            self._record_drift(telemetry, terminated)
        config = self.reward_config
        if config.destination_weight > 0:
            distance = destination_distance(telemetry["root_pose_xyzw"], self.destination)
            scaled = distance / config.destination_distance_m
            if config.destination_kernel == "exponential":
                closeness = torch.exp(-scaled)
            elif config.destination_kernel == "rational":
                closeness = 1 / (1 + scaled.pow(config.destination_power))
                if config.destination_linear_from_m is not None:
                    closeness = linear_tail(distance, closeness, config)
            else:
                closeness = (1 - scaled).clamp_min(0)
            quiet = (command == 0).all(-1).to(closeness.device)
            if config.destination_quiet == "flat":
                closeness = torch.where(quiet, torch.ones_like(closeness), closeness)
            elif config.destination_quiet == "off":
                closeness = torch.where(quiet, torch.zeros_like(closeness), closeness)
            alive = (~terminated).to(closeness.device).to(closeness)
            components["destination"] = (config.destination_weight * alive * closeness).to(reward)
            reward = reward + components["destination"]
            if config.speed_curriculum_start_mps is not None:
                self._curriculum_step(components["destination"], terminated, distance)
            if config.end_destination_weight is not None:
                # The same gap on destination_v9's narrower kernel, weighted by the share of the run done.
                scaled = distance / config.end_destination_distance_m
                share = (self.run_elapsed.to(distance) / self.run_length.to(distance)).clamp(max=1)
                if config.end_destination_kernel == "linear":
                    shape = (1 - scaled).clamp_min(-config.end_destination_floor)
                else:
                    shape = 1 / (1 + scaled.pow(config.end_destination_power))
                closeness = share * alive * shape
                components["end_destination"] = (config.end_destination_weight * closeness).to(
                    reward
                )
                reward = reward + components["end_destination"]
        if config.planar_error_weight is not None:
            components["planar_error"] = self._planar_error(command, telemetry).to(reward)
            reward = reward + components["planar_error"]
        if config.hold_weight is not None:
            components["hold"] = self._hold(command, terminated).to(reward)
            reward = reward + components["hold"]
        if config.heading_weight is not None:
            actual = planar_pose(telemetry["root_pose_xyzw"]).to(self.destination)[:, 2]
            gap = torch.remainder(self.destination[:, 2] - actual + math.pi, 2 * math.pi) - math.pi
            closeness = 1 / (1 + (gap.abs() / config.heading_width_rad).pow(config.heading_power))
            alive = (~terminated).to(closeness.device).to(closeness)
            components["heading"] = (config.heading_weight * alive * closeness).to(reward)
            reward = reward + components["heading"]
        return reward, components, telemetry

    def _planar_error(self, command, telemetry):
        """Charge each walking control its planar speed error over the evaluator's bound, measured on the
        root's centre-of-mass velocity as the evaluator reads it: squared, or linear up to
        ``planar_error_cap`` when one is set."""
        config = self.reward_config
        pose = telemetry["root_pose_xyzw"].to(torch.float32)
        current = getattr(self.env, "current", {})
        if "root_velocity" in current:
            world = current["root_velocity"][:, :3].to(pose)
            planar = navigation(inverse_rotate(pose[:, 3:], world))[:, :2]
        else:
            planar = telemetry["linear_velocity_nav"].to(pose)[:, :2]
        command = command.to(pose)
        speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
        error = torch.linalg.vector_norm(planar - command[:, :2], dim=-1)
        bound = torch.clamp(0.25 * speed, min=0.025)
        moving = (command != 0).any(-1).to(pose)
        ratio = error / bound
        if config.planar_error_cap is None:
            charge = ratio.square()
        else:
            charge = ratio.clamp_max(config.planar_error_cap)
        return -config.planar_error_weight * moving * charge

    def _hold(self, command, terminated):
        """Anchor the joints' running mean a grace period into a zero command and charge its drift."""
        config = self.reward_config
        q = self.env.telemetry["joint_position_rad"].detach().to(self.hold_anchor)
        age = self.command_age.to(q.device)
        quiet = (command.to(q.device) == 0).all(-1)
        self.hold_mean = torch.where(
            (age <= 1)[:, None],
            q,
            self.hold_mean + (q - self.hold_mean) / config.hold_smoothing_controls,
        )
        anchor = quiet & (age == config.hold_grace_controls)
        self.hold_anchor = torch.where(anchor[:, None], self.hold_mean, self.hold_anchor)
        drift = (
            ((self.hold_mean - self.hold_anchor).abs() - config.hold_deadband_rad)
            / config.hold_span_rad
        ).clamp(0, 1)
        active = quiet & (age >= config.hold_grace_controls) & ~terminated.to(q.device)
        return -config.hold_weight * active.to(q) * drift.mean(-1)

    def declaration(self):
        result = super().declaration()
        result["reward_version"] = result["reward"]["version"] = REWARD_VERSION
        here = Path(__file__).parent
        result["experiment_source_sha256"] = {
            name: hashlib.sha256((here / name).read_bytes()).hexdigest()
            for name in ("destination_task.py", "mdp/destination.py")
        }
        return result


def experiment_task(name):
    """The task class of a named experiment, for the surrogate's --task module:factory(text)."""
    chosen = EXPERIMENTS[name]["reward"]

    class ExperimentTask(DestinationTrainingTask):
        def __init__(self, env, config=None, output_dir=None, reward_config=None):
            super().__init__(
                env, config, output_dir, chosen if reward_config is None else reward_config
            )

    ExperimentTask.__name__ = ExperimentTask.__qualname__ = "ExperimentTask"
    return ExperimentTask


def replace_reward(name, **changes):
    """A named experiment's reward with fields changed, for sensitivity runs."""
    return replace(EXPERIMENTS[name]["reward"], **changes)
