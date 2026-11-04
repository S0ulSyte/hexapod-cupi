"""Follow a speed-scaled gait phase per replica and read the tripod swing schedule from it."""

import torch

from locomotion.omniwalk.mdp.commands import GAIT_CLOCK_SCALINGS
from locomotion.omniwalk.mdp.rewards import TRIPOD_A

# Phase comparisons allow for the float64 sum of a cycle's steps landing just below a whole control.
PHASE_TOLERANCE = 1e-9


def gait_cycle(command, scaling):
    """Each row's cycle in controls and tripod swing share at its planar command speed."""
    speeds, periods, swings = (
        torch.tensor(values, dtype=torch.float64, device=command.device)
        for values in GAIT_CLOCK_SCALINGS[scaling]
    )
    # Rounded to the micrometre per second: a float32 0.075 m/s command reads 0.0750000003.
    speed = torch.linalg.vector_norm(command[:, :2].to(torch.float64), dim=-1)
    speed = torch.round(speed * 1e6) / 1e6
    speed = speed.clamp(float(speeds[0]), float(speeds[-1]))
    upper = torch.searchsorted(speeds, speed).clamp(1, len(speeds) - 1)
    weight = (speed - speeds[upper - 1]) / (speeds[upper] - speeds[upper - 1])

    def between(values):
        return values[upper - 1] + weight * (values[upper] - values[upper - 1])

    return between(periods), between(swings)


def phase_swing(phase, swing, mirror=None):
    """Six booleans per row: the feet the schedule wants off the floor at ``phase`` cycles.

    ``mirror`` rows swap the tripods: the fixed first-tripod convention gives the gait a
    handedness, and a clockwise turn fought it on every checkpoint ever scored (the
    turn_-0.40 wall; decisions.md, turn chirality). ``None`` keeps the historical schedule
    byte-identical."""
    fraction = torch.remainder(phase, 1.0)
    fraction = torch.where(fraction > 1 - PHASE_TOLERANCE, torch.zeros_like(fraction), fraction)
    swing = swing.to(fraction)
    first = torch.tensor(TRIPOD_A, device=phase.device)
    swing_a = fraction < swing - PHASE_TOLERANCE
    swing_b = (fraction >= 0.5 - PHASE_TOLERANCE) & (fraction < 0.5 + swing - PHASE_TOLERANCE)
    if mirror is not None:
        first = first[None, :] ^ mirror[:, None].to(first.device)
    return torch.where(first, swing_a[:, None], swing_b[:, None])


class GaitPhase:
    """Each replica's gait phase in cycles: zero at an episode start, then one control over the cycle of
    the command that each control ran under."""

    def __init__(self, num_envs, scaling, device):
        if scaling not in GAIT_CLOCK_SCALINGS:
            raise ValueError(f"Unknown gait clock scaling {scaling!r}")
        self.scaling = scaling
        self.phase = torch.zeros(num_envs, dtype=torch.float64, device=device)
        self.command = torch.zeros(num_envs, 3, device=device)

    def reset(self, indices):
        self.phase[indices] = 0

    def advance(self, command):
        period, _ = gait_cycle(command.to(self.phase.device), self.scaling)
        self.phase = torch.remainder(self.phase + 1 / period, 1.0)

    def swing(self, command, mirror_turns=False):
        """The scheduled swing at the current phase under ``command``, and that command's cycle in seconds."""
        command = command.to(self.phase.device)
        period, swing = gait_cycle(command, self.scaling)
        mirror = (command[:, 2] < 0) if mirror_turns else None
        return phase_swing(self.phase, swing, mirror), period * 0.02

    def follow(self, command, episode_steps):
        """For evaluation: advance under the previous control's command, restart at an episode start and
        return the phase the policy sees before this control."""
        self.advance(self.command)
        fresh = episode_steps.to(self.phase.device) == 0
        self.phase = torch.where(fresh, torch.zeros_like(self.phase), self.phase)
        self.command = command.detach().clone().to(self.command)
        return self.phase
