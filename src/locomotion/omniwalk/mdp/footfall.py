"""Footfall lookahead inputs (``terrain_v1``): the ground each leg's next touchdown will land on.

For each leg, the time to its next touchdown comes from the tripod schedule (``gait.py``): tripod A
touches down when the phase reaches the swing share, tripod B half a cycle later. The body's planar
pose is advanced along the command for that long (one midpoint-heading step, the integration scheme
of ``destination.advance_destination``), the leg's current body-frame toe offset is rotated to the
advanced heading (a planar approximation of the body frame), and the terrain height is sampled there
and at four ring points ``RING_M`` away along world x and y. Heights are relative to the support
height — the mean world height of toes in floor contact (``CONTACT_FORCE_N``), smoothed by
``SMOOTHING`` like ``slope.py``'s tracker and snapped to the mean toe height on a fresh episode —
clamped to ``CLAMP_M`` and scaled over ``destination.DISTANCE_SCALE_M``: 6 legs x 5 samples = 30
values on the actor and the critic, appended after the nine ``ghost_v11`` values.

The terrain is read duck-typed off the environment (``env.terrain``, a ``surrogate.terrain.Terrain``;
this module imports nothing from ``locomotion.surrogate``). With no terrain — the native robot, or a
flat-floor surrogate — every value is zero, so flat suites and native evaluation stay valid.
``ppo_config`` records ``destination_input="terrain_v1"`` and evaluation rebuilds the same inputs
(``surrogate/evaluate.py``'s ``TerrainV1Tracker`` below).
"""

import torch

from locomotion.omniwalk.mdp.commands import GAIT_CLOCK_SCALINGS
from locomotion.omniwalk.mdp.destination import DISTANCE_SCALE_M, GhostV11Tracker, planar_pose
from locomotion.omniwalk.mdp.gait import GaitPhase, gait_cycle
from locomotion.omniwalk.mdp.rewards import TRIPOD_A

FOOTFALL_FEATURES = 30
RING_M = 0.03
CLAMP_M = 0.2
CONTACT_FORCE_N = 1.0
SMOOTHING = 0.2
CONTROL_DT_S = 0.02
# Sample offsets in world metres: the predicted touchdown and a ring around it.
RING_XY = ((0.0, 0.0), (RING_M, 0.0), (-RING_M, 0.0), (0.0, RING_M), (0.0, -RING_M))


class TerrainGrid:
    """A ``surrogate.terrain.Terrain``'s grid as tensors, sampled bilinearly at world x, y points."""

    def __init__(self, terrain):
        self.heights = torch.as_tensor(terrain.heights, dtype=torch.float64)
        self.x0, self.y0 = float(terrain.xs[0]), float(terrain.ys[0])
        self.resolution = float(terrain.resolution)
        self.nx, self.ny = int(terrain.nx), int(terrain.ny)

    def sample(self, x, y):
        """Heights at world x, y tensors of any shape (the floor's level outside the grid)."""
        u = ((x.double() - self.x0) / self.resolution).clamp(0, self.nx - 1)
        v = ((y.double() - self.y0) / self.resolution).clamp(0, self.ny - 1)
        i = u.floor().long().clamp(max=self.nx - 2)
        j = v.floor().long().clamp(max=self.ny - 2)
        a, b = u - i, v - j
        h = self.heights
        return (
            (h[j, i] * (1 - a) + h[j, i + 1] * a) * (1 - b)
            + (h[j + 1, i] * (1 - a) + h[j + 1, i + 1] * a) * b
        ).float()


def touchdown_seconds(phase, command, scaling):
    """Per-leg seconds until the next touchdown, [n, 6], from the tripod schedule at ``command``."""
    period_controls, swing = gait_cycle(command, scaling)
    fraction = torch.remainder(phase.double(), 1.0)
    swing = swing.to(fraction)
    td_a = torch.remainder(swing - fraction, 1.0)
    td_b = torch.remainder(0.5 + swing - fraction, 1.0)
    first = torch.tensor(TRIPOD_A, device=fraction.device)
    cycles = torch.where(first, td_a[:, None], td_b[:, None])
    return cycles * (period_controls.to(fraction) * CONTROL_DT_S)[:, None]


def touchdown_points(root, command, toe_body, seconds):
    """Predicted world x, y of each leg's next touchdown, [n, 6, 2]: the body's planar pose advanced
    along the command for ``seconds``, carrying the leg's current body-frame toe offset."""
    pose = planar_pose(root).double()
    command = command.double()
    t = seconds.double()
    mid = pose[:, 2:3] + command[:, 2:3] * t / 2
    yaw_t = pose[:, 2:3] + command[:, 2:3] * t
    # Navigation (x, y) is body (-y, x), as destination.advance_destination maps commands to the world.
    bx, by = command[:, 1:2].expand_as(t), (-command[:, 0:2][:, :1]).expand_as(t)
    c, s = torch.cos(mid), torch.sin(mid)
    body_x = pose[:, 0:1] + (c * bx - s * by) * t
    body_y = pose[:, 1:2] + (s * bx + c * by) * t
    ct, st = torch.cos(yaw_t), torch.sin(yaw_t)
    ox, oy = toe_body[:, :, 0].double(), toe_body[:, :, 1].double()
    return torch.stack((body_x + ct * ox - st * oy, body_y + st * ox + ct * oy), -1)


class SupportHeight:
    """Per-replica support height: the mean world z of toes in floor contact, smoothed and held, snapped
    to the mean height of all toes on rows that have no estimate yet (a fresh episode)."""

    def __init__(self, num_envs):
        self.height = torch.zeros(num_envs, dtype=torch.float64)
        self.known = torch.zeros(num_envs, dtype=torch.bool)

    def reset(self, fresh):
        self.known = self.known & ~fresh.to(self.known.device)

    def update(self, toe_world, contact_force):
        toes = toe_world[:, :, 2].double()
        if contact_force is None:
            contact = torch.zeros(toes.shape, dtype=torch.bool)
        else:
            contact = torch.linalg.vector_norm(contact_force.double(), dim=-1) > CONTACT_FORCE_N
        n = contact.double().sum(-1)
        measured = (toes * contact).sum(-1) / n.clamp_min(1)
        valid = n > 0
        smoothed = torch.where(
            valid, self.height + SMOOTHING * (measured - self.height), self.height
        )
        snap = torch.where(valid, measured, toes.mean(-1))
        self.height = torch.where(self.known, smoothed, snap)
        self.known = self.known | valid
        return self.height


def footfall_features(grid, reference, root, command, toe_body, phase, scaling):
    """The 30 ``terrain_v1`` values, [n, 30], for a support-height ``reference`` ([n], metres)."""
    seconds = touchdown_seconds(phase, command, scaling)
    points = touchdown_points(root, command, toe_body, seconds)
    ring = torch.tensor(RING_XY, dtype=torch.float64)
    x = points[:, :, None, 0] + ring[None, None, :, 0]
    y = points[:, :, None, 1] + ring[None, None, :, 1]
    relative = grid.sample(x, y) - reference[:, None, None].float()
    scaled = relative.clamp(-CLAMP_M, CLAMP_M) / DISTANCE_SCALE_M
    return scaled.reshape(root.shape[0], FOOTFALL_FEATURES)


def refresh_support(support, seen, env, episode_steps, state):
    """Advance the support height once per control: observations can be rebuilt several times for one
    control, and the smoothing must not advance again (``env.total_controls`` marks the control)."""
    controls = getattr(env, "total_controls", None)
    if controls is None or controls != seen:
        support.reset(episode_steps == 0)
        telemetry = getattr(env, "telemetry", None) or {}
        support.update(state["toe_world"], telemetry.get("tibia_floor_force_world_n"))
    return controls


class FootfallInputs:
    """The training-side ``terrain_v1`` block for ``DestinationVecEnv``: reads the task's env duck-typed."""

    def __init__(self, task, scaling):
        if scaling not in GAIT_CLOCK_SCALINGS:
            raise ValueError(
                "terrain_v1 needs the task's speed-scaled gait clock (gait_clock_scaling)"
            )
        self.task, self.scaling = task, scaling
        terrain = getattr(getattr(task, "env", None), "terrain", None)
        self.grid = None if terrain is None else TerrainGrid(terrain)
        self.support = SupportHeight(task.num_envs)
        self.seen = None

    def features(self, command=None):
        task = self.task
        if self.grid is None:
            return torch.zeros(task.num_envs, FOOTFALL_FEATURES)
        phase = getattr(task, "gait_clock_phase", None)
        if phase is None:
            raise ValueError("terrain_v1 on a terrain needs a task with the speed-scaled clock")
        state = task.current
        self.seen = refresh_support(
            self.support, self.seen, getattr(task, "env", None), task.episode_steps, state
        )
        return footfall_features(
            self.grid,
            self.support.height,
            state["root"],
            task.commands if command is None else command,
            state["toe_body"],
            phase,
            self.scaling,
        )


class TerrainV1Tracker(GhostV11Tracker):
    """Evaluation-side ``terrain_v1``: the nine ``ghost_v11`` values plus the 30 footfall values, rebuilt
    from the env the policy runs in (``surrogate/evaluate.py`` passes it after construction)."""

    def __init__(self, num_envs, device, *, env=None, scaling=None):
        super().__init__(num_envs, device)
        if scaling not in GAIT_CLOCK_SCALINGS:
            raise ValueError("terrain_v1 needs the recorded gait_clock_scaling")
        self.env, self.gait = env, GaitPhase(num_envs, scaling, device)
        self.scaling = scaling
        terrain = getattr(env, "terrain", None)
        self.grid = None if terrain is None else TerrainGrid(terrain)
        self.support = SupportHeight(num_envs)
        self.seen = None

    def inputs(self, pose, command, episode_steps):
        ghost = super().inputs(pose, command, episode_steps)
        phase = self.gait.follow(command, episode_steps)
        if self.grid is None:
            zeros = torch.zeros(ghost.shape[0], FOOTFALL_FEATURES)
            return torch.cat((ghost, zeros.to(ghost)), -1)
        state = self.env.current
        self.seen = refresh_support(self.support, self.seen, self.env, episode_steps, state)
        terrain = footfall_features(
            self.grid,
            self.support.height,
            state["root"],
            command,
            state["toe_body"],
            phase,
            self.scaling,
        )
        return torch.cat((ghost, terrain.to(ghost)), -1)
