"""Leg-odometry ghost inputs (``terrain_v2_odom``): the ghost gap from quantities the robot has.

The nine ghost_v11 values normally come from the simulator's exact world pose, which the robot
cannot supply (plain GPS, metre-level). This module rebuilds the same nine features from
robot-knowable quantities instead: body-frame toe positions (pure joint forward kinematics —
reconstructed here from the environment's world toes and pose, which is kinematically the same
numbers), foot contact from toe force, and the gyro's yaw increment (taken as the true heading
delta per control: an ideal rate gyro, which real IMUs approach at these horizons). A toe in
stance is world-fixed, so the body's planar displacement each control is the mean over stance
toes of R(theta_prev) @ toe_prev - R(theta) @ toe. The ghost itself is the integrated command
(``advance_destination``), restarted at the ODOMETRY pose when a command begins — the same
restart convention as the true destination, so the feature semantics, scales and layout match
``ghost_v11`` exactly and a terrain_v2 policy can warm-start unchanged.

Slip is inherited by construction: measured drift on recorded runs is 0.3-6 mm/s (worst
crabbing sideways), against the 0.1 m feature scale.
"""

import torch

from locomotion.omniwalk.mdp.destination import (
    advance_destination,
    ghost_v10_features,
    planar_pose,
)

CONTACT_FORCE_N = 0.5


def _heading(root):
    return planar_pose(root)[:, 2]


def _body_planar_toes(root, toe_world):
    """Planar toe positions in the body frame: joint forward kinematics, reconstructed."""
    q = root[:, 3:7]
    offset = toe_world - root[:, None, :3]
    x, y, z, w = (-q[:, 0:1], -q[:, 1:2], -q[:, 2:3], q[:, 3:4])
    axis = torch.stack((x, y, z), -1)
    t = 2 * torch.cross(axis.expand_as(offset), offset, dim=-1)
    body = offset + w[..., None] * t + torch.cross(axis.expand_as(offset), t, dim=-1)
    return body[..., :2]


def _rotate(theta, points):
    c, s = torch.cos(theta)[:, None], torch.sin(theta)[:, None]
    x, y = points[..., 0], points[..., 1]
    return torch.stack((c * x - s * y, s * x + c * y), -1)


class OdometryGhost:
    """Batched leg odometry plus the command-integrated ghost it is measured against."""

    def __init__(self, num_envs, device):
        self.xy = torch.zeros(num_envs, 2, device=device)
        self.theta = torch.zeros(num_envs, device=device)
        self.previous_xy = torch.zeros(num_envs, 2, device=device)
        self.previous_theta = torch.zeros(num_envs, device=device)
        self.destination = torch.zeros(num_envs, 3, device=device)
        self.prev_toes = torch.zeros(num_envs, 6, 2, device=device)
        self.prev_contact = torch.zeros(num_envs, 6, dtype=torch.bool, device=device)
        self.prev_heading = torch.zeros(num_envs, device=device)

    def reset(self, rows, root, toe_world, tibia_force):
        """Align the odometry frame with the true pose on reset rows (an arbitrary frame choice)."""
        root = root.to(self.xy)
        pose = planar_pose(root)
        self.xy[rows] = pose[rows, :2]
        self.theta[rows] = pose[rows, 2]
        self.previous_xy[rows] = pose[rows, :2]
        self.previous_theta[rows] = pose[rows, 2]
        self.destination[rows] = pose[rows]
        self.prev_toes[rows] = _body_planar_toes(root, toe_world.to(root))[rows]
        self.prev_contact[rows] = (
            torch.linalg.vector_norm(tibia_force.to(root), dim=-1)[rows] > CONTACT_FORCE_N
        )
        self.prev_heading[rows] = pose[rows, 2]

    def advance(self, root, toe_world, tibia_force, command, started):
        """One control: integrate the gyro, dead-reckon from stance toes, move the ghost."""
        root = root.to(self.xy)
        self.previous_xy = self.xy.clone()
        self.previous_theta = self.theta.clone()
        heading = _heading(root)
        delta = torch.remainder(heading - self.prev_heading + torch.pi, 2 * torch.pi) - torch.pi
        self.prev_heading = heading
        theta_prev = self.theta.clone()
        self.theta = self.theta + delta
        toes = _body_planar_toes(root, toe_world.to(root))
        contact = torch.linalg.vector_norm(tibia_force.to(root), dim=-1) > CONTACT_FORCE_N
        stance = (contact & self.prev_contact).to(self.xy)
        steps = _rotate(theta_prev, self.prev_toes) - _rotate(self.theta, toes)
        weight = stance[..., None]
        self.xy = self.xy + (steps * weight).sum(1) / weight.sum(1).clamp_min(1.0)
        self.prev_toes, self.prev_contact = toes, contact
        started = started.to(self.destination.device)
        fresh = torch.stack((self.xy[:, 0], self.xy[:, 1], self.theta), -1)
        self.destination = torch.where(started[:, None], fresh, self.destination)
        self.destination = advance_destination(self.destination, command.to(self.destination))

    def pose(self, xy, theta):
        zeros = torch.zeros_like(theta)
        return torch.stack(
            (
                xy[:, 0],
                xy[:, 1],
                zeros,
                zeros,
                zeros,
                torch.sin(theta / 2),
                torch.cos(theta / 2),
            ),
            -1,
        )

    def features(self, pending, elapsed, command):
        """The nine ghost_v11 values from the odometry pose; pending rows read the ghost at the body."""
        here = torch.stack((self.xy[:, 0], self.xy[:, 1], self.theta), -1)
        pending = pending.to(self.destination.device)
        destination = torch.where(pending[:, None], here, self.destination)
        elapsed = torch.where(pending, torch.zeros_like(elapsed), elapsed)
        return ghost_v10_features(
            self.pose(self.xy, self.theta),
            self.pose(self.previous_xy, self.previous_theta),
            destination,
            elapsed,
            command,
            along_track=True,
        )


class OdomTerrainTracker:
    """Evaluation-side ``terrain_v2_odom``: the terrain blocks from the TerrainV2Tracker, with
    the nine ghost values recomputed from leg odometry, exactly as training's wrapper serves
    them. Constructed like the terrain trackers (``surrogate/evaluate.py`` passes the env)."""

    def __init__(self, num_envs, device, *, env=None, scaling=None, mirror=False):
        from locomotion.omniwalk.mdp.footfall import TerrainV2Tracker

        self.inner = TerrainV2Tracker(num_envs, device, env=env, scaling=scaling, mirror=mirror)
        self.env = env
        self.odom = OdometryGhost(num_envs, device)
        self._fresh = True

    def inputs(self, pose, command, episode_steps):
        previous = self.inner.command.clone()
        full = self.inner.inputs(pose, command, episode_steps)
        root = self.env.current["root"]
        toes = self.env.telemetry["toe_xyz_world"]
        force = self.env.telemetry["tibia_floor_force_world_n"]
        starts = episode_steps.to(self.odom.theta.device) == 0
        if self._fresh:
            self.odom.reset(torch.arange(len(pose)), root, toes, force)
            self._fresh = False
        elif bool(starts.any()):
            self.odom.reset(starts.nonzero(as_tuple=False).flatten(), root, toes, force)
        fresh = starts | (command.to(previous) != previous).any(-1)
        self.odom.advance(root, toes, force, command, fresh)
        features = self.odom.features(fresh, self.inner.elapsed, command)
        from locomotion.omniwalk.mdp.destination import GHOST_V11_FEATURES

        return torch.cat((features.to(full), full[..., GHOST_V11_FEATURES:]), -1)
