"""Slope inputs: the robot's tilt along a slope from its front and rear toes, and the policy inputs it corrects.

Whenever a front leg (lf or rf) and a rear leg (lr or rr) of a replica touch the ground, the height difference
between their toes over the horizontal distance between them gives the replica's tilt along the slope, smoothed
over about 0.1 s, held while no front and rear leg touch at once and limited to 20 degrees. Then the policy
inputs change in all five history frames of the 231-value observation:

- ``gravity``: the gravity inputs become gravity relative to the sloped ground instead of the vertical;
- ``femurs``: each femur pitch input becomes its measured angle plus the tilt about that joint's axis, nearly
  the whole tilt with opposite signs for the front and rear femurs and almost none for the middle ones;
- ``joints``: as ``femurs``, and each tibia input gains the tilt about its own axis too.

Training (``destination_task.py``) and surrogate scenes (``surrogate/scene.py``) use this one tracker.
"""

import math

import torch

from locomotion.omniwalk.mdp.frames import inverse_rotate, navigation, rotate

MODES = ("gravity", "femurs", "joints")
FRONT_LEGS, REAR_LEGS = (0, 3), (2, 5)
LIMIT_RAD = math.radians(20)
SMOOTHING = 0.2
GRAVITY, JOINTS = slice(3, 6), slice(6, 24)
LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")


def multiply(a, b):
    """Hamilton products of xyzw quaternions, broadcast over leading axes."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ),
        -1,
    )


def about_z(angle):
    zero = torch.zeros_like(angle)
    return torch.stack((zero, zero, torch.sin(angle / 2), torch.cos(angle / 2)), -1)


class SlopeTracker:
    """Per replica: the tilt along the slope and the corrected inputs of the last five controls."""

    def __init__(self, num_envs, joints, mode):
        if mode not in MODES:
            raise ValueError(f"Slope inputs are one of {MODES}")
        self.mode, self.num_envs = mode, num_envs
        named = {j["name"]: j for j in joints}
        # [leg, part] joint frames (coxa, femur, tibia), xyzw.
        self.frames = torch.tensor(
            [
                [
                    named[f"{leg}_{part}"]["quaternion_xyzw"]
                    for part in ("coxa_yaw", "femur_pitch", "tibia_pitch")
                ]
                for leg in LEGS
            ],
            dtype=torch.float64,
        )
        self.pitch = torch.zeros(num_envs, dtype=torch.float64)
        self.direction = torch.zeros(num_envs, 2, dtype=torch.float64)
        width = 3 if mode == "gravity" else 18
        self.history = torch.zeros(num_envs, 5, width)

    def update(self, toes, contact):
        """Refresh each replica's tilt from its toes (world metres, [n, 6, 3]) and floor contacts ([n, 6])."""
        toes, contact = toes.to(torch.float64), contact.to(torch.float64)
        front, rear = contact[:, FRONT_LEGS], contact[:, REAR_LEGS]
        n_front, n_rear = front.sum(-1), rear.sum(-1)
        front_toe = (toes[:, FRONT_LEGS] * front[..., None]).sum(1) / n_front.clamp_min(1)[:, None]
        rear_toe = (toes[:, REAR_LEGS] * rear[..., None]).sum(1) / n_rear.clamp_min(1)[:, None]
        span = front_toe - rear_toe
        distance = torch.linalg.vector_norm(span[:, :2], dim=-1)
        valid = (n_front > 0) & (n_rear > 0) & (distance >= 0.1)
        measured = torch.atan2(span[:, 2], distance).clamp(-LIMIT_RAD, LIMIT_RAD)
        self.pitch = torch.where(
            valid, self.pitch + SMOOTHING * (measured - self.pitch), self.pitch
        )
        self.direction = torch.where(
            valid[:, None], span[:, :2] / distance.clamp_min(1e-9)[:, None], self.direction
        )

    def values(self, quaternion, q):
        """This control's corrected gravity ([n, 3]) or joint offsets ([n, 18], in JOINT_NAMES order)."""
        quaternion = quaternion.to(torch.float64)
        if self.mode == "gravity":
            # The ground's downward normal leans away from the rise.
            c, s = torch.cos(self.pitch), torch.sin(self.pitch)
            down = torch.cat((s[:, None] * self.direction, -c[:, None]), -1)
            return navigation(inverse_rotate(quaternion, down)).to(torch.float32)
        # Rising ground turns the body about the horizontal axis across the rise (direction x up).
        across = torch.stack(
            (self.direction[:, 1], -self.direction[:, 0], torch.zeros_like(self.pitch)), -1
        )
        tilt = inverse_rotate(quaternion, self.pitch[:, None] * across)
        q = q.to(torch.float64).reshape(-1, 6, 3)
        up = torch.tensor([0.0, 0.0, 1.0], dtype=torch.float64).expand(len(q), 6, 3)
        out = torch.zeros(len(q), 6, 3, dtype=torch.float64)
        frame = self.frames[None, :, 0].expand(len(q), -1, -1)
        for part in (1, 2):
            frame = multiply(
                multiply(frame, about_z(q[:, :, part - 1])),
                self.frames[None, :, part].expand(len(q), -1, -1),
            )
            if part == 1 or self.mode == "joints":
                out[:, :, part] = (rotate(frame, up) * tilt[:, None]).sum(-1)
        return out.reshape(-1, 18).to(torch.float32)

    def push(self, quaternion, q, rows=None):
        """Add this control's corrected values; ``rows`` restarts those replicas' history with them."""
        value = self.values(quaternion, q)
        if rows is None:
            self.history = torch.cat((self.history[:, 1:], value[:, None]), 1)
        else:
            self.history[rows] = value[rows, None].expand(-1, 5, -1)

    def reset(self, rows):
        self.pitch[rows] = 0.0
        self.direction[rows] = 0.0

    def apply(self, observation):
        """The 231-value (or longer) observations with the corrected inputs in all five history frames."""
        out = observation.clone()
        for k in range(5):
            if self.mode == "gravity":
                out[:, 42 * k + GRAVITY.start : 42 * k + GRAVITY.stop] = self.history[:, k].to(out)
            else:
                out[:, 42 * k + JOINTS.start : 42 * k + JOINTS.stop] += self.history[:, k].to(out)
        return out
