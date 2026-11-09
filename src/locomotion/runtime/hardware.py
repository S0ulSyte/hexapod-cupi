"""Map motors and the IMU onto the canonical joints and body frame.

These numbers come from the physical robot (ARCHITECTURE: verify joint mapping on hardware), so
nothing here has defaults: a calibration file names, per canonical joint, its motor id, the sign
of the motor angle and the motor angle at the joint's zero; and the IMU's rotation into the URDF
body frame (forward = body -Y, left = body +X, up = body +Z).
"""

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass(frozen=True)
class JointMap:
    joint_names: tuple
    motor_ids: tuple
    sign: np.ndarray
    zero_rad: np.ndarray

    @classmethod
    def from_record(cls, record, joint_names):
        """``record['joints']`` maps each canonical name to {motor_id, sign (+1/-1), zero_rad}."""
        joints = record["joints"]
        if set(joints) != set(joint_names):
            raise ValueError("The joint map must name exactly the 18 canonical joints")
        rows = [joints[name] for name in joint_names]
        ids = tuple(row["motor_id"] for row in rows)
        if len(set(ids)) != len(ids):
            raise ValueError("Two joints share a motor id")
        sign = np.array([float(row["sign"]) for row in rows])
        if not np.all(np.isin(sign, (-1.0, 1.0))):
            raise ValueError("Joint signs must be +1 or -1")
        zero = np.array([float(row["zero_rad"]) for row in rows])
        return cls(tuple(joint_names), ids, sign, zero)

    def to_joint(self, motor_position, motor_velocity):
        """Canonical (q, dq) from motor readings ordered like ``motor_ids``."""
        position = np.asarray(motor_position, dtype=np.float64)
        return self.sign * (position - self.zero_rad), self.sign * np.asarray(motor_velocity, float)

    def to_motor(self, joint_target):
        """Motor position targets ordered like ``motor_ids``; gains need no sign."""
        return self.zero_rad + self.sign * np.asarray(joint_target, dtype=np.float64)


@dataclass(frozen=True)
class ImuMount:
    """``body_from_imu``: the 3x3 rotation taking IMU-frame vectors to body-frame vectors."""

    body_from_imu: np.ndarray

    @classmethod
    def from_record(cls, record):
        rotation = np.asarray(record["body_from_imu"], dtype=np.float64)
        if rotation.shape != (3, 3) or not np.allclose(rotation @ rotation.T, np.eye(3), atol=1e-6):
            raise ValueError("body_from_imu must be a 3x3 rotation")
        if np.linalg.det(rotation) < 0:
            raise ValueError("body_from_imu must be proper (determinant +1)")
        return cls(rotation)

    def gyro_body(self, gyro_imu):
        return self.body_from_imu @ np.asarray(gyro_imu, dtype=np.float64)

    def gravity_body(self, world_from_imu_xyzw):
        """Unit gravity in the body frame from the IMU's attitude quaternion (world z up)."""
        x, y, z, w = np.asarray(world_from_imu_xyzw, dtype=np.float64)
        norm = np.sqrt(x * x + y * y + z * z + w * w)
        x, y, z, w = x / norm, y / norm, z / norm, w / norm
        # Third row of world_from_imu = world z axis in IMU coordinates; gravity is its negative.
        down_imu = -np.array([2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)])
        return self.body_from_imu @ down_imu

    def yaw(self, world_from_imu_xyzw):
        """The body heading about world z (any fixed offset is harmless: only changes are used)."""
        x, y, z, w = np.asarray(world_from_imu_xyzw, dtype=np.float64)
        world_from_imu = np.array(
            [
                [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
            ]
        )
        world_from_body = world_from_imu @ self.body_from_imu.T
        return float(np.arctan2(world_from_body[1, 0], world_from_body[0, 0]))


def load_calibration(path, joint_names):
    record = json.loads(Path(path).read_text())
    return JointMap.from_record(record, joint_names), ImuMount.from_record(record["imu"])
