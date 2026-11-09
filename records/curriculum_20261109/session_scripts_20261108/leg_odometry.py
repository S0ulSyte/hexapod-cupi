"""Leg odometry on recorded traces: how close the robot can place itself to the ghost
using only what it has on board.

Inputs used, all robot-knowable: body-frame toe positions (pure joint forward kinematics —
reconstructed here from the recorded world toes and pose, which is kinematically identical),
foot contact, and the gyro's yaw rate. A toe in stance is world-fixed, so between controls
the body's planar displacement is the mean of R(theta_prev) @ toe_prev - R(theta) @ toe over
stance toes, with theta integrated from the gyro. The ghost is the integrated command
(DestinationTracker semantics: restart at the body on a command change). Diagnostic only.
"""

import glob
import json
import math

import numpy as np

C = "/Users/new/Desktop/cupi/hexapod-surrogate/runs/curriculum/full96_tw_s3_u600"
DT = 0.02
CASES = [
    "static__translate_0.10_0deg",
    "static__translate_0.10_90deg",
    "static__translate_0.15_270deg",
    "static__gentle_arc_0deg_+0.05",
    "static__combined_0deg_+0.20",
    "static__turn_+0.20",
    "stop__arc_left",
    "stop__forward",
    "transition__full_program",
]


def quat_to_yaw_matrix(q):
    """Planar rotation of the body's heading from the xyzw quaternion."""
    x, y, z, w = q
    yaw = math.atan2(-1 + 2 * (x * x + z * z), 2 * (w * z - x * y)) + math.pi / 2
    # The exact convention cancels in the comparison; only consistency matters.
    c, s = math.cos(yaw), math.sin(yaw)
    return np.array([[c, -s], [s, c]]), yaw


def rotate_inverse(q, v):
    """Rotate world vectors into the body frame (conjugate quaternion rotation)."""
    x, y, z, w = -q[0], -q[1], -q[2], q[3]
    t = 2 * np.cross(np.array([x, y, z]), v)
    return v + w * t + np.cross(np.array([x, y, z]), t)


def rot2(theta):
    c, s = math.cos(theta), math.sin(theta)
    return np.array([[c, -s], [s, c]])


def ghost_track(pose, command):
    """DestinationTracker's ghost: integrated command, restarting at the body on a change."""
    out = np.zeros((len(pose), 3))
    g = np.array([pose[0, 0], pose[0, 1], 0.0])
    for j in range(len(pose)):
        if j and (command[j] != command[j - 1]).any():
            x, y, z, w = pose[j - 1, 3:]
            yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
            g = np.array([pose[j - 1, 0], pose[j - 1, 1], yaw])
        forward, left, turn = command[j]
        bx, by = left, -forward
        middle = g[2] + turn * DT / 2
        g = g + np.array(
            [
                (math.cos(middle) * bx - math.sin(middle) * by) * DT,
                (math.sin(middle) * bx + math.cos(middle) * by) * DT,
                turn * DT,
            ]
        )
        out[j] = g
    return out


def odometry(trace):
    pose = trace["root_pose_xyzw"][:, 0].astype(float)
    toes_w = trace["toe_xyz_world_m"][:, 0].astype(float)
    contact = trace["distal_contact"][:, 0].astype(bool)
    gyro_z = trace["gyro_body_rad_s"][:, 0, 2].astype(float)
    n = len(pose)
    # Body-frame toes: pure forward kinematics, reconstructed from the recording.
    toes_b = np.stack(
        [rotate_inverse(pose[t, 3:], toes_w[t] - pose[t, :3]) for t in range(n)]
    )
    x, y, z, w = pose[0, 3:]
    theta = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
    p = pose[0, :2].copy()
    est = np.zeros((n, 2))
    est[0] = p
    for t in range(1, n):
        theta_prev = theta
        theta += gyro_z[t] * DT
        stance = contact[t] & contact[t - 1]
        if stance.any():
            prev = toes_b[t - 1, stance, :2] @ rot2(theta_prev).T
            now = toes_b[t, stance, :2] @ rot2(theta).T
            p = p + (prev - now).mean(0)
        est[t] = p
    return est, pose


def main():
    print(f"{'case':34s} {'odometry end drift':>19s} {'true end gap':>13s} {'odo end gap':>12s} {'mean |gap err|':>14s}")
    for case in CASES:
        hit = None
        for rp in sorted(glob.glob(f"{C}/evaluation/batch_*/report.json")):
            r = json.loads(open(rp).read())
            if r["assigned_case_ids"][0].endswith(case):
                hit = rp
                break
        if hit is None:
            print(f"{case:34s} not found")
            continue
        trace = np.load(hit.replace("report.json", "control_trace.npz"))
        est, pose = odometry(trace)
        command = trace["command"][:, 0].astype(float)
        ghost = ghost_track(pose, command)
        true_gap = np.hypot(*(pose[:, :2] - ghost[:, :2]).T)
        odo_gap = np.hypot(*(est - ghost[:, :2]).T)
        drift = np.hypot(*(est - pose[:, :2]).T)
        print(
            f"{case:34s} {drift[-1] * 1000:16.1f} mm {true_gap[-1] * 1000:10.1f} mm "
            f"{odo_gap[-1] * 1000:9.1f} mm {np.abs(odo_gap - true_gap).mean() * 1000:11.1f} mm"
        )


if __name__ == "__main__":
    main()
