#!/usr/bin/env python3
"""Write a scene file for the viewer's Scene page from the output of ``locomotion.surrogate.scene``.

Every robot becomes one case in the run file's layout (``run_view.py``): leg points in millimetres every second
control, foot contacts, heading, command and the ghost. Here all robots share the world's navigation axes
(forward, left) instead of each starting at the origin, each ghost starts at its robot's start pose, and the
file adds the terrain's height grid (int16 millimetres) after the leg points and contacts, then each robot's
body pose in MuJoCo's world (x, y, z, quaternion xyzw) and 18 joint angles per frame (float32), from which the
Scene page poses the robot's CAD parts.

Usage: uv run python viewer/scene_view.py <scene output folder> --output scene_view.json
"""

from __future__ import annotations

import argparse
import base64
import json
from pathlib import Path

import numpy as np

try:
    from viewer.run_view import (
        CONTROL_SECONDS,
        LEGS,
        ROOT,
        STEP,
        compress,
        leg_points,
        navigation,
        yaw,
    )
except ModuleNotFoundError:  # run as a script from viewer/
    from run_view import CONTROL_SECONDS, LEGS, ROOT, STEP, compress, leg_points, navigation, yaw

SCHEMA = "hexapod_scene_view_v1"


def ghost(pose, command, start):
    """The destination after each control from the start (world x, y, yaw); it restarts at the body when
    the command changes, as the destination experiments train it."""
    out, g = np.zeros((len(pose), 3)), np.asarray(start, dtype=float)
    for j in range(len(pose)):
        if j and (command[j] != command[j - 1]).any():
            g = np.array([pose[j - 1, 0], pose[j - 1, 1], yaw(pose[j - 1])])
        forward, left, turn = command[j]
        bx, by = left, -forward
        middle = g[2] + turn * CONTROL_SECONDS / 2
        g = g + np.array(
            [
                (np.cos(middle) * bx - np.sin(middle) * by) * CONTROL_SECONDS,
                (np.sin(middle) * bx + np.cos(middle) * by) * CONTROL_SECONDS,
                turn * CONTROL_SECONDS,
            ]
        )
        out[j] = g
    return out


def robot_case(trace, row, joints, lateral_m=0.0):
    pose = trace["root_pose_xyzw"].astype(float)
    command = trace["command"].astype(float)
    x, y, _, heading = row["start_world_xy_ground_yaw"]
    toes = trace["toe_xyz_world_m"].astype(float)
    if lateral_m:
        # Display-only lane offset (--spread): the world's x axis is the course's left.
        pose = pose.copy()
        pose[:, 0] += lateral_m
        toes = toes.copy()
        toes[..., 0] += lateral_m
        x += lateral_m
    points = leg_points(pose, trace["joint_position_rad"].astype(float), toes, joints)
    nav = np.concatenate((navigation(points[..., :2]), points[..., 2:]), -1)
    destination = ghost(pose, command, (x, y, heading))
    index = np.arange(0, len(pose), STEP)
    return {
        "name": row["name"],
        "checkpoint": row["checkpoint"],
        "checkpoint_sha256": row["checkpoint_sha256"],
        "fell_at_s": row["fell_at_s"],
        "controls": len(pose),
        "frames": len(index),
        "command": np.round(command[index], 4).tolist(),
        "heading_deg": np.round(np.degrees(np.unwrap(yaw(pose[index]))), 2).tolist(),
        "commanded_heading_deg": np.round(np.degrees(destination[index, 2]), 2).tolist(),
        "ghost_mm": np.round(navigation(destination[index, :2]) * 1000, 1).tolist(),
        "ghost_distance_mm": np.round(
            np.hypot(*(pose[index, :2] - destination[index, :2]).T) * 1000, 2
        ).tolist(),
        "_points": np.round(nav[index] * 1000).astype("<i2"),
        "_contacts": (trace["toe_contact"][index] * (1 << np.arange(6))).sum(-1).astype("u1"),
        "_pose": pose[index].astype("<f4"),
        "_joints": trace["joint_position_rad"][index].astype("<f4"),
    }


def build(folder, model=None, spread=0.0):
    folder = Path(folder)
    summary = json.loads((folder / "scene.json").read_text())
    active = json.loads((ROOT / "robot/active_model.json").read_text())
    model = Path(model) if model else ROOT / active["model"]["path"]
    joints = {j["name"]: j for j in json.loads(model.read_text())["joints"]}
    cases, blocks, contacts, poses, angles = [], [], [], [], []
    count = len(summary["robots"])
    for k, row in enumerate(summary["robots"]):
        with np.load(folder / row["trace"]) as trace:
            case = robot_case(trace, row, joints, lateral_m=(k - (count - 1) / 2) * spread)
        case["points_offset"] = sum(len(b) for b in blocks)
        case["contacts_offset"] = sum(len(c) for c in contacts)
        case["pose_offset"] = sum(len(b) for b in poses)
        case["joints_offset"] = sum(len(b) for b in angles)
        blocks.append(case.pop("_points").tobytes())
        contacts.append(case.pop("_contacts").tobytes())
        poses.append(case.pop("_pose").tobytes())
        angles.append(case.pop("_joints").tobytes())
        cases.append(case)
    points, contact_bytes = b"".join(blocks), b"".join(contacts)
    with np.load(folder / "terrain.npz") as grid:
        heights, xs, ys = grid["heights"], grid["xs"], grid["ys"]
    if spread:
        # Widen the drawn ground to cover the lanes: tile the course's centre cross-section
        # (the profile every trial actually walked) across the new width, display only.
        half = abs(spread) * (count - 1) / 2 + 0.9
        if half > float(xs[-1]):
            resolution = float(xs[1] - xs[0])
            centre = heights[:, [int(np.argmin(np.abs(xs)))]]
            xs = np.arange(-half, half + resolution / 2, resolution)
            heights = np.repeat(centre, len(xs), 1)
            summary["scene"]["terrain"]["size_m"][1] = round(2 * half, 2)
            for feature in summary["scene"]["terrain"].get("features", []):
                size = feature.get("size")
                if isinstance(size, list) and len(size) == 2:
                    size[1] = round(2 * half, 2)
    terrain_bytes = np.round(heights * 1000).astype("<i2").tobytes()
    # Padding keeps the terrain's int16 view and the float32 views aligned.
    pad = b"\0" * ((len(points) + len(contact_bytes)) % 2)
    terrain_base = len(points) + len(contact_bytes) + len(pad)
    pad4 = b"\0" * (-(terrain_base + len(terrain_bytes)) % 4)
    pose_base = terrain_base + len(terrain_bytes) + len(pad4)
    pose_bytes, joint_bytes = b"".join(poses), b"".join(angles)
    return {
        "schema": SCHEMA,
        "name": summary["scene"].get("name", folder.name),
        "scene": summary["scene"],
        "scope": summary["scope"],
        "seconds": summary["seconds"],
        "frame_seconds": STEP * CONTROL_SECONDS,
        "legs": list(LEGS),
        "contacts_base": len(points),
        "pose_base": pose_base,
        "joints_base": pose_base + len(pose_bytes),
        "terrain": {
            "base": terrain_base,
            "nx": len(xs),
            "ny": len(ys),
            "x0_m": float(xs[0]),
            "y0_m": float(ys[0]),
            "resolution_m": float(xs[1] - xs[0]),
        },
        "data": base64.b64encode(
            compress(points + contact_bytes + pad + terrain_bytes + pad4 + pose_bytes + joint_bytes)
        ).decode(),
        "cases": cases,
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("folder", type=Path, help="Output folder of locomotion.surrogate.scene")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--model", type=Path, help="Robot model JSON; the active model by default")
    parser.add_argument(
        "--spread",
        type=float,
        default=0.0,
        help="Metres between robots across the course: display-only lanes, so near-coincident"
        " starts (an evaluation's repeated trials) play side by side instead of stacked",
    )
    args = parser.parse_args(argv)
    value = build(args.folder, args.model, args.spread)
    args.output.write_text(json.dumps(value, separators=(",", ":")) + "\n")
    print(f"{args.output}: {len(value['cases'])} robots, {value['seconds']:.1f} s")


if __name__ == "__main__":
    main()
