"""Run robots through a scene on the surrogate: a terrain, and each robot's checkpoint, start and commands.

A scene file (schema ``hexapod_scene_v1``) is JSON:

    {"schema": "hexapod_scene_v1", "name": "...", "seconds": 60,
     "terrain": {"size_m": [6, 6], "resolution_m": 0.02, "features": [...]},
     "robots": [{"name": "v29", "checkpoint": "path/to/checkpoint_update000600.pt",
                 "start": {"forward_m": 0, "left_m": 0, "heading_deg": 0},
                 "commands": [{"seconds": 4, "command": [0.1, 0, 0]}, ...]}]}

Positions are in the viewer's navigation frame (forward, left; ``terrain.py`` names the features).
``checkpoint`` paths are relative to the scene file; ``"policy": "stand"`` instead holds the standing pose.
Each command is [forward m/s, left m/s, turn rad/s] in the robot's frame and lasts its ``seconds``; after the
last one the robot stands until the scene ends (``seconds``, by default the longest command list). Commands
may run any length: the scene ignores the environment's episode timeout and never resets a robot.

Every robot runs alone on its own copy of the terrain, so robots pass through each other. Each starts above
the highest ground within 0.25 m, at least 0.35 m inside the terrain: the terrain is the whole floor, and a
robot that walks off its edge falls. The policy path is the surrogate evaluation's (``DeterministicPolicy``:
actor mean, no noise). A scene is design evidence only: the touchdown model is off, because it assumes the
flat floor, and the policies were trained on flat ground.

``"slope_inputs"`` on a robot corrects what the policy feels for a slope it walks along. Whenever a front leg
(lf or rf) and a rear leg (lr or rr) touch the ground, the height difference between their toes over the
distance between them gives the robot's tilt along the slope (smoothed over about 0.1 s, held while no front and
rear leg touch at once, limited to 20 degrees; the trace records it). With ``"gravity"`` (or ``true``) the gravity
inputs become gravity relative to that sloped ground. With ``"femurs"`` the gravity inputs stay and each femur
pitch input becomes its measured angle plus the tilt about that joint's axis: nearly the whole tilt, with
opposite signs, for the front and rear femurs, which reach forward and back, and almost none for the middle
ones, whose axes lie across a pitch. ``"joints"`` adds the tilt about each tibia's axis to the tibia inputs too,
although a tibia's angle is measured from its femur. Every mode changes all five history frames.

The output folder holds ``scene.json`` (the scene with resolved checkpoint paths), ``terrain.npz`` and one
``robot_<k>.npz`` trace per robot; ``viewer/scene_view.py`` turns it into a file for the viewer's Scene page.

Usage: uv run python -m locomotion.surrogate.scene scene.json --output folder
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

SCHEMA = "hexapod_scene_v1"
CONTROL_SECONDS = 0.02
START_RADIUS_M = 0.25
EDGE_MARGIN_M = 0.35
FALL_TILT_RAD = 0.85
FRONT_LEGS, REAR_LEGS = (0, 3), (2, 5)
SLOPE_LIMIT_RAD = math.radians(20)
SLOPE_SMOOTHING = 0.2
GRAVITY = slice(3, 6)
JOINTS = slice(6, 24)
FEMURS = (1, 4, 7, 10, 13, 16)
SLOPE_MODES = (True, "gravity", "femurs", "joints")


def _quaternion(xyzw):
    return torch.tensor(xyzw, dtype=torch.float64)


def _multiply(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return torch.stack(
        (
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        )
    )


class SlopeInputs:
    """The robot's tilt along a slope from its front and rear toes, and the policy inputs it corrects:
    gravity relative to the sloped ground (``gravity``), or the femur angles (``femurs``) or femur and tibia
    angles (``joints``) plus the tilt about each joint's axis."""

    def __init__(self, mode=True, joints=None):
        if mode not in SLOPE_MODES:
            raise ValueError(f"slope_inputs is one of {SLOPE_MODES}")
        self.mode = "gravity" if mode is True else mode
        self.pitch, self.direction = 0.0, np.array([0.0, 0.0])
        self.frames = None
        # Each leg's joint frames, for the femur and tibia pitch axes in the body.
        named = {j["name"]: j for j in joints or []}
        self.legs = (
            [
                tuple(
                    _quaternion(named[f"{leg}_{part}"]["quaternion_xyzw"])
                    for part in ("coxa_yaw", "femur_pitch", "tibia_pitch")
                )
                for leg in ("lf", "lm", "lr", "rf", "rm", "rr")
            ]
            if joints
            else []
        )

    def update(self, toes, contact):
        """Refresh the pitch estimate from the toes (world metres) that touch the ground."""
        front = [i for i in FRONT_LEGS if contact[i]]
        rear = [i for i in REAR_LEGS if contact[i]]
        if not front or not rear:
            return
        span = toes[front].mean(0) - toes[rear].mean(0)
        distance = float(np.hypot(span[0], span[1]))
        if distance < 0.1:
            return
        measured = float(np.clip(math.atan2(span[2], distance), -SLOPE_LIMIT_RAD, SLOPE_LIMIT_RAD))
        self.pitch += SLOPE_SMOOTHING * (measured - self.pitch)
        self.direction = span[:2] / distance

    def gravity(self, quaternion):
        """Gravity relative to the sloped ground, in the body's navigation axes (the policy's inputs 3 to 5)."""
        from locomotion.omniwalk.mdp.frames import inverse_rotate, navigation

        # The ground's downward normal leans away from the rise: rising forward, it points down and forward.
        c, s = math.cos(self.pitch), math.sin(self.pitch)
        down = torch.tensor(
            [[s * self.direction[0], s * self.direction[1], -c]], dtype=torch.float64
        )
        return navigation(inverse_rotate(quaternion.to(torch.float64), down)).to(torch.float32)[0]

    def joint_offsets(self, quaternion, q):
        """The tilt about each femur's (and, for ``joints``, each tibia's) pitch axis (rad), in JOINT_NAMES order;
        zero for the coxas."""
        from locomotion.omniwalk.mdp.frames import inverse_rotate, rotate

        # Rising ground turns the body about the horizontal axis across the rise: direction x up.
        across = torch.tensor([[self.direction[1], -self.direction[0], 0.0]], dtype=torch.float64)
        tilt = inverse_rotate(quaternion.to(torch.float64), self.pitch * across)[0]
        out = torch.zeros(18)
        up = torch.tensor([[0.0, 0.0, 1.0]], dtype=torch.float64)
        for leg, frames in enumerate(self.legs):
            frame = torch.tensor([0.0, 0.0, 0.0, 1.0], dtype=torch.float64)
            for part, joint in enumerate(frames):
                frame = _multiply(frame, joint)
                if part and (part == 1 or self.mode == "joints"):
                    axis = rotate(frame[None], up)[0]
                    out[3 * leg + part] = float(tilt @ axis)
                angle = float(q[3 * leg + part])
                turn = torch.tensor(
                    [0.0, 0.0, math.sin(angle / 2), math.cos(angle / 2)], dtype=torch.float64
                )
                frame = _multiply(frame, turn)
        return out

    def push(self, quaternion, q=None):
        """Add this control's corrected inputs (gravity, or the joint offsets) to the five-frame history."""
        value = (
            self.gravity(quaternion)
            if self.mode == "gravity"
            else self.joint_offsets(quaternion, q)
        )
        self.frames = (
            value.repeat(5, 1) if self.frames is None else torch.cat((self.frames[1:], value[None]))
        )

    def apply(self, observation):
        out = observation.clone()
        for k in range(5):
            if self.mode == "gravity":
                out[0, 42 * k + GRAVITY.start : 42 * k + GRAVITY.stop] = self.frames[k]
            else:
                out[0, 42 * k + JOINTS.start : 42 * k + JOINTS.stop] += self.frames[k]
        return out


def schedule(robot, controls):
    """The command of each control: each entry for its seconds, then a zero command."""
    rows, out = [], np.zeros((controls, 3))
    for entry in robot.get("commands", []):
        command = entry.get("command")
        if (
            not isinstance(command, (list, tuple))
            or len(command) != 3
            or not all(isinstance(x, (int, float)) and math.isfinite(x) for x in command)
            or not isinstance(entry.get("seconds"), (int, float))
            or entry["seconds"] <= 0
        ):
            raise ValueError(
                f"{robot['name']}: each command is [forward, left, turn] with seconds > 0"
            )
        rows += [command] * round(entry["seconds"] / CONTROL_SECONDS)
    rows = np.asarray(rows[:controls], dtype=float).reshape(-1, 3)
    out[: len(rows)] = rows
    return out


def scene_seconds(scene):
    planned = [
        sum(entry["seconds"] for entry in robot.get("commands", [])) for robot in scene["robots"]
    ]
    return float(scene.get("seconds") or max(planned + [1.0]))


def validate(scene):
    from locomotion.surrogate import terrain

    if scene.get("schema") != SCHEMA:
        raise ValueError(f"A scene file has schema {SCHEMA}")
    terrain.validate(scene.get("terrain", {}))
    robots = scene.get("robots")
    if not isinstance(robots, list) or not robots:
        raise ValueError("A scene has at least one robot")
    names = [robot.get("name") for robot in robots]
    if len(set(names)) != len(names) or not all(isinstance(n, str) and n for n in names):
        raise ValueError("Every robot has its own nonempty name")
    (cf, cl), (length, width) = scene["terrain"].get("center_m", [0, 0]), scene["terrain"]["size_m"]
    for robot in robots:
        if robot.get("slope_inputs", False) not in (False, None, *SLOPE_MODES):
            raise ValueError(
                f"{robot['name']}: slope_inputs is false, true, 'gravity', 'femurs' or 'joints'"
            )
        if ("checkpoint" in robot) == (robot.get("policy") == "stand"):
            raise ValueError(f"{robot['name']}: give a checkpoint or policy 'stand', not both")
        start = robot.get("start", {})
        if (
            abs(start.get("forward_m", 0.0) - cf) > length / 2 - EDGE_MARGIN_M
            or abs(start.get("left_m", 0.0) - cl) > width / 2 - EDGE_MARGIN_M
        ):
            raise ValueError(
                f"{robot['name']}: start at least {EDGE_MARGIN_M} m inside the terrain"
            )
    if not 0 < scene_seconds(scene) <= 3600:
        raise ValueError("A scene lasts up to an hour")


class Robot:
    """One robot's environment and deterministic policy, as the surrogate evaluation builds them."""

    def __init__(self, robot, terrain, seed):
        from locomotion.omniwalk import task as task_module
        from locomotion.surrogate.env import ContactModel, SurrogateConfig, SurrogateEnv

        start = robot.get("start", {})
        forward, left = float(start.get("forward_m", 0.0)), float(start.get("left_m", 0.0))
        heading = math.radians(float(start.get("heading_deg", 0.0)))
        ground = terrain.highest_under(forward, left, START_RADIUS_M)
        # World x is left and y is -forward; the identity orientation faces forward.
        self.start = (left, -forward, ground, heading)
        cfg = SurrogateConfig(num_envs=1, seed=seed, episode_seconds=120.0)
        self.env = SurrogateEnv(
            cfg,
            threads=0,
            contact=ContactModel(touchdown_overshoot=False),
            terrain=None if terrain.flat else terrain,
            start=self.start,
        )
        mode = robot.get("slope_inputs")
        self.slope = SlopeInputs(mode, self.env.model["joints"]) if mode else None
        self.policy, self.checkpoint_sha256 = None, None
        if "checkpoint" in robot:
            self.policy, self.checkpoint_sha256 = self._policy(
                Path(robot["checkpoint"]), task_module, seed
            )

    def _policy(self, path, task_module, seed):
        import locomotion.rl.ppo as algorithms
        from locomotion.surrogate.evaluate import DeterministicPolicy, task_and_wrapper
        from locomotion.surrogate.train import build_wrapper, resolve, wrapper_options

        record = path.with_suffix(".json")
        identity = json.loads(record.read_text())["identity"] if record.exists() else {}
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        config = identity.get("ppo_config")
        if config is None:
            raise ValueError(f"{path} has no JSON record with its PPO configuration")
        options = wrapper_options(config)
        task_spec, wrapper_spec = task_and_wrapper(
            SimpleNamespace(task=None, wrapper=None), identity
        )
        task = resolve(task_spec)(self.env, task_module.TaskConfig(seed=seed), None)
        wrapped = build_wrapper(resolve(wrapper_spec), task, options)
        observations = wrapped.get_observations()
        if "proprio_history" in config["obs_groups"]["actor"]:
            from locomotion.amp.networks import actor_observation

            observations.update(actor_observation(wrapped.current["obs"]))
        actor, _ = algorithms.load_actor(checkpoint, config, observations, "cpu")
        policy = DeterministicPolicy(
            actor, self.env, options, adapter=getattr(wrapped, "policy_for_evaluation", None)
        )
        return policy, digest

    def run(self, commands):
        """Step through ``commands`` from the start pose: the trace the viewer draws."""
        env, n = self.env, len(commands)
        trace = {
            "root_pose_xyzw": np.zeros((n, 7)),
            "joint_position_rad": np.zeros((n, 18)),
            "toe_xyz_world_m": np.zeros((n, 6, 3)),
            "toe_contact": np.zeros((n, 6), bool),
            "command": commands.astype(np.float32),
            "fallen": np.zeros(n, bool),
            "slope_pitch_rad": np.zeros(n, np.float32),
        }
        env.commands.zero_()
        env.reset()
        if self.slope is not None:
            self.slope.push(env.current["root"][:, 3:7], env.current["q"][0])
        for k in range(n):
            env.commands.copy_(torch.as_tensor(commands[k : k + 1], dtype=torch.float32))
            observation = env.observations(env.current)["obs"]
            if self.slope is not None:
                observation = self.slope.apply(observation)
            if self.policy is None:
                action = torch.zeros(1, 18)
            else:
                action = self.policy(observation)
            env.step(action)
            state, telemetry = env.current, env.telemetry
            trace["root_pose_xyzw"][k] = state["root"][0].numpy()
            trace["joint_position_rad"][k] = state["q"][0].numpy()
            trace["toe_xyz_world_m"][k] = telemetry["toe_xyz_world"][0].numpy()
            force = telemetry["tibia_floor_force_world_n"][0]
            trace["toe_contact"][k] = (torch.linalg.vector_norm(force, dim=-1) > 1.0).numpy()
            trace["fallen"][k] = bool(state["gravity"][0, 2] > -math.cos(FALL_TILT_RAD))
            if self.slope is not None:
                self.slope.update(trace["toe_xyz_world_m"][k], trace["toe_contact"][k])
                self.slope.push(state["root"][:, 3:7], state["q"][0])
                trace["slope_pitch_rad"][k] = self.slope.pitch
        return trace


def run_scene(path, output, seed=20260917):
    """Run every robot of the scene at ``path`` and write the output folder; returns its summary."""
    from locomotion.surrogate.terrain import Terrain

    path, output = Path(path), Path(output)
    scene = json.loads(path.read_text())
    for robot in scene.get("robots", []):
        if "checkpoint" in robot:
            robot["checkpoint"] = str((path.parent / robot["checkpoint"]).resolve())
    validate(scene)
    output.mkdir(parents=True, exist_ok=False)
    terrain = Terrain(scene["terrain"])
    controls = round(scene_seconds(scene) / CONTROL_SECONDS)
    np.savez_compressed(
        output / "terrain.npz", heights=terrain.heights, xs=terrain.xs, ys=terrain.ys
    )
    rows = []
    for k, spec in enumerate(scene["robots"]):
        started = time.monotonic()
        robot = Robot(spec, terrain, seed)
        trace = robot.run(schedule(spec, controls))
        np.savez_compressed(output / f"robot_{k}.npz", **trace)
        fallen = np.flatnonzero(trace["fallen"])
        rows.append(
            {
                "name": spec["name"],
                "trace": f"robot_{k}.npz",
                "checkpoint": spec.get("checkpoint"),
                "checkpoint_sha256": robot.checkpoint_sha256,
                "slope_inputs": spec.get("slope_inputs") or None,
                "start_world_xy_ground_yaw": list(robot.start),
                "fell_at_s": None if not len(fallen) else round(fallen[0] * CONTROL_SECONDS, 2),
                "wall_seconds": round(time.monotonic() - started, 1),
            }
        )
        robot.env.close()
        print(json.dumps(rows[-1]), flush=True)
    summary = {
        "schema": "hexapod_scene_output_v1",
        "scene": scene,
        "seconds": controls * CONTROL_SECONDS,
        "seed": seed,
        "robots": rows,
        "scope": "CPU MuJoCo surrogate with the touchdown model off; design evidence only.",
    }
    (output / "scene.json").write_text(json.dumps(summary, indent=1) + "\n")
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("scene", type=Path)
    parser.add_argument("--output", type=Path, required=True, help="New directory.")
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--threads", type=int, default=2, help="Torch threads (1 to 6).")
    args = parser.parse_args(argv)
    from locomotion.surrogate.train import process_threads

    with process_threads(args.threads):
        run_scene(args.scene, args.output, args.seed)


if __name__ == "__main__":
    main()
