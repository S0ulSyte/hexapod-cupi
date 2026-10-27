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
        }
        env.commands.zero_()
        env.reset()
        for k in range(n):
            env.commands.copy_(torch.as_tensor(commands[k : k + 1], dtype=torch.float32))
            observation = env.observations(env.current)
            if self.policy is None:
                action = torch.zeros(1, 18)
            else:
                action = self.policy(observation["obs"])
            env.step(action)
            state, telemetry = env.current, env.telemetry
            trace["root_pose_xyzw"][k] = state["root"][0].numpy()
            trace["joint_position_rad"][k] = state["q"][0].numpy()
            trace["toe_xyz_world_m"][k] = telemetry["toe_xyz_world"][0].numpy()
            force = telemetry["tibia_floor_force_world_n"][0]
            trace["toe_contact"][k] = (torch.linalg.vector_norm(force, dim=-1) > 1.0).numpy()
            trace["fallen"][k] = bool(state["gravity"][0, 2] > -math.cos(FALL_TILT_RAD))
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
