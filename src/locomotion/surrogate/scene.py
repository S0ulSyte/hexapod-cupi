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

``"slope_inputs"`` on a robot (``"gravity"`` or ``true``, ``"femurs"`` or ``"joints"``) corrects what the policy
feels for a slope it walks along, as ``locomotion/omniwalk/mdp/slope.py`` describes and as training with the
same option does; the trace records the tilt estimate. A checkpoint trained with slope inputs records its mode
in its reward configuration, and a robot without ``slope_inputs`` then uses it.

A command entry may instead hold a pose without the policy: ``{"seconds": 2, "hold": "stand"}`` moves the
joints to the standing pose, ``"hold": "upside_down"`` to the standing pose of a robot on its back (tibias
folded over, toes beyond the top of the body). ``"upside_down": "roll"`` (or ``true``) on a command runs the
policy for a robot that rolled onto its back, ``"pitch"`` for one that tumbled over its front or rear, as
``UpsideDown`` describes. ``"start": {"upside_down": true}`` starts a robot on its back, rolled over, in that
pose. ``{"seconds": 6, "hold": "get_up"}`` gets a robot lying flat up onto its feet with the same moves
mirrored through the body's plane, ending in a level six-toed stance about two thirds of standing height
(``GET_UP_POSES`` says why); a walking policy finishes the rise when it takes over. ``"start": {"flat":
true}`` starts a robot lying flat on the ground, right side up, its legs folded over the top. ``"push": {"force": [forward, left,
up], "torque": [forward, left, up]}`` on any entry pushes the body at its centre of mass for the entry's
seconds, in newtons and newton metres along the robot's heading.

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

from locomotion.omniwalk.mdp.slope import SlopeTracker

SCHEMA = "hexapod_scene_v1"
CONTROL_SECONDS = 0.02
START_RADIUS_M = 0.25
EDGE_MARGIN_M = 0.35
FALL_TILT_RAD = 0.85
SLOPE_MODES = (True, "gravity", "femurs", "joints")
# What each control does: the policy, the policy upside down, or a held pose.
POLICY, UPSIDE_DOWN, HOLD_STAND, HOLD_UPSIDE_DOWN, HOLD_GET_UP = range(5)
HOLDS = {"stand": HOLD_STAND, "upside_down": HOLD_UPSIDE_DOWN, "get_up": HOLD_GET_UP}
FLIPS = {True: "roll", "roll": "roll", "pitch": "pitch"}
# The upside-down pose of a leg: femur and tibia angles mirrored about these, so a femur at the standing
# -0.3 rad sits at 0 and a tibia at 0.4 at 2.79, folded over the top. The exact mirror (femur about -0.70,
# tibia about 3.85) puts the standing tibia at 3.45, past its 3.14 rad stop; this one keeps the policy's whole
# tibia range inside the stop and the body 19 cm up. Upside down on flat ground at 0.1 m/s (forward, left and
# 0.25 rad/s turns) Spark v29 u600 tracked within 2% here; with the femur about -0.4 to -0.7 it lagged or stalled.
UPSIDE_DOWN_FEMUR_RAD = -0.3
UPSIDE_DOWN_TIBIA_RAD = 3.19
UPSIDE_DOWN_LIFT_M = 0.1
# Getting up off its back ("hold": "upside_down"), as (femur, tibia) of a leg. Every leg swings at once from
# the upright pose to grip close in, the femur swung down and the tibia nearly straight ("plant", the toes 11 cm
# out from the femur joints, where the motors' 1.6 N m cap lifts the body), which lifts the body off its shell;
# they push until the toes are as deep as in the standing pose ("lift"). Then the four outer legs, and after
# them the middle two, step into the standing pose: the femurs lift the toes 8 cm ("raise"), the tibias swing
# them over ("step") and they come down, as a straight swing pushed them deeper than either end and lifted the
# body off the other legs, and swinging the tibias while lifting did too (the loaded femurs lag). The body dips
# about 2.5 cm while the outer four step. With the tibias folded over instead (toes 15 cm out), lifting stalled
# with the body 11 cm up. The femurs stop 0.1 rad short of their 1.396 rad stop: at 1.39 the middle ones were
# pressed past it under load. The femurs run at the 1.6 N m cap for much of the lift, and the femur links
# brush the ground while planted.
STAND_UP_POSES = {
    "plant": (1.3, 0.36),
    "lift": (1.3, 0.848),
    "raise": (0.7, 0.848),
    "step": (-0.2, 2.0),
}
# Getting up onto its feet from lying flat ("hold": "get_up"): the same sequence mirrored through the body's
# plane (femur and tibia about UPSIDE_DOWN_FEMUR_RAD and UPSIDE_DOWN_TIBIA_RAD) where the joints reach. The
# exact mirror of the grip is out of reach (the tibia's 3.14 rad stop blocks tucking the toes under the femur
# joints), so the toes grip about 0.37 m out, where the press stalls at the motors' 1.6 N m cap (73 N robot,
# about 0.13 m femur lever) with the body 0.066 m up; and a raised toe can only be placed 0.40 m out or more,
# wide of the 0.35/0.26 m standing footprint, so the steps land wide and the final stand holds a level,
# six-toed stance at about two thirds of standing height (tilt under 1 degree). "step" parks the toe as near
# the footprint as the arc allows; steps parked further out ((-0.1, 1.19), the plain mirror) ended lower, and
# crouch-style plants that skip the grip ((-0.5, 0.6)) barely lifted the body at all (0.017 m). A walking
# policy finishes the rise on takeover: Spark v29 u600 reached 0.092 m within half a second and walked 0.595 m
# of a commanded 0.6 m straight out of the hold.
GET_UP_POSES = {
    "plant": (-1.6, 2.83),
    "lift": (-1.6, 2.342),
    "raise": (-1.0, 2.342),
    "step": (-0.2, 0.55),
}
# Lying flat ("start": {"flat": true}): the body dropped from just above where it rests on the ground.
FLAT_HEIGHT_M = 0.02
# The steps into the standing pose: the four outer legs, then the middle two.
STAND_UP_STEP_GROUPS = ((0, 2, 3, 5), (1, 4))
# Each move lasts its largest joint travel at the environment's target slew (0.04 rad a control) and a short
# settle; the two moves that lift the body, held at the torque cap, get longer. Getting up takes about 4.6 s.
STAND_UP_SLEW_RAD_S = 2.0
STAND_UP_SETTLE_S = 0.05
STAND_UP_PUSH_S = 0.3


def stand_up_keyframes(begin, end, poses=STAND_UP_POSES):
    """Each leg's (seconds into the hold, (femur, tibia)) keyframes of getting up, from the pose ``begin``
    through ``poses`` to the standing pose ``end``: off its back (``STAND_UP_POSES``, from the upright standing
    pose to the upside-down one) or onto its feet from lying flat (``GET_UP_POSES``, the other way)."""
    keys = [[(0.0, tuple(begin))] for _ in range(6)]

    def move(legs, pose, start, extra=0.0):
        """Move ``legs`` to ``pose`` from ``start``; returns when they get there."""
        travel = max(abs(a - b) for leg in legs for a, b in zip(pose, keys[leg][-1][1]))
        end = start + travel / STAND_UP_SLEW_RAD_S + STAND_UP_SETTLE_S + extra
        for leg in legs:
            if start > keys[leg][-1][0]:
                keys[leg].append((start, keys[leg][-1][1]))
            keys[leg].append((end, tuple(pose)))
        return end

    every = range(6)
    at = move(every, poses["plant"], 0.0, STAND_UP_PUSH_S)
    at = move(every, poses["lift"], at, STAND_UP_PUSH_S)
    for group in STAND_UP_STEP_GROUPS:
        at = move(group, poses["raise"], at)
        at = move(group, poses["step"], at)
        at = move(group, end, at)
    return [[(round(t, 3), pose) for t, pose in leg] for leg in keys]


def schedule(robot, controls):
    """The command of each control: each entry for its seconds, then a zero command."""
    rows, out = [], np.zeros((controls, 3))
    for entry in robot.get("commands", []):
        command = entry.get("command", [0, 0, 0] if "hold" in entry else None)
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
        if entry.get("hold", "stand") not in HOLDS:
            raise ValueError(f"{robot['name']}: hold is 'stand', 'upside_down' or 'get_up'")
        if entry.get("upside_down", "roll") not in FLIPS:
            raise ValueError(f"{robot['name']}: upside_down is true, 'roll' or 'pitch'")
        push = entry.get("push", {})
        if (
            not isinstance(push, dict)
            or set(push) - {"force", "torque"}
            or not all(
                isinstance(v, (list, tuple))
                and len(v) == 3
                and all(isinstance(x, (int, float)) and math.isfinite(x) for x in v)
                for v in push.values()
            )
        ):
            raise ValueError(
                f"{robot['name']}: a push has force and torque, each [forward, left, up]"
            )
        rows += [command] * round(entry["seconds"] / CONTROL_SECONDS)
    rows = np.asarray(rows[:controls], dtype=float).reshape(-1, 3)
    out[: len(rows)] = rows
    return out


def plan(robot, controls):
    """What each control does: ``phase`` (POLICY, UPSIDE_DOWN or a hold; the policy after the last entry),
    ``flip`` (``"roll"`` or ``"pitch"`` for an upside-down control, else None), ``push`` ([n, 6] force and
    torque along the robot's heading) and ``elapsed`` (seconds into its entry at the control's end)."""
    phase, flip, push, elapsed = [], [], [], []
    for entry in robot.get("commands", []):
        count = round(entry["seconds"] / CONTROL_SECONDS)
        elapsed += list((np.arange(count) + 1) * CONTROL_SECONDS)
        if "hold" in entry:
            phase += [HOLDS[entry["hold"]]] * count
        else:
            phase += [UPSIDE_DOWN if entry.get("upside_down") else POLICY] * count
        flip += [FLIPS[entry["upside_down"]] if entry.get("upside_down") else None] * count
        wrench = entry.get("push", {})
        push += [
            list(wrench.get("force", [0, 0, 0])) + list(wrench.get("torque", [0, 0, 0]))
        ] * count
    out = {
        "phase": np.full(controls, POLICY),
        "flip": [None] * controls,
        "push": np.zeros((controls, 6)),
        "elapsed": np.zeros(controls),
    }
    k = min(len(phase), controls)
    out["phase"][:k], out["flip"][:k], out["elapsed"][:k] = phase[:k], flip[:k], elapsed[:k]
    if k:
        out["push"][:k] = np.asarray(push[:k], dtype=float)
    return out


class UpsideDown:
    """The policy's view of a robot on its back, as an upright robot turned half a turn.

    ``"roll"`` turns it about its forward axis, for a robot that rolled over: the turn carries each left leg to
    the right, so the policy's lf leg is the robot's rf, and the policy's forward is the robot's. ``"pitch"``
    turns it about its left axis, for a robot that tumbled over its front or rear: the front legs become the
    rear ones, so the policy's lf leg is the robot's lr, and the policy faces the way the robot's rear faces
    (the way it was walking before a forward tumble). Either way the top of the body is the policy's bottom.
    The joint angles mirror the robot's: coxa angles change sign, femur and tibia angles mirror about
    ``UPSIDE_DOWN_FEMUR_RAD`` and ``UPSIDE_DOWN_TIBIA_RAD``; gravity and the body rates turn with the body. The
    robot holds targets about the upside-down standing pose, so the policy's actions keep their range.
    """

    def __init__(self, neutral, flip="roll"):
        legs = [3, 4, 5, 0, 1, 2] if flip == "roll" else [2, 1, 0, 5, 4, 3]
        self.index = torch.tensor([3 * leg + part for leg in legs for part in range(3)])
        self.sign = torch.tensor([-1.0, -1.0, -1.0] * 6)
        self.offset = torch.tensor([0.0, UPSIDE_DOWN_FEMUR_RAD, UPSIDE_DOWN_TIBIA_RAD] * 6)
        self.upright = neutral.clone()
        # The robot's joint angles that put the policy's joints at its standing pose.
        self.neutral = torch.empty_like(neutral)
        self.neutral[self.index] = self.sign * (neutral - self.offset)
        # Body vectors in navigation axes (forward, left, up) as the turned body's.
        self.axes = torch.tensor([1.0, -1.0, -1.0] if flip == "roll" else [-1.0, 1.0, -1.0])
        # Half a turn about the body's y (back) or x (left) axis, xyzw.
        self.turn = torch.tensor([0.0, 1.0, 0.0, 0.0] if flip == "roll" else [1.0, 0.0, 0.0, 0.0])

    def joints(self, values):
        """Joint offsets or rates of the robot ([n, 18]) as the policy's: legs swapped, signs mirrored."""
        return self.sign * values[:, self.index]

    def observation(self, observation):
        """The policy's 231 values for the robot's, which the environment builds about ``self.neutral``."""
        out = observation.clone()
        for k in range(5):
            frame = 42 * k
            out[:, frame : frame + 6] *= self.axes.repeat(2)
            out[:, frame + 6 : frame + 24] = self.joints(observation[:, frame + 6 : frame + 24])
            out[:, frame + 24 : frame + 42] = self.joints(observation[:, frame + 24 : frame + 42])
        out[:, 213:231] = self.joints(observation[:, 213:231])
        return out

    def action(self, action):
        """The robot's action, about ``self.neutral``, for the policy's."""
        out = torch.empty_like(action)
        out[:, self.index] = self.sign * action
        return out

    def root(self, root):
        """The policy's root pose for the robot's: the same position, the body turned half a turn."""
        from locomotion.surrogate.env import quaternion_multiply

        out = root.clone()
        out[:, 3:7] = quaternion_multiply(root[:, 3:7], self.turn.expand(len(root), -1))
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
        if start.get("upside_down") and start.get("flat"):
            raise ValueError(f"{robot['name']}: start upside_down or flat, not both")
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
        self.policy, self.checkpoint_sha256, trained = None, None, None
        if "checkpoint" in robot:
            self.policy, self.checkpoint_sha256, trained = self._policy(
                Path(robot["checkpoint"]), task_module, seed
            )
        mode = robot.get("slope_inputs", trained)
        mode = "gravity" if mode is True else mode
        self.slope_mode = mode or None
        self.slope = SlopeTracker(1, self.env.model["joints"], mode) if mode else None
        self.upside_down = {flip: UpsideDown(self.env.neutral, flip) for flip in ("roll", "pitch")}
        if start.get("upside_down"):
            # On its back in the upside-down standing pose, dropped from just above it; MuJoCo's qpos is wxyz.
            qpos = self.env._qpos0
            qpos[2] += UPSIDE_DOWN_LIFT_M
            qpos[3:7] = (0.0, -math.sin(heading / 2), math.cos(heading / 2), 0.0)
            neutral = self.upside_down["roll"].neutral
            qpos[self.env._qpos_columns] = neutral.numpy().astype(np.float64)
            self.env.neutral = neutral.clone()
        elif start.get("flat"):
            # Lying flat, right side up: the legs folded over the top (the pose getting up on its back ends
            # in), the body dropped from just above where it rests on the ground.
            qpos = self.env._qpos0
            qpos[2] = ground + FLAT_HEIGHT_M
            neutral = self.upside_down["roll"].neutral
            qpos[self.env._qpos_columns] = neutral.numpy().astype(np.float64)
            self.env.neutral = neutral.clone()

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
            actor,
            self.env,
            options,
            adapter=getattr(wrapped, "policy_for_evaluation", None),
            schedule_mirror=bool(
                getattr(getattr(task, "reward_config", None), "schedule_mirror_turns", None)
                or False
            ),
        )
        return policy, digest, getattr(task.reward_config, "slope_inputs", None)

    def run(self, commands, steps=None):
        """Step through ``commands`` (and ``steps``, what each control does, as ``plan`` gives it) from the
        start pose: the trace the viewer draws."""
        env, n = self.env, len(commands)
        if steps is None:
            steps = {
                "phase": np.full(n, POLICY),
                "flip": [None] * n,
                "push": np.zeros((n, 6)),
                "elapsed": np.zeros(n),
            }
        phase = steps["phase"]
        trace = {
            "root_pose_xyzw": np.zeros((n, 7)),
            "joint_position_rad": np.zeros((n, 18)),
            "toe_xyz_world_m": np.zeros((n, 6, 3)),
            "toe_contact": np.zeros((n, 6), bool),
            "command": commands.astype(np.float32),
            "fallen": np.zeros(n, bool),
            "slope_pitch_rad": np.zeros(n, np.float32),
            "phase": np.asarray(phase, np.int8),
        }
        env.commands.zero_()
        env.reset()
        if self.slope is not None:
            self.slope.push(env.current["root"][:, 3:7], env.current["q"], rows=[0])
        # The upright neutral and the upside-down one are each the same for both flips.
        upright, held_over = self.upside_down["roll"].upright, self.upside_down["roll"].neutral
        pushed = bool(np.any(steps["push"]))
        frames = {
            HOLD_UPSIDE_DOWN: stand_up_keyframes(upright[1:3].tolist(), held_over[1:3].tolist()),
            HOLD_GET_UP: stand_up_keyframes(
                held_over[1:3].tolist(), upright[1:3].tolist(), GET_UP_POSES
            ),
        }
        ends = {HOLD_UPSIDE_DOWN: held_over, HOLD_GET_UP: upright}
        for hold, name in ((HOLD_UPSIDE_DOWN, "upside_down"), (HOLD_GET_UP, "get_up")):
            took = max(leg[-1][0] for leg in frames[hold])
            last = [
                k for k in range(n) if phase[k] == hold and (k + 1 == n or phase[k + 1] != hold)
            ]
            if any(steps["elapsed"][k] < took - 1e-9 for k in last):
                raise ValueError(f"Getting up takes {took:.2f} s; hold '{name}' longer")
        start = upright
        for k in range(n):
            flipped = phase[k] in (UPSIDE_DOWN, HOLD_UPSIDE_DOWN)
            env.neutral = held_over if flipped else upright
            if phase[k] in (HOLD_UPSIDE_DOWN, HOLD_GET_UP):
                if k == 0 or phase[k - 1] != phase[k]:
                    start = env.held[0].clone()
                env.neutral = self._getting_up(
                    frames[phase[k]], start, ends[phase[k]], steps["elapsed"][k]
                )
            if pushed:
                env.applied_wrench = self._wrench(steps["push"][k])
            if k and (phase[k], steps["flip"][k]) != (phase[k - 1], steps["flip"][k - 1]):
                self._restart_destination()
            env.commands.copy_(torch.as_tensor(commands[k : k + 1], dtype=torch.float32))
            observation = env.observations(env.current)["obs"]
            if self.slope is not None and not flipped:
                observation = self.slope.apply(observation)
            if self.policy is None or phase[k] in (HOLD_STAND, HOLD_UPSIDE_DOWN, HOLD_GET_UP):
                action = torch.zeros(1, 18)
            elif phase[k] == UPSIDE_DOWN:
                upside_down = self.upside_down[steps["flip"][k]]
                # The policy reads the root pose for its destination inputs: give it the turned body's.
                current = env.current
                env.current = dict(current, root=upside_down.root(current["root"]))
                try:
                    action = upside_down.action(self.policy(upside_down.observation(observation)))
                finally:
                    env.current = current
            else:
                action = self.policy(observation)
            env.step(action)
            state, telemetry = env.current, env.telemetry
            trace["root_pose_xyzw"][k] = state["root"][0].numpy()
            trace["joint_position_rad"][k] = state["q"][0].numpy()
            trace["toe_xyz_world_m"][k] = telemetry["toe_xyz_world"][0].numpy()
            force = telemetry["tibia_floor_force_world_n"][0]
            trace["toe_contact"][k] = (torch.linalg.vector_norm(force, dim=-1) > 1.0).numpy()
            up = -1.0 if flipped else 1.0
            trace["fallen"][k] = bool(up * state["gravity"][0, 2] > -math.cos(FALL_TILT_RAD))
            if self.slope is not None:
                self.slope.update(
                    torch.as_tensor(trace["toe_xyz_world_m"][k : k + 1]),
                    torch.as_tensor(trace["toe_contact"][k : k + 1]),
                )
                self.slope.push(state["root"][:, 3:7], state["q"])
                trace["slope_pitch_rad"][k] = float(self.slope.pitch[0])
        return trace

    def _restart_destination(self):
        """Start the policy's destination inputs afresh from the body where it is: after a hold or a turn
        over, the destination of the last command (and the last pose) lie wherever the robot was then."""
        tracker = getattr(self.policy, "tracker", None)
        if tracker is not None:
            # Unequal to any command, so the next control restarts.
            tracker.command = torch.full_like(tracker.command, float("nan"))
            if hasattr(tracker, "pose"):
                tracker.pose = None

    @staticmethod
    def _getting_up(frames, start, end, elapsed):
        """The held targets ``elapsed`` seconds into getting up: each leg from ``start`` through its keyframes
        in ``frames`` to the standing pose ``end`` (upside-down off its back, upright from lying flat)."""
        pose = end.clone()
        for leg, keys in enumerate(frames):
            first = slice(3 * leg, 3 * leg + 3)
            begin = start[first].clone()
            points = [(0.0, begin)]
            for at, (femur, tibia) in keys[1:]:
                points.append((at, torch.tensor([end[3 * leg], femur, tibia])))
            pose[first] = points[-1][1]
            for (a, p), (b, q) in zip(points, points[1:]):
                if elapsed <= b:
                    pose[first] = p + (q - p) * min(1.0, (elapsed - a) / max(b - a, 1e-9))
                    break
        return pose

    def _wrench(self, push):
        """A push along the robot's heading ([force, torque], forward, left, up) in world axes, [1, 6]."""
        if not np.any(push):
            return np.zeros((1, 6))
        x, y, z, w = self.env.current["root"][0, 3:7].tolist()
        # The body's forward axis (its -y) in the world, whose forward is -y and left x.
        heading = math.atan2(-2 * (x * y - w * z), 1 - 2 * (x * x + z * z))
        c, s = math.cos(heading), math.sin(heading)

        def world(v):
            f, left, up = v
            return [f * s + left * c, -(f * c - left * s), up]

        return np.asarray([world(push[:3]) + world(push[3:])])


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
        trace = robot.run(schedule(spec, controls), plan(spec, controls))
        np.savez_compressed(output / f"robot_{k}.npz", **trace)
        fallen = np.flatnonzero(trace["fallen"])
        rows.append(
            {
                "name": spec["name"],
                "trace": f"robot_{k}.npz",
                "checkpoint": spec.get("checkpoint"),
                "checkpoint_sha256": robot.checkpoint_sha256,
                "slope_inputs": robot.slope_mode,
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
