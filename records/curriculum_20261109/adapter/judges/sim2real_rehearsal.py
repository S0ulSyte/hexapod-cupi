"""End-to-end rehearsal of the deployment plan, with the simulator standing in for hardware.

The recorded plan (decisions.md, sim2real phase 1): train the compensated actor under
randomized dynamics, deploy it on the real robot, let the instrument identify the real world,
then hardcode the identified parameters back into the simulator. Here a hidden world W* — every
parameter drawn once, at random, from the training ranges — plays the real robot. Phases:

CALIBRATE   Fit the linear readout in the randomized simulator (known draws), as the hardware
            plan fits it before deployment.
DEPLOY      The composite walks flat and hill-7 courses in W* ("once it can walk with the
            adapter"); its walk record is the deployment gate.
IDENTIFY    Episode latents from walking in W* go through the readout: per-parameter estimates
            with spread. The hidden truth is only read afterwards, to score the estimate
            against guessing the range center.
HARDCODE    Parameters whose held-out instrument quality cleared --trust-r2 are written back
            into the simulator as fixed values; the rest stay canonical — the plan's "freeze
            the slow ones you can measure, keep re-estimating the rest".
VALIDATE    The frozen core walks the same courses in the nominal sim, in W*, and in the
            calibrated sim: calibration succeeds where the calibrated sim predicts W*'s
            behavior better than the nominal sim does. The composite's W* walks are the
            deployment validation.

Judges reused, none changed: terrain_eval's scene machinery and crossed/walked/falls metrics.

Usage (from the hexapod-cupi checkout):
  uv run python sim2real_rehearsal.py --checkpoint composite.pt --trust-json readout.json
      --output DIR [--world-seed 777003] [--episodes 192] [--rows 16] [--trust-r2 0.15]
"""

import argparse
import json
import math
import sys
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

import locomotion.omniwalk.task as task_module
import locomotion.surrogate.env as surrogate_env
from locomotion.rl import ppo as algorithms
from locomotion.rl.sim2real import TARGET_FIELDS
from locomotion.surrogate.env import (
    ContactModel,
    DynamicsRandomization,
    SurrogateConfig,
    SurrogateEnv,
    sha,
)
from locomotion.surrogate.scene import run_scene
from locomotion.surrogate.train import build_wrapper, resolve, wrapper_options

CORE_CHECKPOINT = (
    "/Users/new/Desktop/cupi/hexapod-surrogate/runs/native_eval/checkpoints/"
    "terrain_isaac_u600_002/checkpoint_update000600.pt"
)
TRAINING_RANGES = dict(
    friction_scale=(0.6, 1.4),
    mass_scale=(0.9, 1.1),
    motor_scale=(0.9, 1.1),
    kd_scale=(0.8, 1.2),
    action_delay_controls=(0, 1),
    sag_per_minute=(0.0, 0.1),
    sag_floor=0.85,
)
STARTS = [(-0.1, -0.06), (-0.05, 0.04), (0.0, 0.0), (0.05, -0.03), (0.1, 0.06), (0.15, -0.08)]
SPEED, WALK_SECONDS, CROSSED_M = 0.10, 40, 3.6
SCENE_SEED = 20261103


def hill(grade):
    rise = round(1.2 * math.tan(math.radians(grade)), 4)
    return [
        {"type": "ramp", "center": [1.1, 0], "size": [1.2, 1.6], "rise": rise},
        {"type": "box", "center": [2.2, 0], "size": [1.0, 1.6], "height": rise},
        {"type": "ramp", "center": [3.3, 0], "size": [1.2, 1.6], "rise": -rise},
        {"type": "box", "center": [3.3, 0], "size": [1.2, 1.6], "height": rise},
    ]


COURSES = {
    "flat 5mm": [{"type": "noise", "amplitude": 0.005, "wavelength": 0.3, "seed": 5}],
    "hill 7 deg": hill(7),
}

OriginalEnv = surrogate_env.SurrogateEnv
ACTIVE = {"randomization": None}


class WorldEnv(OriginalEnv):
    """The stock surrogate inside the currently active world, touchdown model off."""

    def __init__(self, *args, **kwargs):
        contact = kwargs.get("contact") or surrogate_env.ContactModel()
        kwargs["contact"] = replace(contact, touchdown_overshoot=False)
        kwargs["randomization"] = ACTIVE["randomization"]
        super().__init__(*args, **kwargs)


surrogate_env.SurrogateEnv = WorldEnv


def draw_world(seed):
    """W*: one hidden value per parameter, uniform over the training ranges."""
    rng = np.random.default_rng(seed)
    base = DynamicsRandomization(**TRAINING_RANGES)
    values = {}
    for name in TARGET_FIELDS:
        low, high = base.field_range(name)
        if name == "action_delay_controls":
            values[name] = int(rng.integers(int(low), int(high) + 1))
        else:
            values[name] = float(rng.uniform(low, high))
    world = DynamicsRandomization(
        friction_scale=(values["friction_scale"],) * 2,
        mass_scale=(values["mass_scale"],) * 2,
        motor_scale=(values["motor_scale"],) * 2,
        kd_scale=(values["kd_scale"],) * 2,
        action_delay_controls=(values["action_delay_controls"],) * 2,
        sag_per_minute=(values["sag_per_minute"],) * 2,
        sag_floor=TRAINING_RANGES["sag_floor"],
    )
    return world, values


def load_policy(checkpoint_path):
    record = json.loads(checkpoint_path.with_suffix(".json").read_text())
    if sha(checkpoint_path) != record["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes differ from the paired record")
    identity = record["identity"]
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    return identity, checkpoint


def slope_terrain(grade_deg):
    """A full-field ramp: walking on it spends the torque budget the flat floor never asks for."""
    from locomotion.surrogate.terrain import Terrain

    rise = round(12 * math.tan(math.radians(grade_deg)), 4)
    return Terrain(
        {
            "size_m": [12.0, 12.0],
            "center_m": [0.0, 0.0],
            "resolution_m": 0.04,
            "features": [
                {"type": "ramp", "center": [0.0, 0.0], "size": [12.0, 12.0], "rise": rise}
            ],
        }
    )


def collect_latents(
    identity, checkpoint, randomization, episodes, rows, seed, settle=50, terrain=None
):
    """Per-episode mean latents with their episodes' draws, walking under ``randomization``."""
    torch.manual_seed(seed)
    np.random.seed(seed)
    env = OriginalEnv(
        SurrogateConfig(num_envs=rows, seed=seed, episode_seconds=16.0),
        threads=2,
        contact=replace(ContactModel(), touchdown_overshoot=False),
        randomization=randomization,
        terrain=terrain,
        spawn=None if terrain is None else terrain.spawner(3.5, seed),
    )
    try:
        config = identity["ppo_config"]
        task = resolve(identity["task"])(env, task_module.TaskConfig(seed=seed), None)
        wrapped = build_wrapper(resolve(identity["wrapper"]), task, wrapper_options(config))
        actor, _ = algorithms.load_actor(checkpoint, config, wrapped.get_observations(), "cpu")
        latent_width = actor.identification[-1].out_features
        sums = torch.zeros(rows, latent_width)
        counts = torch.zeros(rows)
        draws = {name: np.array(env.randomization_draws[name]) for name in TARGET_FIELDS}
        xs, ys = [], []

        def finalize(selected):
            for row in selected.tolist():
                if counts[row] >= 1:
                    xs.append((sums[row] / counts[row]).numpy().copy())
                    ys.append([draws[name][row] for name in TARGET_FIELDS])
                sums[row] = 0.0
                counts[row] = 0.0
                for name in TARGET_FIELDS:
                    draws[name][row] = env.randomization_draws[name][row]

        observation = wrapped.get_observations()
        with torch.inference_mode():
            while len(xs) < episodes:
                latent = actor.dynamics_latent(observation["policy"])
                settled = env.episode_steps >= settle
                sums[settled] += latent[settled]
                counts[settled] += 1
                observation, _, done, _ = wrapped.step(actor(observation))
                selected = done.nonzero(as_tuple=False).flatten()
                if len(selected):
                    finalize(selected)
        return np.asarray(xs)[:episodes], np.asarray(ys)[:episodes]
    finally:
        env.close()


def walk_courses(checkpoint_path, world, output, tag):
    """terrain_eval-style walks (6 starts per course) for one checkpoint in one world."""
    ACTIVE["randomization"] = world
    table = {}
    for course, features in COURSES.items():
        robots = [
            {
                "name": f"{tag} {j}",
                "checkpoint": str(checkpoint_path.resolve()),
                "start": {"forward_m": f, "left_m": left, "heading_deg": 0},
                "commands": [
                    {"seconds": 2, "command": [0, 0, 0]},
                    {"seconds": WALK_SECONDS - 2, "command": [SPEED, 0, 0]},
                ],
            }
            for j, (f, left) in enumerate(STARTS)
        ]
        scene = {
            "schema": "hexapod_scene_v1",
            "name": f"{course} {tag}",
            "seconds": WALK_SECONDS,
            "terrain": {
                "size_m": [6.2, 2.4],
                "center_m": [2.3, 0],
                "resolution_m": 0.02,
                "features": features,
            },
            "robots": robots,
        }
        folder = output / f"{tag}_{course.replace(' ', '_').replace('.', 'p')}"
        path = output / f"{folder.name}.json"
        path.write_text(json.dumps(scene, indent=1))
        summary = run_scene(path, folder, seed=SCENE_SEED)
        rows = []
        for row in summary["robots"]:
            trace = np.load(folder / row["trace"])
            forward = -trace["root_pose_xyzw"][:, 1]
            walked = float(forward[-1] - forward[0])
            rows.append(
                {
                    "crossed": bool(walked >= CROSSED_M),
                    "walked_m": round(walked, 2),
                    "fell_at_s": row["fell_at_s"],
                }
            )
        table[course] = rows
        print(
            f"walk [{course}] {tag}: crossed {sum(r['crossed'] for r in rows)}/6, "
            f"falls {sum(r['fell_at_s'] is not None for r in rows)}, "
            f"walked {[r['walked_m'] for r in rows]}",
            flush=True,
        )
    return table


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--trust-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--world-seed", type=int, default=777003)
    parser.add_argument("--episodes", type=int, default=192)
    parser.add_argument("--identify-episodes", type=int, default=64)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--trust-r2", type=float, default=0.15)
    parser.add_argument(
        "--identify-slope",
        type=float,
        default=0.0,
        help="Calibrate and identify on a ramp of this grade (degrees) instead of the flat "
        "floor: rehearsal 001 showed walking data never binds the torque ceiling, so motor "
        "and sag stay invisible on the flat.",
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    identity, checkpoint = load_policy(args.checkpoint)
    trust_report = json.loads(args.trust_json.read_text())["readout"]
    trusted = sorted(
        name
        for name, row in trust_report.items()
        if row["r2"] is not None and row["r2"] >= args.trust_r2
    )
    print("trusted (held-out R2 >=", args.trust_r2, "):", trusted or "none", flush=True)

    print("\n== CALIBRATE: fit the readout in the randomized simulator ==", flush=True)
    terrain = slope_terrain(args.identify_slope) if args.identify_slope else None
    if terrain is not None:
        print(f"identification ground: {args.identify_slope:g}-degree ramp", flush=True)
    ranges = DynamicsRandomization(**TRAINING_RANGES)
    x, y = collect_latents(
        identity, checkpoint, ranges, args.episodes, args.rows, seed=20261104, terrain=terrain
    )
    design = np.concatenate([x, np.ones((len(x), 1))], 1)
    coefficients, *_ = np.linalg.lstsq(design, y, rcond=None)
    fit = design @ coefficients
    train_r2 = 1 - (y - fit).var(0) / y.var(0)
    for i, name in enumerate(TARGET_FIELDS):
        print(f"  calibration fit {name:>22}: train R2 {train_r2[i]:.3f}", flush=True)

    print("\n== DEPLOY: the composite walks in the hidden world ==", flush=True)
    world, truth = draw_world(args.world_seed)
    composite_walks = walk_courses(args.checkpoint, world, args.output, "composite_Wstar")
    flat_crossed = sum(r["crossed"] for r in composite_walks["flat 5mm"])
    falls = sum(
        r["fell_at_s"] is not None for rows in composite_walks.values() for r in rows
    )
    deployed = flat_crossed >= 5 and falls == 0
    print(f"deployment gate (flat >= 5/6 crossed, no falls): {'PASS' if deployed else 'FAIL'}")

    print("\n== IDENTIFY: the instrument reads the hidden world ==", flush=True)
    zx, _ = collect_latents(
        identity,
        checkpoint,
        world,
        args.identify_episodes,
        args.rows,
        seed=20261105,
        terrain=terrain,
    )
    estimates = np.concatenate([zx, np.ones((len(zx), 1))], 1) @ coefficients
    estimate = {name: float(estimates[:, i].mean()) for i, name in enumerate(TARGET_FIELDS)}
    spread = {name: float(estimates[:, i].std()) for i, name in enumerate(TARGET_FIELDS)}
    center = {name: sum(ranges.field_range(name)) / 2 for name in TARGET_FIELDS}
    print(f"{'parameter':>22}  {'estimate':>9}  {'spread':>7}  {'truth':>7}  {'center':>7}")
    scores = {}
    for name in TARGET_FIELDS:
        error = abs(estimate[name] - truth[name])
        prior = abs(center[name] - truth[name])
        scores[name] = {
            "estimate": estimate[name],
            "spread": spread[name],
            "truth": truth[name],
            "error": error,
            "center_error": prior,
            "beats_center": bool(error < prior),
        }
        print(
            f"{name:>22}  {estimate[name]:9.4f}  {spread[name]:7.4f}  "
            f"{truth[name]:7.4f}  {center[name]:7.4f}"
            + ("   << beats the center guess" if error < prior else "")
        )

    print("\n== HARDCODE: identified parameters write back into the simulator ==", flush=True)
    calibrated_values = {}
    for name in TARGET_FIELDS:
        low, high = ranges.field_range(name)
        if name in trusted:
            calibrated_values[name] = float(np.clip(estimate[name], low, high))
        else:
            calibrated_values[name] = None
    calibrated = DynamicsRandomization(
        friction_scale=((calibrated_values["friction_scale"] or 1.0),) * 2,
        mass_scale=((calibrated_values["mass_scale"] or 1.0),) * 2,
        motor_scale=((calibrated_values["motor_scale"] or 1.0),) * 2,
        kd_scale=((calibrated_values["kd_scale"] or 1.0),) * 2,
        action_delay_controls=(int(round(calibrated_values["action_delay_controls"] or 0)),) * 2,
        sag_per_minute=((calibrated_values["sag_per_minute"] or 0.0),) * 2,
        sag_floor=TRAINING_RANGES["sag_floor"],
    )
    print("calibrated simulator:", {k: v for k, v in calibrated_values.items() if v is not None})

    print("\n== VALIDATE: does the calibrated simulator predict the hidden world? ==", flush=True)
    core = Path(CORE_CHECKPOINT)
    walks = {
        "core_Wstar": walk_courses(core, world, args.output, "core_Wstar"),
        "core_nominal": walk_courses(core, None, args.output, "core_nominal"),
        "core_calibrated": walk_courses(core, calibrated, args.output, "core_calibrated"),
    }
    validation = {}
    for course in COURSES:
        real = np.array([r["walked_m"] for r in walks["core_Wstar"][course]])
        nominal = np.array([r["walked_m"] for r in walks["core_nominal"][course]])
        cal = np.array([r["walked_m"] for r in walks["core_calibrated"][course]])
        validation[course] = {
            "nominal_gap_m": float(np.abs(nominal - real).mean()),
            "calibrated_gap_m": float(np.abs(cal - real).mean()),
            "closer": bool(np.abs(cal - real).mean() <= np.abs(nominal - real).mean()),
        }
        print(
            f"  [{course}] |sim - W*| walked gap: nominal {validation[course]['nominal_gap_m']:.3f} m"
            f" -> calibrated {validation[course]['calibrated_gap_m']:.3f} m"
            f" ({'closer' if validation[course]['closer'] else 'NOT closer'})"
        )

    record = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha(args.checkpoint),
        "core_checkpoint": CORE_CHECKPOINT,
        "world_seed": args.world_seed,
        "training_ranges": TRAINING_RANGES,
        "trusted": trusted,
        "trust_r2": args.trust_r2,
        "calibration_train_r2": {n: float(train_r2[i]) for i, n in enumerate(TARGET_FIELDS)},
        "truth": truth,
        "identification": scores,
        "calibrated": asdict(calibrated),
        "deployment": {"walks": composite_walks, "pass": deployed},
        "validation_walks": walks,
        "validation": validation,
    }
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    print("\nrecorded:", args.output / "results.json")


if __name__ == "__main__":
    main(sys.argv[1:])
