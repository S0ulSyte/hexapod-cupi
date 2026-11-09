"""Fit the linear readout from the compensator's latent to the drawn dynamics.

The compensated actor was trained only by the task reward; this suite measures whether its
latent nevertheless identifies the world it is in. It runs the composite deterministically in
the randomized surrogate (flat floor, the campaign's training ranges, touchdown model off),
collects one latent mean per episode after a settling period, pairs it with the environment's
recorded draws for that episode, fits ordinary least squares on a training split, and reports
held-out R-squared and mean absolute error per drawn parameter. Nothing here feeds training:
it is the measurement half of the design — identification quality, judged out of sample.

Usage (from the hexapod-cupi checkout):
  uv run python sim2real_readout.py --checkpoint composite.pt --output DIR
                                    [--episodes 128] [--rows 16] [--settle 50]
"""

import argparse
import json
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np
import torch

import locomotion.omniwalk.task as task_module
from locomotion.rl import ppo as algorithms
from locomotion.surrogate.env import (
    ContactModel,
    DynamicsRandomization,
    SurrogateConfig,
    SurrogateEnv,
    sha,
)
from locomotion.surrogate.train import build_wrapper, resolve, wrapper_options

TRAINING_RANGES = dict(
    friction_scale=(0.6, 1.4),
    mass_scale=(0.9, 1.1),
    motor_scale=(0.9, 1.1),
    kd_scale=(0.8, 1.2),
    action_delay_controls=1,
    sag_per_minute=(0.0, 0.1),
    sag_floor=0.85,
)
TARGETS = (
    "friction_scale",
    "mass_scale",
    "motor_scale",
    "kd_scale",
    "action_delay_controls",
    "sag_per_minute",
)


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--episodes", type=int, default=128)
    parser.add_argument("--rows", type=int, default=16)
    parser.add_argument("--settle", type=int, default=50)
    parser.add_argument("--seed", type=int, default=20261103)
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)

    record = json.loads(args.checkpoint.with_suffix(".json").read_text())
    if sha(args.checkpoint) != record["checkpoint_sha256"]:
        raise ValueError("Checkpoint bytes differ from the paired record")
    identity = record["identity"]
    config = identity["ppo_config"]
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = SurrogateEnv(
        SurrogateConfig(num_envs=args.rows, seed=args.seed, episode_seconds=16.0),
        threads=2,
        contact=replace(ContactModel(), touchdown_overshoot=False),
        randomization=DynamicsRandomization(**TRAINING_RANGES),
    )
    task = resolve(identity["task"])(env, task_module.TaskConfig(seed=args.seed), None)
    wrapped = build_wrapper(resolve(identity["wrapper"]), task, wrapper_options(config))
    actor, _ = algorithms.load_actor(checkpoint, config, wrapped.get_observations(), "cpu")

    sums = torch.zeros(args.rows, actor.mlp.encoder[-1].out_features)
    counts = torch.zeros(args.rows)
    draws = {name: np.array(env.randomization_draws[name]) for name in TARGETS}
    samples_x, samples_y = [], []

    def finalize(rows):
        for row in rows.tolist():
            if counts[row] >= 1:
                samples_x.append((sums[row] / counts[row]).numpy().copy())
                samples_y.append([draws[name][row] for name in TARGETS])
            sums[row] = 0.0
            counts[row] = 0.0
            for name in TARGETS:
                draws[name][row] = env.randomization_draws[name][row]

    observation = wrapped.get_observations()
    with torch.inference_mode():
        while len(samples_x) < args.episodes:
            latent = actor.dynamics_latent(observation["policy"])
            settled = env.episode_steps >= args.settle
            sums[settled] += latent[settled]
            counts[settled] += 1
            action = actor(observation)
            observation, _, done, _ = wrapped.step(action)
            rows = done.nonzero(as_tuple=False).flatten()
            if len(rows):
                finalize(rows)

    x = np.asarray(samples_x)[: args.episodes]
    y = np.asarray(samples_y)[: args.episodes]
    split = int(0.75 * len(x))
    design = np.concatenate([x, np.ones((len(x), 1))], axis=1)
    coefficients, *_ = np.linalg.lstsq(design[:split], y[:split], rcond=None)
    predicted = design[split:] @ coefficients
    held = y[split:]
    report = {}
    for i, name in enumerate(TARGETS):
        residual = held[:, i] - predicted[:, i]
        variance = held[:, i].var()
        report[name] = {
            "r2": float(1.0 - residual.var() / variance) if variance > 0 else None,
            "mae": float(np.abs(residual).mean()),
            "target_std": float(held[:, i].std()),
        }
    result = {
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": record["checkpoint_sha256"],
        "ranges": TRAINING_RANGES,
        "episodes": len(x),
        "held_out": len(held),
        "settle_controls": args.settle,
        "seed": args.seed,
        "readout": report,
    }
    (args.output / "readout.json").write_text(json.dumps(result, indent=2) + "\n")
    print(f"episodes {len(x)} (held out {len(held)}), settle {args.settle} controls")
    for name, row in report.items():
        print(
            f"{name:>22}: R2 {row['r2'] if row['r2'] is None else round(row['r2'], 3)}"
            f"  MAE {row['mae']:.4f}  (target std {row['target_std']:.4f})"
        )


if __name__ == "__main__":
    main(sys.argv[1:])
