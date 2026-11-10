"""Score checkpoints under matched dynamics draws, without weakening any judge.

Two halves, split by what each judge certifies. The SCORED half runs eight injected
translate-0.10 bearing cases (1000 controls, the evaluator's own omni_static bounds and native
screens) under draws of friction, mass and action delay only: the 400 Hz capture certifies the
canonical motor law, so motor-side draws would rightly fail its recurrence check and are kept
out. The WALKED half runs the terrain_eval scene machinery (which records traces and falls,
with no canonical-motor capture) on a flat course with 5 mm roughness under draws of ALL axes,
motor strength, PD damping and in-episode sag included; its judges are terrain_eval's own:
crossed = walked >= 3.6 m of the 3.8 m commanded, plus falls. Matching: a draw is a seed, and
the environment's per-row (seed, row, episode) streams give every checkpoint the same drawn
world for the same case or start. The repository's manifest, scorer, bounds and scene code are
untouched; this file injects cases and wraps ``SurrogateEnv`` construction.

Usage (from the hexapod-cupi checkout):
  uv run python sim2real_robust_eval.py --output DIR name=checkpoint.pt [name=ckpt ...]
                                        [--draws 6] [--skip-scored] [--skip-walked]
"""

import argparse
import json
import math
import sys
from dataclasses import replace
from pathlib import Path

import numpy as np

import locomotion.evaluation.runner as runner
import locomotion.surrogate.env as surrogate_env
from locomotion.surrogate import evaluate
from locomotion.surrogate.scene import run_scene

SPEED = 0.10
BEARINGS = 8
BASE_SEED = 20261103
STARTS = [(-0.1, -0.06), (-0.05, 0.04), (0.0, 0.0), (0.05, -0.03), (0.1, 0.06), (0.15, -0.08)]
WALK_SECONDS, CROSSED_M = 40, 3.6
# The campaign's training ranges (overrides_s2r_u600_002.json / the s2r legs' --randomize flags).
TRAINING_RANGES = dict(
    friction_scale=(0.6, 1.4),
    mass_scale=(0.9, 1.1),
    motor_scale=(0.9, 1.1),
    kd_scale=(0.8, 1.2),
    action_delay_controls=1,
    sag_per_minute=(0.0, 0.1),
    sag_floor=0.85,
)
# The scored half keeps the motor law canonical (the capture certifies it bit for bit).
SCORED_RANGES = {
    **TRAINING_RANGES,
    "motor_scale": (1.0, 1.0),
    "kd_scale": (1.0, 1.0),
    "sag_per_minute": (0.0, 0.0),
}

original_cases = runner.learning_probe_cases


def required():
    cases = []
    for i in range(BEARINGS):
        angle = 2 * math.pi * i / BEARINGS
        cases.append(
            {
                "case_id": f"learning:s2r__translate_{SPEED}_{i * 360 / BEARINGS:g}deg",
                "profile": "omni_static",
                "command": [SPEED * math.cos(angle), SPEED * math.sin(angle), 0.0],
                "controls": 1000,
            }
        )
    return cases


def cases():
    return original_cases() + required()


runner.learning_probe_cases = cases

OriginalEnv = surrogate_env.SurrogateEnv
ACTIVE = {"randomization": None}


class RandomizedEnv(OriginalEnv):
    """The stock surrogate with the suite's randomization and the touchdown model off."""

    def __init__(self, *args, **kwargs):
        contact = kwargs.get("contact") or surrogate_env.ContactModel()
        kwargs["contact"] = replace(contact, touchdown_overshoot=False)
        kwargs["randomization"] = ACTIVE["randomization"]
        super().__init__(*args, **kwargs)


surrogate_env.SurrogateEnv = RandomizedEnv


def scored_half(args, named, results):
    case_ids = [case["case_id"] for case in required()]
    ACTIVE["randomization"] = surrogate_env.DynamicsRandomization(**SCORED_RANGES)
    for k in range(args.draws):
        seed = BASE_SEED + k
        for name, checkpoint in named:
            directory = args.output / f"scored_{name}_draw{k:02d}"
            evaluate.main(
                [
                    "--checkpoint",
                    checkpoint,
                    "--output",
                    str(directory),
                    "--cases",
                    ",".join(case_ids),
                    "--seed",
                    str(seed),
                    "--video",
                    "none",
                    "--no-contact-consensus",
                ]
            )
            summary = json.loads((directory / "surrogate_summary.json").read_text())
            rows = {
                case["case_id"]: {
                    "pass": case["pass"],
                    "planar_error_mps": case["metrics"].get("planar_error_mps")
                    if case.get("metrics")
                    else None,
                    "demand_over_rating": case["metrics"].get(
                        "computed_demand_over_rating_fraction"
                    )
                    if case.get("metrics")
                    else None,
                }
                for case in summary["cases"]
            }
            entry = {"seed": seed, "passes": sum(r["pass"] for r in rows.values()), "cases": rows}
            results["scored"].setdefault(name, {})[f"draw{k:02d}"] = entry
            print(f"scored {name} draw{k:02d}: {entry['passes']}/{len(case_ids)}", flush=True)


def hill(grade):
    rise = round(1.2 * math.tan(math.radians(grade)), 4)
    return [
        {"type": "ramp", "center": [1.1, 0], "size": [1.2, 1.6], "rise": rise},
        {"type": "box", "center": [2.2, 0], "size": [1.0, 1.6], "height": rise},
        {"type": "ramp", "center": [3.3, 0], "size": [1.2, 1.6], "rise": -rise},
        {"type": "box", "center": [3.3, 0], "size": [1.2, 1.6], "height": rise},
    ]


# terrain_eval's hill-7 course joins the flat walk: weakened motors and sag should bite where
# the torque budget is spent, and the matched draws say whether the compensator buys it back.
WALK_COURSES = {
    "flat 5mm": [{"type": "noise", "amplitude": 0.005, "wavelength": 0.3, "seed": 5}],
    "hill 7 deg": hill(7),
}


def walked_half(args, named, results):
    ACTIVE["randomization"] = surrogate_env.DynamicsRandomization(**TRAINING_RANGES)
    for course, features in WALK_COURSES.items():
        walked_course(args, named, results, course, features)


def walked_course(args, named, results, course, features):
    tag = course.replace(" ", "_").replace(".", "p")
    for k in range(args.draws):
        seed = BASE_SEED + k
        for name, checkpoint in named:
            robots = [
                {
                    "name": f"{name} {j}",
                    "checkpoint": str(Path(checkpoint).resolve()),
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
                "name": f"{course} {name} draw{k:02d}",
                "seconds": WALK_SECONDS,
                "terrain": {
                    "size_m": [6.2, 2.4],
                    "center_m": [2.3, 0],
                    "resolution_m": 0.02,
                    "features": features,
                },
                "robots": robots,
            }
            folder = args.output / f"walked_{tag}_{name}_draw{k:02d}"
            path = args.output / f"{folder.name}.json"
            path.write_text(json.dumps(scene, indent=1))
            summary = run_scene(path, folder, seed=seed)
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
            entry = {"seed": seed, "crossed": sum(r["crossed"] for r in rows), "starts": rows}
            results["walked"].setdefault(name, {}).setdefault(course, {})[f"draw{k:02d}"] = entry
            print(
                f"walked [{course}] {name} draw{k:02d}: crossed {entry['crossed']}/6, "
                f"falls {sum(r['fell_at_s'] is not None for r in rows)}, "
                f"walked {[r['walked_m'] for r in rows]}",
                flush=True,
            )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--draws", type=int, default=6)
    parser.add_argument("--skip-scored", action="store_true")
    parser.add_argument("--skip-walked", action="store_true")
    parser.add_argument("checkpoints", nargs="+", metavar="name=checkpoint")
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=False)
    named = [item.split("=", 1) for item in args.checkpoints]
    results = {"scored": {}, "walked": {}}
    if not args.skip_scored:
        scored_half(args, named, results)
    if not args.skip_walked:
        walked_half(args, named, results)
    record = {
        "scored_ranges": SCORED_RANGES,
        "walked_ranges": TRAINING_RANGES,
        "speed": SPEED,
        "bearings": BEARINGS,
        "base_seed": BASE_SEED,
        "results": results,
    }
    (args.output / "results.json").write_text(json.dumps(record, indent=2) + "\n")
    print("\n== matched-draw totals ==")
    for name, draws in results["scored"].items():
        total = sum(d["passes"] for d in draws.values())
        print(f"scored {name:>24}: {total}/{len(draws) * BEARINGS}")
    for name, courses in results["walked"].items():
        for course, draws in courses.items():
            total = sum(d["crossed"] for d in draws.values())
            print(f"walked [{course}] {name:>24}: {total}/{len(draws) * len(STARTS)} crossed")


if __name__ == "__main__":
    main(sys.argv[1:])
