"""Build results.json in this folder from a copy of the destination_v9 96-case evaluation attempt.

Each case copies its pass/fail, failed bounds, recorded controls and mean planar and yaw velocity from the
evaluator's summary.json, and its command from allocation.json. A case counts as inside the training range
when its command needs no planar speed above 0.05 m/s and no yaw rate above 0.2 rad/s, the fastest the task
at the v9 training commit 976344cb draws (walks at 0.025 and 0.05 m/s, turns at 0.2 rad/s, arcs at 0.04 m/s
and 0.15 rad/s); the transition program counts as outside. Groups follow the case-id prefixes.

Usage: python3 build_results.py <attempt folder holding binding.json and run/standing/evaluation>
"""

import hashlib
import json
import math
import sys
from pathlib import Path

HERE = Path(__file__).parent
ATTEMPT = Path(sys.argv[1])
EVALUATION = ATTEMPT / "run/standing/evaluation"
TRAINED_SPEED_MPS, TRAINED_YAW_RAD_S = 0.05, 0.2
GROUPS = [
    ("Stand, quiet and zero-speed cases", lambda c: c["profile"] != "transition" and not any(c["command"] or [1])),
    ("Turn in place, and stop after one", lambda c: c["case_id"].startswith(("static:turn", "stop:turn"))),
    ("Walk 0.10 m/s, 16 directions", lambda c: c["case_id"].startswith("static:translate_0.10")),
    ("Walk 0.20 m/s, 16 directions", lambda c: c["case_id"].startswith("static:translate_0.20")),
    ("Combined walk and turn", lambda c: c["case_id"].startswith("static:combined")),
    ("Gentle arcs, 0.15 m/s", lambda c: c["case_id"].startswith("static:gentle_arc")),
    ("Tight arcs, 0.2 m/s and 0.4 rad/s", lambda c: c["case_id"].startswith("static:tight_arc")),
    ("Stops after a moving command", lambda c: c["case_id"].startswith("stop:")),
    ("Formal forward cases and the transition program", lambda c: c["case_id"].startswith(("formal:", "transition:"))),
]


def sha(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def inside(case):
    command = case.get("command")
    # The transition program changes its command through the episode; its first command is zero.
    if command is None or case["profile"] == "transition":
        return False
    return math.hypot(command[0], command[1]) <= TRAINED_SPEED_MPS + 1e-9 and abs(command[2]) <= TRAINED_YAW_RAD_S + 1e-9


summary = json.loads((EVALUATION / "summary.json").read_text())
allocation = json.loads((EVALUATION / "allocation.json").read_text())
commands = {c["case_id"]: c for c in allocation["full_required_cases"]}
cases = []
for result in summary["results"]:
    case = commands[result["case_id"]]
    metrics = result.get("metrics") or {}
    velocity = metrics.get("mean_velocity_mps")
    cases.append(
        {
            "case_id": result["case_id"],
            "profile": case["profile"],
            "command": case.get("command"),
            "inside_training_range": inside(case),
            "pass": bool(result["pass"]),
            "failed_bounds": result.get("failed_bounds") or [],
            "recorded_controls": result.get("recorded_controls"),
            "requested_controls": case["controls"],
            "mean_velocity_mps": None if not velocity else [round(v, 4) for v in velocity[:2]],
            "mean_yaw_rate_rad_s": None if metrics.get("mean_yaw_rate_rad_s") is None else round(metrics["mean_yaw_rate_rad_s"], 4),
        }
    )
taken, groups = set(), []
for name, member in GROUPS:
    rows = [c for c in cases if c["case_id"] not in taken and member(c)]
    taken.update(c["case_id"] for c in rows)
    groups.append({"group": name, "passed": sum(c["pass"] for c in rows), "cases": len(rows)})
assert len(taken) == len(cases) == len(summary["required_cases"]), "Every case belongs to one group"
binding = json.loads((ATTEMPT / "binding.json").read_text())
within = [c for c in cases if c["inside_training_range"]]
results = {
    "schema": "hexapod_destination_full96_results_v1",
    "scope": "The 96-case manifest on the destination_v9 checkpoint: native Isaac on Spark, one robot, the actor mean without sampling noise, one trial per case, thresholds unchanged. The checks and the pass/fail are the evaluator's; this file copies them.",
    "attempt": ATTEMPT.name,
    "spark_path": "/srv/cupi/hexapod/runs/julian/" + ATTEMPT.name,
    "binding_sha256": sha(ATTEMPT / "binding.json"),
    "source_commit": "976344cb3405177954d8a223d88afbedce4417c2",
    "source_freeze_sha256": binding["source_freeze_sha256"],
    "checkpoint_sha256": allocation["checkpoint_sha256"],
    "summary_sha256": sha(EVALUATION / "summary.json"),
    "wall_hours": round(summary["wall_seconds"] / 3600, 2),
    "allocation_limit_reached": summary["allocation_limit_reached"],
    "missing_cases": summary["missing_cases"],
    "stage2_complete": summary["stage2_complete"],
    "passed": sum(c["pass"] for c in cases),
    "cases": len(cases),
    "inside_training_range": {"passed": sum(c["pass"] for c in within), "cases": len(within)},
    "groups": groups,
    "earlier_attempt": "eval_full96_dest_v9_u600_seed20260917_20261023_001 ran the same pack contents on a GPU shared with other Isaac jobs at about 1.1 s per control; it was stopped after its first case so the run could fit its 6 h allocation, and its folder stays on Spark.",
    "case_results": cases,
}
(HERE / "results.json").write_text(json.dumps(results, indent=1) + "\n")
print(results["passed"], "/", results["cases"], "inside training range", results["inside_training_range"])
for g in groups:
    print(f"{g['group']:52s} {g['passed']}/{g['cases']}")
