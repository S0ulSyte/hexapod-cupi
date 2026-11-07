"""Build results.json in this folder from the surrogate evaluation folders and the native v9 96-case run.

The surrogate speed suite is the 13 learning probes plus walks (omni_static, 1000 controls) and walk-then-stop
cases (stop_to_stand, 1050 controls) at 0.05 to 0.175 m/s in 8 directions; the 96 required cases are
locomotion.evaluation.runner.case_manifest(), run as surrogate learning probes with the native 90 s episode
timeout. Pass/fail and failed checks copy each case's report.json. The end gap is the planar distance to the
ghost after the last control (viewer/run_view.py: the ghost starts at the reset pose, follows the command and
restarts at the body when the command changes, as training builds it).

Every consumed report must declare the label's intended checkpoint, the scope's seed and one evaluation
source shared by the whole build, and every capture read must match its report-declared hash; a mismatch
rejects the build before anything is written, and the verified bindings are published with the reductions.

Usage: uv run python build_results.py <runs/curriculum folder> <native v9 96-case run/standing folder>
"""

import hashlib
import io
import json
import statistics
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).parent
sys.path.insert(0, str(HERE.parents[2]))
from viewer.run_view import ghost  # noqa: E402

# The one seed the published scope declares, and each label's intended checkpoint, from the
# retained checkpoint declarations (carry/runs/native_eval/checkpoints).
SCOPE_SEED = 20260917
INTENDED_CHECKPOINTS = {
    "destination_v9": "16048b8f7b31c61ba6630742475f796e24c4bc43bd17bc61f9b40bb0d37b6d33",
    "destination_v25_update1350": "b0a2ccf3d82ccaa1c38f66b36340ceec7b6ced969af281aca9f99263c9c3d529",
    "destination_v29_update300": "3983a56956843fa435a157764c8119f6e469954cd462c6cb0f57e9d9d3f97b94",
    "destination_v29_update600": "ed73eb07c1f9c364e33fd6673c53d2a1c4a54d4086bda1e05ca72b680f36ccf8",
}
SPEED_SUITES = [
    ("destination_v9", "suite_v9"),
    ("destination_v25_update1350", "suite_spark_v25_u1350"),
    ("destination_v29_update300", "suite_spark_v29_u300"),
    ("destination_v29_update600", "suite_spark_v29_u600"),
]
REQUIRED = ("destination_v29_update600", ["full96_spark_v29_u600", "full96_spark_v29_u600_transition"])


def cases(folder, checkpoint, sources):
    """Each batch's case, result, batch folder and report, rejected unless the report declares the
    intended checkpoint, the scope seed and the build's one evaluation source."""
    for batch in sorted(Path(folder, "evaluation").glob("batch_*")):
        report = json.loads((batch / "report.json").read_text())
        if report.get("seed") != SCOPE_SEED:
            raise ValueError(f"{batch}: seed {report.get('seed')} is not the scope's {SCOPE_SEED}")
        if report.get("checkpoint_sha256") != checkpoint:
            raise ValueError(
                f"{batch}: checkpoint {report.get('checkpoint_sha256')} is not the intended {checkpoint}"
            )
        sources.add(report.get("source_sha256"))
        if len(sources) != 1 or None in sources:
            raise ValueError(f"{batch}: evaluation source differs across the published reports")
        yield report["assigned_case_ids"][0].split(":", 1)[1], report["results"][0], batch, report


def verified_trace(batch, report):
    """The batch's capture, rejected unless its bytes match the report-declared hash."""
    declared = (report.get("files") or {}).get("control_trace.npz")
    data = (batch / "control_trace.npz").read_bytes()
    if declared is None or hashlib.sha256(data).hexdigest() != declared:
        raise ValueError(f"{batch}: control_trace.npz does not match its report-declared hash")
    return np.load(io.BytesIO(data))


def speed_suite(folder, checkpoint, sources):
    rows, digest = {}, hashlib.sha256()
    for case, result, batch, report in cases(folder, checkpoint, sources):
        digest.update((batch / "report.json").read_bytes())
        failed = result.get("failed_bounds") or []
        trace = verified_trace(batch, report)
        pose = trace["root_pose_xyzw"][:, 0].astype(float)
        command = trace["command"][:, 0].astype(float)
        end = float(np.hypot(*(pose[-1, :2] - ghost(pose, command)[-1, :2])) * 1000)
        rows[case] = {
            "pass": bool(result["pass"]),
            "fell": "terminated" in failed or bool(trace["terminated"][:, 0].any()),
            "end_gap_mm": round(end, 2),
            "failed_bounds": failed,
        }
    groups = {"probes": [v for k, v in rows.items() if not k.startswith(("walk_", "stop_"))]}
    for k, v in rows.items():
        if k.startswith(("walk_", "stop_")):
            kind, speed, _ = k.split("_")
            groups.setdefault(f"{kind} {speed}", []).append(v)
    table = {
        name: {
            "cases": len(v),
            "passed": sum(x["pass"] for x in v),
            "fell": sum(x["fell"] for x in v),
            "median_end_gap_mm": round(statistics.median(x["end_gap_mm"] for x in v), 2),
            "worst_end_gap_mm": round(max(x["end_gap_mm"] for x in v), 2),
        }
        for name, v in groups.items()
    }
    return {
        "checkpoint_sha256": checkpoint,
        "seed": SCOPE_SEED,
        "reports_sha256": digest.hexdigest(),
        "groups": table,
        "cases": rows,
    }


def group(case):
    kind, name = case.split(":", 1)
    if kind != "static":
        return kind
    return "static " + "_".join(p for p in name.split("_") if not p.endswith("deg"))


def required(rows):
    table = {}
    for case, row in rows.items():
        entry = table.setdefault(group(case), {"cases": 0, "passed": 0})
        entry["cases"] += 1
        entry["passed"] += row["pass"]
    return {"passed": sum(r["pass"] for r in rows.values()), "cases": len(rows), "groups": table, "cases_detail": rows}


def main(runs, native):
    sources = set()
    result = {
        "schema": "hexapod_destination_curriculum_results_v1",
        "scope": "Surrogate (CPU MuJoCo) evaluations: design evidence, not acceptance evidence. The native "
        "evaluator does not yet rebuild the ghost_v11 inputs or the speed-scaled gait clock, so the native "
        "trainings of destination_v25 and destination_v29 were evaluated on the surrogate. One trial per case, "
        "the actor mean without sampling noise, seed 20260917, without the extra contact-model checks.",
        "speed_suite": {
            name: speed_suite(Path(runs, folder), INTENDED_CHECKPOINTS[name], sources)
            for name, folder in SPEED_SUITES
        },
    }
    rows = {}
    for folder in REQUIRED[1]:
        for case, res, _, _ in cases(Path(runs, folder), INTENDED_CHECKPOINTS[REQUIRED[0]], sources):
            rows[case.replace("__", ":", 1)] = {"pass": bool(res["pass"]), "failed_bounds": res.get("failed_bounds") or []}
    result["evaluation_source_sha256"] = sources.pop()
    state = json.loads(Path(native, "state.json").read_text())
    native_checkpoint = state.get("checkpoint_sha256")
    if native_checkpoint is not None and native_checkpoint != INTENDED_CHECKPOINTS["destination_v9"]:
        raise ValueError(
            f"Native v9 state declares checkpoint {native_checkpoint}, "
            f"not the intended {INTENDED_CHECKPOINTS['destination_v9']}"
        )
    v9 = {r["case_id"]: {"pass": bool(r.get("pass")), "failed_bounds": r.get("failed_bounds") or []} for r in state["evaluation"]["results"]}
    if set(rows) != set(v9) or len(v9) != 96:
        raise ValueError("The surrogate run and the native v9 run must cover the same 96 required cases")
    result["required_96"] = {
        REQUIRED[0] + "_surrogate": {
            "checkpoint_sha256": INTENDED_CHECKPOINTS[REQUIRED[0]],
            "seed": SCOPE_SEED,
            **required(rows),
        },
        "destination_v9_native": {
            "checkpoint_sha256": native_checkpoint,
            **required(v9),
        },
    }
    (HERE / "results.json").write_text(json.dumps(result, indent=1) + "\n")


if __name__ == "__main__":
    main(*sys.argv[1:3])
