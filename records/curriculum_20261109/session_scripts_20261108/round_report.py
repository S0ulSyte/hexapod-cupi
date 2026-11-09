"""Compact per-round judging report: each tag vs a control, with margins.

Usage: uv run python round_report.py CONTROL_TAG TAG [TAG ...]
Reads full96_<tag>/surrogate_summary.json and omni15_<tag>/surrogate_summary.json
under the curriculum folder, so one short printout replaces the ad-hoc analysis.
"""

import json
import sys
from pathlib import Path

C = Path("/Users/new/Desktop/cupi/hexapod-surrogate/runs/curriculum")
EXCL = ("gentle_arc", "tight_arc", "combined", "transition", "translate_0.20",
        "formal__0.20", "formal__0.30", "forward_fast")
GOAL_SKIP = ("tight_arc", "translate_0.20", "formal__0.20", "formal__0.30", "forward_fast")


def load(tag):
    s = json.loads((C / f"full96_{tag}" / "surrogate_summary.json").read_text())
    return {c["case_id"]: c for c in s["cases"]}


def margin(c):
    m = c["metrics"]
    parts = []
    for b in c.get("failed_bounds", []):
        v = m.get(b)
        parts.append(f"{b.split('_')[0]} {v:.4f}" if isinstance(v, float) else b.replace("_", " ")[:30])
    return ", ".join(parts)


def short(k):
    return k.split("__", 1)[1]


def subset(d):
    cs = [c for c in d.values() if not any(e in c["case_id"] for e in EXCL)]
    return sum(c["pass"] for c in cs)


def omni(tag):
    p = C / f"omni15_{tag}" / "surrogate_summary.json"
    if not p.exists():
        return "-", []
    cases = [c for c in json.loads(p.read_text())["cases"] if "omni15" in c["case_id"]]
    miss = [f"{c['case_id'].split('omni15_')[-1]} [{margin(c)}]" for c in cases if not c["pass"]]
    return f"{sum(c['pass'] for c in cases)}/{len(cases)}", miss


def main():
    control, *tags = sys.argv[1:]
    ctrl = load(control)
    print(f"control {control}: total {sum(c['pass'] for c in ctrl.values())}/96, subset {subset(ctrl)}/36")
    for tag in tags:
        d = load(tag)
        o, omiss = omni(tag)
        print(f"\n=== {tag}: total {sum(c['pass'] for c in d.values())}/96, subset {subset(d)}/36, omni {o}")
        for k in sorted(d):
            if d[k]["pass"] and not ctrl[k]["pass"]:
                print(f"  + {short(k)}")
        for k in sorted(d):
            if not d[k]["pass"] and ctrl[k]["pass"]:
                print(f"  - {short(k)} [{margin(d[k])}]")
        print("  goal-scope misses:")
        for k in sorted(d):
            c = d[k]
            if c["pass"] or any(e in k for e in GOAL_SKIP):
                continue
            print(f"    {short(k)} [{margin(c)}]")
        for line in omiss:
            print(f"  omni miss: {line}")


if __name__ == "__main__":
    main()
