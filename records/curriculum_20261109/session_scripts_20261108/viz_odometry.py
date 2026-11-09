"""Render schedule-inferred vs sensed-contact leg odometry per case as one static page."""

import glob
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
import leg_odometry as L  # noqa: E402

from locomotion.omniwalk.mdp.gait import GaitPhase  # noqa: E402

OUT = Path(__file__).parent / "viz_odometry.html"
TRIPOD_A = np.array([True, False, True, False, True, False])
SWING = 0.4
CASES = [
    "static__translate_0.10_0deg",
    "static__translate_0.10_90deg",
    "static__gentle_arc_0deg_+0.05",
    "static__combined_0deg_+0.20",
    "stop__forward",
    "transition__full_program",
]
BLUE, ORANGE, BLACK, GHOST = "#1668c9", "#e07b00", "#222", "#c62828"


def gait_contact(command):
    gait = GaitPhase(1, "speed_v1", "cpu")
    n = len(command)
    inferred = np.ones((n, 6), dtype=bool)
    for t in range(n):
        cmd = torch.tensor(command[t : t + 1], dtype=torch.float32)
        cycles = gait.follow(cmd, torch.tensor([t]))
        moving = bool((cmd != 0).any())
        fraction = float(cycles[0]) % 1.0
        swing = np.where(TRIPOD_A, fraction < SWING, 0.5 <= fraction < 0.5 + SWING)
        inferred[t] = ~(swing & moving)
    return inferred


def with_contact(trace, contact):
    data = {k: trace[k] for k in trace.files}
    data["distal_contact"] = contact[:, None, :]

    class D:
        files = list(data)

        def __getitem__(self, k):
            return data[k]

    return L.odometry(D())


def path_svg(tracks, size=300):
    xs = np.concatenate([t[:, 0] for t, *_ in tracks])
    ys = np.concatenate([t[:, 1] for t, *_ in tracks])
    span = max(1e-6, float(max(np.ptp(xs), np.ptp(ys)))) * 1.15
    cx, cy = (xs.min() + xs.max()) / 2, (ys.min() + ys.max()) / 2

    def P(p):
        return (size / 2 - (p[1] - cy) / span * size, size / 2 - (p[0] - cx) / span * size)

    parts = [f'<svg viewBox="0 0 {size} {size}" class="path"><rect width="{size}" height="{size}" fill="#fafafa" stroke="#ddd"/>']
    for track, color, width, dash in tracks:
        pts = " ".join(f"{a:.1f},{b:.1f}" for a, b in (P(p) for p in track[:: max(1, len(track) // 500)]))
        parts.append(f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="{width}" stroke-dasharray="{dash}"/>')
    sx, sy = P(tracks[0][0][0])
    parts.append(f'<circle cx="{sx:.1f}" cy="{sy:.1f}" r="3" fill="#222"/>')
    parts.append(f'<text x="6" y="12" font-size="9" fill="#777">top-down, start &#9679; (span {span:.2f} m)</text></svg>')
    return "".join(parts)


def drift_svg(t, sensed, schedule, width=460, height=170):
    L_, R, T, B = 48, 10, 20, 22
    pw, ph = width - L_ - R, height - T - B
    ymax = max(5.0, float(max(sensed.max(), schedule.max())) * 1.1)

    def line(y, color):
        pts = " ".join(
            f"{L_ + t[i] / t[-1] * pw:.1f},{T + (ymax - y[i]) / ymax * ph:.1f}"
            for i in range(0, len(t), max(1, len(t) // 600))
        )
        return f'<polyline points="{pts}" fill="none" stroke="{color}" stroke-width="1.6"/>'

    ticks = "".join(
        f'<line x1="{L_}" x2="{L_ + pw}" y1="{T + (ymax - v) / ymax * ph:.0f}" y2="{T + (ymax - v) / ymax * ph:.0f}" stroke="#eee"/>'
        f'<text x="{L_ - 5}" y="{T + (ymax - v) / ymax * ph + 3:.0f}" font-size="9" fill="#666" text-anchor="end">{v:.0f}</text>'
        for v in np.linspace(0, ymax, 4)
    )
    return (
        f'<svg viewBox="0 0 {width} {height}" style="width:100%;height:auto;background:#fff">'
        f'<text x="{L_}" y="13" font-size="11" fill="#222" font-weight="600">odometry error vs truth (mm) over the run</text>'
        f'<rect x="{L_}" y="{T}" width="{pw}" height="{ph}" fill="none" stroke="#ccc"/>{ticks}'
        f"{line(sensed, BLUE)}{line(schedule, ORANGE)}"
        f'<text x="{L_ + pw - 4}" y="{T + 12}" font-size="10" fill="{BLUE}" text-anchor="end">sensed contact</text>'
        f'<text x="{L_ + pw - 4}" y="{T + 24}" font-size="10" fill="{ORANGE}" text-anchor="end">schedule-inferred</text></svg>'
    )


def main():
    sections = []
    for case in CASES:
        hit = None
        for rp in sorted(glob.glob(f"{L.C}/evaluation/batch_*/report.json")):
            if json.loads(open(rp).read())["assigned_case_ids"][0].endswith(case):
                hit = rp
                break
        if hit is None:
            continue
        trace = np.load(hit.replace("report.json", "control_trace.npz"))
        command = trace["command"][:, 0].astype(float)
        true_contact = trace["distal_contact"][:, 0].astype(bool)
        inferred = gait_contact(command)
        acc = float((inferred == true_contact).mean())
        est_s, pose = with_contact(trace, true_contact)
        est_g, _ = with_contact(trace, inferred)
        ghost = L.ghost_track(pose, command)
        t = np.arange(len(pose)) * 0.02
        drift_s = np.hypot(*(est_s - pose[:, :2]).T) * 1000
        drift_g = np.hypot(*(est_g - pose[:, :2]).T) * 1000
        paths = path_svg(
            [
                (pose[:, :2], BLACK, 1.8, ""),
                (est_s, BLUE, 1.4, ""),
                (est_g, ORANGE, 1.4, ""),
                (ghost[:, :2], GHOST, 1.0, "4 3"),
            ]
        )
        name = case.split("__", 1)[-1]
        sections.append(f"""
<section><h2>{name}</h2>
<p class="what">Contact inference accuracy {acc * 100:.1f}%. End drift: sensed {drift_s[-1]:.0f} mm,
schedule-inferred {drift_g[-1]:.0f} mm. Black = true body path, <b style="color:{BLUE}">blue</b> = odometry
with sensed contact, <b style="color:{ORANGE}">orange</b> = with schedule-inferred contact,
dashed red = the ghost (integrated command).</p>
<div class="row">{paths}<div class="grow">{drift_svg(t, drift_s, drift_g)}</div></div></section>""")
    html = f"""<!doctype html>
<html><head><meta charset="utf-8"><title>Leg odometry: sensed vs schedule-inferred contact</title>
<style>
body {{ font: 14px/1.5 -apple-system, sans-serif; color: #1d1d1f; max-width: 1060px; margin: 0 auto; padding: 18px 16px 50px; background: #fff; }}
h1 {{ font-size: 20px; }} h2 {{ font-size: 15px; margin: 0 0 4px; }}
p.what {{ color: #444; margin: 2px 0 8px; }}
section {{ border: 1px solid #e3e3e3; border-radius: 10px; padding: 12px 14px; margin: 12px 0; }}
.row {{ display: flex; gap: 14px; align-items: flex-start; flex-wrap: wrap; }}
svg.path {{ width: 300px; flex: 0 0 auto; }}
.grow {{ flex: 1 1 420px; min-width: 300px; }}
</style></head><body>
<h1>Leg odometry: sensed contact vs gait-schedule inference</h1>
<p>Each case from the recorded tw_s3_u600 evaluation traces. "Sensed" gates the stance vote with the real
toe-force contact; "schedule-inferred" assumes a toe is planted unless its tripod's swing window
(phase &lt; 0.4 for tripod A, 0.5&ndash;0.9 for tripod B, zero command = all planted) says otherwise.
Diagnostic only.</p>
{"".join(sections)}
</body></html>"""
    OUT.write_text(html)
    print("wrote", OUT, len(html))


if __name__ == "__main__":
    main()
