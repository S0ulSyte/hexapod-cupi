"""Fit the one-leg simulator to a stand recording: joint mapping, delay and actuator parameters.

    uv run python -m locomotion.sysid.leg_fit RECORDING [--output DIR]

RECORDING is the stem of a ``leg_record.py`` output (``.npz`` + ``.json``). Two stages:

1. **Statics: mapping and stiffness.** On settled samples (the last 0.3 s of every hold and step, small
   velocity) the motor torque balances gravity: sign kp_scale kp (target - q) = G(q_urdf). Least squares
   over the femur and tibia gives each joint's sign, motor-frame offset and kp_scale. The coxa yaw axis
   is vertical on the level stand, so gravity cannot identify its mapping: it keeps sign +1 and the
   offset that centres its measured range on the model's range, and the report says so.
2. **Dynamics: delay and the rest.** Each joint's own excitation window (steps and chirp) is replayed
   with all three joints simulated, and Nelder-Mead fits that joint's delay, kd_scale, armature and
   Coulomb friction to the measured position. The three delays are fitted independently, so their
   agreement is itself a check; the reported delay is their median.

Writes ``<output>/fit.json`` (parameters, per-joint RMS error for the nominal training actuator and for
the fit, the delay in 20 ms controls for ``DynamicsRandomization``) and ``fit.html`` (overlays).
"""

import argparse
import itertools
import json
import math
from pathlib import Path

import numpy as np

from locomotion.sysid.leg_sim import (
    LEG,
    MOTORS,
    LegParams,
    LegSim,
    active_model_path,
    load_recording,
)

SETTLE_S, SETTLED_VEL = 0.3, 0.05
CONTROL_DT = 0.02


def _segments(rec):
    segs = rec["meta"]["segments"]
    end = max(rec["motors"][m]["t_read"][-1] for m in MOTORS)
    for i, s in enumerate(segs):
        s["end_s"] = segs[i + 1]["start_s"] if i + 1 < len(segs) else end
    return segs


def settled_samples(rec):
    """(q_motor [n,3], target [n,3]) at the settled end of every hold and step segment."""
    rows_q, rows_t = [], []
    for s in _segments(rec):
        if " hold" not in s["name"] and " step" not in s["name"]:
            continue
        a, b = s["end_s"] - SETTLE_S, s["end_s"] - 0.02
        q, t = [], []
        for m in MOTORS:
            c = rec["motors"][m]
            w = (c["t_read"] >= a) & (c["t_read"] <= b) & (np.abs(c["vel"]) < SETTLED_VEL)
            if w.sum() < 3:
                break
            q.append(np.median(c["pos"][w]))
            t.append(np.median(c["target"][w]))
        else:
            rows_q.append(q)
            rows_t.append(t)
    return np.array(rows_q), np.array(rows_t)


def range_offset(model_limits, motor_limits, sign):
    """The offset that centres a joint's measured motor range on its model range."""
    mid_model = 0.5 * (model_limits[0] + model_limits[1])
    mid_motor = 0.5 * (motor_limits[0] + motor_limits[1])
    return mid_motor - mid_model / sign


def fit_statics(sim, rec, model_limits):
    """Signs, offsets and kp_scales from gravity; returns (params, report)."""
    from scipy.optimize import least_squares

    q, target = settled_samples(rec)
    motor_limits = {m: tuple(rec["meta"]["limits"][str(m)]) for m in MOTORS}
    kp = rec["meta"]["gains"]["kp"]
    best, by_sign = None, {}
    for signs in itertools.product((1.0, -1.0), repeat=2):  # femur, tibia
        sign = np.array([1.0, *signs])
        x0 = [range_offset(model_limits[i], motor_limits[m], sign[i]) for i, m in enumerate(MOTORS)]

        def residual(x, sign=sign, coxa=x0[0]):
            offset = np.array([coxa, x[0], x[1]])
            scale = np.array([1.0, x[2], x[3]])
            out = []
            for qm, tm in zip(q, target):
                g = sim.gravity_torque(sign * (qm - offset))
                motor = sign * scale * kp * (tm - qm)
                out.extend((motor - g)[1:])
            return np.array(out)

        res = least_squares(
            residual,
            [x0[1], x0[2], 1.0, 1.0],
            bounds=([x0[1] - 1.0, x0[2] - 1.0, 0.3, 0.3], [x0[1] + 1.0, x0[2] + 1.0, 3.0, 3.0]),
            loss="soft_l1",
            f_scale=0.05,
        )
        rms = float(np.sqrt(np.mean(res.fun**2)))
        by_sign[f"femur {signs[0]:+.0f}, tibia {signs[1]:+.0f}"] = rms
        if best is None or rms < best[0]:
            best = (
                rms,
                sign,
                np.array([x0[0], res.x[0], res.x[1]]),
                np.array([1.0, res.x[2], res.x[3]]),
            )
    rms, sign, offset, kp_scale = best
    params = LegParams(sign=sign, offset=offset, kp_scale=kp_scale)
    report = {
        "settled_samples": int(len(q)),
        "torque_residual_rms_nm": rms,
        "residual_rms_nm_by_sign": by_sign,
        "sign_note": "a sign is identified only when its hypothesis clearly beats the flipped one; the "
        "tibia's gravity torque is under 0.05 Nm, so friction can leave its sign ambiguous",
        "coxa_mapping": "not identifiable from gravity on a level stand: sign +1, offset centres the "
        "measured range on the model range; verify by eye (a + move should swing the foot toward "
        "the model's +coxa_yaw, counter-clockwise seen from above)",
    }
    return params, report


def joint_window(rec, m):
    segs = [s for s in _segments(rec) if s["joint"] == m]
    return (segs[0]["start_s"], segs[-1]["end_s"]) if segs else None


def rms_error(sim_out, rec, m, t0, t1):
    c = rec["motors"][m]
    keep = (c["t_read"] >= t0) & (c["t_read"] <= t1)
    return float(np.sqrt(np.mean((sim_out[m]["pos"] - c["pos"][keep]) ** 2)))


def fit_dynamics(sim, rec, params, max_delay=0.1):
    from scipy.optimize import minimize

    fitted = params.copy()
    delays, report = [], {}
    for i, m in enumerate(MOTORS):
        window = joint_window(rec, m)
        if window is None:
            continue
        t0, t1 = window

        def cost(x, i=i, m=m, t0=t0, t1=t1):
            delay, kd_scale, armature, friction = x
            if not (
                0 <= delay <= max_delay
                and 0.2 <= kd_scale <= 5
                and 0 <= armature <= 0.01
                and 0 <= friction <= 0.5
            ):
                return 1e3
            p = fitted.copy(delay_s=float(delay))
            p.kd_scale[i], p.armature[i], p.frictionloss[i] = kd_scale, armature, friction
            return rms_error(sim.replay(rec, p, t0, t1), rec, m, t0, t1)

        starts = [(d, 1.0, 5e-4, 0.01) for d in (0.0, 0.02, 0.04)]
        best = min(
            (
                minimize(
                    cost,
                    s,
                    method="Nelder-Mead",
                    options={"xatol": 1e-4, "fatol": 1e-6, "maxfev": 150},
                )
                for s in starts
            ),
            key=lambda r: r.fun,
        )
        delay, fitted.kd_scale[i], fitted.armature[i], fitted.frictionloss[i] = best.x
        delays.append(float(delay))
        report[LEG[i]] = {
            "delay_s": float(delay),
            "window_rms_rad": float(best.fun),
            "evaluations": int(best.nfev),
        }
    fitted.delay_s = float(np.median(delays)) if delays else 0.0
    return fitted, report


def overlay_html(rec, nominal_out, fit_out, fit_json):
    """A self-contained page: measured, nominal-sim and fitted-sim position per joint (inline SVG)."""
    parts = []
    for i, m in enumerate(MOTORS):
        c = rec["motors"][m]
        t = c["t_read"]
        series = [
            ("target", c["target"], "#999"),
            ("measured", c["pos"], "#111"),
            ("nominal sim", nominal_out[m]["pos"], "#d1495b"),
            ("fitted sim", fit_out[m]["pos"], "#2e86de"),
        ]
        lo = min(float(np.min(s)) for _, s, _ in series)
        hi = max(float(np.max(s)) for _, s, _ in series)
        span = max(hi - lo, 1e-3)
        W, H = 1100, 220

        def path(values):
            xs = (t - t[0]) / max(t[-1] - t[0], 1e-6) * W
            ys = H - (values - lo) / span * H
            step = max(1, len(xs) // 3000)
            return "M" + " L".join(f"{x:.1f},{y:.1f}" for x, y in zip(xs[::step], ys[::step]))

        lines = "".join(
            f'<path d="{path(v)}" fill="none" stroke="{col}" stroke-width="1.2"/>'
            for _, v, col in series
        )
        legend = " · ".join(f'<span style="color:{col}">{name}</span>' for name, _, col in series)
        nom = fit_json["rms_rad"]["nominal"][LEG[i]]
        fit = fit_json["rms_rad"]["fitted"][LEG[i]]
        parts.append(
            f"<h3>{LEG[i]} (motor {m}) — RMS nominal {math.degrees(nom):.2f}°, fitted {math.degrees(fit):.2f}°</h3>"
            f"<div>{legend}; {math.degrees(lo):.0f}° … {math.degrees(hi):.0f}° motor frame</div>"
            f'<svg viewBox="0 0 {W} {H}" width="100%" style="background:#fafafa;border:1px solid #ddd">{lines}</svg>'
        )
    p = fit_json["fitted"]
    return (
        "<!doctype html><meta charset=utf-8><title>Leg actuator fit</title>"
        "<body style='font:14px system-ui;margin:20px;max-width:1150px'>"
        f"<h2>Leg stand actuator fit: {fit_json['recording']}</h2>"
        f"<p>delay {p['delay_s'] * 1000:.1f} ms ({fit_json['delay_controls']:.2f} controls of 20 ms); "
        f"kp_scale {np.round(p['kp_scale'], 2).tolist()}, kd_scale {np.round(p['kd_scale'], 2).tolist()}, "
        f"armature {np.round(p['armature'], 5).tolist()} kg m², friction {np.round(p['frictionloss'], 3).tolist()} Nm</p>"
        + "".join(parts)
        + "</body>"
    )


def fit(stem, output=None):
    rec = load_recording(stem)
    sim = LegSim()
    joints = {j["name"]: j for j in json.loads(active_model_path().read_text())["joints"]}
    model_limits = [(joints[n]["lower"], joints[n]["upper"]) for n in LEG]
    statics, statics_report = fit_statics(sim, rec, model_limits)
    fitted, dyn_report = fit_dynamics(sim, rec, statics)
    nominal = statics.copy(kp_scale=np.ones(3))  # the training actuator on the identified mapping
    nominal_out, fit_out = sim.replay(rec, nominal), sim.replay(rec, fitted)
    t0 = min(rec["motors"][m]["t_send"][0] for m in MOTORS)
    t1 = max(rec["motors"][m]["t_read"][-1] for m in MOTORS)
    result = {
        "schema": "hexapod_leg_sysid_fit_v1",
        "recording": Path(str(stem)).name,
        "recording_meta": {
            k: rec["meta"][k]
            for k in (
                "started",
                "outcome",
                "fake",
                "bus",
                "rate_hz",
                "cycle_s_median",
                "read_after_send_s_median",
            )
            if k in rec["meta"]
        },
        "fitted": fitted.to_json(),
        "delay_controls": fitted.delay_s / CONTROL_DT,
        "statics": statics_report,
        "dynamics": dyn_report,
        "rms_rad": {
            "nominal": {
                LEG[i]: rms_error(nominal_out, rec, m, t0, t1) for i, m in enumerate(MOTORS)
            },
            "fitted": {LEG[i]: rms_error(fit_out, rec, m, t0, t1) for i, m in enumerate(MOTORS)},
        },
        "training_mapping": {
            "action_delay_controls": int(round(fitted.delay_s / CONTROL_DT)),
            "note": "DynamicsRandomization.action_delay_controls draws whole 20 ms controls; the fitted "
            "delay includes the bus round trip and the feedback read, as the policy would see them",
        },
    }
    if output is not None:
        out = Path(output)
        out.mkdir(parents=True, exist_ok=True)
        (out / "fit.json").write_text(json.dumps(result, indent=1) + "\n")
        (out / "fit.html").write_text(overlay_html(rec, nominal_out, fit_out, result))
    return result


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording", help="stem of a leg_record.py output (.npz/.json)")
    parser.add_argument("--output", help="directory for fit.json and fit.html")
    args = parser.parse_args(argv)
    result = fit(args.recording, args.output)
    f = result["fitted"]
    print(
        f"delay {f['delay_s'] * 1000:.1f} ms ({result['delay_controls']:.2f} controls); "
        f"per-joint delays {[round(v['delay_s'] * 1000, 1) for v in result['dynamics'].values()]} ms"
    )
    for name in LEG:
        print(
            f"  {name:16s} RMS nominal {math.degrees(result['rms_rad']['nominal'][name]):6.2f}°  "
            f"fitted {math.degrees(result['rms_rad']['fitted'][name]):6.2f}°"
        )


if __name__ == "__main__":
    main()
