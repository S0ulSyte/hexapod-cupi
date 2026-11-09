"""Fit the one-leg simulator to a stand recording: joint mapping, delay and actuator parameters.

    uv run python -m locomotion.sysid.leg_fit RECORDING [--output DIR]

RECORDING is the stem of a ``leg_record.py`` output (``.npz`` + ``.json``). Two stages:

1. **Statics: joint mapping.** On settled samples (the last 0.3 s of every hold and step, small
   velocity) the motor's own torque reading balances gravity: sign torque_motor = G(q_urdf). A joint's
   gravity torque is close to a sinusoid, so a flipped sign with a shifted offset fits it too: each sign
   hypothesis keeps its offset within 0.8 rad of the one that lines the calibrated motor range up with
   the model range, and least squares on gravity decides inside that window. The report keeps the
   residual of every sign hypothesis. The coxa yaw axis is vertical on the level stand, so gravity
   cannot identify its mapping: it keeps sign +1 and the offset that centres its measured range on the
   model's range, and the report says so.
2. **Dynamics: delay and the actuator.** Each joint's own excitation window (steps and chirp) is
   replayed with all three joints simulated. A grid over the delay (0-100 ms, 5 ms) with a quick
   stiffness/damping fit at each point finds the delay; Nelder-Mead then refines that joint's delay,
   kp_scale, kd_scale, armature and Coulomb friction together. The cost is the position RMS (rad) plus
   0.1 rad/Nm times the torque RMS against the motor's torque reading, so stiffness is pinned by
   torque and not traded against inertia. The three delays are fitted independently, so their
   agreement is a check; the reported delay is their median.

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
RANGE_WINDOW = 0.8  # rad an offset may sit from the range-aligned one
LOADED_RESIDUAL_NM = (
    0.16  # static residual that means the leg did not hang free (free: 0.11, loaded: 0.22)
)
CONTROL_DT = 0.02


def _segments(rec):
    segs = rec["meta"]["segments"]
    end = max(rec["motors"][m]["t_read"][-1] for m in MOTORS)
    for i, s in enumerate(segs):
        s["end_s"] = segs[i + 1]["start_s"] if i + 1 < len(segs) else end
    return segs


def settled_samples(rec):
    """(q_motor [n,3], torque_motor [n,3]) at the settled end of every hold and step segment."""
    rows_q, rows_tau = [], []
    for s in _segments(rec):
        if " hold" not in s["name"] and " step" not in s["name"]:
            continue
        a, b = s["end_s"] - SETTLE_S, s["end_s"] - 0.02
        q, tau = [], []
        for m in MOTORS:
            c = rec["motors"][m]
            w = (c["t_read"] >= a) & (c["t_read"] <= b) & (np.abs(c["vel"]) < SETTLED_VEL)
            if w.sum() < 3:
                break
            q.append(np.median(c["pos"][w]))
            tau.append(np.median(c["torque"][w]))
        else:
            rows_q.append(q)
            rows_tau.append(tau)
    return np.array(rows_q), np.array(rows_tau)


def range_offset(model_limits, motor_limits, sign):
    """The offset that centres a joint's measured motor range on its model range."""
    mid_model = 0.5 * (model_limits[0] + model_limits[1])
    mid_motor = 0.5 * (motor_limits[0] + motor_limits[1])
    return mid_motor - mid_model / sign


def fit_statics(sim, rec, model_limits, signs=None):
    """Signs and offsets of the femur and tibia from gravity; returns (params, report)."""
    from scipy.optimize import least_squares

    q, tau = settled_samples(rec)
    motor_limits = {m: tuple(rec["meta"]["limits"][str(m)]) for m in MOTORS}
    best, by_sign = None, {}
    hypotheses = itertools.product((1.0, -1.0), repeat=2) if signs is None else [tuple(signs[1:])]
    for signs in hypotheses:  # femur, tibia
        sign = np.array([1.0, *signs])
        x0 = [range_offset(model_limits[i], motor_limits[m], sign[i]) for i, m in enumerate(MOTORS)]
        fits = []
        # A single joint's gravity torque is close to a sinusoid, so a flipped sign with a shifted
        # offset reproduces it: each hypothesis is held within RANGE_WINDOW of the offset that lines
        # its calibrated motor range up with the model range, and gravity decides inside that.
        low = [x0[1] - RANGE_WINDOW, x0[2] - RANGE_WINDOW]
        high = [x0[1] + RANGE_WINDOW, x0[2] + RANGE_WINDOW]
        for shift in itertools.product((-0.5, 0.0, 0.5), repeat=2):

            def residual(x, sign=sign, coxa=x0[0]):
                offset = np.array([coxa, x[0], x[1]])
                out = []
                for qm, tm in zip(q, tau):
                    g = sim.gravity_torque(sign * (qm - offset))
                    out.extend((sign * tm - g)[1:])
                return np.array(out)

            fits.append(
                least_squares(
                    residual,
                    [x0[1] + shift[0], x0[2] + shift[1]],
                    bounds=(low, high),
                    loss="soft_l1",
                    f_scale=0.02,
                )
            )
        res = min(fits, key=lambda r: r.cost)
        rms = float(np.sqrt(np.mean(res.fun**2)))
        by_sign[f"femur {signs[0]:+.0f}, tibia {signs[1]:+.0f}"] = rms
        if best is None or rms < best[0]:
            best = (rms, sign, np.array([x0[0], res.x[0], res.x[1]]))
    rms, sign, offset = best
    params = LegParams(sign=sign, offset=offset)
    torque_span = {LEG[i]: float(np.ptp(tau[:, i])) for i in range(3)} if len(tau) else {}
    report = {
        "settled_samples": int(len(q)),
        "torque_residual_rms_nm": rms,
        "residual_rms_nm_by_sign": by_sign,
        "settled_torque_span_nm": torque_span,
        # The free leg's own gravity torque is at most about 0.3 Nm (femur) and 0.05 Nm (tibia): a
        # residual near that size means something else carried load, e.g. the foot on the floor.
        "loaded_warning": rms > LOADED_RESIDUAL_NM,
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


def rms_error(sim_out, rec, m, t0, t1, key="pos"):
    c = rec["motors"][m]
    keep = (c["t_read"] >= t0) & (c["t_read"] <= t1)
    return float(np.sqrt(np.mean((sim_out[m][key] - c[key][keep]) ** 2)))


TORQUE_WEIGHT = 0.1  # rad per Nm in the dynamics cost


def fit_dynamics(sim, rec, params, max_delay=0.1):
    from scipy.optimize import minimize

    fitted = params.copy(tau_max=np.full(3, 5.5))  # the RS05 peak; the stand never reaches it
    delays, report = [], {}
    for i, m in enumerate(MOTORS):
        window = joint_window(rec, m)
        if window is None:
            continue
        t0, t1 = window

        def cost(x, i=i, m=m, t0=t0, t1=t1):
            delay, kp_scale, kd_scale, armature, friction = x
            if not (
                0 <= delay <= max_delay
                and 0.3 <= kp_scale <= 3
                and 0.1 <= kd_scale <= 10
                and 0 <= armature <= 0.01
                and 0 <= friction <= 0.5
            ):
                return 1e3
            p = fitted.copy(delay_s=float(delay))
            p.kp_scale[i], p.kd_scale[i] = kp_scale, kd_scale
            p.armature[i], p.frictionloss[i] = armature, friction
            out = sim.replay(rec, p, t0, t1)
            return rms_error(out, rec, m, t0, t1) + TORQUE_WEIGHT * rms_error(
                out, rec, m, t0, t1, "torque"
            )

        grid = []
        for delay in np.arange(0.0, max_delay + 1e-9, 0.005):
            r = minimize(
                lambda y, d=delay: cost((d, y[0], y[1], 2e-4, 0.02)),
                (1.0, 1.0),
                method="Nelder-Mead",
                options={"xatol": 1e-3, "fatol": 1e-6, "maxfev": 40},
            )
            grid.append((float(r.fun), float(delay), r.x))
        _, d0, (kp0, kd0) = min(grid, key=lambda g: g[0])
        best = minimize(
            cost,
            (d0, kp0, kd0, 2e-4, 0.02),
            method="Nelder-Mead",
            options={"xatol": 1e-5, "fatol": 1e-7, "maxfev": 300},
        )
        delay, kp_s, kd_s, arm, fric = best.x
        fitted.kp_scale[i], fitted.kd_scale[i] = kp_s, kd_s
        fitted.armature[i], fitted.frictionloss[i] = arm, fric
        delays.append(float(delay))
        p = fitted.copy(delay_s=float(delay))
        out = sim.replay(rec, p, t0, t1)
        report[LEG[i]] = {
            "delay_s": float(delay),
            "window_pos_rms_rad": rms_error(out, rec, m, t0, t1),
            "window_torque_rms_nm": rms_error(out, rec, m, t0, t1, "torque"),
            "delay_grid_cost": {f"{g[1] * 1000:.0f}ms": round(g[0], 5) for g in grid},
            "evaluations": int(best.nfev) + 40 * len(grid),
        }
    fitted.delay_s = float(np.median(delays)) if delays else 0.0
    return fitted, report


CARRIAGE_BOUNDS = {"mass": (0.1, 2.0), "friction": (0.0, 20.0), "femur": 0.15, "tibia": 0.3}


def fit_carriage(rail, recs, params, bounds=CARRIAGE_BOUNDS):
    """The rail fixture from recordings with the toe on the floor: carriage mass, rail friction and the
    femur and tibia offsets (the toe's height and lever arms follow them), with the actuator fixed."""
    from scipy.optimize import minimize

    windows = [
        (rec, m, *joint_window(rec, m)) for rec in recs for m in (12, 13) if joint_window(rec, m)
    ]

    def candidate(x):
        mass, friction, d_femur, d_tibia = x
        offset = params.offset + np.array([0.0, d_femur, d_tibia])
        return params.copy(carriage_mass=float(mass), rail_friction=float(friction), offset=offset)

    def cost(x):
        mass, friction, d_femur, d_tibia = x
        if not (
            bounds["mass"][0] <= mass <= bounds["mass"][1]
            and bounds["friction"][0] <= friction <= bounds["friction"][1]
            and abs(d_femur) <= bounds["femur"]
            and abs(d_tibia) <= bounds["tibia"]
        ):
            return 1e3
        p, total = candidate(x), 0.0
        for rec, m, t0, t1 in windows:
            out = rail.replay(rec, p, t0, t1)
            for k in (12, 13):
                total += rms_error(out, rec, k, t0, t1) + TORQUE_WEIGHT * rms_error(
                    out, rec, k, t0, t1, "torque"
                )
        return total / len(windows)

    # A massless carriage on the slide is numerically unstable and physically wrong: the mass is held
    # in a plausible range, and the offsets may only move a little from the free-hanging statics.
    starts = [
        (mass, f, 0.0, d)
        for mass in (0.2, 0.5, 1.0)
        for f in (0.5, 3.0)
        for d in (-0.15, 0.0, 0.15)
    ]
    scored = sorted((cost(s), s) for s in starts)
    best = minimize(
        cost,
        scored[0][1],
        method="Nelder-Mead",
        options={"xatol": 1e-3, "fatol": 1e-5, "maxfev": 160},
    )
    fitted = candidate(best.x)
    report = {
        "recordings": [Path(r["stem"]).name for r in recs],
        "carriage_mass_kg": float(best.x[0]),
        "rail_friction_n": float(best.x[1]),
        "femur_offset_shift_rad": float(best.x[2]),
        "tibia_offset_shift_rad": float(best.x[3]),
        "cost": float(best.fun),
        "start_costs": {str(tuple(round(v, 2) for v in s)): round(c, 4) for c, s in scored},
        "evaluations": int(best.nfev) + len(starts),
        "bounds": bounds,
        "note": "the stand CAD is not in the repository: the carriage is a point mass on a frictional "
        "vertical slide, the toe a 12 mm sphere at the model's toe point, the floor a plane",
    }
    return fitted, report


def fit_floor(floor, recs, params):
    """Recordings with the carriage resting still and the toe pressing on the floor (confirmed by the
    camera height tracker): one floor height below the mount shared by every recording, the toe
    friction and a small tibia offset shift, with the actuator fixed. Every joint window counts."""
    from scipy.optimize import minimize

    windows = [
        (rec, m, *joint_window(rec, m)) for rec in recs for m in MOTORS if joint_window(rec, m)
    ]

    def candidate(x):
        height, friction, d_tibia = x
        offset = params.offset + np.array([0.0, 0.0, d_tibia])
        return params.copy(floor_height=float(height), toe_friction=float(friction), offset=offset)

    def cost(x):
        height, friction, d_tibia = x
        if not (-0.2 <= height <= -0.02 and 0.1 <= friction <= 2.0 and abs(d_tibia) <= 0.15):
            return 1e3
        p, total = candidate(x), 0.0
        for rec, m, t0, t1 in windows:
            out = floor.replay(rec, p, t0, t1)
            for k in MOTORS:
                total += rms_error(out, rec, k, t0, t1) + TORQUE_WEIGHT * rms_error(
                    out, rec, k, t0, t1, "torque"
                )
        return total / len(windows)

    starts_h = []
    for rec in recs:
        q0 = params.sign * (np.array([rec["motors"][m]["pos"][0] for m in MOTORS]) - params.offset)
        starts_h.append(floor.toe_height_below_mount(q0))
    h0 = float(np.median(starts_h))
    starts = [(h0 + dh, f, 0.0) for dh in (-0.005, 0.0, 0.005) for f in (0.5, 1.0)]
    scored = sorted((cost(s), s) for s in starts)
    best = minimize(
        cost,
        scored[0][1],
        method="Nelder-Mead",
        options={"xatol": 1e-4, "fatol": 1e-5, "maxfev": 120},
    )
    fitted = candidate(best.x)
    report = {
        "recordings": [Path(r["stem"]).name for r in recs],
        "floor_height_below_mount_m": float(best.x[0]),
        "toe_friction": float(best.x[1]),
        "tibia_offset_shift_rad": float(best.x[2]),
        "start_toe_heights_m": [round(v, 4) for v in starts_h],
        "cost": float(best.fun),
        "evaluations": int(best.nfev) + len(starts),
        "evidence": "camera height tracker: the carriage moved under 2 mm while the leg carried up to 0.9 Nm",
    }
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


def _rms_table(sim, rec, params):
    t0 = min(rec["motors"][m]["t_send"][0] for m in MOTORS)
    t1 = max(rec["motors"][m]["t_read"][-1] for m in MOTORS)
    out = sim.replay(rec, params)
    return out, {LEG[i]: rms_error(out, rec, m, t0, t1) for i, m in enumerate(MOTORS)}


def fit(stem, output=None, signs=None, loaded=()):
    """Stages 1-2 on ``stem`` (the leg hanging free), stage 3 on ``loaded`` (toe on the floor)."""
    rec = load_recording(stem)
    sim = LegSim()
    joints = {j["name"]: j for j in json.loads(active_model_path().read_text())["joints"]}
    model_limits = [(joints[n]["lower"], joints[n]["upper"]) for n in LEG]
    statics, statics_report = fit_statics(sim, rec, model_limits, signs)
    fitted, dyn_report = fit_dynamics(sim, rec, statics)
    nominal = statics.copy(kp_scale=np.ones(3))  # the training actuator on the identified mapping
    nominal_out, rms_nominal = _rms_table(sim, rec, nominal)
    fit_out, rms_fitted = _rms_table(sim, rec, fitted)
    result = {
        "schema": "hexapod_leg_sysid_fit_v2",
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
        "signs_given": None if signs is None else [float(v) for v in signs],
        "statics": statics_report,
        "dynamics": dyn_report,
        "rms_rad": {"nominal": rms_nominal, "fitted": rms_fitted},
    }
    if loaded:
        rail = LegSim(fixture="rail")
        recs = [load_recording(s) for s in loaded]
        fitted, result["carriage"] = fit_carriage(rail, recs, fitted)
        result["rail_rms_rad"] = {
            Path(r["stem"]).name: {
                "nominal": _rms_table(
                    rail,
                    r,
                    nominal.copy(
                        carriage_mass=fitted.carriage_mass,
                        rail_friction=fitted.rail_friction,
                        offset=fitted.offset,
                    ),
                )[1],
                "fitted": _rms_table(rail, r, fitted)[1],
            }
            for r in recs
        }
        _, result["rms_rad"]["fitted_after_carriage"] = _rms_table(sim, rec, fitted)
    result["fitted"] = fitted.to_json()
    result["delay_controls"] = fitted.delay_s / CONTROL_DT
    result["training_mapping"] = {
        "action_delay_controls": int(round(fitted.delay_s / CONTROL_DT)),
        "note": "DynamicsRandomization.action_delay_controls draws whole 20 ms controls; the fitted "
        "delay includes the bus round trip and the feedback read, as the policy would see them",
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
    parser.add_argument(
        "--signs", help="fixed joint signs, e.g. 1,1,-1 (default: search femur and tibia)"
    )
    parser.add_argument(
        "--loaded", nargs="*", default=(), help="recordings with the toe on the floor"
    )
    args = parser.parse_args(argv)
    signs = None if args.signs is None else tuple(float(v) for v in args.signs.split(","))
    result = fit(args.recording, args.output, signs, args.loaded)
    f = result["fitted"]
    if result["statics"]["loaded_warning"]:
        print(
            "WARNING: settled torques are not explained by the free leg's gravity "
            f"(residual {result['statics']['torque_residual_rms_nm']:.2f} Nm): the leg was loaded, "
            "e.g. the foot carried the carriage. Re-record with the foot off the floor."
        )
    print(
        f"delay {f['delay_s'] * 1000:.1f} ms ({result['delay_controls']:.2f} controls); "
        f"per-joint delays {[round(v['delay_s'] * 1000, 1) for v in result['dynamics'].values()]} ms"
    )
    for name in LEG:
        print(
            f"  {name:16s} RMS nominal {math.degrees(result['rms_rad']['nominal'][name]):6.2f}°  "
            f"fitted {math.degrees(result['rms_rad']['fitted'][name]):6.2f}°"
        )
    if "carriage" in result:
        c = result["carriage"]
        print(
            f"carriage {c['carriage_mass_kg']:.3f} kg, rail friction {c['rail_friction_n']:.2f} N, "
            f"offset shifts femur {c['femur_offset_shift_rad']:+.3f} tibia {c['tibia_offset_shift_rad']:+.3f} rad"
        )
        for rec, table in result["rail_rms_rad"].items():
            print(
                f"  {rec}: "
                + ", ".join(
                    f"{n.split('_')[1]} {math.degrees(table['nominal'][n]):.2f}° -> {math.degrees(table['fitted'][n]):.2f}°"
                    for n in LEG
                )
            )


if __name__ == "__main__":
    main()
