"""Write viewer run files (the Runs page) for a stand recording: the real leg and the simulated one.

    uv run python -m locomotion.sysid.stand_view RECORDING FIT_JSON --fixture rail|clamped --output DIR

Two files share case ids, so the Runs page plays them side by side on one clock: ``<name>_real.json``
(measured joint angles) and ``<name>_sim.json`` (the fitted simulator replaying the same commands).
Each case is one joint's excitation window, plus the whole recording. The scene is the hexapod's
viewer scene: the lf leg is drawn from the model's joint frames through the fit's joint mapping; the
other five legs are drawn as their mounts only, because the stand carries one leg. The body height is
the carriage: simulated on the rail fixture, and for the real leg inferred from the toe resting on the
floor (the stand has no carriage sensor in these recordings), or fixed when clamped. Each case carries
its own charts: the three joint angles (target, real, simulated) and, on the rail, the carriage height.
"""

import argparse
import base64
import hashlib
import json
from pathlib import Path

import numpy as np

from locomotion.sysid.leg_fit import joint_window
from locomotion.sysid.leg_sim import (
    LEG,
    MOTORS,
    LegParams,
    LegSim,
    active_model_path,
    load_recording,
    stance_path,
)

LEGS = ("lf", "lm", "lr", "rf", "rm", "rr")
PARTS = ("coxa_yaw", "femur_pitch", "tibia_pitch")
FRAME_S = 0.02
CLAMPED_HEIGHT_M = 0.30
SCHEMA = "hexapod_run_view_v1"


def _qmul(a, b):
    ax, ay, az, aw = a
    bx, by, bz, bw = b
    return np.array(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ]
    )


def _rot(q, v):
    u, w = np.asarray(q[:3]), q[3]
    t = 2 * np.cross(u, v)
    return v + w * t + np.cross(u, t)


def leg_points(joints, toe_local, q_lf, height):
    """(6, 4, 3) points in metres, body frame shifted up by ``height``: lf from its joint angles, the
    other legs as their coxa mount (all four points at the mount)."""
    out = np.zeros((6, 4, 3))
    for i, leg in enumerate(LEGS):
        rotation, position = np.array([0.0, 0.0, 0.0, 1.0]), np.array([0.0, 0.0, height])
        for k, part in enumerate(PARTS):
            j = joints[f"{leg}_{part}"]
            position = position + _rot(rotation, np.asarray(j["xyz"], float))
            out[i, k] = position
            if leg != "lf":
                out[i, :] = position
                break
            angle = q_lf[k]
            rotation = _qmul(rotation, np.asarray(j["quaternion_xyzw"], float))
            rotation = _qmul(rotation, np.array([0.0, 0.0, np.sin(angle / 2), np.cos(angle / 2)]))
        else:
            out[i, 3] = position + _rot(rotation, np.asarray(toe_local, float))
    return out


def _nav_mm(points):
    """Body (x left, y back, z up) metres to the viewer's (forward, left, up) millimetres."""
    return np.stack((-points[..., 1], points[..., 0], points[..., 2]), -1) * 1000


def build(stem, fit_json, fixture):
    rec = load_recording(stem)
    fit = json.loads(Path(fit_json).read_text())
    params = LegParams.from_json(fit["fitted"])
    sim = LegSim(fixture=fixture)
    model = json.loads(active_model_path().read_text())
    joints = {j["name"]: j for j in model["joints"]}
    toe = json.loads(stance_path().read_text())["toe_local_points_m"][0]
    motors = rec["motors"]
    t_start = max(motors[m]["t_read"][0] for m in MOTORS)
    t_end = min(motors[m]["t_read"][-1] for m in MOTORS)
    out = sim.replay(rec, params)
    stand = out["stand"]
    times = np.arange(t_start, t_end, FRAME_S)
    real_m = np.stack([np.interp(times, motors[m]["t_read"], motors[m]["pos"]) for m in MOTORS], -1)
    sim_m = np.stack([np.interp(times, out[m]["t"], out[m]["pos"]) for m in MOTORS], -1)
    target_m = np.stack(
        [np.interp(times, motors[m]["t_send"], motors[m]["target"]) for m in MOTORS], -1
    )

    def urdf(q_motor):
        return params.sign * (q_motor - params.offset)

    if fixture == "rail":
        sim_height = np.interp(times, stand["t"], stand["height"])
        sim_contact = np.interp(times, stand["t"], stand["contact"].astype(float)) > 0.5
        real_height = np.array([-sim.toe_height_below_mount(urdf(q)) for q in real_m])
        real_contact = np.ones(len(times), bool)  # assumed: the toe rests on the floor
    elif fixture == "floor":
        # The mount is still (the camera tracker saw the carriage move under 2 mm); the floor sits at
        # the fitted height below it, so the scene's floor is where the toe presses.
        sim_height = real_height = np.full(len(times), -params.floor_height)
        sim_contact = np.interp(times, stand["t"], stand["contact"].astype(float)) > 0.5
        real_contact = np.ones(len(times), bool)  # assumed: the toe presses on the floor
    else:
        sim_height = real_height = np.full(len(times), CLAMPED_HEIGHT_M)
        sim_contact = real_contact = np.zeros(len(times), bool)
    cases_spec = [("whole recording", t_start, t_end)]
    for i, m in enumerate(MOTORS):
        w = joint_window(rec, m)
        if w:
            cases_spec.append((f"{LEG[i]} excitation", max(w[0], t_start), min(w[1], t_end)))
    files = {}
    for kind in ("real", "sim"):
        q_m = real_m if kind == "real" else sim_m
        height = real_height if kind == "real" else sim_height
        contact = real_contact if kind == "real" else sim_contact
        points_all = np.stack([leg_points(joints, toe, urdf(q), h) for q, h in zip(q_m, height)])
        blobs, contacts, cases = [], [], []
        for name, a, b in cases_spec:
            idx = np.flatnonzero((times >= a) & (times < b))
            frames = len(idx)
            pts = np.round(_nav_mm(points_all[idx])).astype("<i2")
            con = (contact[idx].astype(np.uint8) * 1).astype("u1")  # bit 0: the lf toe
            rel = lambda arr, i: np.round(np.degrees(arr[idx, i] - real_m[idx[0], i]), 2).tolist()  # noqa: E731
            charts = []
            for i in range(3):
                series = [
                    {"values": rel(target_m, i), "colour": "--muted", "dashed": True},
                    {"values": rel(real_m, i), "colour": "--ink"},
                ]
                if kind == "sim":
                    series.append({"values": rel(sim_m, i), "colour": "--run-0"})
                charts.append(
                    {
                        "title": f"{LEG[i]} — dashed target, white real"
                        + (", blue simulated" if kind == "sim" else ""),
                        "unit": "°",
                        "series": series,
                    }
                )
            if fixture == "rail":
                hs = [
                    {
                        "values": np.round(
                            (real_height[idx] - real_height[idx][0]) * 1000, 2
                        ).tolist(),
                        "colour": "--ink",
                        "dashed": True,
                    }
                ]
                if kind == "sim":
                    hs.append(
                        {
                            "values": np.round(
                                (sim_height[idx] - real_height[idx][0]) * 1000, 2
                            ).tolist(),
                            "colour": "--run-0",
                        }
                    )
                charts.append(
                    {
                        "title": "carriage height — dashed: inferred from the toe on the floor"
                        + (", blue simulated" if kind == "sim" else ""),
                        "unit": "mm",
                        "series": hs,
                    }
                )
            failed = []
            if kind == "sim":
                for i in range(3):
                    failed.append(
                        {
                            "check": f"{LEG[i]} position RMS vs real, deg",
                            "value": float(
                                np.degrees(np.sqrt(np.mean((sim_m[idx, i] - real_m[idx, i]) ** 2)))
                            ),
                            "bound": "—",
                        }
                    )
            cases.append(
                {
                    "case_id": f"stand:{name}",
                    "batch": Path(stem).name,
                    "pass": kind == "real" or all(f["value"] < 1.0 for f in failed),
                    "failed": failed,
                    "controls": frames,
                    "fell": False,
                    "mean_velocity_mps": None,
                    "mean_yaw_rate_rad_s": None,
                    "frames": frames,
                    "command": [[0.0, 0.0, 0.0]] * frames,
                    "heading_deg": [0.0] * frames,
                    "commanded_heading_deg": [0.0] * frames,
                    "ghost_mm": [[0.0, 0.0]] * frames,
                    "ghost_distance_mm": [0.0] * frames,
                    "ghost_distance_end_mm": 0.0,
                    "charts": charts,
                    "points_offset": sum(p.nbytes for p in blobs),
                    "contacts_offset": sum(c.nbytes for c in contacts),
                }
            )
            blobs.append(pts)
            contacts.append(con)
        blob = b"".join(p.tobytes() for p in blobs)
        import gzip
        import io

        buffer = io.BytesIO()
        with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as stream:
            stream.write(blob + b"".join(c.tobytes() for c in contacts))
        files[kind] = {
            "schema": SCHEMA,
            "source": "leg stand (measured)"
            if kind == "real"
            else f"leg stand (simulated, {fixture})",
            "name": f"{Path(stem).name} {kind}",
            "summary_sha256": None,
            "model_sha256": hashlib.sha256(active_model_path().read_bytes()).hexdigest(),
            "frame_seconds": FRAME_S,
            "legs": list(LEGS),
            "contacts_base": len(blob),
            "data": base64.b64encode(buffer.getvalue()).decode(),
            "cases": cases,
            "cases_without_trace": [],
        }
    return files


FITTED_COLOUR, DEFAULT_COLOUR, OVERLAY_ALPHA = "#3fb871", "#e5534b", 0.45


def default_params(fitted):
    """The simulator before fitting: the training actuator (no delay, nominal gains, no friction or
    armature, the 1.6 Nm ceiling) on the same joint mapping and fixture."""
    return fitted.copy(
        delay_s=0.0,
        kp_scale=np.ones(3),
        kd_scale=np.ones(3),
        armature=np.zeros(3),
        frictionloss=np.zeros(3),
        damping=np.zeros(3),
        tau_max=np.full(3, 1.6),
    )


def build_overlay(stem, fit_json, fixture):
    """One run file: the real leg opaque, the fitted simulator translucent green and the default
    simulator translucent red on top of it, with joint charts in the same colours."""
    rec = load_recording(stem)
    fitted = LegParams.from_json(json.loads(Path(fit_json).read_text())["fitted"])
    default = default_params(fitted)
    sim = LegSim(fixture=fixture)
    joints = {j["name"]: j for j in json.loads(active_model_path().read_text())["joints"]}
    toe = json.loads(stance_path().read_text())["toe_local_points_m"][0]
    motors = rec["motors"]
    t_start = max(motors[m]["t_read"][0] for m in MOTORS)
    t_end = min(motors[m]["t_read"][-1] for m in MOTORS)
    times = np.arange(t_start, t_end, FRAME_S)

    def on_frames(out):
        return np.stack([np.interp(times, out[m]["t"], out[m]["pos"]) for m in MOTORS], -1)

    real_m = np.stack([np.interp(times, motors[m]["t_read"], motors[m]["pos"]) for m in MOTORS], -1)
    target_m = np.stack(
        [np.interp(times, motors[m]["t_send"], motors[m]["target"]) for m in MOTORS], -1
    )
    fit_m, def_m = on_frames(sim.replay(rec, fitted)), on_frames(sim.replay(rec, default))
    if fixture == "floor":
        height = -fitted.floor_height
    elif fixture == "clamped":
        height = CLAMPED_HEIGHT_M
    else:
        raise ValueError("overlays support the clamped and floor fixtures (a still mount)")

    def points(q_m):
        return np.stack(
            [leg_points(joints, toe, fitted.sign * (q - fitted.offset), height) for q in q_m]
        )

    pts = {"real": points(real_m), "fit": points(fit_m), "default": points(def_m)}
    cases_spec = [("whole recording", t_start, t_end)]
    for i, m in enumerate(MOTORS):
        w = joint_window(rec, m)
        if w:
            cases_spec.append((f"{LEG[i]} excitation", max(w[0], t_start), min(w[1], t_end)))
    blobs, contacts, cases, offset = [], [], [], 0

    def add(array):
        nonlocal offset
        here = offset
        blobs.append(array)
        offset += array.nbytes
        return here

    for name, a, b in cases_spec:
        idx = np.flatnonzero((times >= a) & (times < b))
        frames = len(idx)

        def rel(arr, i):
            return np.round(np.degrees(arr[idx, i] - real_m[idx[0], i]), 2).tolist()

        def rms(arr, i):
            return float(np.degrees(np.sqrt(np.mean((arr[idx, i] - real_m[idx, i]) ** 2))))

        main = add(np.round(_nav_mm(pts["real"][idx])).astype("<i2"))
        overlays = [
            {
                "label": "fitted sim",
                "colour": FITTED_COLOUR,
                "alpha": OVERLAY_ALPHA,
                "points_offset": add(np.round(_nav_mm(pts["fit"][idx])).astype("<i2")),
            },
            {
                "label": "default sim (before fitting)",
                "colour": DEFAULT_COLOUR,
                "alpha": OVERLAY_ALPHA,
                "points_offset": add(np.round(_nav_mm(pts["default"][idx])).astype("<i2")),
            },
        ]
        charts = [
            {
                "title": f"{LEG[i]} — dashed target, white real, green fitted, red default",
                "unit": "°",
                "series": [
                    {"values": rel(target_m, i), "colour": "--muted", "dashed": True},
                    {"values": rel(def_m, i), "colour": DEFAULT_COLOUR},
                    {"values": rel(fit_m, i), "colour": FITTED_COLOUR},
                    {"values": rel(real_m, i), "colour": "--ink"},
                ],
            }
            for i in range(3)
        ]
        failed = [
            {
                "check": f"{LEG[i]} RMS vs real, deg: fitted",
                "value": rms(fit_m, i),
                "bound": f"default {rms(def_m, i):.2f}",
            }
            for i in range(3)
        ]
        contacts.append(np.full(frames, 1 if fixture == "floor" else 0, "u1"))
        cases.append(
            {
                "case_id": f"stand:{name}",
                "batch": Path(stem).name,
                "pass": True,
                "failed": failed,
                "controls": frames,
                "fell": False,
                "mean_velocity_mps": None,
                "mean_yaw_rate_rad_s": None,
                "frames": frames,
                "command": [[0.0, 0.0, 0.0]] * frames,
                "heading_deg": [0.0] * frames,
                "commanded_heading_deg": [0.0] * frames,
                "ghost_mm": [[0.0, 0.0]] * frames,
                "ghost_distance_mm": [0.0] * frames,
                "ghost_distance_end_mm": 0.0,
                "charts": charts,
                "overlays": overlays,
                "points_offset": main,
            }
        )
    blob = b"".join(x.tobytes() for x in blobs)
    at = 0
    for c, con in zip(cases, contacts):
        c["contacts_offset"] = at
        at += con.nbytes
    import gzip
    import io

    buffer = io.BytesIO()
    with gzip.GzipFile(fileobj=buffer, mode="wb", compresslevel=9, mtime=0) as stream:
        stream.write(blob + b"".join(c.tobytes() for c in contacts))
    return {
        "schema": SCHEMA,
        "source": f"leg stand: real (opaque), fitted (green), default (red), {fixture}",
        "name": f"{Path(stem).name} overlay",
        "summary_sha256": None,
        "model_sha256": hashlib.sha256(active_model_path().read_bytes()).hexdigest(),
        "frame_seconds": FRAME_S,
        "legs": list(LEGS),
        "contacts_base": len(blob),
        "data": base64.b64encode(buffer.getvalue()).decode(),
        "cases": cases,
        "cases_without_trace": [],
    }


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording")
    parser.add_argument("fit_json")
    parser.add_argument("--fixture", choices=("rail", "clamped", "floor"), required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument(
        "--overlay", action="store_true", help="one file: real, fitted and default on top"
    )
    args = parser.parse_args(argv)
    args.output.mkdir(parents=True, exist_ok=True)
    if args.overlay:
        path = args.output / f"{Path(args.recording).name}_overlay.json"
        value = build_overlay(args.recording, args.fit_json, args.fixture)
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        print(path)
        return
    for kind, value in build(args.recording, args.fit_json, args.fixture).items():
        path = args.output / f"{Path(args.recording).name}_{kind}.json"
        path.write_text(json.dumps(value, separators=(",", ":")) + "\n")
        print(path)


if __name__ == "__main__":
    main()
