"""Render a stand recording on the hexapod model: the real leg beside the simulated one.

    uv run python -m locomotion.sysid.leg_video RECORDING FIT_JSON --output VIDEO.mp4 [--speed 0.5]

The surrogate's full robot model (hull meshes from the approved geometry) is posed with the body fixed
and level as the stand holds it and the other five legs at the neutral stance. The lf leg plays back
three panels on one clock: the measured joint angles, the fitted simulator's replay of the same
commands, and the two blended (real solid, simulated as a ghost). Both panels share one joint mapping,
so their difference is the dynamics alone.

``--mapping stance`` (the default) is for display when the fit's mapping is not trustworthy (a loaded
recording): signs from the taught jump poses (crouch -> extended lowers the femur motor angle and
straightens the tibia while lowering its motor angle; coxa assumed +1) and offsets that put the
recording's start pose at the robot's neutral stance. ``--mapping fit`` uses the fitted mapping.
"""

import argparse
import json
import subprocess
import tempfile
from pathlib import Path

import numpy as np

from locomotion.sysid.leg_sim import LEG, LINKS, MOTORS, LegParams, LegSim, load_recording

PANEL = (640, 480)
BODY_HEIGHT_M = 0.30  # well above the floor: the stand leg hangs free
SIM_RGBA = (0.16, 0.47, 0.84, 1.0)  # the simulated leg, blue
STANCE_SIGN = np.array([1.0, 1.0, -1.0])  # from the taught crouch and extended poses


def _segment_at(segments, t):
    name = ""
    for s in segments:
        if s["start_s"] <= t:
            name = s["name"]
    return name


def render(
    stem,
    fit_json,
    output,
    speed=0.5,
    fps=25,
    azimuth=250.0,
    elevation=-10.0,
    distance=0.8,
    mapping="stance",
    preview=None,
):
    import cv2
    import mujoco

    from locomotion.surrogate.env import (
        DEFAULT_GEOMETRY,
        DEFAULT_MODEL,
        DEFAULT_STANCE,
        SurrogateConfig,
        SurrogateEnv,
    )

    rec = load_recording(stem)  # (DEFAULT_STANCE imported above)
    fit = json.loads(Path(fit_json).read_text())
    params = LegParams.from_json(fit["fitted"])
    if mapping == "stance":
        stance = json.loads(DEFAULT_STANCE.read_text())["nominal_joint_position_rad"][:3]
        start = np.array([rec["motors"][m]["pos"][0] for m in MOTORS])
        params = params.copy(sign=STANCE_SIGN, offset=start - np.asarray(stance) / STANCE_SIGN)
    sim_out = LegSim().replay(rec, params)
    with tempfile.TemporaryDirectory() as tmp:
        env = SurrogateEnv(
            SurrogateConfig(num_envs=1),
            None,
            DEFAULT_MODEL,
            DEFAULT_GEOMETRY,
            Path(tmp) / "native",
            reference_metadata=json.loads(DEFAULT_STANCE.read_text()),
            threads=0,
        )
    model = env.mj_model
    columns = [
        model.jnt_qposadr[mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, n)] for n in LEG
    ]
    leg_bodies = {mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, n) for n in LINKS}
    leg_geoms = np.array([g for g in range(model.ngeom) if model.geom_bodyid[g] in leg_bodies])
    default_rgba = model.geom_rgba.copy()
    base = env.local_qpos(0).copy()
    base[0:3] = (0.0, 0.0, BODY_HEIGHT_M)
    base[3:7] = (1.0, 0.0, 0.0, 0.0)
    data = mujoco.MjData(model)
    renderer = mujoco.Renderer(model, height=PANEL[1], width=PANEL[0])
    camera = mujoco.MjvCamera()
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.distance, camera.azimuth, camera.elevation = distance, azimuth, elevation
    femur_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "lf_femur")

    def pose(q_motor, sim=False, mask=False):
        q = base.copy()
        q[columns] = params.sign * (np.asarray(q_motor) - params.offset)
        data.qpos[:] = q
        mujoco.mj_forward(model, data)
        model.geom_rgba[:] = default_rgba
        if sim:
            model.geom_rgba[leg_geoms] = SIM_RGBA
        camera.lookat[:] = data.xpos[femur_id] + np.array([0.05, -0.05, -0.09])
        renderer.update_scene(data, camera)
        image = renderer.render().copy()
        if not mask:
            return image, None
        renderer.enable_segmentation_rendering()
        renderer.update_scene(data, camera)
        seg = renderer.render()
        renderer.disable_segmentation_rendering()
        hit = (seg[..., 1] == mujoco.mjtObj.mjOBJ_GEOM) & np.isin(seg[..., 0], leg_geoms)
        return image, hit

    motors = rec["motors"]
    t0 = max(motors[m]["t_read"][0] for m in MOTORS)
    t1 = min(motors[m]["t_read"][-1] for m in MOTORS)
    if preview:
        t1 = min(t1, t0 + preview)
    times = np.arange(t0, t1, speed / fps)
    real = np.stack([np.interp(times, motors[m]["t_read"], motors[m]["pos"]) for m in MOTORS], -1)
    simq = np.stack([np.interp(times, sim_out[m]["t"], sim_out[m]["pos"]) for m in MOTORS], -1)
    dense = np.arange(t0, t1, 0.01)
    trace = {
        "real": np.stack(
            [np.interp(dense, motors[m]["t_read"], motors[m]["pos"]) for m in MOTORS], -1
        ),
        "sim": np.stack([np.interp(dense, sim_out[m]["t"], sim_out[m]["pos"]) for m in MOTORS], -1),
        "target": np.stack(
            [np.interp(dense, motors[m]["t_send"], motors[m]["target"]) for m in MOTORS], -1
        ),
    }
    width, strip_h = 3 * PANEL[0], 170
    raw = str(Path(output).with_suffix(".raw.mp4"))
    writer = cv2.VideoWriter(raw, cv2.VideoWriter_fourcc(*"mp4v"), fps, (width, PANEL[1] + strip_h))
    font = cv2.FONT_HERSHEY_SIMPLEX
    names = ("coxa yaw", "femur pitch", "tibia pitch")

    def label(img, text, org, scale=0.7, thick=2):
        cv2.putText(img, text, org, font, scale, (0, 0, 0), thick + 3, cv2.LINE_AA)
        cv2.putText(img, text, org, font, scale, (255, 255, 255), thick, cv2.LINE_AA)

    def strip(t):
        img = np.full((strip_h, width, 3), 250, np.uint8)
        window = (dense >= t - 1.5) & (dense <= t + 1.5)
        for i in range(3):
            x0, w, h = i * PANEL[0] + 50, PANEL[0] - 70, strip_h - 50
            values = np.concatenate([trace[k][window, i] for k in trace])
            lo, hi = values.min(), values.max()
            span = max(hi - lo, np.radians(2.0))
            mid = 0.5 * (hi + lo)
            lo, hi = mid - 0.6 * span, mid + 0.6 * span
            for key, colour, th in (
                ("target", (150, 150, 150), 1),
                ("sim", (214, 120, 42), 2),
                ("real", (20, 20, 20), 2),
            ):
                xs = x0 + (dense[window] - (t - 1.5)) / 3.0 * w
                ys = 30 + (hi - trace[key][window, i]) / (hi - lo) * h
                cv2.polylines(img, [np.c_[xs, ys].astype(np.int32)], False, colour, th, cv2.LINE_AA)
            cx = int(x0 + 0.5 * w)
            cv2.line(img, (cx, 26), (cx, 30 + h), (120, 120, 120), 1)
            cv2.putText(
                img,
                f"{names[i]}   {np.degrees(hi - lo):.0f} deg tall, 3 s wide",
                (x0, 18),
                font,
                0.48,
                (40, 40, 40),
                1,
                cv2.LINE_AA,
            )
        cv2.putText(
            img,
            "black real   blue sim   grey target",
            (width - 330, strip_h - 8),
            font,
            0.45,
            (90, 90, 90),
            1,
            cv2.LINE_AA,
        )
        return img

    for k, t in enumerate(times):
        a, _ = pose(real[k])
        b, hit = pose(simq[k], sim=True, mask=True)
        overlay = a.copy()
        overlay[hit] = (0.45 * a[hit] + 0.55 * b[hit]).astype(np.uint8)
        panels = []
        for img, title in (
            (a, "REAL leg (measured)"),
            (b, "SIM leg (fitted actuator)"),
            (overlay, "OVERLAY: real + blue sim ghost"),
        ):
            img = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            label(img, title, (14, 32))
            panels.append(img)
        err = np.degrees(simq[k] - real[k])
        lines = [f"t {t - t0:5.2f} s  (x{speed:g} speed)", _segment_at(rec["meta"]["segments"], t)]
        lines += [f"{n:12s} sim - real {err[i]:+5.1f} deg" for i, n in enumerate(names)]
        box_h = 24 * len(lines) + 12
        cv2.rectangle(
            panels[2], (8, PANEL[1] - box_h - 8), (360, PANEL[1] - 8), (255, 255, 255), -1
        )
        for i, line in enumerate(lines):
            y = PANEL[1] - box_h + 14 + 24 * i
            cv2.putText(panels[2], line, (16, y), font, 0.52, (30, 30, 30), 1, cv2.LINE_AA)
        writer.write(np.vstack((np.hstack(panels), strip(t))))
    writer.release()
    renderer.close()
    subprocess.run(
        [
            "ffmpeg",
            "-v",
            "error",
            "-y",
            "-i",
            raw,
            "-c:v",
            "libx264",
            "-pix_fmt",
            "yuv420p",
            "-crf",
            "24",
            str(output),
        ],
        check=True,
    )
    Path(raw).unlink()
    return output


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("recording")
    parser.add_argument("fit_json")
    parser.add_argument("--output", required=True)
    parser.add_argument(
        "--speed", type=float, default=0.5, help="playback speed (0.5 = half speed)"
    )
    parser.add_argument("--azimuth", type=float, default=250.0)
    parser.add_argument("--elevation", type=float, default=-10.0)
    parser.add_argument("--distance", type=float, default=0.8)
    parser.add_argument("--mapping", choices=("stance", "fit"), default="stance")
    parser.add_argument("--preview", type=float, help="render only the first N recording seconds")
    args = parser.parse_args(argv)
    print(
        render(
            args.recording,
            args.fit_json,
            args.output,
            args.speed,
            azimuth=args.azimuth,
            elevation=args.elevation,
            distance=args.distance,
            mapping=args.mapping,
            preview=args.preview,
        )
    )


if __name__ == "__main__":
    main()
