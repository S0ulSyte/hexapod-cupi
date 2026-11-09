"""Record a scripted excitation of the lf leg on the stand, under the policy's own motor law.

Standalone (numpy + motorbridge): copy it to the stand computer and run it there, with nothing else on
the bus (stop leg_app.py / leg_jog.py first):

    ~/rebot-venv/bin/python leg_record.py --plan             what it will do, from the live pose
    ~/rebot-venv/bin/python leg_record.py --fake --out DIR   the whole loop on simulated motors
    ~/rebot-venv/bin/python leg_record.py --out ~/sysid      the real run (asks you to type GO)

Every motor runs RobStride MIT mode with the gains training uses (robots/rs05.py, robots/hexapod.py):
stiffness 12 Nm/rad and the per-joint damping, zero velocity target, no feedforward. Targets change at
50 Hz like the policy's, and every target step is slew-limited to 0.040 rad per 20 ms like the policy's
limiter; commands are re-sent and feedback read at --rate Hz (default 100) to sample the response.

One joint at a time is excited while the other two hold the start pose: a hold, four steps
(+A, back, -A, back), a logarithmic chirp, and a final hold. Amplitudes shrink to stay inside each
joint's measured limits (~/leg_limits.json) less a margin.

Guards, any of which stops everything (zero stiffness, then every motor disabled): a joint more than
0.4 rad from its target, outside its measured limits by 0.05 rad, a torque above --max-torque, a
motor temperature above 70 C, feedback silent for 0.2 s, Ctrl+C or SIGTERM. The leg should hang free
(carriage resting on its stop, foot off the floor): the simulator replays it with a fixed base.

Output: <out>/leg_<time>.npz (one row per motor per cycle) and <out>/leg_<time>.json (setup, gains,
start pose, segment table, bus and timing statistics).
"""

import argparse
import json
import math
import os
import signal
import sys
import time

import numpy as np

IDS = (11, 12, 13)
NAMES = {11: "lf_coxa_yaw", 12: "lf_femur_pitch", 13: "lf_tibia_pitch"}
HOST = 0xFD
MECH_POS, RUN_MODE, MIT_MODE = 0x7019, 0x7005, 0
KP = 12.0  # robots/rs05.py STIFFNESS_NM_PER_RAD
KD = {11: 0.44197696391810243, 12: 0.2456657243160222, 13: 0.10552064613953134}  # robots/hexapod.py
CONTROL_DT, SLEW = 0.02, 0.04  # policy period (s) and target slew (rad per control)
LIMITS = os.path.expanduser("~/leg_limits.json")
MARGIN_MIN, MARGIN_FRAC = math.radians(7), 0.10
TRACK_MAX, LIMIT_TOL, TEMP_MAX, SILENT_MAX = 0.4, 0.05, 70.0, 0.2


def load_limits(path):
    with open(path) as f:
        d = json.load(f)
    return {m: (d["joints"][str(m)]["min"], d["joints"][str(m)]["max"]) for m in IDS}


def safe_band(limits, m):
    lo, hi = limits[m]
    margin = max(MARGIN_MIN, MARGIN_FRAC * (hi - lo))
    return lo + margin, hi - margin


def plan(p0, limits, step=0.15, chirp=0.08, f0=0.3, f1=3.0, chirp_s=8.0):
    """-> list of segments (name, joint, seconds, f(t) -> excited joint's offset from p0, rad)."""
    segs = []
    for m in IDS:
        slo, shi = safe_band(limits, m)
        room = min(p0[m] - slo, shi - p0[m])
        if room < 0.03:
            segs.append(
                (f"skip {NAMES[m]}: start pose within 0.03 rad of its safe band", m, 0.0, None)
            )
            continue
        a, c = min(step, room), min(chirp, room)
        k = math.log(f1 / f0) / chirp_s

        def chirp_f(t, c=c, k=k):
            ramp = min(1.0, t / 0.5, (chirp_s - t) / 0.5)
            phase = 2 * math.pi * f0 * (math.exp(k * t) - 1) / k
            return c * max(0.0, ramp) * math.sin(phase)

        segs += [
            (f"{NAMES[m]} hold", m, 1.0, lambda t: 0.0),
            (f"{NAMES[m]} step +{math.degrees(a):.1f} deg", m, 0.8, lambda t, a=a: a),
            (f"{NAMES[m]} step back", m, 0.8, lambda t: 0.0),
            (f"{NAMES[m]} step -{math.degrees(a):.1f} deg", m, 0.8, lambda t, a=a: -a),
            (f"{NAMES[m]} step back", m, 0.8, lambda t: 0.0),
            (f"{NAMES[m]} chirp {math.degrees(c):.1f} deg {f0:g}-{f1:g} Hz", m, chirp_s, chirp_f),
            (f"{NAMES[m]} hold", m, 0.6, lambda t: 0.0),
        ]
    return segs


class Stop(Exception):
    pass


class FakeMotor:
    """A second-order joint with a transport delay, for --fake runs without hardware."""

    def __init__(self, mid, p, delay=0.012):
        self.mid, self.p, self.v, self.on, self.cmd, self.queue, self.delay = (
            mid,
            p,
            0.0,
            False,
            None,
            [],
            delay,
        )
        self.t = time.monotonic()

    def _advance(self):
        now = time.monotonic()
        tau = getattr(self, "tau", 0.0)
        while self.t < now:
            dt = min(0.001, now - self.t)
            while self.queue and self.queue[0][0] <= self.t - self.delay:
                self.cmd = self.queue.pop(0)[1]
            tau = 0.0
            if self.on and self.cmd:
                pos, _, kp, kd, _ = self.cmd
                tau = max(-5.5, min(5.5, kp * (pos - self.p) - kd * self.v))
            self.v += (tau - 0.02 * self.v) / 2e-3 * dt
            self.p += self.v * dt
            self.t += dt
        self.tau = tau if self.on else 0.0

    def ensure_mode(self, mode, timeout_ms=1000):
        pass

    def robstride_get_param_i8(self, key, timeout_ms=1000):
        return MIT_MODE

    def robstride_get_param_f32(self, key, timeout_ms=1000):
        self._advance()
        return self.p

    def enable(self):
        self.on = True

    def disable(self):
        self.on = False

    def send_mit(self, pos, vel, kp, kd, tau):
        self._advance()
        self.queue.append((time.monotonic(), (pos, vel, kp, kd, tau)))

    def get_state(self):
        self._advance()

        class S:
            pass

        s = S()
        s.pos, s.vel, s.torq, s.t_mos, s.status_code = self.p, self.v, self.tau, 35.0, 0
        return s


def release(ms):
    """Zero stiffness, then disable every motor; never raises (it runs on every way out)."""
    for m in ms.values():
        try:
            st = m.get_state()
            m.send_mit(st.pos if st else 0.0, 0.0, 0.0, 0.5, 0.0)
        except Exception:
            pass
    time.sleep(0.2)
    for m in ms.values():
        try:
            m.disable()
        except Exception:
            pass


def record(ms, limits, args, log=print):
    p0 = {mid: m.robstride_get_param_f32(MECH_POS, 200) for mid, m in ms.items()}
    for mid in IDS:
        lo, hi = limits[mid]
        if not lo <= p0[mid] <= hi:
            raise Stop(
                f"{NAMES[mid]} reads {p0[mid]:+.3f}, outside its measured limits: recalibrate"
            )
    segs = plan(p0, limits, args.step, args.chirp)
    from_mode = None
    try:
        from motorbridge.models import Mode

        from_mode = Mode.MIT
    except ImportError:
        pass
    for mid, m in ms.items():
        m.ensure_mode(from_mode if from_mode is not None else 1, 500)
        if m.robstride_get_param_i8(RUN_MODE, 200) != MIT_MODE:
            raise Stop(f"{NAMES[mid]}: MIT mode not confirmed; not enabling")
    rows, segtab = [], []
    period = 1.0 / args.rate
    held = dict(p0)  # slew-limited targets
    last_seen = {mid: time.monotonic() for mid in IDS}
    last_state = {}
    t_start = time.monotonic()
    wall_start = time.time()  # wall clock at t_start, to align with other loggers (camera height)
    for m in ms.values():
        m.enable()
    try:
        # Ramp the stiffness in over 0.5 s at the start pose, then run the segments back to back.
        schedule = [("ramp in", None, 0.5, None)] + [s for s in segs if s[3] is not None]
        for s in segs:
            if s[3] is None:
                log(s[0])
        for index, (name, joint, seconds, f) in enumerate(schedule):
            seg_t0 = time.monotonic()
            segtab.append(
                {
                    "index": index,
                    "name": name,
                    "joint": joint,
                    "start_s": seg_t0 - t_start,
                    "seconds": seconds,
                }
            )
            log(f"  {name} ({seconds:g} s)")
            next_control = seg_t0
            while (t := time.monotonic() - seg_t0) < seconds:
                cycle_t0 = time.monotonic()
                if cycle_t0 >= next_control:  # a new 50 Hz target, slew-limited like the policy's
                    next_control += CONTROL_DT
                    for mid in IDS:
                        want = p0[mid] + (f(t) if (f is not None and mid == joint) else 0.0)
                        held[mid] += max(-SLEW, min(SLEW, want - held[mid]))
                kp = KP * min(1.0, t / seconds) if name == "ramp in" else KP
                for mid, m in ms.items():
                    t_send = time.monotonic() - t_start
                    m.send_mit(held[mid], 0.0, kp, KD[mid], 0.0)
                    st = m.get_state()
                    t_read = time.monotonic() - t_start
                    if st is not None and (
                        mid not in last_state or (st.pos, st.vel, st.torq) != last_state[mid]
                    ):
                        last_state[mid] = (st.pos, st.vel, st.torq)
                        last_seen[mid] = time.monotonic()
                        fresh = 1
                    else:
                        fresh = 0
                    if st is None:
                        if time.monotonic() - last_seen[mid] > SILENT_MAX:
                            raise Stop(f"{NAMES[mid]}: no feedback for {SILENT_MAX} s")
                        continue
                    # A joint at rest legitimately repeats its quantized feedback, so a repeat is only
                    # flagged (fresh = 0), never a stop; a missing state is the silence guard above.
                    last_seen[mid] = time.monotonic()
                    rows.append(
                        (
                            t_send,
                            t_read,
                            mid,
                            index,
                            held[mid],
                            kp,
                            KD[mid],
                            st.pos,
                            st.vel,
                            st.torq,
                            st.t_mos,
                            fresh,
                        )
                    )
                    lo, hi = limits[mid]
                    if not lo - LIMIT_TOL <= st.pos <= hi + LIMIT_TOL:
                        raise Stop(f"{NAMES[mid]} left its measured limits at {st.pos:+.3f}")
                    if abs(st.pos - held[mid]) > TRACK_MAX:
                        raise Stop(
                            f"{NAMES[mid]} is {math.degrees(st.pos - held[mid]):+.0f} deg off target"
                        )
                    if abs(st.torq) > args.max_torque:
                        raise Stop(
                            f"{NAMES[mid]} torque {st.torq:+.2f} Nm above {args.max_torque} Nm"
                        )
                    if st.t_mos > TEMP_MAX:
                        raise Stop(f"{NAMES[mid]} at {st.t_mos:.0f} C")
                time.sleep(max(0.0, period - (time.monotonic() - cycle_t0)))
        outcome = "complete"
    except Stop as e:
        outcome = f"stopped: {e}"
    except KeyboardInterrupt:
        outcome = "stopped: Ctrl+C"
    finally:
        release(ms)
    return p0, rows, segtab, outcome, wall_start


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--out", default=os.path.expanduser("~/sysid"))
    parser.add_argument("--limits", default=LIMITS)
    parser.add_argument(
        "--rate", type=float, default=100.0, help="command/feedback cycles per second"
    )
    parser.add_argument("--step", type=float, default=0.15, help="step amplitude, rad")
    parser.add_argument("--chirp", type=float, default=0.08, help="chirp amplitude, rad")
    parser.add_argument("--max-torque", type=float, default=4.5, help="Nm; RS05 peak is 5.5")
    parser.add_argument(
        "--plan", action="store_true", help="print the excitation from the live pose and exit"
    )
    parser.add_argument("--fake", action="store_true", help="simulated motors, no bus")
    parser.add_argument("--yes", action="store_true", help="skip the typed GO confirmation")
    args = parser.parse_args(argv)
    if not 20 <= args.rate <= 400 or not 0 < args.step <= 0.3 or not 0 < args.chirp <= 0.15:
        sys.exit("--rate 20..400 Hz, --step <= 0.3 rad, --chirp <= 0.15 rad")
    limits = load_limits(args.limits)
    signal.signal(signal.SIGTERM, lambda *a: (_ for _ in ()).throw(KeyboardInterrupt()))

    def run(ms, bus):
        p0 = {mid: m.robstride_get_param_f32(MECH_POS, 200) for mid, m in ms.items()}
        segs = plan(p0, limits, args.step, args.chirp)
        total = 0.5 + sum(s[2] for s in segs)
        print(
            "start pose (motor frame, rad): " + ", ".join(f"{NAMES[m]} {p0[m]:+.3f}" for m in IDS)
        )
        for name, _, seconds, f in segs:
            print(f"  {name:48s} {seconds:4.1f} s")
        print(f"total {total:.1f} s at {args.rate:g} Hz; gains kp {KP} Nm/rad, kd {KD}")
        if args.plan:
            return
        if not args.fake and not args.yes:
            print("The leg must hang free with a hand near the power. Ctrl+C stops at any time.")
            if input("Type GO to move the motors: ").strip() != "GO":
                print("not started")
                return
        t_wall = time.strftime("%Y%m%dT%H%M%S")
        p0, rows, segtab, outcome, wall_start = record(ms, limits, args)
        os.makedirs(args.out, exist_ok=True)
        stem = os.path.join(args.out, f"leg_{t_wall}{'_fake' if args.fake else ''}")
        cols = (
            "t_send",
            "t_read",
            "motor",
            "segment",
            "target",
            "kp",
            "kd",
            "pos",
            "vel",
            "torque",
            "t_mos",
            "fresh",
        )
        data = np.array(rows, dtype=np.float64).reshape(-1, len(cols))
        np.savez_compressed(stem + ".npz", **{c: data[:, i] for i, c in enumerate(cols)})
        fresh = data[:, 11] > 0
        meta = {
            "schema": "hexapod_leg_sysid_recording_v1",
            "started": t_wall,
            "outcome": outcome,
            "fake": bool(args.fake),
            "bus": bus,
            "motor_ids": list(IDS),
            "joint_names": NAMES,
            "frame": "RobStride motor frame (mechPos 0x7019), rad",
            "gains": {"kp": KP, "kd": KD, "vel_target": 0.0, "feedforward": 0.0},
            "control_dt_s": CONTROL_DT,
            "slew_rad": SLEW,
            "rate_hz": args.rate,
            "start_pose": p0,
            "limits": limits,
            "segments": segtab,
            "wall_clock_at_t0": wall_start,
            "rows": int(len(data)),
            "fresh_fraction": float(fresh.mean()) if len(data) else 0.0,
            "cycle_s_median": float(np.median(np.diff(data[data[:, 2] == IDS[0], 0])))
            if len(data) > 9
            else None,
            "read_after_send_s_median": float(np.median(data[:, 1] - data[:, 0]))
            if len(data)
            else None,
            "columns": cols,
        }
        with open(stem + ".json", "w") as f:
            json.dump(meta, f, indent=1, default=str)
        print(f"{outcome}: {len(data)} rows -> {stem}.npz / .json")

    if args.fake:
        ms = {mid: FakeMotor(mid, p) for mid, p in zip(IDS, (0.988, 4.9, 2.4))}
        run(ms, "fake")
        return
    from motorbridge import Controller

    with Controller("vcan0") as ctrl:
        ms = {mid: ctrl.add_robstride_motor(mid, HOST, "rs-05") for mid in IDS}
        release(ms)
        run(ms, "vcan0 via rs_bridge (CH340 USB-CAN, 921600 baud)")


if __name__ == "__main__":
    main()
