"""Leg-stand system identification: the fit recovers known parameters from a synthetic recording."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locomotion.sysid import leg_record
from locomotion.sysid.leg_sim import MOTORS, LegParams, LegSim

LIMITS = {11: (-0.029, 2.227), 12: (3.063, 6.632), 13: (0.965, 4.786)}
TRUTH = LegParams(
    delay_s=0.023,
    sign=np.array([1.0, -1.0, 1.0]),
    offset=np.array([1.1, 4.6, 2.1]),
    kp_scale=np.array([1.0, 0.9, 1.0]),
    kd_scale=np.array([1.4, 1.3, 1.5]),
    tau_max=np.array([5.5, 5.5, 5.5]),
    frictionloss=np.array([0.01, 0.015, 0.01]),
    armature=np.array([4e-4, 4e-4, 4e-4]),
)


def synthetic_recording(directory, params=TRUTH, rate=100.0, seed=0):
    """A recording as leg_record.py writes it, with positions from the simulator under ``params``."""
    p0 = {11: 1.0, 12: 4.7, 13: 2.3}
    segs = leg_record.plan(p0, LIMITS, chirp_s=4.0)
    schedule = [("ramp in", None, 0.5, None)] + segs
    period, rows, table, t = 1.0 / rate, [], [], 0.0
    held = dict(p0)
    for index, (name, joint, seconds, f) in enumerate(schedule):
        table.append(
            {"index": index, "name": name, "joint": joint, "start_s": t, "seconds": seconds}
        )
        s, next_control = 0.0, 0.0
        while s < seconds:
            if s >= next_control:
                next_control += leg_record.CONTROL_DT
                for m in MOTORS:
                    want = p0[m] + (f(s) if (f is not None and m == joint) else 0.0)
                    held[m] += max(-leg_record.SLEW, min(leg_record.SLEW, want - held[m]))
            kp = leg_record.KP * min(1.0, s / seconds) if name == "ramp in" else leg_record.KP
            for k, m in enumerate(MOTORS):
                ts = t + s + 0.001 * k
                rows.append(
                    [ts, ts + 0.0006, m, index, held[m], kp, leg_record.KD[m], 0, 0, 0, 30, 1]
                )
            s += period
        t += seconds
    data = np.array(rows)
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
    stem = Path(directory) / "leg_synthetic"
    meta = {
        "segments": table,
        "limits": {str(m): LIMITS[m] for m in MOTORS},
        "gains": {"kp": leg_record.KP, "kd": leg_record.KD},
        "started": "synthetic",
        "outcome": "complete",
    }
    rec = {"meta": meta, "motors": {}}
    for m in MOTORS:
        r = data[:, 2] == m
        rec["motors"][m] = {c: data[r, i] for i, c in enumerate(cols) if c != "motor"}
    out = LegSim().replay(rec, params, q0_motor=np.array([p0[m] for m in MOTORS]))
    rng = np.random.default_rng(seed)
    for m in MOTORS:
        r = data[:, 2] == m
        data[r, 7] = out[m]["pos"] + rng.normal(0, 4e-4, r.sum())  # about the RS05 feedback quantum
        data[r, 8] = np.gradient(out[m]["pos"], data[r, 1])
        data[r, 9] = out[m]["torque"]
    np.savez_compressed(str(stem) + ".npz", **{c: data[:, i] for i, c in enumerate(cols)})
    Path(str(stem) + ".json").write_text(json.dumps(meta))
    return stem


class LegSysid(unittest.TestCase):
    def test_plan_stays_inside_the_safe_band(self):
        p0 = {11: 1.0, 12: 3.15, 13: 2.3}  # coxa tilt below its safe band: it is skipped
        segs = leg_record.plan(p0, LIMITS)
        self.assertTrue(any(s[0].startswith("skip lf_femur_pitch") for s in segs))
        for name, joint, seconds, f in segs:
            if f is None:
                continue
            lo, hi = leg_record.safe_band(LIMITS, joint)
            values = [p0[joint] + f(t) for t in np.linspace(0, seconds, 200)]
            self.assertGreaterEqual(min(values), lo - 1e-9, name)
            self.assertLessEqual(max(values), hi + 1e-9, name)

    def test_fit_recovers_known_parameters(self):
        from locomotion.sysid.leg_fit import fit

        with tempfile.TemporaryDirectory() as directory:
            stem = synthetic_recording(directory)
            result = fit(stem, Path(directory) / "fit")
            self.assertTrue((Path(directory) / "fit" / "fit.html").exists())
        f = LegParams.from_json(result["fitted"])
        print("\nsynthetic fit:", json.dumps({k: result[k] for k in ("delay_controls", "rms_rad")}))
        self.assertAlmostEqual(f.delay_s, TRUTH.delay_s, delta=0.004)
        self.assertEqual(f.sign[1], TRUTH.sign[1])
        self.assertAlmostEqual(f.offset[1], TRUTH.offset[1], delta=0.03)
        for name, rms in result["rms_rad"]["fitted"].items():
            self.assertLess(rms, math.radians(0.3), name)
            self.assertLess(rms, result["rms_rad"]["nominal"][name], name)


if __name__ == "__main__":
    unittest.main()
