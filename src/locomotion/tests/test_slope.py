"""Slope inputs: the tilt from the front and rear toes, and the gravity or joint inputs it corrects."""

import json
import math
import unittest
from pathlib import Path

import torch

from locomotion.omniwalk.mdp.slope import SlopeTracker

ROBOT = Path(__file__).resolve().parents[3] / "robot/hexapod_mkii_updated_v1"
MODEL = json.loads((ROBOT / "model_rs05_mass_corrected.json").read_text())
STANCE = json.loads((ROBOT / "stance.json").read_text())
IDENTITY = torch.tensor([[0.0, 0.0, 0.0, 1.0]])
# Toes lf, lm, lr, rf, rm, rr on flat ground; forward is world -y.
TOES = torch.tensor(
    [
        [
            [x, y, 0.01]
            for x, y in (
                (0.1, -0.15),
                (0.2, 0),
                (0.1, 0.15),
                (-0.1, -0.15),
                (-0.2, 0),
                (-0.1, 0.15),
            )
        ]
    ],
    dtype=torch.float64,
)
FRONT_AND_REAR = torch.tensor([[True, False, True, True, False, True]])


def settled(mode, rise=0.03):
    """A tracker whose front toes stand ``rise`` metres above its rear ones, 0.3 m behind."""
    slope = SlopeTracker(1, MODEL["joints"], mode)
    toes = TOES.clone()
    toes[0, [0, 3], 2] += rise
    for _ in range(60):
        slope.update(toes, FRONT_AND_REAR)
    return slope


class SlopeTests(unittest.TestCase):
    def test_the_front_and_rear_toes_give_the_tilt(self):
        with self.assertRaises(ValueError):
            SlopeTracker(1, MODEL["joints"], "knees")
        slope = SlopeTracker(2, MODEL["joints"], "gravity")
        slope.update(TOES.expand(2, -1, -1), torch.ones(2, 6, dtype=torch.bool))
        self.assertEqual(slope.pitch.tolist(), [0.0, 0.0])
        slope = settled("gravity")
        self.assertAlmostEqual(float(slope.pitch[0]), math.atan2(0.03, 0.3), places=6)
        # Without a front and a rear toe down at once, the estimate holds.
        slope.update(TOES * 0, torch.tensor([[True, True, False, False, True, False]]))
        self.assertAlmostEqual(float(slope.pitch[0]), math.atan2(0.03, 0.3), places=6)
        slope.reset([0])
        self.assertEqual(float(slope.pitch[0]), 0.0)

    def test_gravity_relative_to_the_slope(self):
        slope = settled("gravity")
        torch.testing.assert_close(
            SlopeTracker(1, MODEL["joints"], "gravity").values(IDENTITY, None),
            torch.tensor([[0.0, 0.0, -1.0]]),
        )
        # A body pitched nose-up with the ramp feels gravity as on flat ground.
        half = float(slope.pitch[0]) / 2
        nose_up = torch.tensor([[-math.sin(half), 0.0, 0.0, math.cos(half)]])
        torch.testing.assert_close(
            slope.values(nose_up, None), torch.tensor([[0.0, 0.0, -1.0]]), atol=1e-6, rtol=0
        )
        slope.push(IDENTITY, None, rows=[0])
        corrected = slope.apply(torch.zeros(1, 231))
        self.assertEqual(
            corrected[0, 42 * 4 + 3 : 42 * 4 + 6].tolist(), slope.values(IDENTITY, None)[0].tolist()
        )
        self.assertEqual(int((corrected != 0).sum()), 5 * 2)

    def test_the_joint_modes_add_the_tilt_about_each_pitch_axis(self):
        q = torch.tensor([STANCE["nominal_joint_position_rad"]])
        rise = 0.3 * math.tan(math.radians(5))
        femurs = settled("femurs", rise).values(IDENTITY, q)[0]
        degrees = [math.degrees(float(femurs[i])) for i in (1, 4, 7, 10, 13, 16)]
        # Ground rising 5 degrees forward: the front femurs (lf, rf) gain about 3.9 degrees, the rear ones lose as
        # much, and the middle ones, whose axes point forward, almost nothing.
        for front, middle, rear in (
            (degrees[0], degrees[1], degrees[2]),
            (degrees[3], degrees[4], degrees[5]),
        ):
            self.assertAlmostEqual(front, 3.86, delta=0.05)
            self.assertAlmostEqual(rear, -3.86, delta=0.05)
            self.assertLess(abs(middle), 0.1)
        self.assertEqual(float(femurs[0::3].abs().sum() + femurs[2::3].abs().sum()), 0.0)
        joints = settled("joints", rise).values(IDENTITY, q)[0]
        # Each tibia gains the tilt about its own axis, which lies along its femur's.
        torch.testing.assert_close(joints[1::3], femurs[1::3])
        torch.testing.assert_close(joints[2::3], femurs[1::3], atol=1e-6, rtol=0)
        slope = settled("joints", rise)
        slope.push(IDENTITY, q, rows=[0])
        corrected = slope.apply(torch.zeros(1, 231))
        self.assertAlmostEqual(float(corrected[0, 42 * 2 + 6 + 1]), float(joints[1]), places=6)
        self.assertEqual(int((corrected[0, :210] != 0).sum()), 5 * 12)
        # Later controls shift into the history; the newest frame is the last. Turning every coxa 0.5 rad
        # turns the femur and tibia axes, and so their share of the tilt.
        slope.push(IDENTITY, q + torch.tensor([0.5, 0.0, 0.0] * 6))
        self.assertFalse(torch.equal(slope.history[0, 4], slope.history[0, 3]))


if __name__ == "__main__":
    unittest.main()
