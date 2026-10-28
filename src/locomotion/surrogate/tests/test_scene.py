"""Scenes on the surrogate: terrain heights, the height-field floor, start poses and the scene runner."""

import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np
import torch

from locomotion.surrogate.env import ContactModel, SurrogateConfig, SurrogateEnv
from locomotion.surrogate.scene import (
    HOLD_STAND,
    HOLD_UPSIDE_DOWN,
    POLICY,
    STAND_UP_STEP_GROUPS,
    UPSIDE_DOWN,
    UpsideDown,
    plan,
    run_scene,
    schedule,
    stand_up_keyframes,
)
from locomotion.surrogate.terrain import Terrain, feature_height, validate


def heights(feature, points):
    return [round(float(feature_height(feature, f, left)), 6) for f, left in points]


class TerrainTests(unittest.TestCase):
    def test_each_feature_has_its_shape(self):
        box = {"type": "box", "center": [1, 0], "size": [0.4, 0.2], "height": 0.03}
        self.assertEqual(
            heights(box, [(1, 0), (1.19, 0.09), (1.21, 0), (1, 0.11)]), [0.03, 0.03, 0, 0]
        )
        # Turned 90 degrees, the box is 0.4 m along left and 0.2 m along forward.
        box["heading_deg"] = 90
        self.assertEqual(heights(box, [(1, 0.19), (1.09, 0), (1.11, 0)]), [0.03, 0.03, 0])
        ramp = {"type": "ramp", "center": [0, 0], "size": [1, 1], "rise": 0.1}
        self.assertEqual(
            heights(ramp, [(-0.5, 0), (0, 0), (0.25, 0), (0.6, 0)]), [0, 0.05, 0.075, 0]
        )
        steps = {"type": "steps", "center": [0, 0], "size": [0.9, 1], "rise": 0.09, "count": 3}
        self.assertEqual(
            heights(steps, [(-0.4, 0), (-0.1, 0), (0.1, 0), (0.44, 0)]), [0.03, 0.06, 0.06, 0.09]
        )
        bump = {"type": "bump", "center": [0, 0], "radius": 0.2, "height": 0.02}
        self.assertEqual(heights(bump, [(0, 0), (0.1, 0), (0, 0.2)]), [0.02, 0.01, 0])
        hill = {"type": "hill", "center": [1, 0], "radius": 1.5, "top_radius": 0.5, "height": 0.2}
        self.assertEqual(
            heights(hill, [(1, 0), (1, 0.5), (2, 0), (1, -1.25), (2.6, 0)]),
            [0.2, 0.2, 0.1, 0.05, 0],
        )
        noise = {"type": "noise", "amplitude": 0.01, "wavelength": 0.25, "seed": 3}
        grid = feature_height(noise, *np.meshgrid(np.linspace(-2, 2, 81), np.linspace(-2, 2, 81)))
        self.assertLessEqual(float(np.abs(grid).max()), 0.01)
        self.assertGreater(float(np.ptp(grid)), 0.01)
        self.assertTrue(
            np.array_equal(
                grid,
                feature_height(noise, *np.meshgrid(np.linspace(-2, 2, 81), np.linspace(-2, 2, 81))),
            )
        )

    def test_terrains_are_checked(self):
        good = {"size_m": [2, 2], "features": []}
        validate(good)
        for bad in (
            {"size_m": [0.1, 2]},
            {"size_m": [2, 2], "resolution_m": 0.5},
            {"size_m": [2, 2], "features": [{"type": "crater"}]},
            {
                "size_m": [2, 2],
                "features": [
                    {"type": "hill", "center": [0, 0], "radius": 1, "top_radius": 1, "height": 0.1}
                ],
            },
            {"size_m": [2, 2], "features": [{"type": "box", "center": [0, 0], "size": [1, 1]}]},
            {
                "size_m": [2, 2],
                "features": [
                    {"type": "steps", "center": [0, 0], "size": [1, 1], "rise": 0.1, "count": 1.5}
                ],
            },
            {
                "size_m": [2, 2],
                "features": [{"type": "noise", "amplitude": 0.01, "wavelength": 0.2, "seed": -1}],
            },
        ):
            with self.assertRaises(ValueError):
                validate(bad)

    def test_the_grid_lies_in_mujoco_world_axes(self):
        # A box 1 m forward and 0.5 m left sits at world x = 0.5, y = -1.
        t = Terrain(
            {
                "size_m": [4, 2],
                "resolution_m": 0.05,
                "features": [
                    {"type": "box", "center": [1, 0.5], "size": [0.2, 0.2], "height": 0.04}
                ],
            }
        )
        self.assertEqual((t.ny, t.nx), (81, 41))
        row, column = np.argwhere(t.heights > 0).mean(0)
        self.assertAlmostEqual(t.ys[int(round(row))], -1.0, places=6)
        self.assertAlmostEqual(t.xs[int(round(column))], 0.5, places=6)
        self.assertAlmostEqual(t.height_at(1, 0.5), 0.04)
        self.assertAlmostEqual(t.highest_under(1.25, 0.5, 0.25), 0.04)
        self.assertEqual(t.highest_under(1.5, 0.5, 0.25), 0.0)
        self.assertTrue(Terrain({"size_m": [2, 2]}).flat)


class SceneTests(unittest.TestCase):
    def test_commands_run_their_seconds_then_stand(self):
        robot = {
            "name": "r",
            "commands": [
                {"seconds": 0.04, "command": [0.1, 0, 0]},
                {"seconds": 0.02, "command": [0, 0, 0.2]},
            ],
        }
        np.testing.assert_allclose(
            schedule(robot, 5), [[0.1, 0, 0], [0.1, 0, 0], [0, 0, 0.2], [0, 0, 0], [0, 0, 0]]
        )
        with self.assertRaises(ValueError):
            schedule({"name": "r", "commands": [{"seconds": 1, "command": [0.1, 0]}]}, 5)

    def test_a_robot_stands_on_a_height_field_box_at_its_start(self):
        terrain = Terrain(
            {
                "size_m": [2, 2],
                "resolution_m": 0.02,
                "features": [
                    {"type": "box", "center": [0.3, 0.2], "size": [0.8, 0.8], "height": 0.04}
                ],
            }
        )
        with self.assertRaises(ValueError):
            SurrogateEnv(SurrogateConfig(num_envs=1), threads=0, terrain=terrain)
        tops = []
        for ground, start in ((None, None), (terrain, (0.2, -0.3, 0.04, math.pi / 2))):
            env = SurrogateEnv(
                SurrogateConfig(num_envs=1, episode_seconds=20.0),
                threads=0,
                contact=ContactModel(touchdown_overshoot=False),
                terrain=ground,
                start=start,
            )
            env.commands.zero_()
            env.reset()
            for _ in range(100):
                env.step(torch.zeros(1, 18))
            root = env.current["root"][0]
            force = env.telemetry["tibia_floor_force_world_n"][0]
            # All six feet carry the robot's weight through the floor's contact sensors.
            self.assertEqual(int((torch.linalg.vector_norm(force, dim=-1) > 1).sum()), 6)
            tops.append(float(root[2]))
            if start is not None:
                self.assertAlmostEqual(float(root[0]), 0.2, delta=0.01)
                self.assertAlmostEqual(float(root[1]), -0.3, delta=0.01)
                self.assertAlmostEqual(
                    2 * math.atan2(float(root[5]), float(root[6])), math.pi / 2, delta=0.02
                )
            env.close()
        self.assertAlmostEqual(tops[1] - tops[0], 0.04, delta=0.002)

    def test_the_runner_writes_each_robots_trace_on_the_terrain(self):
        scene = {
            "schema": "hexapod_scene_v1",
            "name": "two standers",
            "seconds": 0.4,
            "terrain": {
                "size_m": [3, 3],
                "resolution_m": 0.05,
                "features": [{"type": "box", "center": [1, 0], "size": [0.8, 0.8], "height": 0.03}],
            },
            "robots": [
                {
                    "name": "on the box",
                    "policy": "stand",
                    "start": {"forward_m": 1, "left_m": 0, "heading_deg": 0},
                },
                {
                    "name": "beside it",
                    "policy": "stand",
                    "start": {"forward_m": -0.5, "left_m": 0.6, "heading_deg": 90},
                },
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            path = Path(folder, "scene.json")
            path.write_text(json.dumps(scene))
            near_edge = {**scene["robots"][1], "start": {"forward_m": 1.3, "left_m": 0}}
            for bad in (
                {"robots": []},
                {"robots": [scene["robots"][0]] * 2},
                {"schema": "other"},
                {"robots": [near_edge]},
            ):
                Path(folder, "bad.json").write_text(json.dumps({**scene, **bad}))
                with self.assertRaises(ValueError):
                    run_scene(Path(folder, "bad.json"), Path(folder, "bad_out"))
            summary = run_scene(path, Path(folder, "out"))
            self.assertEqual([r["name"] for r in summary["robots"]], ["on the box", "beside it"])
            self.assertEqual(
                summary["robots"][0]["start_world_xy_ground_yaw"][:3], [0.0, -1.0, 0.03]
            )
            first = np.load(Path(folder, "out", "robot_0.npz"))
            self.assertEqual(first["root_pose_xyzw"].shape, (20, 7))
            self.assertGreater(first["root_pose_xyzw"][-1, 2], 0.11)
            self.assertTrue(first["toe_contact"][-1].all())
            self.assertFalse(first["fallen"].any())
            grid = np.load(Path(folder, "out", "terrain.npz"))
            self.assertEqual(grid["heights"].shape, (61, 61))


class UpsideDownTests(unittest.TestCase):
    def setUp(self):
        self.env = SurrogateEnv(SurrogateConfig(num_envs=1), threads=0)

    def test_the_policy_reads_a_robot_on_its_back_as_an_upright_one(self):
        for flip in ("roll", "pitch"):
            view = UpsideDown(self.env.neutral, flip)
            # A target the policy asks for lands where the policy then reads it.
            asked = torch.linspace(-1, 1, 18)[None]
            held = view.neutral + 0.35 * view.action(asked)
            self.assertTrue(torch.allclose(view.joints(held - view.neutral) / 0.35, asked))
            # On its back, gravity points out of the body's top; the policy feels it below.
            observation = torch.zeros(1, 231)
            observation[:, 5] = 1.0
            self.assertEqual(view.observation(observation)[0, 3:6].tolist(), [0.0, 0.0, -1.0])
            # The turned root stands upright: its body z is the world's up.
            on_back = {"roll": [0.0, 1.0, 0.0, 0.0], "pitch": [1.0, 0.0, 0.0, 0.0]}[flip]
            root = torch.tensor([[0.0, 0.0, 0.2, *on_back]])
            x, y, z, w = view.root(root)[0, 3:7].tolist()
            self.assertAlmostEqual(1 - 2 * (x * x + y * y), 1.0, places=6)
            # The legs swap sides (roll) or ends (pitch): the policy's lf leg is the robot's rf or lr.
            self.assertEqual(int(view.index[0]), {"roll": 9, "pitch": 6}[flip])

    def test_entries_hold_poses_turn_over_and_push(self):
        robot = {
            "name": "r",
            "commands": [
                {"seconds": 0.04, "hold": "stand", "push": {"force": [80, 0, 0]}},
                {"seconds": 0.04, "hold": "upside_down"},
                {"seconds": 0.04, "command": [0.1, 0, 0], "upside_down": "pitch"},
            ],
        }
        steps = plan(robot, 8)
        self.assertEqual(
            steps["phase"].tolist(),
            [HOLD_STAND] * 2 + [HOLD_UPSIDE_DOWN] * 2 + [UPSIDE_DOWN] * 2 + [POLICY] * 2,
        )
        self.assertEqual(steps["flip"], [None] * 4 + ["pitch"] * 2 + [None] * 2)
        self.assertEqual(steps["push"][:, 0].tolist(), [80, 80, 0, 0, 0, 0, 0, 0])
        np.testing.assert_allclose(steps["elapsed"][:4], [0.02, 0.04, 0.02, 0.04])
        self.assertEqual(schedule(robot, 8)[4:6, 0].tolist(), [0.1, 0.1])
        for bad in (
            {"seconds": 1, "hold": "sit"},
            {"seconds": 1, "command": [0, 0, 0], "upside_down": "yaw"},
            {"seconds": 1, "command": [0, 0, 0], "push": {"force": [1, 2]}},
            {"seconds": 1, "command": [0, 0, 0], "push": {"spin": [1, 2, 3]}},
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                schedule({"name": "r", "commands": [bad]}, 50)

    def test_getting_up_ends_with_every_leg_in_the_upside_down_standing_pose(self):
        standing = (0.0, 2.79)
        legs = stand_up_keyframes((-0.3, 0.4), standing)
        for keys in legs:
            times = [at for at, _ in keys]
            self.assertEqual(times, sorted(times))
            self.assertEqual(keys[-1][1], standing)
            # No move outruns the 2 rad/s target slew.
            for (a, before), (b, after) in zip(keys, keys[1:]):
                travel = max(abs(u - v) for u, v in zip(before, after))
                self.assertGreaterEqual(b - a + 1e-9, travel / 2.0)
        self.assertLess(max(keys[-1][0] for keys in legs), 5.0)
        # Every leg swings to its grip and pushes up at once.
        self.assertEqual(len({tuple(keys[:3]) for keys in legs}), 1)
        # The outer legs step into the standing pose together, then the middle two.
        outer, middle = STAND_UP_STEP_GROUPS
        self.assertEqual(sorted(outer + middle), list(range(6)))
        self.assertEqual(len({tuple(legs[leg][-4:]) for leg in outer}), 1)
        self.assertEqual(legs[middle[0]], legs[middle[1]])
        self.assertGreaterEqual(legs[middle[0]][-4][0], legs[outer[0]][-1][0])

    def test_a_hold_too_short_to_get_up_is_refused(self):
        scene = {
            "schema": "hexapod_scene_v1",
            "name": "short",
            "seconds": 2,
            "terrain": {"size_m": [2, 2], "features": []},
            "robots": [
                {
                    "name": "r",
                    "policy": "stand",
                    "commands": [{"seconds": 2, "hold": "upside_down"}],
                }
            ],
        }
        with tempfile.TemporaryDirectory() as folder:
            Path(folder, "s.json").write_text(json.dumps(scene))
            with self.assertRaises(ValueError):
                run_scene(Path(folder, "s.json"), Path(folder, "out"))

    def test_an_applied_force_lifts_the_body(self):
        self.env.reset()
        start = float(self.env.current["root"][0, 2])
        self.env.applied_wrench = np.array([[0.0, 0.0, 300.0, 0.0, 0.0, 0.0]])
        for _ in range(10):
            self.env.step(torch.zeros(1, 18))
        self.assertGreater(float(self.env.current["root"][0, 2]), start + 0.05)


if __name__ == "__main__":
    unittest.main()
