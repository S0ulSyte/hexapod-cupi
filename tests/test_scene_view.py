"""Check the viewer's scene files and the fixtures its node test reads.

``viewer/test/fixtures/sample_terrain.json`` holds terrain heights that ``locomotion.surrogate.terrain`` computes
at sample points, for ``viewer/src/scene/terrain.js`` to match; ``sample_scene_view.json`` is the scene file of
the two-robot scene below. After a deliberate format change, rewrite both with
``uv run python -c "from tests.test_scene_view import write_fixtures; write_fixtures()"``.
"""

import base64
import gzip
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from locomotion.surrogate.scene import run_scene
from locomotion.surrogate.terrain import Terrain, feature_height
from viewer import scene_view

ROOT = Path(__file__).resolve().parents[1]
FIXTURES = ROOT / "viewer/test/fixtures"
TERRAIN = {
    "size_m": [3, 2],
    "center_m": [0.5, 0],
    "resolution_m": 0.02,
    "features": [
        {"type": "box", "center": [1, -0.4], "size": [0.6, 0.3], "height": 0.03, "heading_deg": 30},
        {"type": "ramp", "center": [0, 0.5], "size": [0.8, 0.6], "rise": 0.05, "heading_deg": -20},
        {
            "type": "steps",
            "center": [-0.5, -0.3],
            "size": [0.6, 0.4],
            "rise": 0.04,
            "count": 4,
            "heading_deg": 90,
        },
        {"type": "bump", "center": [1.4, 0.4], "radius": 0.3, "height": 0.025},
        {"type": "noise", "amplitude": 0.006, "wavelength": 0.23, "seed": 12345},
    ],
}
SCENE = {
    "schema": "hexapod_scene_v1",
    "name": "fixture",
    "seconds": 0.2,
    "terrain": {
        "size_m": [1.6, 1.4],
        "resolution_m": 0.1,
        "features": [{"type": "box", "center": [0.3, 0.2], "size": [0.8, 0.7], "height": 0.02}],
    },
    "robots": [
        {
            "name": "a",
            "policy": "stand",
            "start": {"forward_m": 0.3, "left_m": 0.2, "heading_deg": 0},
        },
        {
            "name": "b",
            "policy": "stand",
            "start": {"forward_m": -0.4, "left_m": -0.3, "heading_deg": 90},
        },
    ],
}


def terrain_samples():
    rng = np.random.default_rng(20260917)
    points = rng.uniform([-1.2, -1.1], [2.2, 1.1], size=(200, 2)).round(4).tolist()
    return {
        "terrain": TERRAIN,
        "points": points,
        "features": [
            [round(float(feature_height(f, p[0], p[1])), 9) for p in points]
            for f in TERRAIN["features"]
        ],
    }


def scene_file(folder):
    path = Path(folder, "scene.json")
    path.write_text(json.dumps(SCENE))
    run_scene(path, Path(folder, "out"))
    return scene_view.build(Path(folder, "out"))


def write_fixtures():
    (FIXTURES / "sample_terrain.json").write_text(json.dumps(terrain_samples()) + "\n")
    with tempfile.TemporaryDirectory() as folder:
        value = scene_file(folder)
    (FIXTURES / "sample_scene_view.json").write_text(
        json.dumps(value, separators=(",", ":")) + "\n"
    )


class SceneViewTests(unittest.TestCase):
    def test_the_terrain_fixture_matches_the_terrain_module(self):
        fixture = json.loads((FIXTURES / "sample_terrain.json").read_text())
        self.assertEqual(fixture, json.loads(json.dumps(terrain_samples())))

    def test_a_scene_file_holds_every_robot_and_the_simulated_terrain(self):
        with tempfile.TemporaryDirectory() as folder:
            value = scene_file(folder)
        self.assertEqual(value["schema"], scene_view.SCHEMA)
        self.assertEqual(value["scene"]["name"], "fixture")
        self.assertEqual([c["name"] for c in value["cases"]], ["a", "b"])
        a, b = value["cases"]
        self.assertEqual((a["frames"], a["controls"]), (5, 10))
        # Each ghost starts at its robot's start: b stands 0.4 m back and 0.3 m right, facing left.
        np.testing.assert_allclose(a["ghost_mm"][0], [300, 200], atol=0.1)
        np.testing.assert_allclose(b["ghost_mm"][0], [-400, -300], atol=0.1)
        self.assertAlmostEqual(b["commanded_heading_deg"][0], 90, places=1)
        blob = gzip.decompress(base64.b64decode(value["data"]))
        t = value["terrain"]
        cells = t["ny"] * t["nx"]
        grid = np.frombuffer(blob[t["base"] : t["base"] + 2 * cells], dtype="<i2").reshape(
            t["ny"], t["nx"]
        )
        expected = np.round(Terrain(SCENE["terrain"]).heights * 1000)
        np.testing.assert_array_equal(grid, expected)
        # Robot a stands on the 20 mm box and b on the floor: their toe points sit 20 mm apart in height.
        points = np.frombuffer(blob[: value["contacts_base"]], dtype="<i2").reshape(-1, 6, 4, 3)
        toes_a, toes_b = points[a["frames"] - 1, :, 3, 2], points[-1, :, 3, 2]
        self.assertTrue(np.all(np.abs(toes_a - toes_b - 20) <= 2))
        # Each robot's body pose (world x, y, z and quaternion) and joint angles follow, every second control.
        pose = np.frombuffer(blob, "<f4", 5 * 7, value["pose_base"] + b["pose_offset"]).reshape(
            5, 7
        )
        np.testing.assert_allclose(pose[0, :2], [-0.3, 0.4], atol=0.01)
        self.assertAlmostEqual(2 * np.arctan2(pose[0, 5], pose[0, 6]), np.pi / 2, places=2)
        angles = np.frombuffer(
            blob, "<f4", 5 * 18, value["joints_base"] + a["joints_offset"]
        ).reshape(5, 18)
        stance = json.loads((ROOT / "robot/hexapod_mkii_updated_v1/stance.json").read_text())
        np.testing.assert_allclose(angles[-1], stance["nominal_joint_position_rad"], atol=0.05)
        # The node test's fixture is this file; its physics may differ in the last digits between builds.
        fixture = json.loads((FIXTURES / "sample_scene_view.json").read_text())
        for key in ("schema", "scene", "terrain", "contacts_base", "frame_seconds"):
            self.assertEqual(fixture[key], value[key])
        self.assertEqual([(c["name"], c["frames"]) for c in fixture["cases"]], [("a", 5), ("b", 5)])


if __name__ == "__main__":
    unittest.main()
