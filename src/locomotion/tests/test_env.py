"""CPU tests of independent motor arithmetic and selected-row reset isolation."""

import unittest
from pathlib import Path

import numpy as np
import torch

from locomotion.amp.features import extract_features
from locomotion.evaluation.capture import diagnostic_servo
from locomotion.omniwalk.adapter import OmniEnv
from locomotion.omniwalk.env_cfg import EnvConfig
from locomotion.omniwalk.mdp.actions import emitted_target, executed_action_feature
from locomotion.omniwalk.mdp.frames import inverse_rotate, rotate
from locomotion.robots.hexapod import JOINT_NAMES, KD, verify_assets
from locomotion.robots.rs05 import motor_force
from locomotion.tests.test_direct_environment import load_isaac


class MotorTests(unittest.TestCase):
    def test_curve_matches_canonical_numpy_recurrence(self):
        rng = np.random.default_rng(809)
        q = rng.uniform(-1, 1, (64, 18)).astype(np.float32)
        dq = rng.uniform(-60, 60, (64, 18)).astype(np.float32)
        dq[0, :7] = np.array([0, 70, 275, 340, 450, 477, 480]) * (2 * np.pi / 60)
        target = rng.uniform(-1, 1, (64, 18)).astype(np.float32)
        expected = diagnostic_servo(
            q, dq, target, np.full(18, 12, np.float32), np.asarray(KD, np.float32)
        )
        actual = motor_force(
            torch.from_numpy(q), torch.from_numpy(dq), torch.from_numpy(target), torch.tensor(KD)
        )
        for a, b in zip(actual, expected):
            np.testing.assert_array_equal(a.numpy(), b)
        self.assertLessEqual(float(actual[1].abs().max()), 1.60001)

    def test_actual_target_stays_inside_double_precision_slew_and_limits(self):
        held = torch.linspace(-0.7, 0.7, 18)[None].repeat(4, 1)
        action = torch.full_like(held, 9.0)
        lower, upper = torch.full((18,), -0.72), torch.full((18,), 0.72)
        neutral = torch.zeros(18)
        for direction in [1.0, -1.0, 1.0, -1.0]:
            target = emitted_target(action * direction, held, neutral, lower, upper)
            self.assertTrue(bool(((target.double() - held.double()).abs() <= 0.04).all()))
            self.assertTrue(bool(((target >= lower) & (target <= upper)).all()))
            held = target

    def test_body_world_roundtrip(self):
        q = torch.tensor([[0.0, 0.0, 2**-0.5, 2**-0.5]])
        v = torch.tensor([[1.0, 0.0, 0.0]])
        torch.testing.assert_close(rotate(q, v), torch.tensor([[0.0, 1.0, 0.0]]), atol=3e-7, rtol=0)
        torch.testing.assert_close(inverse_rotate(q, rotate(q, v)), v, atol=4e-7, rtol=0)

    def test_observed_executed_action_recovers_hidden_limiter_state(self):
        neutral = torch.tensor([0.0, -0.3, 0.4] * 6)
        old_hold = neutral[None].clone()
        raw = torch.ones_like(old_hold)
        target = emitted_target(
            raw, old_hold, neutral, torch.full((18,), -2.0), torch.full((18,), 2.0)
        )
        feature = executed_action_feature(target, neutral, 0.35)
        self.assertTrue(bool((feature < raw).all()))
        torch.testing.assert_close(neutral + 0.35 * feature, target, atol=2e-8, rtol=0)
        # Distinct previous holds can yield distinct targets for identical raw
        # actions. The observation must expose that difference to the policy.
        second_target = emitted_target(
            raw, old_hold + 0.01, neutral, torch.full((18,), -2.0), torch.full((18,), 2.0)
        )
        second_feature = executed_action_feature(second_target, neutral, 0.35)
        self.assertFalse(torch.equal(feature, second_feature))


class ResetTests(unittest.TestCase):
    def test_selected_reset_preserves_other_physics_and_history_rows(self):
        # Exercise real reset against a backend double that enforces selected writes.
        class DirectRLEnv:
            def _reset_idx(self, env_ids):
                self.episode_length_buf[env_ids] = 0

        module = load_isaac(direct_env=DirectRLEnv)

        class Warp:
            uint32 = None

            @staticmethod
            def array(x, **kwargs):
                return torch.as_tensor(x, dtype=torch.long)

        module.wp = Warp()
        isaac = module.IsaacEnv.__new__(module.IsaacEnv)
        isaac.device = "cpu"
        isaac.num_envs = 3
        isaac.torch = lambda x: torch.as_tensor(x, dtype=torch.float32)
        isaac.neutral = torch.tensor([0.0, -0.3, 0.4] * 6)
        isaac.reset_height = 0.1
        isaac.origins = torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0], [4.0, 0.0, 0.0]])
        isaac.inverse_joint_index = torch.arange(18)
        isaac.held = torch.ones(3, 18)
        isaac.previous_action = torch.ones(3, 18)
        isaac.episode_length_buf = torch.tensor([7, 8, 9])
        isaac.history = torch.arange(3 * 5 * 42, dtype=torch.float32).reshape(3, 5, 42)
        isaac.commands = torch.zeros(3, 3)
        isaac.native_array = lambda x: x
        values = {
            "get_root_transforms": torch.tensor([[1.0, 2.0, 0.2, 0.0, 0.0, 0.0, 1.0]]).repeat(3, 1),
            "get_root_velocities": torch.ones(3, 6),
            "get_dof_positions": torch.ones(3, 18),
            "get_dof_velocities": torch.ones(3, 18),
            "get_dof_actuation_forces": torch.ones(3, 18),
        }

        class View:
            def __getattr__(self, name):
                getter = name.replace("set_", "get_")
                if getter in values:
                    return lambda data, ids: values[getter].__setitem__(ids, data[ids])
                return lambda data, ids: None

        isaac.view = View()
        isaac.get = lambda name: values[name].clone()

        class Robot:
            def write_root_link_pose_to_sim_index(self, *, root_pose, env_ids):
                values["get_root_transforms"][env_ids] = root_pose

            def write_root_com_velocity_to_sim_index(self, *, root_velocity, env_ids):
                values["get_root_velocities"][env_ids] = root_velocity

            def write_joint_position_to_sim_index(self, *, position, env_ids):
                values["get_dof_positions"][env_ids] = position

            def write_joint_velocity_to_sim_index(self, *, velocity, env_ids):
                values["get_dof_velocities"][env_ids] = velocity

            def set_joint_position_target_index(self, **kwargs):
                pass

            def set_joint_velocity_target_index(self, **kwargs):
                pass

            def set_joint_effort_target_index(self, **kwargs):
                pass

        isaac.robot = Robot()
        isaac.read_state = lambda: {
            "q": values["get_dof_positions"].clone(),
            "dq": values["get_dof_velocities"].clone(),
            "root": values["get_root_transforms"].clone(),
            "angular": torch.zeros(3, 3),
            "linear": torch.zeros(3, 3),
            "gravity": torch.tensor([[0.0, 0.0, -1.0]]).repeat(3, 1),
            "toe_body": torch.zeros(3, 6, 3),
        }
        env = OmniEnv.__new__(OmniEnv)
        env.isaac = isaac
        env.device = "cpu"
        env.num_envs = 3
        env.cfg = EnvConfig(record_motion_features=True)
        env.native_errors = []
        env.torch, env.neutral, env.reset_height, env.origins = (
            isaac.torch,
            isaac.neutral,
            isaac.reset_height,
            isaac.origins,
        )
        env.held, env.previous_action, env.episode_steps = (
            isaac.held,
            isaac.previous_action,
            isaac.episode_length_buf,
        )
        env.history, env.commands = isaac.history, isaac.commands

        class Sim:
            @staticmethod
            def get_physics_step_count():
                return 17

        env.sim = Sim()
        before = {k: v.clone() for k, v in values.items()}
        history = env.history.clone()
        result = env.reset(torch.tensor([1]))
        for name in values:
            torch.testing.assert_close(values[name][[0, 2]], before[name][[0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(env.history[[0, 2]], history[[0, 2]], rtol=0, atol=0)
        torch.testing.assert_close(env.history[1, 0], env.history[1, -1], rtol=0, atol=0)
        torch.testing.assert_close(env.held[1], env.neutral, rtol=0, atol=0)
        self.assertEqual(env.episode_steps.tolist(), [7, 0, 9])
        self.assertIs(env.current, isaac.current)
        self.assertEqual(result["obs"].shape, (3, 231))
        self.assertEqual(result["critic"].shape, (3, 234))
        self.assertNotIn("amp", result)
        self.assertEqual(extract_features(env.current).shape, (3, 61))
        torch.testing.assert_close(result["critic"][:, :231], result["obs"], rtol=0, atol=0)


class ConfigTests(unittest.TestCase):
    def test_fixed_physics_contract(self):
        self.assertEqual(EnvConfig().control_dt, 0.02)
        for kwargs in (
            {"physics_dt": 0.005},
            {"decimation": 4},
            {"target_slew_rad": 0.05},
            {"num_envs": 1025},
            {"num_envs": 0},
        ):
            with self.assertRaises(ValueError):
                EnvConfig(**kwargs)
        self.assertEqual(len(JOINT_NAMES), 18)

    def test_real_canonical_assets(self):
        repo = Path(__file__).resolve().parents[3]
        model = verify_assets(
            repo / "robot/hexapod_mkii_updated_v1/usd",
            repo / "robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json",
        )
        self.assertEqual(len(model["links"]), 19)


if __name__ == "__main__":
    unittest.main()


class FloorDeclarationTests(unittest.TestCase):
    def test_declared_floor_and_offsets_are_bounded_and_recorded(self):
        from locomotion.omniwalk.env_cfg import FLOORS, EnvConfig

        for floor in FLOORS:
            self.assertEqual(EnvConfig(floor=floor).declaration()["floor"], floor)
        declared = EnvConfig().declaration()
        self.assertEqual(
            (
                declared["floor"],
                declared["floor_contact_offset_m"],
                declared["floor_rest_offset_m"],
            ),
            ("isaaclab_mesh", None, None),
        )
        self.assertFalse(EnvConfig().authors_floor_offsets)
        self.assertTrue(
            EnvConfig(floor_contact_offset_m=0.02, floor_rest_offset_m=0.0).authors_floor_offsets
        )
        with self.assertRaises(ValueError):
            EnvConfig(floor="two_triangle_mesh")
        for contact, rest in (
            (0.02, 0.02),
            (0.0, 0.0),
            (0.2, 0.0),
            (0.02, -0.001),
            (0.02, None),
            (None, 0.0),
        ):
            with self.subTest(contact=contact, rest=rest), self.assertRaises(ValueError):
                EnvConfig(floor_contact_offset_m=contact, floor_rest_offset_m=rest)


class TerrainDeclarationTests(unittest.TestCase):
    def test_a_terrain_spec_is_declared_bounded_and_tied_to_the_mesh_floor(self):
        from locomotion.omniwalk.env_cfg import EnvConfig

        declared = EnvConfig(
            terrain_spec_path="/terrain/mixed_v3.json", terrain_spawn=True
        ).declaration()
        self.assertEqual(declared["terrain_spec_path"], "/terrain/mixed_v3.json")
        self.assertEqual(declared["terrain_mesh_resolution_m"], 0.08)
        self.assertTrue(declared["terrain_spawn"])
        self.assertIsNone(EnvConfig().declaration()["terrain_spec_path"])
        for kwargs in (
            dict(terrain_spec_path="/t.json", floor="isaaclab_ground_plane"),
            dict(terrain_spawn=True),
            dict(terrain_spec_path="/t.json", terrain_mesh_resolution_m=0.01),
            dict(terrain_spec_path="/t.json", terrain_mesh_resolution_m=0.5),
        ):
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                EnvConfig(**kwargs)

    def test_the_terrain_mesh_authors_explicit_collider_offsets(self):
        """Unauthored offsets leave PhysX's auto contact margin, which pairs every nearby body
        with swaths of height-field triangles and multiplies sim.step ~50x at 128 replicas
        (terrain_isaac_probe_004, probe_014). The stock-floor offset declaration stays None, so
        flat runs and the 96-case evaluation keep their physics; the authored terrain values
        bind through locomotion/terrain.py's hash in the physics closure, and the admission
        readback checks every chunk against them."""
        import ast

        from locomotion.terrain import TERRAIN_CONTACT_OFFSET_M, TERRAIN_REST_OFFSET_M

        self.assertEqual((TERRAIN_CONTACT_OFFSET_M, TERRAIN_REST_OFFSET_M), (0.02, 0.0))
        self.assertTrue(0 <= TERRAIN_REST_OFFSET_M < TERRAIN_CONTACT_OFFSET_M <= 0.1)
        tree = ast.parse((Path(__file__).resolve().parents[1] / "omniwalk/env.py").read_text())
        authored = {
            ast.unparse(node.args[0])
            for node in ast.walk(tree)
            if isinstance(node, ast.Call)
            and isinstance(node.func, ast.Attribute)
            and node.func.attr in ("CreateContactOffsetAttr", "CreateRestOffsetAttr")
            and node.args
        }
        self.assertEqual(authored, {"TERRAIN_CONTACT_OFFSET_M", "TERRAIN_REST_OFFSET_M"})
        imported = {
            alias.name
            for node in ast.walk(tree)
            if isinstance(node, ast.ImportFrom) and node.module == "locomotion.terrain"
            for alias in node.names
        }
        self.assertLessEqual({"TERRAIN_CONTACT_OFFSET_M", "TERRAIN_REST_OFFSET_M"}, imported)

    def test_chunked_grid_is_the_same_surface_with_shared_seams(self):
        """Chunk colliders exist so broadphase can cull by chunk (one world-sized mesh pairs
        every body with every triangle on this image, terrain_isaac_probe_015/_016); the union
        must be exactly the grid's surface, cell for cell, with seam vertices shared."""
        from locomotion.terrain import Terrain, chunk_grid

        spec = {
            "size_m": [0.5, 0.7],
            "center_m": [0, 0],
            "resolution_m": 0.05,
            "features": [{"type": "box", "center": [0.05, 0.1], "size": [0.2, 0.2], "height": 0.1}],
        }
        terrain = Terrain(spec)
        self.assertEqual((terrain.ny, terrain.nx), (11, 15))
        chunks = list(chunk_grid(terrain.xs, terrain.ys, terrain.heights, 4))
        self.assertEqual(len(chunks), 12)
        self.assertEqual(sum(len(quads) for _, quads in chunks), (11 - 1) * (15 - 1))
        heights = {}
        cells = set()
        for points, quads in chunks:
            for x, y, z in points:
                key = (round(float(x), 6), round(float(y), 6))
                self.assertEqual(heights.setdefault(key, float(z)), float(z))
            for quad in quads:
                corner = points[quad[0]]
                cells.add((round(float(corner[0]), 6), round(float(corner[1]), 6)))
                square = points[quad][:, :2] - corner[:2]
                self.assertEqual(
                    {(round(float(dx), 6), round(float(dy), 6)) for dx, dy in square},
                    {(0.0, 0.0), (0.05, 0.0), (0.05, 0.05), (0.0, 0.05)},
                )
        self.assertEqual(len(cells), (11 - 1) * (15 - 1))
        for (x, y), z in heights.items():
            i = int(round((x - terrain.xs[0]) / 0.05))
            j = int(round((y - terrain.ys[0]) / 0.05))
            self.assertAlmostEqual(z, float(terrain.heights[j, i]), places=5)
