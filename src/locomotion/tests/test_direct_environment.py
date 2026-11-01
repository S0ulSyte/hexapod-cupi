"""Check CPU adapter contracts, without claiming Isaac Lab native parity."""

import ast
import importlib
import sys
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from locomotion.omniwalk.adapter import OmniEnv, reset_rows
from locomotion.omniwalk.mdp.actions import emitted_target, executed_action_feature
from locomotion.omniwalk.mdp.observations import observations, proprio
from locomotion.robots.hexapod import BODY_NAMES, JOINT_NAMES, LEGS

BACKEND = Path(__file__).resolve().parents[1] / "omniwalk"
ISAAC_MODULES = (
    "warp",
    "pxr",
    "omni",
    "omni.physx",
    "isaaclab",
    "isaaclab.envs",
    "isaaclab.sim",
    "isaaclab.sim.schemas",
    "isaaclab.sim.utils",
    "isaaclab_physx",
    "isaaclab_physx.assets",
    "isaaclab_physx.assets.articulation",
    "isaaclab_physx.sim",
    "isaaclab_physx.sim.spawners",
    "isaaclab_physx.sim.spawners.materials",
)


def load_isaac(direct_env=object, articulation=object):
    """Import the Direct environment module (omniwalk/env.py) over stand-ins for Isaac Lab, PhysX, USD and Warp, which this CPU host lacks."""
    stubs = {name: mock.MagicMock(name=name) for name in ISAAC_MODULES}
    stubs["isaaclab.envs"].DirectRLEnv = direct_env
    stubs["isaaclab_physx.assets.articulation"].Articulation = articulation
    package = importlib.import_module("locomotion.omniwalk")
    saved = {name: sys.modules.get(name) for name in (*ISAAC_MODULES, "locomotion.omniwalk.env")}
    bound = package.__dict__.get("isaac")
    sys.modules.update(stubs)
    sys.modules.pop("locomotion.omniwalk.env", None)
    try:
        return importlib.import_module("locomotion.omniwalk.env")
    finally:
        for name, module in saved.items():
            if module is None:
                sys.modules.pop(name, None)
            else:
                sys.modules[name] = module
        if bound is None:
            package.__dict__.pop("isaac", None)
        else:
            package.isaac = bound


def class_node(path, name):
    return next(
        n
        for n in ast.parse(path.read_text()).body
        if isinstance(n, ast.ClassDef) and n.name == name
    )


class DirectEnvironmentContractTests(unittest.TestCase):
    def test_selected_rows_preserve_order_and_empty_selection(self):
        self.assertEqual(reset_rows([3, 1], 4, "cpu").tolist(), [3, 1])
        self.assertEqual(reset_rows([], 4, "cpu").tolist(), [])
        self.assertEqual(reset_rows(None, 4, "cpu").tolist(), [0, 1, 2, 3])
        for rows in [[1, 1], [-1], [4], [[1]]]:
            with self.assertRaises(ValueError):
                reset_rows(rows, 4, "cpu")

    def test_candidate_reuses_current_frames_history_and_observation_order(self):
        # Exercise the shared tensor path with distinct values in each field.
        history = torch.arange(2 * 5 * 42, dtype=torch.float32).reshape(2, 5, 42)
        commands = torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]])
        previous_action = torch.arange(36, dtype=torch.float32).reshape(2, 18)
        state = {"linear": torch.tensor([[2.0, -1.0, 3.0], [5.0, -4.0, 6.0]])}
        obs = observations(history, commands, previous_action, state)
        self.assertEqual(set(obs), {"obs", "critic"})
        self.assertEqual(obs["obs"].shape, (2, 231))
        self.assertEqual(obs["critic"].shape, (2, 234))
        self.assertTrue(torch.equal(obs["obs"][:, :210], history.flatten(1)))
        self.assertTrue(torch.equal(obs["obs"][:, 210:213], commands))
        self.assertTrue(torch.equal(obs["obs"][:, 213:], previous_action))
        self.assertTrue(
            torch.equal(obs["critic"][:, -3:], torch.tensor([[1.0, 2.0, 3.0], [4.0, 5.0, 6.0]]))
        )
        # One frame: navigation body rates, navigation gravity, joint offsets, joint rates.
        frame_state = {
            "angular": torch.tensor([[1.0, 2.0, 3.0]]),
            "gravity": torch.tensor([[4.0, 5.0, 6.0]]),
            "q": torch.full((1, 18), 0.5),
            "dq": torch.full((1, 18), 2.0),
        }
        neutral = torch.arange(18, dtype=torch.float32) / 100
        frame = proprio(frame_state, neutral)
        self.assertEqual(frame.shape, (1, 42))
        self.assertTrue(torch.equal(frame[0, :6], torch.tensor([-2.0, 1.0, 3.0, -5.0, 4.0, 6.0])))
        self.assertTrue(torch.equal(frame[0, 6:24], 0.5 - neutral))
        self.assertTrue(torch.equal(frame[0, 24:], torch.full((18,), 2.0)))

    def test_stock_loop_owns_steps_and_task_owns_resets(self):
        isaac = class_node(BACKEND / "env.py", "IsaacEnv")
        self.assertEqual(ast.unparse(isaac.bases[0]), "DirectRLEnv")
        self.assertNotIn("step", {n.name for n in isaac.body if isinstance(n, ast.FunctionDef)})
        calls = [
            ast.unparse(n.func)
            for n in ast.walk(ast.parse((BACKEND / "env.py").read_text()))
            if isinstance(n, ast.Call)
        ]
        self.assertNotIn("self.sim.step", calls)
        self.assertNotIn("self.scene.update", calls)
        self.assertIn("super()._reset_idx", calls)
        self.assertIn("self._reset_idx", calls)
        self.assertIn("self.robot.write_joint_position_to_sim_index", calls)
        adapter = ast.parse((BACKEND / "adapter.py").read_text())
        adapter_calls = [ast.unparse(n.func) for n in ast.walk(adapter) if isinstance(n, ast.Call)]
        self.assertIn("self.isaac.step", adapter_calls)
        self.assertIn("self.isaac.reset_selected", adapter_calls)

    def test_episode_clock_reuses_the_stock_buffer(self):
        def writes(name, target):
            return [
                n
                for n in ast.walk(ast.parse((BACKEND / name).read_text()))
                if isinstance(n, (ast.Assign, ast.AugAssign))
                and target
                in {
                    ast.unparse(t) for t in (n.targets if isinstance(n, ast.Assign) else [n.target])
                }
            ]

        bindings = writes("adapter.py", "self.episode_steps")
        self.assertEqual(len(bindings), 1)
        self.assertEqual(ast.unparse(bindings[0].value), "isaac.episode_length_buf")
        self.assertEqual(writes("env.py", "self.episode_length_buf"), [])

    def test_stock_collision_filter_keeps_the_shared_ground(self):
        setup = next(
            n
            for n in class_node(BACKEND / "env.py", "IsaacEnv").body
            if isinstance(n, ast.FunctionDef) and n.name == "_setup_scene"
        )
        filters = [
            n
            for n in ast.walk(setup)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "self.scene.filter_collisions"
        ]
        self.assertEqual(len(filters), 1)
        self.assertEqual(ast.literal_eval(filters[0].keywords[0].value), ["/Ground"])
        ground = next(
            n
            for n in ast.walk(setup)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "GroundPlaneCfg"
        )
        self.assertEqual(ast.literal_eval(ground.keywords[0].value), (80.0, 80.0))
        self.assertLess(ground.lineno, filters[0].lineno)
        material = next(
            n
            for n in ast.walk(setup)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "RigidBodyMaterialCfg"
        )
        self.assertEqual(
            {k.arg: ast.literal_eval(k.value) for k in material.keywords},
            {
                "static_friction": 1.0,
                "dynamic_friction": 1.0,
                "restitution": 0.0,
                "friction_combine_mode": "multiply",
                "restitution_combine_mode": "multiply",
            },
        )
        config = ast.parse((BACKEND / "env_cfg.py").read_text())
        scenes = [
            n
            for n in ast.walk(config)
            if isinstance(n, ast.Call) and ast.unparse(n.func) == "InteractiveSceneCfg"
        ]
        self.assertEqual(len(scenes), 2)
        for scene in scenes:
            values = {
                k.arg: ast.literal_eval(k.value)
                for k in scene.keywords
                if k.arg in ("filter_collisions", "replicate_physics")
            }
            self.assertEqual(values, {"replicate_physics": False, "filter_collisions": True})

    def test_force_readback_brackets_physics_at_articulation_hooks(self):
        # Run the production hooks against a call recorder. This verifies
        # hook order, not PhysX contact or torque behavior.
        events = []

        class Articulation:
            def write_data_to_sim(self):
                events.append("submit")

            def update(self, dt):
                events.append("update")

        isaac = load_isaac(articulation=Articulation)
        env = isaac.IsaacEnv.__new__(isaac.IsaacEnv)

        def read_force(name):
            self.assertEqual(name, "get_dof_actuation_forces")
            events.append("read_submitted")
            return torch.tensor([[0.5, 1.0]])

        def capture():
            events.append("capture")
            self.assertTrue(torch.equal(env.native_pre, torch.tensor([[1.0, 0.5]])))

        env.control_active, env.capture, env.joint_index = True, object(), torch.tensor([1, 0])
        env.get, env.collect_substep = read_force, capture
        robot = isaac.CaptureArticulation()
        robot.env = env
        robot.write_data_to_sim()
        events.append("physics")
        robot.update(0.0025)
        self.assertEqual(events, ["submit", "read_submitted", "physics", "update", "capture"])
        events.clear()
        env.control_active = False
        robot.write_data_to_sim()
        robot.update(0.0025)
        self.assertEqual(events, ["submit", "update"])

    def test_capture_hooks_receive_the_task_environment(self):
        env = OmniEnv.__new__(OmniEnv)
        env.device, env.num_envs = "cpu", 1
        received = []

        def step(action):
            env.isaac.capture("state", "q")
            env.isaac.current, env.isaac.telemetry = {"q": 1}, {"terminated": "t", "truncated": "f"}
            return {"policy": "obs", "critic": "critic"}, None, None, None, None

        env.isaac = SimpleNamespace(step=step, total_controls=4)
        env.capture = lambda *args: received.append(args)
        result = env.step(torch.zeros(1, 18))
        self.assertEqual(received, [(env, "state", "q")])
        self.assertEqual(
            result, {"obs": "obs", "critic": "critic", "terminated": "t", "truncated": "f"}
        )
        self.assertEqual((env.current, env.total_controls), ({"q": 1}, 4))
        env.capture = None
        env.isaac.step = lambda action: ({"policy": "obs"}, None, None, None, None)
        env.step(torch.zeros(1, 18))
        self.assertIsNone(env.isaac.capture)

    def test_control_writes_the_tensors_that_the_task_environment_aliases(self):
        # Drive one control through the production hooks. A rebinding in _get_dones would leave
        # OmniEnv.reset writing a history tensor that the observation builder no longer reads.
        class StockLoop:
            """DirectRLEnv.step order: pre-physics, eight write/physics/update substeps, dones, observations."""

            def step(self, action):
                self._pre_physics_step(action)
                for _ in range(8):
                    self._apply_action()
                    self.robot.write_data_to_sim()
                    self.sim.count += 1
                    self.robot.update(0.0025)
                self.episode_length_buf += 1
                terminated, truncated = self._get_dones()
                rewards = self._get_rewards()
                return self._get_observations(), rewards, terminated, truncated, {}

            def _reset_idx(self, env_ids):
                self.episode_length_buf[env_ids] = 0

        class Articulation:
            def write_data_to_sim(self):
                pass

            def update(self, dt):
                pass

            def __getattr__(self, name):
                if name.startswith(("write_", "set_")):
                    return lambda **kwargs: None
                raise AttributeError(name)

        module = load_isaac(direct_env=StockLoop, articulation=Articulation)
        module.wp = SimpleNamespace(
            to_torch=lambda x: x,
            from_torch=lambda x: x,
            uint32=None,
            array=lambda x, **kwargs: torch.as_tensor(x),
        )
        n, neutral = 2, torch.linspace(-0.3, 0.3, 18)
        rest = {
            "q": neutral.expand(n, -1).clone(),
            "dq": torch.zeros(n, 18),
            "root": torch.tensor(
                [[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0], [2.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]]
            ),
            "linear": torch.zeros(n, 3),
            "angular": torch.zeros(n, 3),
            "gravity": torch.tensor([[0.0, 0.0, -1.0]]).repeat(n, 1),
            "toe_world": torch.zeros(n, 6, 3),
            "toe_body": torch.zeros(n, 6, 3),
        }
        moved = {
            **rest,
            "q": (neutral + 0.03).expand(n, -1).clone(),
            "dq": torch.full((n, 18), 0.5),
            "angular": torch.tensor([[0.1, 0.2, 0.3], [0.4, 0.5, 0.6]]),
            "linear": torch.tensor([[0.01, -0.02, 0.0], [0.0, 0.03, 0.0]]),
        }

        class Isaac(module.IsaacEnv):
            def __init__(
                self, config, asset, model_path, geometry_path, output, *, reference_metadata=None
            ):
                self.config, self.device, self.num_envs = config, config.device, config.num_envs
                self.torch = lambda x: torch.as_tensor(x, dtype=torch.float32)
                self.neutral, self.lower, self.upper = neutral, neutral - 1.0, neutral + 1.0
                self.reset_height, self.origins = (
                    0.1,
                    torch.tensor([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]]),
                )
                self.state = rest
                self.read_state = lambda: {key: value.clone() for key, value in self.state.items()}
                self.joint_index = self.inverse_joint_index = torch.arange(18)
                self.native_body_names = list(BODY_NAMES)
                self.toe_index = torch.tensor(
                    [self.native_body_names.index(leg + "_tibia") for leg in LEGS]
                )
                self.contact_index = torch.arange(19 * n)
                self.ground_collision_paths = ["/Ground/mesh"]
                self.contact = SimpleNamespace(
                    get_contact_force_matrix=lambda dt: torch.zeros(19 * n, 1, 3)
                )
                self.motor = SimpleNamespace(
                    joint_names=list(JOINT_NAMES),
                    **{
                        name: torch.zeros(n, 18)
                        for name in (
                            "pre_position",
                            "pre_velocity",
                            "computed_effort",
                            "applied_effort",
                            "ceiling",
                        )
                    },
                )
                self.view = SimpleNamespace(
                    get_dof_actuation_forces=lambda: torch.ones(n, 18),
                    set_dof_actuation_forces=lambda data, ids: None,
                )
                self.sim = SimpleNamespace(count=0)
                self.sim.get_physics_step_count = lambda: self.sim.count
                self.robot = module.CaptureArticulation()
                self.robot.env = self
                self.history, self.previous_action = torch.zeros(n, 5, 42), torch.zeros(n, 18)
                self.held, self.commands = neutral.expand(n, -1).clone(), torch.zeros(n, 3)
                self.episode_length_buf = torch.zeros(n, dtype=torch.long)
                self.total_controls, self.telemetry, self.capture, self.control_active = (
                    0,
                    {},
                    None,
                    False,
                )
                self.native_errors = []
                self.terrain = SimpleNamespace(
                    heights=[[0.0, 0.01], [0.02, 0.03]],
                    xs=(-1.0, 1.0),
                    ys=(-1.0, 1.0),
                    resolution=2.0,
                    nx=2,
                    ny=2,
                    highest_under=lambda x, y, radius: 0.0,
                )
                # OmniEnv aliases these; this control reads none of them.
                for name in (
                    "output",
                    "model",
                    "reference_metadata",
                    "geometry_path",
                    "geometry_meta",
                    "sensor_map",
                    "roots",
                    "joint_names",
                    "body_names",
                    "native_joint_names",
                    "body_index",
                    "toe_local",
                    "root_com_local",
                ):
                    setattr(self, name, None)

        config = SimpleNamespace(
            num_envs=n,
            device="cpu",
            action_scale_rad=0.35,
            target_slew_rad=0.04,
            physics_dt=0.0025,
            episode_seconds=20.0,
            control_dt=0.02,
            record_motion_features=False,
            terrain_spawn=False,
        )
        with mock.patch.dict(
            sys.modules, {"locomotion.omniwalk.env": SimpleNamespace(IsaacEnv=Isaac)}
        ):
            env = OmniEnv(config, None, None, None, None)
        isaac = env.isaac
        # The wrapper's footfall inputs read the terrain duck-typed off the task's env — this
        # adapter — so the alias must carry IsaacEnv's grid (zeros trained wherever it is lost).
        self.assertIs(env.terrain, isaac.terrain)
        from locomotion.omniwalk.mdp.footfall import FootfallInputs

        footfall = FootfallInputs(SimpleNamespace(env=env, num_envs=n), "speed_v1", profile=True)
        self.assertIsNotNone(footfall.grid)
        # A task-side write between controls must reach the tensors that IsaacEnv shifts.
        env.history.copy_(torch.arange(n * 5 * 42, dtype=torch.float32).reshape(n, 5, 42) / 100)
        env.commands[:] = torch.tensor([[0.05, 0.0, 0.0], [0.0, 0.0, 0.2]])
        history, held = env.history.clone(), env.held.clone()
        isaac.state = moved
        action = torch.linspace(-1.0, 1.0, n * 18).reshape(n, 18)
        result = env.step(action)
        self.assertEqual(
            (isaac.sim.count, isaac.total_controls, env.episode_steps.tolist()), (8, 1, [1, 1])
        )
        for name in ("held", "previous_action", "history", "commands"):
            self.assertIs(getattr(env, name), getattr(isaac, name), name)
        target = emitted_target(action, held, neutral, neutral - 1.0, neutral + 1.0, 0.35, 0.04)
        self.assertFalse(torch.equal(target, held))
        torch.testing.assert_close(env.held, target, rtol=0, atol=0)
        torch.testing.assert_close(
            env.previous_action, executed_action_feature(target, neutral, 0.35), rtol=0, atol=0
        )
        shifted = torch.cat((history[:, 1:], proprio(moved, neutral)[:, None]), dim=1)
        torch.testing.assert_close(env.history, shifted, rtol=0, atol=0)
        expected = observations(
            shifted, env.commands, executed_action_feature(target, neutral, 0.35), moved
        )
        for key in ("obs", "critic"):
            torch.testing.assert_close(result[key], expected[key], rtol=0, atol=0)
            torch.testing.assert_close(
                env.observations(env.current)[key], expected[key], rtol=0, atol=0
            )

    def test_slew_and_action_feature_share_current_arithmetic(self):
        q = torch.zeros((2, 18))
        held = q.clone()
        neutral = torch.zeros(18)
        for _ in range(12):
            target = emitted_target(torch.ones_like(q), held, neutral, neutral - 0.3, neutral + 0.3)
            self.assertTrue(bool(((target.double() - held.double()).abs() <= 0.040).all()))
            self.assertTrue(
                torch.equal(executed_action_feature(target, neutral, 0.35), target / 0.35)
            )
            held = target
        self.assertTrue(torch.equal(held, torch.full_like(held, 0.3)))


if __name__ == "__main__":
    unittest.main()
