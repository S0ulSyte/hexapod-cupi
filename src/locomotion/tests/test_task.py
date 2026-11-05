"""Compare the active task with the frozen baseline source archive."""

import hashlib
import importlib
import json
import sys
import tarfile
import tempfile
import types
import unittest
from dataclasses import asdict, replace
from pathlib import Path

import torch

import locomotion.omniwalk.mdp.rewards as reward
import locomotion.omniwalk.task as task
from locomotion.evaluation.task_metrics import TaskMetrics
from locomotion.omniwalk.mdp.commands import TaskConfig, command_bank
from locomotion.tests.task_double import GaitDouble, telemetry

ROOT = Path(__file__).resolve().parents[3]
FIXTURE = Path(__file__).parent / "fixtures/refactor_b2a6168c.tar.gz"
FIXTURE_SHA = "f0afe51e180bc1d5c5f6099b8225e77cd63afdbed78e4830dd837767ae43b6fc"
SPEC = (
    "forward_draw_fraction=0,joint_margin_weight=1.0,yaw_tracking_weight=0.5,"
    "turn_draw_fraction=0.3,gate_window_controls=120,yaw_gate_min_command_rad_s=0.18,"
    "yaw_gate_bonus=0.25,yaw_gate_floor=0.5,stride_tracking_weight=0.3,"
    "stride_length_weight=2,termination_weight=22"
)


class ActiveTaskTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if hashlib.sha256(FIXTURE.read_bytes()).hexdigest() != FIXTURE_SHA:
            raise ValueError("Frozen source archive changed")
        cls.directory = tempfile.TemporaryDirectory()
        cls.addClassCleanup(cls.directory.cleanup)
        with tarfile.open(FIXTURE) as archive:
            archive.extractall(cls.directory.name, filter="data")
        name = "_active_task_baseline"
        package = types.ModuleType(name)
        package.__path__ = [cls.directory.name]
        sys.modules[name] = package
        cls.old = importlib.import_module(name + ".task_v4_yaw_gate")

        def cleanup():
            for key in list(sys.modules):
                if key == name or key.startswith(name + "."):
                    del sys.modules[key]

        cls.addClassCleanup(cleanup)

    def exact(self, actual, expected):
        if isinstance(expected, torch.Tensor):
            torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
        elif isinstance(expected, dict):
            self.assertEqual(actual.keys(), expected.keys())
            for key in expected:
                self.exact(actual[key], expected[key])
        elif isinstance(expected, (list, tuple)):
            self.assertEqual(len(actual), len(expected))
            for left, right in zip(actual, expected):
                self.exact(left, right)
        else:
            self.assertEqual(actual, expected)

    def test_default_config_matches_recorded_checkpoint(self):
        result = json.loads(
            (
                ROOT / "site/assets/yaw_gated_stride_native_20261012_001/native_results.json"
            ).read_text()
        )
        args = result["training"]["c48_long"]["command_args"]
        self.assertEqual(args[args.index("--reward-options") + 1], SPEC)
        expected = asdict(self.old.reward_config(SPEC))
        del expected["forward_draw_fraction"]
        del expected["yaw_gate_on"]
        active = asdict(reward.REWARD_CONFIG)
        # Fields added since the frozen baseline must be exactly these and stay at their inert
        # defaults, so the recorded checkpoint's options still reproduce the active default reward.
        self.assertEqual(
            set(active) - set(expected),
            {
                "swing_clearance_weight",
                "swing_clearance_target_m",
                "swing_clearance_rough_m",
                "landing_lift_weight",
                "landing_lift_target_m",
                "landing_lift_rise_m",
                "schedule_mirror_turns",
                "situation_tracking",
                "situation_speed_ref_mps",
                "situation_yaw_ref_rad_s",
                "quiet_hold_weight",
                "quiet_hold_scale_rad",
                "quiet_hold_grace_controls",
            },
        )
        self.assertEqual(active["swing_clearance_weight"], 0.0)
        self.assertEqual(active["landing_lift_weight"], 0.0)
        self.assertEqual(active["situation_tracking"], 0.0)
        self.assertEqual(active["quiet_hold_weight"], 0.0)
        self.assertIsNone(active["schedule_mirror_turns"])
        self.assertEqual({k: v for k, v in active.items() if k in expected}, expected)
        baseline = importlib.import_module("_active_task_baseline.task")
        self.assertEqual(command_bank(TaskConfig()), baseline.command_bank(baseline.TaskConfig()))

    def test_reward_rng_episode_reset_and_reporting_match(self):
        traces = []
        for current in (False, True):
            env = GaitDouble(3)
            wrapped = (
                task.TrainingTask(env, TaskConfig(seed=20260917))
                if current
                else self.old.variant(SPEC)(env, self.old.task_v4.TaskConfig(seed=20260917))
            )
            # The active task reports through its metrics observer; the frozen task reports itself.
            reporter = TaskMetrics(wrapped) if current else wrapped
            wrapped.reset()
            trace = []
            for step in range(300):
                if step in (6, 99, 201):
                    wrapped.reset(torch.tensor([1]))
                wrapped.remaining_controls[step % 3] = 1
                # Exercise liftoff, landing and both translation/yaw command windows.
                env.forces[:, step % 6] = 0 if step % 2 else 24
                env.world[:, :, 0] += 0.0002
                output = wrapped.step(torch.full((3, 18), step / 2000.0))
                trace.append(
                    (
                        {key: value.clone() for key, value in output.items()},
                        wrapped.rng.get_state().clone(),
                        wrapped.remaining_controls.clone(),
                        {key: value.clone() for key, value in wrapped.last_components.items()},
                        {
                            key: value
                            for key, value in reporter.status(reset_interval=step % 17 == 0).items()
                            if key != "proximity"
                        },
                    )
                )
            traces.append(trace)
        self.exact(traces[1], traces[0])

    def test_pure_reward_components_match_randomized_telemetry(self):
        generator = torch.Generator().manual_seed(20260917)
        commands = torch.tensor(command_bank(TaskConfig()) + [[0.0, 0.0, 0.0]])
        n = len(commands)
        values = telemetry(commands)
        for key, value in list(values.items()):
            if value.is_floating_point() and key not in ("command", "root_pose_xyzw"):
                values[key] = torch.randn(value.shape, generator=generator)
        values.update(
            gate_yaw_rate_rad_s=torch.randn(n, generator=generator),
            stride_velocity_nav=torch.randn(n, 3, generator=generator),
            stride_touchdown=torch.rand(n, 6, generator=generator) > 0.5,
            stride_displacement_nav=torch.randn(n, 6, 2, generator=generator),
        )
        terminated = torch.arange(n) % 2 == 0
        expected = self.old.measured_reward_v4_yaw_gate(
            values, commands, terminated, self.old.reward_config(SPEC), 0.1
        )
        actual = reward.measured_reward(values, commands, terminated, reward.REWARD_CONFIG, 0.1)
        self.exact(actual, expected)

    def test_situation_tracking_prices_each_axis_by_its_command(self):
        commands = torch.tensor(
            [[0.0, 0.0, 0.0], [0.2, 0.0, 0.0], [0.0, 0.0, -0.4], [0.15, 0.0, 0.05]]
        )
        values = telemetry(commands)
        # Perfect tracking keeps both kernels at one, so the factors are read off directly.
        values["tracked_velocity_nav"] = commands.clone()
        terminated = torch.zeros(len(commands), dtype=torch.bool)
        base = reward.base_reward(values, commands, terminated, reward.REWARD_CONFIG, 0.1)[1]
        config = replace(reward.REWARD_CONFIG, situation_tracking=1.0, termination_weight=44.0)
        config.validate()
        priced = reward.base_reward(values, commands, terminated, config, 0.1)[1]
        torch.testing.assert_close(
            priced["linear_tracking"],
            base["linear_tracking"] * torch.tensor([1.0, 2.0, 1.0, 1.75]),
        )
        torch.testing.assert_close(
            priced["yaw_tracking"],
            base["yaw_tracking"] * torch.tensor([1.0, 1.0, 2.0, 1.125]),
        )
        changed = {key for key in base if not torch.equal(base[key], priced[key])}
        self.assertEqual(changed, {"linear_tracking", "yaw_tracking"})

    def test_quiet_hold_prices_wander_from_the_anchor_while_still(self):
        commands = torch.tensor([[0.0, 0.0, 0.0], [0.0, 0.0, 0.0], [0.1, 0.0, 0.0]])
        values = telemetry(commands)
        values["joint_position_rad"] = torch.zeros(3, 18)
        anchor = torch.zeros(3, 18)
        anchor[0] = -0.01  # one scale of deviation on every joint of the settled stand
        anchor[2] = -0.02
        values.update(
            quiet_anchor_joint_rad=anchor,
            command_age_controls=torch.tensor([100, 10, 100]),
            gate_yaw_rate_rad_s=torch.zeros(3),
            stride_velocity_nav=torch.zeros(3, 3),
            stride_touchdown=torch.zeros(3, 6, dtype=torch.bool),
            stride_displacement_nav=torch.zeros(3, 6, 2),
        )
        config = replace(reward.REWARD_CONFIG, quiet_hold_weight=3.0)
        _, components = reward.measured_reward(
            values, commands, torch.zeros(3, dtype=torch.bool), config, 0.1
        )
        # Row 0 is still and one scale off its anchor; row 1 sits inside the grace window;
        # row 2 is moving, however far from its stale anchor.
        torch.testing.assert_close(components["quiet_hold"], torch.tensor([-3.0, 0.0, 0.0]))
        # With the hold's own longer settle, age 100 has not yet reached the hold.
        _, delayed = reward.measured_reward(
            values,
            commands,
            torch.zeros(3, dtype=torch.bool),
            replace(config, quiet_hold_grace_controls=120),
            0.1,
        )
        torch.testing.assert_close(delayed["quiet_hold"], torch.zeros(3))

    def test_quiet_hold_telemetry_flows_through_the_task(self):
        env = GaitDouble(3)
        wrapped = task.TrainingTask(
            env,
            TaskConfig(seed=20260917),
            reward_config=replace(reward.REWARD_CONFIG, quiet_hold_weight=3.0),
        )
        wrapped.reset()
        for _ in range(3):
            wrapped.remaining_controls[:] = 2
            wrapped.step(torch.zeros(3, 18))
        self.assertIn("quiet_hold", wrapped.last_components)

    def test_terminal_rows_keep_the_final_command_until_selected_reset(self):
        env = GaitDouble(3)
        original = env.step

        def terminal(action):
            result = original(action)
            result["terminated"][0] = True
            result["truncated"][1] = True
            return result

        env.step = terminal
        wrapped = task.TrainingTask(env)
        wrapped.reset()
        before = env.commands.clone()
        wrapped.remaining_controls[:] = 1
        result = wrapped.step(torch.zeros(3, 18))
        self.exact(result["obs"][:2, 210:213], before[:2])
        self.exact(env.commands[:2], before[:2])
        self.assertEqual(len(env.reset_calls), 1)
        wrapped.reset(torch.tensor([0, 1]))
        self.assertEqual(env.reset_calls[-1].tolist(), [0, 1])


if __name__ == "__main__":
    unittest.main()
