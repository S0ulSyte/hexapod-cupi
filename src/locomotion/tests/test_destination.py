"""Experiment destination_v1: the destination, its pay, the five inputs, evaluation and the widened start."""

import math
import unittest
from dataclasses import asdict, replace
from types import SimpleNamespace

import torch

from locomotion.omniwalk.agents.rsl_rl_ppo_cfg import EXPERIMENT_INPUTS, ppo_config
from locomotion.omniwalk.destination_task import (
    EXPERIMENTS,
    DestinationRewardConfig,
    DestinationTrainingTask,
)
from locomotion.omniwalk.mdp.commands import TaskConfig
from locomotion.omniwalk.mdp.destination import (
    DestinationTracker,
    advance_destination,
    destination_features,
)
from locomotion.omniwalk.mdp.footfall import (
    RING_XY,
    TerrainGrid,
    touchdown_points,
    touchdown_seconds,
)
from locomotion.omniwalk.mdp.rewards import REWARD_CONFIG, measured_reward
from locomotion.rl.destination_wrapper import DestinationVecEnv
from locomotion.rl.ppo import initialize_learner, widen_inputs
from locomotion.tests.task_double import GaitDouble, telemetry

FORWARD, LEFT = torch.tensor([[0.05, 0.0, 0.0]]), torch.tensor([[0.0, 0.05, 0.0]])
SETTINGS = EXPERIMENTS["destination_v1"]


def plain(value, **kw):
    return value


def pose_at(x, y, yaw):
    return torch.tensor([[x, y, 0.1, 0.0, 0.0, math.sin(yaw / 2), math.cos(yaw / 2)]])


def moving_task(motion, command=FORWARD, reward=None, fall_on=None, remaining=1000):
    """A one-row task whose body follows ``motion(k) -> (x, y, yaw)`` after control k."""
    native = GaitDouble(1)
    native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
    task = DestinationTrainingTask(native, TaskConfig(), None, reward or SETTINGS["reward"])
    task.reset()
    native.commands[:] = command
    task.remaining_controls[:] = remaining
    count, original = {"k": 0}, native.step

    def step(action):
        output = original(action)
        count["k"] += 1
        pose = pose_at(*motion(count["k"]))
        native.current["root"], native.telemetry["root_pose_xyzw"] = pose, pose.clone()
        if count["k"] == fall_on:
            output = {**output, "terminated": torch.tensor([True])}
        return output

    native.step = step
    return task, native


def forward(k, slide=0.0):
    # A forward walk at heading 0 moves along world -y; a slide moves along world +x (left).
    return (slide * 0.02 * k, -0.05 * 0.02 * k, 0.0)


class DestinationTests(unittest.TestCase):
    def test_the_destination_follows_the_command_in_the_turning_body_axes(self):
        start = torch.zeros(3, 3)
        start[2, 2] = math.pi / 2
        commands = torch.tensor([[0.05, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.05, 0.0]])
        destination = start
        for _ in range(100):
            destination = advance_destination(destination, commands)
        expected = torch.tensor([[0.0, -0.1, 0.0], [0.0, 0.0, 0.4], [0.0, 0.1, math.pi / 2]])
        torch.testing.assert_close(destination, expected, atol=1e-5, rtol=0)
        arc = torch.zeros(1, 3)
        for _ in range(500):
            arc = advance_destination(arc, torch.tensor([[0.05, 0.0, 0.2]]))
        self.assertAlmostEqual(
            float(torch.linalg.vector_norm(arc[0, :2])), 0.5 * math.sin(1.0), places=4
        )

    def test_the_gap_reads_in_body_navigation_axes(self):
        cases = (
            ((0.0, -0.1, 0.0), (0.0, 0.0, 0.0), [1.0, 0.0, 0.0, 1.0]),
            ((0.1, 0.0, 0.0), (0.0, 0.0, 0.0), [0.0, 1.0, 0.0, 1.0]),
            ((0.0, 0.0, 0.3), (0.0, 0.0, 0.0), [0.0, 0.0, math.sin(0.3), math.cos(0.3)]),
            ((0.1, 0.0, math.pi / 2), (0.0, 0.0, math.pi / 2), [1.0, 0.0, 0.0, 1.0]),
        )
        for destination, body, expected in cases:
            with self.subTest(destination=destination):
                features = destination_features(
                    pose_at(*body), torch.tensor([destination]), torch.tensor([125])
                )
                torch.testing.assert_close(
                    features[0], torch.tensor(expected + [0.5]), atol=1e-5, rtol=0
                )

    def test_the_tracker_restarts_on_a_new_command_or_episode(self):
        tracker, steps = DestinationTracker(1, "cpu"), torch.zeros(1, dtype=torch.long)
        torch.testing.assert_close(
            tracker.inputs(pose_at(0.0, 0.0, 0.0), FORWARD, steps)[0],
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
        )
        for _ in range(50):
            steps += 1
            features = tracker.inputs(pose_at(0.0, 0.0, 0.0), FORWARD, steps)
        torch.testing.assert_close(
            features[0], torch.tensor([0.5, 0.0, 0.0, 1.0, 0.2]), atol=1e-5, rtol=0
        )
        steps += 1
        torch.testing.assert_close(
            tracker.inputs(pose_at(0.0, 0.0, 0.0), LEFT, steps)[0],
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]),
        )


class TaskTests(unittest.TestCase):
    def test_the_experiment_differs_from_the_canonical_reward_only_in_its_fields(self):
        base, chosen = asdict(REWARD_CONFIG), asdict(SETTINGS["reward"])
        changed = {key: chosen[key] for key in chosen if base.get(key) != chosen[key]}
        self.assertEqual(
            changed,
            {
                "termination_weight": 32.0,
                "destination_weight": 1.0,
                "destination_distance_m": 0.03,
                "destination_kernel": "exponential",
                "command_controls": 1000,
            },
        )
        self.assertEqual(
            (EXPERIMENT_INPUTS["destination_v1"], SETTINGS["episode_seconds"]), ("ghost", 20.0)
        )

    def test_destination_v2_only_halves_the_pay_every_centimetre(self):
        first, second = EXPERIMENTS["destination_v1"], EXPERIMENTS["destination_v2"]
        changed = {
            key: value
            for key, value in asdict(second["reward"]).items()
            if asdict(first["reward"])[key] != value
        }
        self.assertEqual(list(changed), ["destination_distance_m"])
        self.assertEqual(
            {key: value for key, value in second.items() if key != "reward"},
            {key: value for key, value in first.items() if key != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v2"], "ghost")
        scale = changed["destination_distance_m"]
        for distance_m, pay in ((0.0, 1.0), (0.005, 0.7071), (0.01, 0.5), (0.02, 0.25)):
            self.assertAlmostEqual(math.exp(-distance_m / scale), pay, places=4)
        second["reward"].validate()

    def test_destination_v3_only_concentrates_the_pay_with_the_rational_kernel(self):
        second, third = EXPERIMENTS["destination_v2"], EXPERIMENTS["destination_v3"]
        changed = {
            key: value
            for key, value in asdict(third["reward"]).items()
            if asdict(second["reward"])[key] != value
        }
        self.assertEqual(
            changed,
            {
                "destination_kernel": "rational",
                "destination_distance_m": 0.0037,
                "destination_power": 1.2,
                "destination_quiet": "flat",
                "hold_weight": 2.0,
                "hold_grace_controls": 100,
                "hold_deadband_rad": 0.005,
                "hold_span_rad": 0.045,
                "hold_smoothing_controls": 50,
            },
        )
        self.assertEqual(
            {key: value for key, value in third.items() if key != "reward"},
            {key: value for key, value in second.items() if key != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v3"], "ghost")
        task, _ = moving_task(lambda k: forward(k, slide=0.02), reward=third["reward"])
        for k in range(1, 26):
            task.step(torch.zeros(1, 18))
            if k in (1, 5, 13, 25):  # 0.4, 2, 5.2 and 10 mm to the side of the destination
                self.assertAlmostEqual(
                    float(task.last_components["destination"][0]),
                    1 / (1 + (0.02 * 0.02 * k / 0.0037) ** 1.2),
                    places=4,
                )
        for bad in (
            dict(destination_power=None),
            dict(destination_power=0.0),
            dict(destination_kernel="exponential"),
            dict(destination_quiet="track"),
            dict(hold_span_rad=None),
            dict(hold_weight=0.0),
            dict(hold_grace_controls=0),
            dict(hold_deadband_rad=-0.001),
            dict(hold_smoothing_controls=101),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(third["reward"], **bad).validate()

    def test_a_zero_command_pays_the_full_destination_and_charges_joint_drift_after_the_grace(self):
        reward = EXPERIMENTS["destination_v3"]["reward"]
        smoothing = reward.hold_smoothing_controls

        def standing(position):
            """Charges and running-mean drift while standing; joint 4 follows ``position(k)``."""
            task, native = moving_task(
                lambda k: (0.001 * k, 0.0, 0.0), command=torch.zeros(1, 3), reward=reward
            )
            start, out, mean, anchor = native.position.clone(), {}, 0.0, None
            for k in range(1, 301):
                native.position[:] = start
                native.position[0, 4] += position(k)
                mean = position(k) if k == 1 else mean + (position(k) - mean) / smoothing
                anchor = mean if k == 100 else anchor
                task.step(torch.zeros(1, 18))
                # The body slides 1 mm per control, yet a zero command pays the full destination.
                self.assertEqual(float(task.last_components["destination"][0]), 1.0)
                out[k] = (float(task.last_components["hold"][0]), mean - (anchor or mean))
            return out

        # Exploration-sized jitter of +-20 mrad averages out and is never charged.
        jitter = standing(lambda k: 0.02 * (-1) ** k)
        self.assertTrue(all(charge == 0.0 for charge, _ in jitter.values()))
        # A creep of 0.2 mrad per control is charged once its running mean leaves the deadband.
        creep = standing(lambda k: 0.0002 * k)
        self.assertTrue(all(creep[k][0] == 0.0 for k in range(1, 125)))
        for k in (150, 200, 300):
            expected = -2.0 * min(max((abs(creep[k][1]) - 0.005) / 0.045, 0), 1) / 18
            self.assertAlmostEqual(creep[k][0], expected, places=5)
        self.assertLess(creep[300][0], -0.07)
        # A moving command pays by distance and charges nothing.
        walking, native = moving_task(lambda k: forward(k, slide=0.02), reward=reward)
        for _ in range(130):
            native.position[0, 4] += 0.01
            walking.step(torch.zeros(1, 18))
        self.assertEqual(float(walking.last_components["hold"][0]), 0.0)
        self.assertLess(float(walking.last_components["destination"][0]), 0.1)

    def test_destination_v4_drops_the_hold_charge_and_pays_nothing_under_a_zero_command(self):
        third, fourth = EXPERIMENTS["destination_v3"], EXPERIMENTS["destination_v4"]
        changed = {
            key: value
            for key, value in asdict(fourth["reward"]).items()
            if asdict(third["reward"])[key] != value
        }
        self.assertEqual(
            changed,
            {
                "destination_quiet": "off",
                "hold_weight": None,
                "hold_grace_controls": None,
                "hold_deadband_rad": None,
                "hold_span_rad": None,
                "hold_smoothing_controls": None,
            },
        )
        self.assertEqual(
            {key: value for key, value in fourth.items() if key != "reward"},
            {key: value for key, value in third.items() if key != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v4"], "ghost")
        # Standing exactly on the destination pays nothing and charges no hold.
        standing, _ = moving_task(
            lambda k: (0.0, 0.0, 0.0), command=torch.zeros(1, 3), reward=fourth["reward"]
        )
        for _ in range(150):
            standing.step(torch.zeros(1, 18))
            self.assertEqual(float(standing.last_components["destination"][0]), 0.0)
            self.assertNotIn("hold", standing.last_components)
        # A moving command still pays by the rational kernel.
        walking, _ = moving_task(lambda k: forward(k, slide=0.02), reward=fourth["reward"])
        for k in range(1, 6):
            walking.step(torch.zeros(1, 18))
        self.assertAlmostEqual(
            float(walking.last_components["destination"][0]),
            1 / (1 + (0.02 * 0.02 * 5 / 0.0037) ** 1.2),
            places=4,
        )

    def test_destination_v5_pays_the_destination_under_a_zero_command_too(self):
        fourth, fifth = EXPERIMENTS["destination_v4"], EXPERIMENTS["destination_v5"]
        changed = {
            key: value
            for key, value in asdict(fifth["reward"]).items()
            if asdict(fourth["reward"])[key] != value
        }
        self.assertEqual(changed, {"destination_quiet": None})
        self.assertEqual(
            {key: value for key, value in fifth.items() if key != "reward"},
            {key: value for key, value in fourth.items() if key != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v5"], "ghost")
        # Standing 1 mm per control off the destination pays by the rational kernel.
        standing, _ = moving_task(
            lambda k: (0.001 * k, 0.0, 0.0), command=torch.zeros(1, 3), reward=fifth["reward"]
        )
        for _ in range(4):
            standing.step(torch.zeros(1, 18))
        self.assertAlmostEqual(
            float(standing.last_components["destination"][0]),
            1 / (1 + (0.004 / 0.0037) ** 1.2),
            places=4,
        )
        self.assertNotIn("hold", standing.last_components)

    def test_destination_v6_follows_a_share_of_walks_with_a_stop_for_the_rest_of_the_command(self):
        fifth, sixth = EXPERIMENTS["destination_v5"], EXPERIMENTS["destination_v6"]
        changed = {
            key: value
            for key, value in asdict(sixth["reward"]).items()
            if asdict(fifth["reward"])[key] != value
        }
        self.assertEqual(
            changed,
            {"stop_fraction": 0.2, "stop_after_min_controls": 200, "stop_after_max_controls": 600},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v6"], "ghost")
        for bad in (
            dict(stop_fraction=0.0),
            dict(stop_after_max_controls=None),
            dict(stop_after_min_controls=700),
            dict(stop_after_max_controls=1000),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(sixth["reward"], **bad).validate()
        # Every moving command stops: it walks 10 to 20 controls, then stands for the rest of 60.
        reward = replace(
            sixth["reward"],
            stop_fraction=1.0,
            stop_after_min_controls=10,
            stop_after_max_controls=20,
            command_controls=60,
        )
        native = GaitDouble(64)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, reward)
        task.reset()
        moving = (native.commands != 0).any(-1)
        self.assertTrue(bool(moving.any()) and bool((~moving).any()))
        walked = task.remaining_controls.clone()
        self.assertTrue(bool(((walked >= 10) & (walked <= 20))[moving].all()))
        self.assertTrue(bool((walked[~moving] == 60).all()))
        first = native.commands.clone()
        # Training truncates the episode when the stop ends at control 60; check up to there.
        for k in range(1, 60):
            task.step(torch.zeros(64, 18))
            stopped = moving & (walked <= k)
            self.assertTrue(bool((native.commands[stopped] == 0).all()))
            self.assertTrue(
                torch.equal(native.commands[~stopped & moving], first[~stopped & moving])
            )
            self.assertTrue(bool((task.remaining_controls[stopped] == 60 - k).all()))
        # A reset drops a stop that its episode no longer reaches.
        task.reset()
        task.stop_pending[:] = True
        task.reset()
        self.assertTrue(bool((task.remaining_controls[(native.commands == 0).all(-1)] == 60).all()))

    def test_destination_v7_pays_for_the_destination_heading_like_the_destination(self):
        fifth, seventh = EXPERIMENTS["destination_v5"], EXPERIMENTS["destination_v7"]
        changed = {
            key: value
            for key, value in asdict(seventh["reward"]).items()
            if asdict(fifth["reward"])[key] != value
        }
        self.assertEqual(
            changed,
            {
                "heading_weight": 1.0,
                "heading_width_rad": math.radians(1.0),
                "heading_power": 1.2,
                "termination_weight": 42.0,
            },
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v7"], "ghost")
        reward = seventh["reward"]
        with self.assertRaises(ValueError):
            replace(reward, termination_weight=41.0).validate()
        for bad in (dict(heading_power=None), dict(heading_width_rad=0.0)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(reward, **bad).validate()
        width = math.radians(1.0)
        # Walking forward, the body turns 0.1 deg per control while the destination keeps its heading.
        walking, _ = moving_task(lambda k: (0.0, -0.001 * k, math.radians(0.1 * k)), reward=reward)
        for k in range(1, 31):
            walking.step(torch.zeros(1, 18))
            if k in (5, 10, 30):
                self.assertAlmostEqual(
                    float(walking.last_components["heading"][0]),
                    1 / (1 + (math.radians(0.1 * k) / width) ** 1.2),
                    places=4,
                )
        # A turn command turns the destination: a body that turns with it is paid in full.
        turning, _ = moving_task(
            lambda k: (0.0, 0.0, 0.2 * 0.02 * k),
            command=torch.tensor([[0.0, 0.0, 0.2]]),
            reward=reward,
        )
        for _ in range(20):
            turning.step(torch.zeros(1, 18))
            self.assertAlmostEqual(float(turning.last_components["heading"][0]), 1.0, places=4)
        # Standing pays too, and the heading gap wraps around +-180 deg.
        standing, _ = moving_task(
            lambda k: (0.0, 0.0, math.radians(359.5)), command=torch.zeros(1, 3), reward=reward
        )
        for _ in range(3):
            standing.step(torch.zeros(1, 18))
        self.assertAlmostEqual(
            float(standing.last_components["heading"][0]),
            1 / (1 + (math.radians(0.5) / width) ** 1.2),
            places=4,
        )

    def test_destination_v8_doubles_every_joint_penalty_of_v7(self):
        seventh, eighth = EXPERIMENTS["destination_v7"], EXPERIMENTS["destination_v8"]
        changed = {
            key: value
            for key, value in asdict(eighth["reward"]).items()
            if asdict(seventh["reward"])[key] != value
        }
        self.assertEqual(
            changed,
            {
                "torque_weight": 0.1,
                "acceleration_weight": 0.1,
                "target_rate_weight": 0.1,
                "torque_limit_weight": 1.0,
                "over_rating_weight": 20.0,
                "joint_margin_weight": 2.0,
                "action_limit_weight": 12.0,
                "quiet_joint_rate_weight": 1.0,
                "quiet_target_weight": 1.0,
                "quiet_strain_weight": 20.0,
            },
        )
        self.assertEqual(
            {key: value for key, value in eighth.items() if key != "reward"},
            {key: value for key, value in seventh.items() if key != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v8"], "ghost")
        eighth["reward"].validate()

    def test_destination_v9_adds_destination_v6_stops_to_v8(self):
        eighth, ninth = EXPERIMENTS["destination_v8"], EXPERIMENTS["destination_v9"]
        changed = {
            key: value
            for key, value in asdict(ninth["reward"]).items()
            if asdict(eighth["reward"])[key] != value
        }
        stops = ("stop_fraction", "stop_after_min_controls", "stop_after_max_controls")
        self.assertEqual(
            changed, {key: getattr(EXPERIMENTS["destination_v6"]["reward"], key) for key in stops}
        )
        self.assertEqual(
            {key: value for key, value in ninth.items() if key != "reward"},
            {key: value for key, value in eighth.items() if key != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v9"], "ghost")
        ninth["reward"].validate()

    def test_every_control_pays_the_exponential_of_the_distance_and_a_fall_pays_nothing(self):
        task, _ = moving_task(lambda k: forward(k, slide=0.02))
        for k in range(1, 301):
            task.step(torch.zeros(1, 18))
            if k in (1, 75, 300):  # 0.4 mm, 3 cm and 12 cm to the side of the destination
                self.assertAlmostEqual(
                    float(task.last_components["destination"][0]),
                    math.exp(-0.02 * 0.02 * k / 0.03),
                    places=4,
                )
        fell, _ = moving_task(forward, fall_on=3)
        for _ in range(3):
            fell.step(torch.zeros(1, 18))
        self.assertEqual(float(fell.last_components["destination"][0]), 0.0)

    def test_commands_last_the_declared_length_and_the_fall_cost_counts_the_weight(self):
        native = GaitDouble(3)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, SETTINGS["reward"])
        task.reset()
        self.assertEqual(task.remaining_controls.tolist(), [1000] * 3)
        with self.assertRaises(ValueError):
            replace(SETTINGS["reward"], termination_weight=31.0).validate()
        for bad in (dict(destination_kernel="gaussian"), dict(command_controls=5)):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(SETTINGS["reward"], **bad).validate()
        with self.assertRaises(ValueError):
            DestinationTrainingTask(native, TaskConfig(), None, REWARD_CONFIG)
        self.assertIsInstance(SETTINGS["reward"], DestinationRewardConfig)


class CurriculumTests(unittest.TestCase):
    def test_destination_v10_is_v9_with_a_walking_speed_curriculum(self):
        nine, ten = (
            EXPERIMENTS["destination_v9"]["reward"],
            EXPERIMENTS["destination_v10"]["reward"],
        )
        changed = {key for key, value in asdict(ten).items() if asdict(nine)[key] != value}
        self.assertEqual(
            changed,
            {
                "speed_curriculum_start_mps",
                "speed_curriculum_step_mps",
                "speed_curriculum_final_mps",
                "speed_curriculum_threshold",
                "speed_curriculum_window_controls",
                "speed_curriculum_new_share",
            },
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v10"], "ghost")
        for bad in (
            dict(speed_curriculum_final_mps=0.16),
            dict(speed_curriculum_threshold=1.5),
            dict(speed_curriculum_window_controls=0),
            dict(speed_curriculum_new_share=None),
        ):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                replace(ten, **bad).validate()

    def test_walks_take_the_top_or_a_slower_speed_and_the_top_rises_past_the_threshold(self):
        native = GaitDouble(4096)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        reward = EXPERIMENTS["destination_v10"]["reward"]
        task = DestinationTrainingTask(native, TaskConfig(), None, reward)
        task.reset()
        command = native.commands
        speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
        walks = (speed > 0) & (command[:, 2] == 0)
        levels = torch.round(speed[walks] / 0.025).long()
        self.assertEqual(task.curriculum_level, 0.075)
        self.assertEqual(set(levels.tolist()), {1, 2, 3})
        self.assertAlmostEqual(float((levels == 3).double().mean()), 0.5, delta=0.05)
        # Walks point every way; arcs keep v9's 0.04 m/s.
        angle = torch.atan2(command[walks, 1], command[walks, 0])
        self.assertEqual(int(torch.histc(angle, 4, -math.pi, math.pi).min() > 0), 1)
        self.assertTrue(
            torch.allclose(speed[(speed > 0) & (command[:, 2] != 0)], torch.tensor(0.04))
        )
        # A window below the threshold holds the level; one above it raises the level and refills.
        window = reward.speed_curriculum_window_controls
        for _ in range(window):
            task._curriculum_step(torch.full((4096,), 0.79))
        self.assertEqual(task.curriculum_status()["level_mps"], 0.075)
        for level in (0.1, 0.125, 0.15, 0.15):
            for _ in range(window):
                task._curriculum_step(torch.full((4096,), 0.85))
            self.assertEqual(task.curriculum_status()["level_mps"], level)
        status = task.curriculum_status()
        self.assertTrue(status["complete"])
        self.assertEqual([a["level_mps"] for a in status["advances"]], [0.075, 0.1, 0.125, 0.15])


class BackoffTests(unittest.TestCase):
    def test_destination_v11_rehearses_more_and_backs_off_a_falling_level(self):
        ten, eleven = (EXPERIMENTS[n]["reward"] for n in ("destination_v10", "destination_v11"))
        changed = {k for k, v in asdict(eleven).items() if asdict(ten)[k] != v}
        self.assertEqual(
            changed, {"speed_curriculum_new_share", "speed_curriculum_backoff_threshold"}
        )
        with self.assertRaises(ValueError):
            replace(eleven, speed_curriculum_backoff_threshold=0.9).validate()
        native = GaitDouble(64)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, eleven)
        window = eleven.speed_curriculum_window_controls
        # The first level never backs off; a raised level that falls below 0.65 drops one step.
        for _ in range(window):
            task._curriculum_step(torch.full((64,), 0.5))
        self.assertEqual(task.curriculum_level, 0.075)
        for pay, level in ((0.85, 0.1), (0.85, 0.125), (0.6, 0.1)):
            for _ in range(window):
                task._curriculum_step(torch.full((64,), pay))
            self.assertEqual(task.curriculum_level, level)
        self.assertTrue(task.curriculum_status()["advances"][-1]["back_off"])

    def test_destination_v13_walks_everywhere_and_pays_0_8_at_5_mm(self):
        twelve, thirteen = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v12", "destination_v13")
        )
        changed = {k for k, v in asdict(thirteen).items() if asdict(twelve)[k] != v}
        self.assertEqual(
            changed,
            {
                "destination_distance_m",
                "speed_curriculum_walks_only",
                "speed_curriculum_window_controls",
            },
        )
        self.assertEqual(thirteen.speed_curriculum_window_controls, 360)
        self.assertEqual(EXPERIMENT_INPUTS["destination_v13"], "ghost")
        width, power = thirteen.destination_distance_m, thirteen.destination_power
        self.assertAlmostEqual(1 / (1 + (0.005 / width) ** power), 0.8)
        with self.assertRaises(ValueError):
            replace(thirteen, speed_curriculum_walks_only=False).validate()
        native = GaitDouble(2048)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, thirteen)
        task.reset()
        command = native.commands
        speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
        # Every row walks without yaw; about a fifth are due to stop.
        self.assertTrue(bool((speed > 0).all()) and bool((command[:, 2] == 0).all()))
        self.assertAlmostEqual(float(task.stop_pending.double().mean()), 0.2, delta=0.03)
        stopping = torch.nonzero(task.stop_pending).flatten()
        task._resample(stopping)
        self.assertTrue(bool((native.commands[stopping] == 0).all()))

    def test_destination_v14_backs_off_only_after_50_updates_at_a_level(self):
        thirteen, fourteen = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v13", "destination_v14")
        )
        changed = {k for k, v in asdict(fourteen).items() if asdict(thirteen)[k] != v}
        self.assertEqual(changed, {"speed_curriculum_backoff_window_controls"})
        self.assertEqual(EXPERIMENT_INPUTS["destination_v14"], "ghost")
        with self.assertRaises(ValueError):
            replace(fourteen, speed_curriculum_backoff_window_controls=100).validate()
        native = GaitDouble(64)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, fourteen)
        # 10 updates above 0.8 raise the level each time, up to 0.125 m/s; the window is read every 7
        # controls, so the first read after 360 controls comes at 364.
        for level in (0.1, 0.125):
            for _ in range(364):
                task._curriculum_step(torch.full((64,), 0.9))
            self.assertEqual(task.curriculum_level, level)
        # Low pay holds the level until 1800 controls have passed there, then lowers it once.
        for _ in range(1799):
            task._curriculum_step(torch.full((64,), 0.5))
        self.assertEqual(task.curriculum_level, 0.125)
        for _ in range(7):
            task._curriculum_step(torch.full((64,), 0.5))
        self.assertEqual(task.curriculum_level, 0.1)
        self.assertTrue(task.curriculum_status()["advances"][-1]["back_off"])

    def test_destination_v15_rises_only_after_a_window_without_falls(self):
        fourteen, fifteen = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v14", "destination_v15")
        )
        changed = {k for k, v in asdict(fifteen).items() if asdict(fourteen)[k] != v}
        self.assertEqual(changed, {"speed_curriculum_advance_without_falls"})
        self.assertEqual(EXPERIMENT_INPUTS["destination_v15"], "ghost")
        native = GaitDouble(64)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, fifteen)
        none, one = torch.zeros(64, dtype=torch.bool), torch.zeros(64, dtype=torch.bool)
        one[3] = True
        # High pay with one fall at control 100 holds the level at the read after 364 controls; the fall
        # leaves the 360-control window at control 460 and the level rises at the next read (462).
        for k in range(364):
            task._curriculum_step(torch.full((64,), 0.9), one if k == 99 else none)
        self.assertEqual(task.curriculum_level, 0.075)
        self.assertEqual(task.curriculum_status()["window_falls"], 1)
        for _ in range(98):
            task._curriculum_step(torch.full((64,), 0.9), none)
        self.assertEqual(task.curriculum_level, 0.1)

    def test_destination_v12_is_v10_with_the_back_off(self):
        ten, twelve = (EXPERIMENTS[n]["reward"] for n in ("destination_v10", "destination_v12"))
        changed = {k for k, v in asdict(twelve).items() if asdict(ten)[k] != v}
        self.assertEqual(changed, {"speed_curriculum_backoff_threshold"})
        self.assertEqual(EXPERIMENT_INPUTS["destination_v12"], "ghost")


class GaitClockTests(unittest.TestCase):
    def test_speed_v1_is_the_fixed_clock_up_to_0_075_and_shortens_the_cycle_above(self):
        from locomotion.omniwalk.mdp.gait import GaitPhase, gait_cycle
        from locomotion.omniwalk.mdp.rewards import scheduled_swing
        from locomotion.rl.wrapper import clock_features

        command = torch.tensor([[0.0, 0.05, 0.0], [-0.075, 0.0, 0.0], [0.0, 0.0, 0.2]])
        gait = GaitPhase(3, "speed_v1", "cpu")
        for k in range(600):
            steps = torch.full((3,), k)
            swing, period = gait.swing(command)
            self.assertTrue(bool((swing == scheduled_swing(steps)).all()), k)
            torch.testing.assert_close(period, torch.full((3,), 1.2, dtype=torch.float64))
            torch.testing.assert_close(
                clock_features(steps, 60, command, gait.phase),
                clock_features(steps, 60, command),
                atol=1e-5,
                rtol=0,
            )
            gait.advance(command)
        walks = torch.tensor([[0.0875, 0.0, 0.0], [0.0, -0.1, 0.0], [0.125, 0.0, 0.0], [0.3, 0, 0]])
        period, swing = gait_cycle(walks, "speed_v1")
        torch.testing.assert_close(
            period, torch.tensor([56.0, 52.0, 44.0, 40.0], dtype=torch.float64)
        )
        torch.testing.assert_close(swing, torch.tensor([0.44, 0.48, 0.5, 0.5], dtype=torch.float64))

    def test_evaluation_follows_the_phase_the_training_task_kept(self):
        from locomotion.omniwalk.mdp.gait import GaitPhase

        trained, followed = GaitPhase(1, "speed_v1", "cpu"), GaitPhase(1, "speed_v1", "cpu")
        speeds = [0.1] * 70 + [0.125] * 50 + [0.0] * 30 + [0.15] * 40
        for k, speed in enumerate(speeds):
            command = torch.tensor([[speed, 0.0, 0.0]])
            seen = followed.follow(command, torch.tensor([k])).clone()
            torch.testing.assert_close(seen, trained.phase)
            trained.advance(command)
        # A 0.125 m/s walk crosses half its 44-control cycle in 22 controls.
        gait = GaitPhase(1, "speed_v1", "cpu")
        for _ in range(22):
            gait.advance(torch.tensor([[0.125, 0.0, 0.0]]))
        self.assertAlmostEqual(float(gait.phase[0]), 0.5)

    def test_destination_v16_scales_the_clock_and_backs_off_a_robot_5_cm_from_its_ghost(self):
        fifteen, sixteen = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v15", "destination_v16")
        )
        changed = {k for k, v in asdict(sixteen).items() if asdict(fifteen)[k] != v}
        self.assertEqual(changed, {"gait_clock_scaling", "speed_curriculum_backoff_distance_m"})
        self.assertEqual(EXPERIMENT_INPUTS["destination_v16"], "ghost")
        with self.assertRaises(ValueError):
            replace(sixteen, gait_clock_scaling="speed_v0").validate()
        native = GaitDouble(64)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, sixteen)
        near, none = torch.zeros(64), torch.zeros(64, dtype=torch.bool)
        for _ in range(364):
            task._curriculum_step(torch.full((64,), 0.9), none, near)
        self.assertEqual(task.curriculum_level, 0.1)
        # A fallen robot's distance does not count (its fall holds the level); a standing one 6 cm away
        # lowers it at the next read of a full window, however high the pay.
        far, fell = near.clone(), none.clone()
        far[5], fell[5] = 0.2, True
        for k in range(364):
            task._curriculum_step(
                torch.full((64,), 0.9), fell if k == 10 else none, far if k == 10 else near
            )
        self.assertEqual(task.curriculum_level, 0.1)
        self.assertEqual(task.curriculum_status()["window_lost"], 0)
        far[5] = 0.06
        for _ in range(7):
            task._curriculum_step(torch.full((64,), 0.9), none, far)
        self.assertEqual(task.curriculum_level, 0.075)
        # The first level holds while any robot stands that far away, and rises once none has for a window.
        for _ in range(364):
            task._curriculum_step(torch.full((64,), 0.9), none, far)
        self.assertEqual(task.curriculum_level, 0.075)
        for _ in range(364):
            task._curriculum_step(torch.full((64,), 0.9), none, near)
        self.assertEqual(task.curriculum_level, 0.1)
        record = task.curriculum_status()["advances"][-2]
        self.assertEqual((record["back_off"], record["reason"]), (True, "distance"))

    def test_destination_v17_never_backs_off_and_rises_after_30_updates(self):
        sixteen, seventeen = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v16", "destination_v17")
        )
        changed = {k for k, v in asdict(seventeen).items() if asdict(sixteen)[k] != v}
        self.assertEqual(
            changed,
            {
                "speed_curriculum_window_controls",
                "speed_curriculum_backoff_threshold",
                "speed_curriculum_backoff_window_controls",
                "speed_curriculum_backoff_distance_m",
            },
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v17"], "ghost")
        self.assertEqual(seventeen.gait_clock_scaling, "speed_v1")
        native = GaitDouble(64)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, seventeen)
        none = torch.zeros(64, dtype=torch.bool)
        # The window is read every 21 controls: 1071 controls are too few, the read at 1092 raises the level.
        for _ in range(1071):
            task._curriculum_step(torch.full((64,), 0.9), none)
        self.assertEqual(task.curriculum_level, 0.075)
        for _ in range(21):
            task._curriculum_step(torch.full((64,), 0.9), none)
        self.assertEqual(task.curriculum_level, 0.1)
        # Low pay, falls and far robots hold the level and never lower it.
        fell, far = none.clone(), torch.full((64,), 0.5)
        fell[0] = True
        for _ in range(5000):
            task._curriculum_step(torch.full((64,), 0.1), fell, far)
        self.assertEqual(task.curriculum_level, 0.1)
        self.assertFalse(any(a.get("back_off") for a in task.curriculum_status()["advances"]))

    def test_destination_v18_rises_only_when_every_finished_walk_passes_the_checks(self):
        from locomotion.omniwalk.destination_task import WALK_CHECKS, walk_failures

        seventeen, eighteen = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v17", "destination_v18")
        )
        changed = {k for k, v in asdict(eighteen).items() if asdict(seventeen)[k] != v}
        self.assertEqual(changed, {"speed_curriculum_pass_checks"})
        self.assertEqual(EXPERIMENT_INPUTS["destination_v18"], "ghost")
        # The evaluator's bounds: 0.0316 m/s of mean planar error fails a 0.125 m/s walk (bound 0.03125)
        # and torque above 1.6 Nm in 0.6% of joint-controls fails any walk.
        sums = torch.tensor([[0.0316, 0.0, 0.0, 0.0, 0.0], [0.031, 0.0, 0.0, 0.0, 0.006]])
        failed = walk_failures(
            sums,
            torch.ones(2),
            torch.full((2,), 0.125),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
        )
        named = [[c for c, f in zip(WALK_CHECKS, row) if f] for row in failed.tolist()]
        self.assertEqual(named, [["planar_error_mps"], ["computed_demand_over_rating_fraction"]])
        native = GaitDouble(4)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, eighteen)
        task.reset()
        command = torch.tensor([[0.1, 0.0, 0.0]]).repeat(4, 1)
        native.commands[:] = command
        # Row 0 walks on the command; row 1 lags 0.05 m/s; both are scored when their commands end.
        velocity = torch.tensor(
            [[0.0, -0.1, 0.0], [0.0, -0.05, 0.0], [0.0, -0.1, 0.0], [0.0, -0.1, 0.0]]
        )
        native.current["root_velocity"] = torch.cat((velocity, torch.zeros(4, 3)), -1)
        identity = torch.tensor([[0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0]]).repeat(4, 1)
        telemetry = {
            "root_pose_xyzw": identity,
            "angular_velocity_body": torch.zeros(4, 3),
            "computed_torque_nm": torch.zeros(4, 18),
            "joint_velocity_rad_s": torch.zeros(4, 18),
        }
        task.command_elapsed[:] = 0
        for _ in range(150):
            task.command_elapsed += 1
            task._record_walks(command, telemetry, torch.zeros(4, dtype=torch.bool))
        self.assertEqual(task.walk_counted.tolist(), [50.0] * 4)
        task._finish_walks(torch.tensor([0, 1]))
        self.assertEqual((task.pending_walks, task.pending_failed), (2, 1))
        self.assertEqual(task.check_failures["planar_error_mps"], 1)
        # A failed walk in the window holds the level however high the pay; a clean window raises it.
        none = torch.zeros(4, dtype=torch.bool)
        task.pending_walks = task.pending_failed = 0
        for k in range(1092):
            if k == 100:
                task.pending_walks, task.pending_failed = 2, 1
            task._curriculum_step(torch.full((4,), 0.9), none)
        self.assertEqual(task.curriculum_level, 0.075)
        self.assertEqual(task.curriculum_status()["window_walks_failed"], 1)
        for k in range(1092):
            task.pending_walks += k == 5
            task._curriculum_step(torch.full((4,), 0.9), none)
        self.assertEqual(task.curriculum_level, 0.1)

    def test_the_status_reports_the_speed_of_walks_at_the_top_level(self):
        native = GaitDouble(4)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(
            native, TaskConfig(), None, EXPERIMENTS["destination_v18"]["reward"]
        )
        # Rows 0 and 1 walk at the 0.075 m/s level (forward, then left); row 2 walks slower; row 3 fell.
        command = torch.tensor(
            [[0.075, 0.0, 0.0], [0.0, 0.075, 0.0], [0.05, 0.0, 0.0], [0.075, 0.0, 0.0]]
        )
        nav = torch.tensor([[0.07, 0.01, 0.0], [0.0, 0.08, 0.0], [0.05, 0.0, 0.0], [0.0, 0.0, 0.0]])
        fell = torch.tensor([False, False, False, True])
        task._record_top_speed(command, {"linear_velocity_nav": nav}, fell)
        status = task.curriculum_status()
        self.assertEqual(status["top_speed_controls"], 2)
        self.assertAlmostEqual(status["top_speed_along_command_sum_mps"], 0.15, places=6)
        # A new level starts the count again.
        for _ in range(1092):
            task.pending_walks += 1
            task._curriculum_step(torch.full((4,), 0.9), torch.zeros(4, dtype=torch.bool))
        self.assertEqual(task.curriculum_level, 0.1)
        self.assertEqual(task.curriculum_status()["top_speed_controls"], 0)

    def test_destination_v19_charges_a_growing_lag_and_doubles_the_planar_bound(self):
        from locomotion.omniwalk.destination_task import WALK_CHECKS, linear_tail, walk_failures

        eighteen, nineteen = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v18", "destination_v19")
        )
        changed = {k for k, v in asdict(nineteen).items() if asdict(eighteen)[k] != v}
        self.assertEqual(
            changed,
            {
                "destination_linear_from_m",
                "destination_penalty_cap",
                "speed_curriculum_planar_bound_scale",
            },
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v19"], "ghost")
        with self.assertRaises(ValueError):
            replace(nineteen, destination_kernel="linear", destination_power=None).validate()
        # The rational kernel up to 20 mm, then its tangent: zero near 49 mm and -1 from 117 mm on.
        distance = torch.tensor([0.005, 0.02, 0.0201, 0.0493, 0.1, 0.2])
        kernel = 1 / (1 + (distance / nineteen.destination_distance_m).pow(1.2))
        pay = linear_tail(distance, kernel, nineteen)
        torch.testing.assert_close(pay[:2], kernel[:2])
        torch.testing.assert_close(
            pay, torch.tensor([0.8, 0.4311, 0.4297, 0.0, -0.7457, -1.0]), atol=2e-3, rtol=0
        )
        # At 0.125 m/s the walking check now allows 62.5 mm/s of mean planar error.
        failed = walk_failures(
            torch.tensor([[0.06, 0, 0, 0, 0], [0.065, 0, 0, 0, 0]]),
            torch.ones(2),
            torch.full((2,), 0.125),
            torch.zeros(2, dtype=torch.bool),
            torch.zeros(2, dtype=torch.bool),
            nineteen.speed_curriculum_planar_bound_scale,
        )
        self.assertEqual(failed[:, WALK_CHECKS.index("planar_error_mps")].tolist(), [False, True])

    def test_the_wrapper_shows_the_task_phase_and_records_the_scaling(self):
        sixteen = EXPERIMENTS["destination_v16"]["reward"]
        task, native = moving_task(lambda k: (0.0, -0.125 * 0.02 * k, 0.0), reward=sixteen)
        with self.assertRaises(ValueError):
            DestinationVecEnv(task, plain, observation_scaling="fixed", gait_clock=60)
        env = DestinationVecEnv(
            task, plain, observation_scaling="fixed", gait_clock=60, gait_clock_scaling="speed_v1"
        )
        native.commands[:] = torch.tensor([[0.125, 0.0, 0.0]])
        task.remaining_controls[:] = 1000
        env.get_observations()
        for _ in range(22):
            observation, *_ = env.step(torch.zeros(1, 18))
        torch.testing.assert_close(
            observation["policy"][0, 231:233], torch.tensor([0.0, -1.0]), atol=1e-5, rtol=0
        )
        # The reward scheduled tripod B's swing at the half cycle, not the fixed clock's control 22.
        self.assertEqual(task.last_components["gait_schedule"].shape, (1,))
        swing, _ = task.gait.swing(native.commands)
        self.assertEqual(swing[0].tolist(), [False, True, False, True, False, True])


class GhostV10Tests(unittest.TestCase):
    def test_the_inputs_are_the_sideways_gap_and_the_velocity_relative_to_the_ghost(self):
        from locomotion.omniwalk.mdp.destination import ghost_v10_features

        # Heading 0: navigation forward is world -y and left is world +x. The ghost is 30 mm ahead and
        # 10 mm to the left; the body moved 2 mm forward in the last control while the ghost walks 0.15 m/s.
        pose, before = pose_at(0.0, 0.0, 0.0), pose_at(0.0, 0.002, 0.0)
        destination = torch.tensor([[0.01, -0.03, 0.0]])
        command = torch.tensor([[0.15, 0.0, 0.0]])
        value = ghost_v10_features(pose, before, destination, torch.tensor([50]), command)
        torch.testing.assert_close(
            value[0], torch.tensor([0.0, 0.1, 0.0, 1.0, 0.2, -0.5, 0.0]), atol=1e-5, rtol=0
        )
        # Under a zero command the whole gap is the sideways input and the velocity is the body's own.
        value = ghost_v10_features(pose, before, destination, torch.tensor([50]), command * 0)
        torch.testing.assert_close(value[0, :2], torch.tensor([0.3, 0.1]), atol=1e-5, rtol=0)
        torch.testing.assert_close(value[0, 5:], torch.tensor([1.0, 0.0]), atol=1e-5, rtol=0)

    def wrapped(self):
        twenty = EXPERIMENTS["destination_v20"]["reward"]
        task, native = moving_task(lambda k: (0.0, -0.03 * 0.02 * k, 0.0), reward=twenty)
        env = DestinationVecEnv(
            task,
            plain,
            observation_scaling="fixed",
            gait_clock=60,
            gait_clock_scaling="speed_v1",
            destination_input="ghost_v10",
        )
        native.commands[:] = FORWARD
        task.remaining_controls[:] = 1000
        native.current["root_velocity"] = torch.zeros(1, 6)  # read by the walking checks
        return env, task, native

    def test_destination_v20_gives_the_policy_seven_ghost_inputs_that_track_the_lag(self):
        nineteen, twenty = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v19", "destination_v20")
        )
        changed = {k for k, v in asdict(twenty).items() if asdict(nineteen)[k] != v}
        self.assertEqual(changed, {"speed_curriculum_planar_bound_scale"})
        self.assertEqual(EXPERIMENT_INPUTS["destination_v20"], "ghost_v10")
        self.assertEqual(
            ppo_config(7, gait_clock=60, destination_input="ghost_v10")["environment_wrapper"][
                "destination_input"
            ],
            "ghost_v10",
        )
        env, task, native = self.wrapped()
        first = env.get_observations()
        self.assertEqual(
            (first["policy"].shape[-1], first["critic"].shape[-1]), (231 + 2 + 7, 234 + 2 + 7)
        )
        for _ in range(100):
            observation, *_ = env.step(torch.zeros(1, 18))
        # Walking straight 0.02 m/s short of the command: no sideways gap, 0.2 below the ghost's speed.
        torch.testing.assert_close(
            observation["policy"][0, -7:],
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.4, -0.2, 0.0]),
            atol=1e-4,
            rtol=0,
        )
        torch.testing.assert_close(observation["critic"][0, -7:], observation["policy"][0, -7:])

    def test_evaluation_rebuilds_the_training_inputs(self):
        from locomotion.omniwalk.mdp.destination import GhostV10Tracker

        env, task, native = self.wrapped()
        tracker, seen, trained = GhostV10Tracker(1, "cpu"), [], []
        for _ in range(30):
            seen.append(
                tracker.inputs(native.current["root"], native.commands, native.episode_steps)[0]
            )
            observation, *_ = env.step(torch.zeros(1, 18))
            trained.append(observation["policy"][0, -7:].clone())
        for k in range(29):
            torch.testing.assert_close(seen[k + 1], trained[k], atol=1e-5, rtol=0)

    def test_ghost_v11_adds_the_along_track_gap_and_rebuilds_in_evaluation(self):
        from locomotion.omniwalk.mdp.destination import (
            GhostV11Tracker,
            destination_features,
            ghost_v10_features,
        )

        self.assertEqual(EXPERIMENTS["destination_v21"], EXPERIMENTS["destination_v20"])
        self.assertEqual(EXPERIMENT_INPUTS["destination_v21"], "ghost_v11")
        # Sideways plus along-track is the ghost gap; the first seven values are ghost_v10's.
        pose, before = pose_at(0.0, 0.0, 0.3), pose_at(0.001, 0.002, 0.3)
        destination = torch.tensor([[0.01, -0.03, 0.35]])
        for command in (torch.tensor([[0.12, 0.05, 0.0]]), torch.zeros(1, 3)):
            value = ghost_v10_features(pose, before, destination, torch.tensor([9]), command, True)
            torch.testing.assert_close(
                value[:, :7],
                ghost_v10_features(pose, before, destination, torch.tensor([9]), command),
            )
            gap = destination_features(pose, destination, torch.tensor([9]))[:, :2]
            torch.testing.assert_close(value[:, :2] + value[:, 7:], gap, atol=1e-6, rtol=0)
        twenty = EXPERIMENTS["destination_v21"]["reward"]
        task, native = moving_task(lambda k: (0.0, -0.03 * 0.02 * k, 0.0), reward=twenty)
        env = DestinationVecEnv(
            task,
            plain,
            observation_scaling="fixed",
            gait_clock=60,
            gait_clock_scaling="speed_v1",
            destination_input="ghost_v11",
        )
        native.commands[:] = FORWARD
        task.remaining_controls[:] = 1000
        native.current["root_velocity"] = torch.zeros(1, 6)
        self.assertEqual(env.get_observations()["policy"].shape[-1], 231 + 2 + 9)
        tracker, seen, trained = GhostV11Tracker(1, "cpu"), [], []
        for _ in range(100):
            seen.append(
                tracker.inputs(native.current["root"], native.commands, native.episode_steps)[0]
            )
            observation, *_ = env.step(torch.zeros(1, 18))
            trained.append(observation["policy"][0, -9:].clone())
        for k in range(99):
            torch.testing.assert_close(seen[k + 1], trained[k], atol=1e-5, rtol=0)
        # 100 controls 0.02 m/s short of the command: 40 mm behind along the path, 0.2 below its speed.
        torch.testing.assert_close(
            trained[-1],
            torch.tensor([0.0, 0.0, 0.0, 1.0, 0.4, -0.2, 0.0, 0.4, 0.0]),
            atol=1e-4,
            rtol=0,
        )


class DriftTests(unittest.TestCase):
    def walk(self, motion):
        """One 200-control forward walk at 0.05 m/s under destination_v22; returns the task after it ends."""
        task, native = moving_task(motion, reward=EXPERIMENTS["destination_v22"]["reward"])
        native.current["root_velocity"] = torch.zeros(1, 6)
        task.remaining_controls[:] = 200
        task.drift_walking[:], task.drift_plan[:], task.drift_halfway[:] = True, 200, False
        for _ in range(200):
            task.step(torch.zeros(1, 18))
        return task

    def test_destination_v22_judges_the_curriculum_on_drift_from_halfway_to_the_end(self):
        twentyone, twentytwo = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v21", "destination_v22")
        )
        changed = {k for k, v in asdict(twentytwo).items() if asdict(twentyone)[k] != v}
        self.assertEqual(changed, {"speed_curriculum_measure"})
        self.assertEqual(EXPERIMENT_INPUTS["destination_v22"], "ghost_v11")
        with self.assertRaises(ValueError):
            replace(twentytwo, speed_curriculum_measure="pay").validate()
        # A walk that starts a second late and then holds the ghost's speed keeps a steady 50 mm gap: no drift.
        late = self.walk(lambda k: (0.0, -0.05 * 0.02 * max(0, k - 50), 0.0))
        self.assertEqual(late.pending_scored, 1)
        self.assertAlmostEqual(late.pending_drift, 0.0, places=5)
        self.assertAlmostEqual(late.pending_score, 1.0, places=4)
        # A walk 0.02 m/s slow drifts 40 mm over its second half and scores the kernel there (0.25).
        slow = self.walk(lambda k: (0.0, -0.03 * 0.02 * k, 0.0))
        self.assertAlmostEqual(slow.pending_drift, 0.04, places=5)
        self.assertAlmostEqual(slow.pending_score, 1 / (1 + (0.04 / 0.015874) ** 1.2), places=3)

    def test_the_window_rises_on_the_mean_drift_score_of_finished_walks(self):
        native = GaitDouble(4)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(
            native, TaskConfig(), None, EXPERIMENTS["destination_v22"]["reward"]
        )
        none = torch.zeros(4, dtype=torch.bool)
        # High pay with no scored walks never rises; walks scoring 0.7 hold it; walks scoring 0.9 raise it.
        for _ in range(1092):
            task.pending_walks += 1
            task._curriculum_step(torch.full((4,), 0.95), none)
        self.assertEqual(task.curriculum_level, 0.075)
        for score in (0.7, 0.9):
            for _ in range(1092):
                task.pending_walks += 1
                task.pending_scored, task.pending_score, task.pending_drift = 1.0, score, 0.004
                task._curriculum_step(torch.full((4,), 0.3), none)
        # It rises once the rolling mean of scored walks passes 0.8, while 0.9s replace the 0.7s.
        self.assertEqual(task.curriculum_level, 0.1)
        self.assertGreater(task.curriculum_advances[-1]["window_mean"], 0.8)
        self.assertLess(task.curriculum_advances[-1]["window_mean"], 0.81)

    def test_destination_v23_charges_the_evaluators_planar_error_each_walking_control(self):
        twentytwo, twentythree = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v22", "destination_v23")
        )
        changed = {k for k, v in asdict(twentythree).items() if asdict(twentytwo)[k] != v}
        self.assertEqual(changed, {"planar_error_weight"})
        self.assertEqual(EXPERIMENT_INPUTS["destination_v23"], "ghost_v11")
        native = GaitDouble(4)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, twentythree)
        # At heading 0 navigation forward is world -y. Rows: on the command; one bound (37.5 mm/s) off at
        # 0.15 m/s; 0.05 m/s off a 0.05 m/s walk (bound 25 mm/s); a zero command, never charged.
        command = torch.tensor(
            [[0.15, 0.0, 0.0], [0.15, 0.0, 0.0], [0.05, 0.0, 0.0], [0.0, 0.0, 0.0]]
        )
        native.current["root_velocity"] = torch.tensor(
            [
                [0.0, -0.15, 0, 0, 0, 0],
                [0.0375, -0.15, 0, 0, 0, 0],
                [0.0, -0.1, 0, 0, 0, 0],
                [0.3, 0.0, 0, 0, 0, 0],
            ]
        )
        telemetry = {"root_pose_xyzw": torch.tensor([[0.0, 0, 0.1, 0, 0, 0, 1.0]]).repeat(4, 1)}
        torch.testing.assert_close(
            task._planar_error(command, telemetry), torch.tensor([0.0, -0.5, -2.0, 0.0])
        )

    def test_destination_v24_continues_the_curriculum_to_0_175(self):
        twentythree, twentyfour = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v23", "destination_v24")
        )
        changed = {k for k, v in asdict(twentyfour).items() if asdict(twentythree)[k] != v}
        self.assertEqual(changed, {"speed_curriculum_final_mps"})
        self.assertEqual(twentyfour.speed_curriculum_final_mps, 0.175)
        self.assertEqual(EXPERIMENT_INPUTS["destination_v24"], "ghost_v11")

    def test_named_experiment_clocks_match_their_rewards_and_reach_the_learner_configuration(self):
        from locomotion.omniwalk.agents.rsl_rl_ppo_cfg import EXPERIMENT_CLOCKS

        for name, settings in EXPERIMENTS.items():
            self.assertEqual(
                EXPERIMENT_CLOCKS.get(name), settings["reward"].gait_clock_scaling, name
            )
        config = ppo_config(
            7,
            action_mean="tanh",
            learning_rate_max=1e-4,
            learning_rate_min=2e-5,
            gait_clock=60,
            destination_input="ghost_v11",
            gait_clock_scaling="speed_v1",
        )
        self.assertEqual(config["algorithm"]["learning_rate_min"], 2e-5)
        self.assertEqual(config["environment_wrapper"]["gait_clock_scaling"], "speed_v1")
        self.assertNotIn("learning_rate_min", ppo_config(7, learning_rate_max=1e-4)["algorithm"])
        for options in (
            {"learning_rate_min": 2e-5},
            {"learning_rate_max": 1e-4, "learning_rate_min": 2e-4},
            {"gait_clock_scaling": "speed_v1"},
        ):
            with self.assertRaises(ValueError):
                ppo_config(7, **options)

    def test_destination_v25_caps_a_linear_planar_error_charge(self):
        twentythree, twentyfive = (
            EXPERIMENTS[n]["reward"] for n in ("destination_v23", "destination_v25")
        )
        changed = {k for k, v in asdict(twentyfive).items() if asdict(twentythree)[k] != v}
        self.assertEqual(changed, {"planar_error_cap"})
        self.assertEqual(EXPERIMENT_INPUTS["destination_v25"], "ghost_v11")
        with self.assertRaises(ValueError):
            replace(twentyfive, planar_error_weight=None).validate()
        native = GaitDouble(4)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, twentyfive)
        # At 0.15 m/s (bound 37.5 mm/s): on the command, at the bound, at three bounds (capped), stopped.
        command = torch.tensor([[0.15, 0.0, 0.0]] * 3 + [[0.0, 0.0, 0.0]])
        native.current["root_velocity"] = torch.tensor(
            [
                [0.0, -0.15, 0, 0, 0, 0],
                [0.0375, -0.15, 0, 0, 0, 0],
                [0.1125, -0.15, 0, 0, 0, 0],
                [0.3, 0.0, 0, 0, 0, 0],
            ]
        )
        telemetry = {"root_pose_xyzw": torch.tensor([[0.0, 0, 0.1, 0, 0, 0, 1.0]]).repeat(4, 1)}
        torch.testing.assert_close(
            task._planar_error(command, telemetry), torch.tensor([0.0, -0.5, -1.0, 0.0])
        )

    def test_destination_v26_walks_at_uniform_speeds_and_stops_half_the_walks(self):
        twentysix = EXPERIMENTS["destination_v26"]["reward"]
        self.assertIsNone(twentysix.speed_curriculum_start_mps)
        self.assertEqual(EXPERIMENT_INPUTS["destination_v26"], "ghost_v11")
        with self.assertRaises(ValueError):
            replace(twentysix, walk_speed_min_mps=0.2).validate()
        native = GaitDouble(4096)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, twentysix)
        task.reset()
        command = native.commands
        speed = torch.linalg.vector_norm(command[:, :2], dim=-1)
        # Every row walks without yaw at 0.05 to 0.15 m/s, spread across the range; about half will stop.
        self.assertTrue(bool((speed >= 0.05 - 1e-6).all()) and bool((speed <= 0.15 + 1e-6).all()))
        self.assertTrue(bool((command[:, 2] == 0).all()))
        self.assertAlmostEqual(float(speed.mean()), 0.1, delta=0.003)
        self.assertAlmostEqual(float(task.stop_pending.double().mean()), 0.5, delta=0.03)
        self.assertIsNone(task.curriculum_status())

    def test_destination_v27_stops_a_quarter_of_the_walks(self):
        twentyseven = EXPERIMENTS["destination_v27"]["reward"]
        self.assertEqual(
            replace(twentyseven, stop_fraction=0.5), EXPERIMENTS["destination_v26"]["reward"]
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v27"], "ghost_v11")
        native = GaitDouble(4096)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, twentyseven)
        task.reset()
        self.assertAlmostEqual(float(task.stop_pending.double().mean()), 0.25, delta=0.03)

    def test_destination_v28_mixes_runs_and_pays_the_ghost_gap_more_towards_each_runs_end(self):
        settings = EXPERIMENTS["destination_v28"]
        eight = settings["reward"]
        self.assertEqual(settings["episode_seconds"], 32.0)
        self.assertEqual(EXPERIMENT_INPUTS["destination_v28"], "ghost_v11")
        with self.assertRaises(ValueError):
            replace(eight, run_mix=(0.5, 0.5, 0.5, 0.0)).validate()
        with self.assertRaises(ValueError):
            replace(eight, termination_weight=42.0).validate()
        native = GaitDouble(8192)
        native.cfg.episode_seconds, native.cfg.declaration = 32.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, eight)
        task.reset()
        command = native.commands
        planar = torch.linalg.vector_norm(command[:, :2], dim=-1)
        turns = (planar == 0) & (command[:, 2] != 0)
        stands = (command == 0).all(-1)
        walks = planar > 0
        self.assertTrue(bool((command[turns, 2].abs() == 0.2).all()))
        self.assertTrue(
            bool((planar[walks] >= 0.05 - 1e-6).all() & (planar[walks] <= 0.15 + 1e-6).all())
        )
        for share, rows in ((0.1, turns), (0.1, stands), (0.5, task.stop_pending), (0.8, walks)):
            self.assertAlmostEqual(float(rows.double().mean()), share, delta=0.015)
        long = task.run_length == 1600
        self.assertTrue(bool((long == (stands & (task.remaining_controls == 1600))).all()))
        self.assertAlmostEqual(float(long.double().mean()), 0.05, delta=0.01)

        # Shortened runs: walks stop after 10 to 20 controls of 60 and long stands last 90.
        short = replace(
            eight,
            command_controls=60,
            long_stand_controls=90,
            stop_after_min_controls=10,
            stop_after_max_controls=20,
        )
        native = GaitDouble(512)
        native.cfg.episode_seconds, native.cfg.declaration = 32.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, short)
        task.reset()
        stopping, length = task.stop_pending.clone(), task.run_length.clone()
        still = (native.commands == 0).all(-1)
        for k in range(1, 61):
            result = task.step(torch.zeros(512, 18))
            # The body stays at the origin, so a stand is at its ghost and earns the run's share done.
            torch.testing.assert_close(
                task.last_components["end_destination"][still], (k / length[still]).float()
            )
            self.assertTrue(bool((result["truncated"] == (length == k)).all()))
            if k == 30:
                self.assertTrue(bool((native.commands[stopping] == 0).all()))
        # The 60-control runs keep their command until the reset that the timeout calls for.
        self.assertTrue(bool((task.remaining_controls[length == 60] == 0).all()))

    def test_destination_v29_pays_the_run_end_gap_linearly_down_to_a_floor(self):
        nine = EXPERIMENTS["destination_v29"]["reward"]
        self.assertEqual(EXPERIMENT_INPUTS["destination_v29"], "ghost_v11")
        for broken in ({"end_destination_floor": None}, {"end_destination_power": 1.2}):
            with self.assertRaises(ValueError):
                replace(nine, **broken).validate()
        native = GaitDouble(4)
        native.cfg.episode_seconds, native.cfg.declaration = 32.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, nine)
        task.reset()
        native.commands[:] = 0
        task.stop_pending[:] = False
        task.run_length[:] = 4
        task.remaining_controls[:] = 4
        task.step(torch.zeros(4, 18))
        # At control 2 of 4, ghosts 0, 12.5, 25 and 75 mm from the body earn half of 1, 0.5, 0 and -1.
        task.destination[:, :2] = native.current["root"][:, :2].to(task.destination)
        task.destination[:, 0] += torch.tensor([0.0, 0.0125, 0.025, 0.075])
        task.command_started[:] = False
        task.step(torch.zeros(4, 18))
        torch.testing.assert_close(
            task.last_components["end_destination"], torch.tensor([0.5, 0.25, 0.0, -0.5])
        )

    def test_destination_v30_is_the_v25_curriculum_with_the_linear_run_end_pay(self):
        thirty = EXPERIMENTS["destination_v30"]["reward"]
        nine = EXPERIMENTS["destination_v29"]["reward"]
        end = ("end_destination_weight", "end_destination_distance_m", "end_destination_kernel")
        self.assertEqual([getattr(thirty, k) for k in end], [getattr(nine, k) for k in end])
        self.assertEqual(
            replace(
                thirty,
                end_destination_weight=None,
                end_destination_distance_m=None,
                end_destination_kernel=None,
                end_destination_floor=None,
                termination_weight=42.0,
            ),
            EXPERIMENTS["destination_v25"]["reward"],
        )
        self.assertEqual(EXPERIMENTS["destination_v30"]["episode_seconds"], 20.0)
        native = GaitDouble(64)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, thirty)
        task.reset()
        self.assertEqual(task.curriculum_status()["level_mps"], 0.075)
        self.assertTrue(bool((task.run_length == 1000).all()))
        native.current["root_velocity"] = torch.zeros(64, 6)
        task.step(torch.zeros(64, 18))
        self.assertIn("end_destination", task.last_components)

    def test_destination_v32_walks_forward_and_feels_the_slope_in_its_joint_inputs(self):
        settings = EXPERIMENTS["destination_v32"]
        two = settings["reward"]
        self.assertEqual(
            (two.slope_inputs, two.walk_forward_only, settings["episode_seconds"]),
            ("joints", True, 20.0),
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v32"], "ghost_v11")
        with self.assertRaises(ValueError):
            replace(two, slope_inputs="knees").validate()
        native = GaitDouble(64)
        native.cfg.episode_seconds, native.cfg.declaration = 20.0, dict
        native.current["q"] = torch.zeros(64, 18)
        task = DestinationTrainingTask(native, TaskConfig(), None, two)
        first = task.reset()
        # Every run walks straight ahead at 0.05 to 0.15 m/s; nothing is tilted yet.
        command = native.commands
        self.assertTrue(bool((command[:, 1:] == 0).all() and (command[:, 0] >= 0.05 - 1e-6).all()))
        self.assertTrue(bool((command[:, 0] <= 0.15 + 1e-6).all()))
        self.assertTrue(torch.equal(first["critic"][:, :231], first["obs"]))
        # Front toes (lf, rf) 30 mm above the rear ones (lr, rr), 0.3 m apart along forward (world -y).
        native.world[:, :, 1] = torch.tensor([-0.15, 0.0, 0.15, -0.15, 0.0, 0.15])
        native.world[:, [0, 3], 2] = 0.03
        native.current["root_velocity"] = torch.zeros(64, 6)
        for _ in range(40):
            result = task.step(torch.zeros(64, 18))
        tilt = math.degrees(float(task.slope.pitch[0]))
        self.assertAlmostEqual(tilt, math.degrees(math.atan2(0.03, 0.3)), delta=0.05)
        raw = native.last_result["obs"]
        offset = (result["obs"] - raw)[:, 42 * 4 + 6 : 42 * 4 + 24]
        # Front femur and tibia inputs gain the tilt, rear ones lose it; coxas and the rest stay.
        self.assertGreater(float(offset[0, 1]), 0.05)
        self.assertLess(float(offset[0, 7]), -0.05)
        self.assertAlmostEqual(float(offset[0, 2]), float(offset[0, 1]), places=4)
        self.assertEqual(float(offset[:, 0::3].abs().sum()), 0.0)
        self.assertTrue(torch.equal(result["critic"][:, :231], result["obs"]))

    def test_the_capped_rate_learner_can_hold_a_learning_rate_floor(self):
        from locomotion.rl.ppo import CappedRatePPO

        learner = CappedRatePPO.__new__(CappedRatePPO)
        learner.learning_rate_max, learner.learning_rate_min = 1e-4, 2e-5
        for value, kept in ((1e-5, 2e-5), (5e-5, 5e-5), (1e-3, 1e-4)):
            learner.learning_rate = value
            self.assertAlmostEqual(learner.learning_rate, kept)


class WrapperTests(unittest.TestCase):
    def wrapped(self, **options):
        task, native = moving_task(lambda k: (0.0, -0.03 * 0.02 * k, 0.0))  # 0.02 m/s short
        env = DestinationVecEnv(task, plain, observation_scaling="fixed", gait_clock=60, **options)
        native.commands[:] = FORWARD
        task.remaining_controls[:] = 1000
        return env, task, native

    def test_both_inputs_gain_five_values_that_track_the_lag(self):
        env, task, native = self.wrapped()
        first = env.get_observations()
        self.assertEqual(
            (first["policy"].shape[-1], first["critic"].shape[-1]), (231 + 2 + 5, 234 + 2 + 5)
        )
        torch.testing.assert_close(first["policy"][0, -5:], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0]))
        for _ in range(100):
            observation, *_ = env.step(torch.zeros(1, 18))
        torch.testing.assert_close(
            observation["policy"][0, -5:],
            torch.tensor([0.4, 0.0, 0.0, 1.0, 0.4]),
            atol=1e-4,
            rtol=0,
        )
        torch.testing.assert_close(observation["critic"][0, -5:], observation["policy"][0, -5:])

    def test_a_command_switch_bootstraps_the_held_destination_and_restarts_the_inputs(self):
        env, task, native = self.wrapped(command_segments="bootstrap")
        env.get_observations()
        for _ in range(50):
            env.step(torch.zeros(1, 18))
        task.bank = LEFT.clone()  # the next draw is a left walk or a stop, either a new command
        task.remaining_controls[:] = 1
        observation, _, done, extras = env.step(torch.zeros(1, 18))
        critic = extras["command_bootstrap_observation"]["critic"]
        # 51 controls at 0.02 m/s short of the command: 0.0204 m behind the held destination.
        torch.testing.assert_close(
            critic[0, -5:], torch.tensor([0.204, 0.0, 0.0, 1.0, 51 / 250]), atol=1e-4, rtol=0
        )
        torch.testing.assert_close(
            observation["critic"][0, -5:], torch.tensor([0.0, 0.0, 0.0, 1.0, 0.0])
        )

    def test_evaluation_rebuilds_the_training_inputs(self):
        from locomotion.scripts.play import actor_input

        env, task, native = self.wrapped()
        args = SimpleNamespace(
            observation_scaling="fixed", gait_clock=60, experiment="destination_v1"
        )
        tracker, seen, trained = DestinationTracker(1, "cpu"), [], []
        for _ in range(30):
            raw = env.current["obs"].clone()
            raw[:, 210:213] = native.commands  # evaluation's observation carries the live command
            seen.append(actor_input(native, args, raw, tracker)[0, -5:].clone())
            observation, *_ = env.step(torch.zeros(1, 18))
            trained.append(observation["policy"][0, -5:].clone())
        for k in range(29):
            torch.testing.assert_close(seen[k + 1], trained[k], atol=1e-5, rtol=0)

    def test_the_learner_configuration_records_the_input(self):
        base = ppo_config(7, gait_clock=60)
        ghost = ppo_config(7, gait_clock=60, destination_input="ghost")
        self.assertEqual(
            ghost["environment_wrapper"],
            dict(base["environment_wrapper"], destination_input="ghost"),
        )
        self.assertEqual(ppo_config(7, gait_clock=60, destination_input="none"), base)


class SurrogateTests(unittest.TestCase):
    def test_surrogate_evaluation_rebuilds_a_native_experiment_checkpoint(self):
        from locomotion.surrogate.evaluate import DEFAULT_TASK, task_and_wrapper
        from locomotion.surrogate.train import resolve

        plain_args = SimpleNamespace(task=None, wrapper=None)
        task, wrapper = task_and_wrapper(plain_args, {"experiment": "destination_v9"})
        self.assertIs(resolve(wrapper), DestinationVecEnv)
        self.assertIsInstance(resolve(task)(GaitDouble(1)), DestinationTrainingTask)
        self.assertEqual(
            resolve(task)(GaitDouble(1)).reward_config, EXPERIMENTS["destination_v9"]["reward"]
        )
        self.assertEqual(
            task_and_wrapper(plain_args, {}), (DEFAULT_TASK, "locomotion.rl.wrapper:OmniVecEnv")
        )
        chosen = SimpleNamespace(task="a:B", wrapper="c:D")
        self.assertEqual(task_and_wrapper(chosen, {"experiment": "destination_v9"}), ("a:B", "c:D"))


class StartTests(unittest.TestCase):
    def test_a_widened_start_computes_what_the_checkpoint_did_and_restarts_the_rate(self):
        torch.manual_seed(0)

        def learner(actor_in, critic_in):
            algorithm = SimpleNamespace(
                actor=torch.nn.Sequential(
                    torch.nn.Linear(actor_in, 3), torch.nn.Tanh(), torch.nn.Linear(3, 2)
                ),
                critic=torch.nn.Sequential(
                    torch.nn.Linear(critic_in, 3), torch.nn.Tanh(), torch.nn.Linear(3, 1)
                ),
            )
            algorithm.optimizer = torch.optim.Adam(
                [*algorithm.actor.parameters(), *algorithm.critic.parameters()], lr=1e-5
            )

            def load(loaded, cfg, strict):
                algorithm.actor.load_state_dict(loaded["actor_state_dict"], strict=strict)
                algorithm.critic.load_state_dict(loaded["critic_state_dict"], strict=strict)
                algorithm.optimizer.load_state_dict(loaded["optimizer_state_dict"])

            algorithm.load = load
            return algorithm

        narrow = learner(4, 5)
        loss = (
            narrow.actor(torch.randn(8, 4)).square().sum()
            + narrow.critic(torch.randn(8, 5)).square().sum()
        )
        loss.backward()
        narrow.optimizer.step()
        saved = {
            "actor_state_dict": narrow.actor.state_dict(),
            "critic_state_dict": narrow.critic.state_dict(),
            "optimizer_state_dict": narrow.optimizer.state_dict(),
        }
        wide = learner(6, 7)
        widened, rate = initialize_learner(wide, saved, 3e-4, "configured")
        self.assertEqual(
            (widened, rate), ([["actor.0.weight", 4, 6], ["critic.0.weight", 5, 7]], 3e-4)
        )
        x = torch.randn(3, 4)
        torch.testing.assert_close(
            wide.actor(torch.cat((x, torch.randn(3, 2)), -1)), narrow.actor(x)
        )
        moment = wide.optimizer.state_dict()["state"][0]["exp_avg"]
        self.assertEqual(tuple(moment.shape), (3, 6))
        torch.testing.assert_close(moment[:, 4:], torch.zeros(3, 2))
        self.assertEqual(wide.optimizer.param_groups[0]["lr"], 3e-4)
        bad = {
            **saved,
            "actor_state_dict": {**saved["actor_state_dict"], "2.weight": torch.zeros(5, 3)},
        }
        with self.assertRaises(ValueError):
            widen_inputs(bad, learner(6, 7))

    def test_a_widened_normalizer_passes_the_added_inputs_unscaled_and_a_missing_one_is_named(self):
        from rsl_rl.modules.normalization import EmpiricalNormalization

        class Actor(torch.nn.Module):
            def __init__(self, inputs, normalize=True):
                super().__init__()
                if normalize:
                    self.obs_normalizer = EmpiricalNormalization(inputs)
                self.mlp = torch.nn.Linear(inputs, 2)

        def learner(inputs, normalize=True):
            algorithm = SimpleNamespace(
                actor=Actor(inputs, normalize),
                critic=torch.nn.Sequential(torch.nn.Linear(inputs, 1)),
            )
            algorithm.optimizer = torch.optim.Adam(
                [*algorithm.actor.parameters(), *algorithm.critic.parameters()]
            )
            return algorithm

        narrow = learner(4)
        narrow.actor.obs_normalizer.update(torch.randn(64, 4) * 3 + 1)
        saved = {
            "actor_state_dict": narrow.actor.state_dict(),
            "critic_state_dict": narrow.critic.state_dict(),
            "optimizer_state_dict": narrow.optimizer.state_dict(),
        }
        widened = widen_inputs(saved, learner(6))
        self.assertEqual(widened, [["actor.mlp.weight", 4, 6], ["critic.0.weight", 4, 6]])
        state = saved["actor_state_dict"]
        for name, fill in (("_mean", 0.0), ("_var", 1.0), ("_std", 1.0)):
            column = state[f"obs_normalizer.{name}"]
            self.assertEqual(tuple(column.shape), (1, 6))
            torch.testing.assert_close(column[:, 4:], torch.full((1, 2), fill))
            torch.testing.assert_close(column[:, :4], getattr(narrow.actor.obs_normalizer, name))
        empirical = {
            "actor_state_dict": narrow.actor.state_dict(),
            "critic_state_dict": narrow.critic.state_dict(),
            "optimizer_state_dict": narrow.optimizer.state_dict(),
        }
        with self.assertRaisesRegex(ValueError, "obs_normalizer._mean"):
            widen_inputs(empirical, learner(6, normalize=False))


class NativeStartTests(unittest.TestCase):
    def test_native_training_starts_only_from_a_declared_checkpoint_of_the_same_robot(self):
        import json
        import tempfile
        from pathlib import Path
        from unittest.mock import patch

        import locomotion.rl.ppo as ppo
        from locomotion.evaluation.checkpoint import STARTING_KEYS
        from locomotion.scripts.train import sha, start_from_checkpoint

        robot = {key: key[0] * 64 for key in STARTING_KEYS}
        identity = {**robot, "physics_config": {"floor": "isaaclab_mesh"}}
        parent = {
            **robot,
            "physics_config": {"floor": "earlier"},
            "experiment": "destination_v8",
            "reward_version": "4",
            "reward_task": "v4_yaw_gate",
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "checkpoint.pt"

            def write(record_identity, infos_identity=None):
                torch.save(
                    {"infos": {"identity": infos_identity or record_identity, "updates": 3500}},
                    path,
                )
                record = {
                    "identity": record_identity,
                    "updates": 3500,
                    "checkpoint_sha256": sha(path),
                }
                path.with_suffix(".json").write_text(json.dumps(record))
                return SimpleNamespace(
                    initialize_from=path,
                    initialize_from_sha256=sha(path),
                    initialize_from_declaration_sha256=sha(path.with_suffix(".json")),
                    initialize_rate="configured",
                    device="cpu",
                )

            config = {"algorithm": {"learning_rate": 3e-4}}
            with patch.object(
                ppo, "initialize_learner", return_value=([["actor.0.weight", 4, 6]], 3e-4)
            ):
                args = write(parent)
                started = dict(identity)
                start_from_checkpoint(SimpleNamespace(alg=None), args, config, started)
                start = started["initialized_from"]
                self.assertEqual(
                    (start["declaration_sha256"], start["experiment"], start["updates"]),
                    (args.initialize_from_declaration_sha256, "destination_v8", 3500),
                )
                self.assertFalse(start["physics_config_equal"])
                cases = {
                    "declaration bytes": (parent, None, "initialize_from_declaration_sha256"),
                    "robot differs: usd_sha256": (dict(parent, usd_sha256="0" * 64), None, None),
                    "Embedded checkpoint declaration": (parent, dict(parent, seed=1), None),
                }
                for message, (record, infos, broken) in cases.items():
                    with self.subTest(message=message):
                        args = write(record, infos)
                        if broken:
                            setattr(args, broken, "0" * 64)
                        with self.assertRaisesRegex(ValueError, message):
                            start_from_checkpoint(
                                SimpleNamespace(alg=None), args, config, dict(identity)
                            )


class PackTests(unittest.TestCase):
    def test_a_training_pack_names_the_experiment_and_mounts_its_starting_checkpoint(self):
        import json
        import tempfile
        from pathlib import Path

        from locomotion.spark.prepare import prepare

        remote = "/srv/cupi/hexapod/runs/julian/destination_fixture"
        start = "/srv/cupi/hexapod/runs/julian/c48/run/standing/checkpoint_update002500.pt"
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = prepare(
                root / "train",
                remote,
                mode="train",
                updates=3500,
                extended_updates=True,
                experiment="destination_v1",
                initialize_from=start,
                initialize_from_sha="c" * 64,
                initialize_from_declaration_sha="d" * 64,
                steps_per_env=36,
                allocation_profile="flat_pilot_v1",
                max_wall_seconds=21600,
            )
            args = binding["command_args"]
            self.assertEqual(args[args.index("--experiment") + 1], "destination_v1")
            self.assertEqual(
                args[args.index("--initialize-from") + 1], "/checkpoint/checkpoint_update002500.pt"
            )
            self.assertEqual(args[args.index("--initialize-from-sha256") + 1], "c" * 64)
            self.assertEqual(args[args.index("--initialize-from-declaration-sha256") + 1], "d" * 64)
            self.assertNotIn("--initialize-rate", args)  # the configured rate is the default
            self.assertIn([str(Path(start).parent), "/checkpoint"], binding["extra_mounts"])
            self.assertEqual(binding["input_files"][start], "c" * 64)
            self.assertEqual(binding["input_files"][start[: -len(".pt")] + ".json"], "d" * 64)
            pack = json.loads((root / "train/PACK.json").read_text())
            self.assertEqual(
                (pack["experiment"], pack["initialize_from_sha256"]), ("destination_v1", "c" * 64)
            )
            evaluation = prepare(
                root / "evaluate",
                remote,
                mode="evaluate",
                experiment="destination_v1",
                checkpoint=start,
                checkpoint_sha="a" * 64,
                checkpoint_declaration_sha="b" * 64,
                steps_per_env=36,
            )
            self.assertIn("--experiment", evaluation["command_args"])
            spec = "/srv/cupi/hexapod/runs/julian/terrain_inputs/terrain/mixed_v3.json"
            terrain = prepare(
                root / "terrain",
                remote,
                mode="train",
                experiment="destination_v1",
                terrain=spec,
                terrain_sha="e" * 64,
                terrain_mesh_resolution=0.08,
                terrain_spawn=True,
                steps_per_env=36,
            )
            args = terrain["command_args"]
            self.assertEqual(args[args.index("--terrain") + 1], "/terrain/mixed_v3.json")
            self.assertEqual(args[args.index("--terrain-mesh-resolution") + 1], "0.08")
            self.assertIn("--terrain-spawn", args)
            self.assertIn([str(Path(spec).parent), "/terrain"], terrain["extra_mounts"])
            self.assertEqual(terrain["input_files"][spec], "e" * 64)
            for index, bad in enumerate(
                (
                    dict(mode="train", experiment="unknown"),
                    dict(mode="diagnostic", experiment="destination_v1"),
                    dict(mode="evaluate", terrain=spec, terrain_sha="e" * 64),
                    dict(mode="train", terrain=spec),
                    dict(mode="train", terrain="/tmp/mixed.json", terrain_sha="e" * 64),
                    dict(mode="train", terrain=spec, terrain_sha="not a hash"),
                    dict(mode="train", terrain_spawn=True),
                    dict(mode="train", terrain_mesh_resolution=0.08),
                    dict(mode="train", initialize_from=start),
                    dict(mode="train", initialize_from=start, initialize_from_sha="c" * 64),
                    dict(
                        mode="train",
                        initialize_from="/tmp/x.pt",
                        initialize_from_sha="c" * 64,
                        initialize_from_declaration_sha="d" * 64,
                    ),
                    dict(
                        mode="train",
                        initialize_from=start,
                        initialize_from_sha="c" * 64,
                        initialize_from_declaration_sha="not a hash",
                    ),
                    dict(
                        mode="evaluate",
                        checkpoint="/tmp/x.pt",
                        checkpoint_sha="a" * 64,
                        checkpoint_declaration_sha="b" * 64,
                    ),
                )
            ):
                with self.subTest(bad=bad), self.assertRaises(ValueError):
                    prepare(root / f"bad{index}", remote, **bad)
                # A refusal writes nothing, so the same output can be prepared again.
                self.assertFalse((root / f"bad{index}").exists())


class LaunchMountTests(unittest.TestCase):
    def test_every_mount_a_training_pack_declares_is_one_the_launcher_accepts(self):
        import ast
        from pathlib import Path

        source = Path(__file__).resolve().parents[1] / "spark/launch.py"
        allowed = {
            node.value
            for node in ast.walk(ast.parse(source.read_text()))
            if isinstance(node, ast.Constant)
            and isinstance(node.value, str)
            and node.value.startswith("/")
        }
        self.assertIn("/checkpoint", allowed)


class TerrainCampaignTests(unittest.TestCase):
    def test_destination_v33_biases_walks_forward_and_v34_adds_the_slope_inputs(self):
        v29, v33, v34 = (
            EXPERIMENTS[n]["reward"]
            for n in ("destination_v29", "destination_v33", "destination_v34")
        )
        self.assertEqual(
            {k for k, v in asdict(v33).items() if asdict(v29)[k] != v}, {"walk_forward_bias"}
        )
        self.assertEqual(v33.walk_forward_bias, 0.5)
        self.assertEqual(
            {k for k, v in asdict(v34).items() if asdict(v33)[k] != v}, {"slope_inputs"}
        )
        self.assertEqual(v34.slope_inputs, "joints")
        self.assertEqual(EXPERIMENT_INPUTS["destination_v33"], "terrain_v1")
        self.assertEqual(EXPERIMENT_INPUTS["destination_v34"], "ghost_v11")
        self.assertEqual(EXPERIMENT_INPUTS["destination_v35"], "terrain_v2")
        self.assertEqual(EXPERIMENTS["destination_v35"]["reward"], v33)
        v36 = EXPERIMENTS["destination_v36"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v36).items() if asdict(v33)[k] != v},
            {"end_destination_distance_m"},
        )
        self.assertEqual(v36.end_destination_distance_m, 0.075)
        self.assertEqual(EXPERIMENT_INPUTS["destination_v36"], "terrain_v2")
        self.assertEqual(EXPERIMENTS["destination_v37"]["reward"], v33)
        self.assertEqual(EXPERIMENT_INPUTS["destination_v37"], "terrain_v2_critic")
        self.assertEqual(EXPERIMENTS["destination_v33"]["episode_seconds"], 32.0)
        with self.assertRaises(ValueError):
            replace(v33, walk_forward_only=True).validate()
        with self.assertRaises(ValueError):
            replace(v33, walk_forward_bias=1.0).validate()

    def test_half_the_biased_walks_face_forward(self):
        task, native = moving_task(forward, reward=EXPERIMENTS["destination_v33"]["reward"])
        hits = 0
        for _ in range(300):
            task._uniform_walks(torch.tensor([0]))
            command = native.commands[0]
            speed = float(torch.linalg.vector_norm(command[:2]))
            self.assertGreaterEqual(speed, 0.05 - 1e-6)
            self.assertLessEqual(speed, 0.15 + 1e-6)
            hits += abs(float(command[1])) < 1e-9 and float(command[0]) > 0
        self.assertGreater(hits / 300, 0.38)
        self.assertLess(hits / 300, 0.62)

    def test_destination_v38_draws_both_turn_rates_and_rebalances_the_mix(self):
        v35 = EXPERIMENTS["destination_v35"]["reward"]
        v38 = EXPERIMENTS["destination_v38"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v38).items() if asdict(v35)[k] != v},
            {"walk_forward_bias", "run_mix", "walk_speed_max_mps", "turn_rate_choices_rad_s"},
        )
        self.assertEqual(v38.walk_forward_bias, 0.25)
        self.assertEqual(v38.run_mix, (0.15, 0.10, 0.45, 0.30))
        self.assertEqual(v38.walk_speed_max_mps, 0.17)
        self.assertEqual(v38.turn_rate_choices_rad_s, (0.2, 0.4))
        self.assertEqual(EXPERIMENT_INPUTS["destination_v38"], "terrain_v2")
        self.assertEqual(EXPERIMENTS["destination_v38"]["episode_seconds"], 32.0)
        v39 = EXPERIMENTS["destination_v39"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v39).items() if asdict(v38)[k] != v}, {"over_rating_weight"}
        )
        self.assertEqual(v39.over_rating_weight, 30.0)
        v40 = EXPERIMENTS["destination_v40"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v40).items() if asdict(v38)[k] != v},
            {"run_mix", "turn_rate_choices_rad_s"},
        )
        self.assertEqual(v40.run_mix, (0.35, 0.10, 0.35, 0.20))
        self.assertEqual(v40.turn_rate_choices_rad_s, (0.3, 0.4))
        v42 = EXPERIMENTS["destination_v42"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v42).items() if asdict(v38)[k] != v},
            {"run_mix", "turn_rate_choices_rad_s"},
        )
        self.assertEqual(v42.run_mix, (0.20, 0.10, 0.40, 0.30))
        self.assertEqual(v42.turn_rate_choices_rad_s, (0.3, 0.4))
        for name in ("destination_v39", "destination_v40", "destination_v42"):
            self.assertEqual(EXPERIMENT_INPUTS[name], "terrain_v2")
        for broken in (
            {"turn_rate_choices_rad_s": (0.2, -0.4)},
            {"turn_rate_choices_rad_s": ()},
            {"turn_rate_choices_rad_s": [0.2, 0.4]},
        ):
            with self.assertRaises(ValueError):
                replace(v38, **broken).validate()
        with self.assertRaises(ValueError):
            # The drawn turn rates need a run mix.
            replace(
                EXPERIMENTS["destination_v26"]["reward"], turn_rate_choices_rad_s=(0.2, 0.4)
            ).validate()
        native = GaitDouble(8192)
        native.cfg.episode_seconds, native.cfg.declaration = 32.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, v38)
        task.reset()
        command = native.commands
        planar = torch.linalg.vector_norm(command[:, :2], dim=-1)
        turns = (planar == 0) & (command[:, 2] != 0)
        walks = planar > 0
        rates = command[turns, 2].abs()

        def near(value):
            return (rates - value).abs() < 1e-6

        self.assertTrue(bool((near(0.2) | near(0.4)).all()))
        # Both rates and both signs are drawn in roughly even shares; the mix holds its shares.
        self.assertAlmostEqual(float(near(0.4).double().mean()), 0.5, delta=0.06)
        self.assertAlmostEqual(float((command[turns, 2] > 0).double().mean()), 0.5, delta=0.06)
        self.assertAlmostEqual(float(turns.double().mean()), 0.15, delta=0.015)
        self.assertAlmostEqual(float(walks.double().mean()), 0.75, delta=0.02)
        self.assertTrue(bool((planar[walks] >= 0.05 - 1e-6).all()))
        self.assertTrue(bool((planar[walks] <= 0.17 + 1e-6).all()))
        # About a quarter of the walks face exactly forward (the rebalanced bias).
        ahead = walks & (command[:, 1].abs() < 1e-9) & (command[:, 0] > 0)
        self.assertAlmostEqual(float(ahead.double().sum() / walks.double().sum()), 0.25, delta=0.03)

    def test_destination_v45_draws_arcing_walks(self):
        v44 = EXPERIMENTS["destination_v44"]["reward"]
        v45 = EXPERIMENTS["destination_v45"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v45).items() if asdict(v44)[k] != v},
            {"arc_fraction", "arc_yaw_choices_rad_s"},
        )
        self.assertEqual(v45.arc_fraction, 0.25)
        self.assertEqual(v45.arc_yaw_choices_rad_s, (0.05, 0.2))
        self.assertEqual(
            {k: v for k, v in EXPERIMENTS["destination_v45"].items() if k != "reward"},
            {k: v for k, v in EXPERIMENTS["destination_v44"].items() if k != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v45"], "terrain_v2")
        v45.validate()
        for broken in (
            {"arc_fraction": 0.0},
            {"arc_fraction": 1.5},
            {"arc_fraction": None},
            {"arc_yaw_choices_rad_s": ()},
            {"arc_yaw_choices_rad_s": (0.05, -0.2)},
            {"arc_yaw_choices_rad_s": [0.05, 0.2]},
        ):
            with self.assertRaises(ValueError):
                replace(v45, **broken).validate()
        with self.assertRaises(ValueError):
            # Arcing walks need a run mix to draw walks from.
            replace(
                EXPERIMENTS["destination_v26"]["reward"],
                arc_fraction=0.25,
                arc_yaw_choices_rad_s=(0.05, 0.2),
            ).validate()
        native = GaitDouble(8192)
        native.cfg.episode_seconds, native.cfg.declaration = 32.0, dict
        task = DestinationTrainingTask(native, TaskConfig(), None, v45)
        task.reset()
        command = native.commands
        planar = torch.linalg.vector_norm(command[:, :2], dim=-1)
        walks = planar > 0
        arcs = walks & (command[:, 2] != 0)
        rates = command[arcs, 2].abs()

        def near(value):
            return (rates - value).abs() < 1e-6

        # Every arcing walk draws one of the two scored tiers, both signs, about a quarter of
        # the walks; the pure turns keep the v44 turn rates and the walk shares hold.
        self.assertTrue(bool((near(0.05) | near(0.2)).all()))
        self.assertAlmostEqual(float(arcs.double().sum() / walks.double().sum()), 0.25, delta=0.03)
        self.assertAlmostEqual(float(near(0.2).double().mean()), 0.5, delta=0.06)
        self.assertAlmostEqual(float((command[arcs, 2] > 0).double().mean()), 0.5, delta=0.06)
        self.assertAlmostEqual(float(walks.double().mean()), 0.70, delta=0.02)
        turns = (planar == 0) & (command[:, 2] != 0)
        turn_rates = command[turns, 2].abs()
        self.assertTrue(
            bool(
                ((turn_rates - 0.3).abs() < 1e-6).logical_or((turn_rates - 0.4).abs() < 1e-6).all()
            )
        )
        self.assertTrue(bool((planar[arcs] >= 0.05 - 1e-6).all()))
        self.assertTrue(bool((planar[arcs] <= 0.17 + 1e-6).all()))

    def test_destination_v46_arcs_the_lift_champion_without_the_turn_mix(self):
        v43 = EXPERIMENTS["destination_v43"]["reward"]
        v46 = EXPERIMENTS["destination_v46"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v46).items() if asdict(v43)[k] != v},
            {"arc_fraction", "arc_yaw_choices_rad_s"},
        )
        self.assertEqual(v46.arc_fraction, 0.25)
        self.assertEqual(v46.arc_yaw_choices_rad_s, (0.05, 0.2))
        self.assertEqual(v46.run_mix, v43.run_mix)
        self.assertEqual(v46.turn_rate_choices_rad_s, v43.turn_rate_choices_rad_s)
        self.assertEqual(
            {k: v for k, v in EXPERIMENTS["destination_v46"].items() if k != "reward"},
            {k: v for k, v in EXPERIMENTS["destination_v43"].items() if k != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v46"], "terrain_v2")
        v46.validate()

    def test_destination_v48_halves_the_turn_share_on_the_arcing_recipe(self):
        v45 = EXPERIMENTS["destination_v45"]["reward"]
        v48 = EXPERIMENTS["destination_v48"]["reward"]
        self.assertEqual({k for k, v in asdict(v48).items() if asdict(v45)[k] != v}, {"run_mix"})
        self.assertEqual(v48.run_mix, (0.10, 0.10, 0.45, 0.35))
        self.assertEqual(
            {k: v for k, v in EXPERIMENTS["destination_v48"].items() if k != "reward"},
            {k: v for k, v in EXPERIMENTS["destination_v45"].items() if k != "reward"},
        )
        self.assertEqual(EXPERIMENT_INPUTS["destination_v48"], "terrain_v2")
        v48.validate()

    def test_destination_v47_mirrors_the_schedule_on_clockwise_turns(self):
        import torch as _torch

        from locomotion.omniwalk.mdp.footfall import touchdown_seconds
        from locomotion.omniwalk.mdp.gait import GaitPhase, phase_swing

        v45 = EXPERIMENTS["destination_v45"]["reward"]
        v47 = EXPERIMENTS["destination_v47"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v47).items() if asdict(v45)[k] != v},
            {"schedule_mirror_turns"},
        )
        self.assertIs(v47.schedule_mirror_turns, True)
        self.assertEqual(EXPERIMENT_INPUTS["destination_v47"], "terrain_v2")
        v47.validate()
        with self.assertRaises(ValueError):
            replace(v47, schedule_mirror_turns=False).validate()
        # The mirrored rows swap tripods exactly; unmirrored rows and the default path are
        # byte-identical to the historical schedule.
        phase = _torch.tensor([0.1, 0.1, 0.6, 0.6], dtype=_torch.float64)
        swing = _torch.full((4,), 0.4, dtype=_torch.float64)
        base = phase_swing(phase, swing)
        mirror = _torch.tensor([False, True, False, True])
        swapped = phase_swing(phase, swing, mirror)
        self.assertTrue(bool(_torch.equal(swapped[::2], base[::2])))
        self.assertTrue(bool(_torch.equal(swapped[1::2], ~base[1::2])))
        self.assertTrue(bool(_torch.equal(phase_swing(phase, swing, None), base)))
        command = _torch.tensor(
            [[0.1, 0.0, 0.3], [0.1, 0.0, -0.3], [0.0, 0.1, 0.3], [0.0, 0.1, -0.3]]
        )
        plain = touchdown_seconds(phase, command, "speed_v1")
        mirrored = touchdown_seconds(phase, command, "speed_v1", mirror=True)
        self.assertTrue(bool(_torch.equal(mirrored[::2], plain[::2])))
        self.assertFalse(bool(_torch.equal(mirrored[1::2], plain[1::2])))
        legs = _torch.tensor([True, False, True, False, True, False])
        self.assertTrue(
            bool(_torch.equal(mirrored[1][legs], plain[1][~legs]))
            and bool(_torch.equal(mirrored[1][~legs], plain[1][legs]))
        )
        gait = GaitPhase(4, "speed_v1", "cpu")
        gait.phase = phase.clone()
        on, _ = gait.swing(command, True)
        off, _ = gait.swing(command, False)
        self.assertTrue(bool(_torch.equal(on[::2], off[::2])))
        self.assertTrue(bool(_torch.equal(on[1::2], ~off[1::2])))

    def test_destination_v49_and_v50_are_the_flat_polish_arms(self):
        v45 = EXPERIMENTS["destination_v45"]["reward"]
        v49 = EXPERIMENTS["destination_v49"]["reward"]
        v50 = EXPERIMENTS["destination_v50"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v49).items() if asdict(v45)[k] != v}, {"roll_pitch_weight"}
        )
        self.assertEqual(v49.roll_pitch_weight, 0.3)
        self.assertEqual(
            {k for k, v in asdict(v50).items() if asdict(v45)[k] != v}, {"walk_forward_bias"}
        )
        self.assertIsNone(v50.walk_forward_bias)
        for name in ("destination_v49", "destination_v50"):
            self.assertEqual(EXPERIMENT_INPUTS[name], "terrain_v2")
            self.assertEqual(
                {k: v for k, v in EXPERIMENTS[name].items() if k != "reward"},
                {k: v for k, v in EXPERIMENTS["destination_v45"].items() if k != "reward"},
            )
            EXPERIMENTS[name]["reward"].validate()

    def test_destination_v51_doubles_the_arc_share_for_the_flat_polish(self):
        v45 = EXPERIMENTS["destination_v45"]["reward"]
        v51 = EXPERIMENTS["destination_v51"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v51).items() if asdict(v45)[k] != v}, {"arc_fraction"}
        )
        self.assertEqual(v51.arc_fraction, 0.5)
        self.assertEqual(v51.arc_yaw_choices_rad_s, v45.arc_yaw_choices_rad_s)
        self.assertEqual(EXPERIMENT_INPUTS["destination_v51"], "terrain_v2")
        self.assertEqual(
            {k: v for k, v in EXPERIMENTS["destination_v51"].items() if k != "reward"},
            {k: v for k, v in EXPERIMENTS["destination_v45"].items() if k != "reward"},
        )
        EXPERIMENTS["destination_v51"]["reward"].validate()

    def test_destination_v52_composes_the_arc_share_and_uniform_bearings(self):
        v45 = EXPERIMENTS["destination_v45"]["reward"]
        v51 = EXPERIMENTS["destination_v51"]["reward"]
        v52 = EXPERIMENTS["destination_v52"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v52).items() if asdict(v51)[k] != v}, {"walk_forward_bias"}
        )
        self.assertEqual(
            {k for k, v in asdict(v52).items() if asdict(v45)[k] != v},
            {"arc_fraction", "walk_forward_bias"},
        )
        self.assertIsNone(v52.walk_forward_bias)
        self.assertEqual(v52.arc_fraction, 0.5)
        self.assertEqual(EXPERIMENT_INPUTS["destination_v52"], "terrain_v2")
        self.assertEqual(
            {k: v for k, v in EXPERIMENTS["destination_v52"].items() if k != "reward"},
            {k: v for k, v in EXPERIMENTS["destination_v45"].items() if k != "reward"},
        )
        EXPERIMENTS["destination_v52"]["reward"].validate()

    def test_destination_v53_puts_weight_on_stand_tilt(self):
        v45 = EXPERIMENTS["destination_v45"]["reward"]
        v53 = EXPERIMENTS["destination_v53"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v53).items() if asdict(v45)[k] != v}, {"tilt_weight"}
        )
        self.assertEqual(v53.tilt_weight, 15.0)
        self.assertEqual(v53.tilt_scale_rad, v45.tilt_scale_rad)
        self.assertEqual(EXPERIMENT_INPUTS["destination_v53"], "terrain_v2")
        self.assertEqual(
            {k: v for k, v in EXPERIMENTS["destination_v53"].items() if k != "reward"},
            {k: v for k, v in EXPERIMENTS["destination_v45"].items() if k != "reward"},
        )
        EXPERIMENTS["destination_v53"]["reward"].validate()

    def test_destination_v41_prices_swing_clearance_over_rough_ground_only(self):
        v38 = EXPERIMENTS["destination_v38"]["reward"]
        v41 = EXPERIMENTS["destination_v41"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v41).items() if asdict(v38)[k] != v}, {"swing_clearance_weight"}
        )
        self.assertEqual(v41.swing_clearance_weight, 0.5)
        self.assertEqual(EXPERIMENT_INPUTS["destination_v41"], "terrain_v2")
        with self.assertRaises(ValueError):
            replace(v41, swing_clearance_target_m=0.0).validate()
        commands = torch.tensor(
            [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]
        )
        values = telemetry(commands)
        values.update(
            gate_yaw_rate_rad_s=torch.zeros(4),
            stride_velocity_nav=torch.zeros(4, 3),
            stride_touchdown=torch.zeros(4, 6, dtype=torch.bool),
            stride_displacement_nav=torch.zeros(4, 6, 2),
        )
        values["tibia_floor_force_world_n"] = torch.zeros(4, 6, 3)
        values["scheduled_swing"] = torch.tensor([[True] * 3 + [False] * 3] * 4)
        terminated = torch.zeros(4, dtype=torch.bool)
        base, parts = measured_reward(values, commands, terminated, v38, 0.1)
        self.assertNotIn("swing_clearance", parts)
        # Swinging airborne toes earn height up to the target on walking rows over rough ground;
        # a smooth-ground row, a turning row and a standing row earn nothing.
        values["toe_clearance_m"] = torch.full((4, 6), 0.035)
        values["toe_clearance_m"][1] = 0.0175
        values["toe_rough_m"] = torch.full((4, 6), 0.02)
        values["toe_rough_m"][1] = 0.004
        paid, parts = measured_reward(values, commands, terminated, v41, 0.1)
        torch.testing.assert_close(parts["swing_clearance"], torch.tensor([0.5, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(paid, base + parts["swing_clearance"])
        missing = {k: v for k, v in values.items() if k not in ("toe_clearance_m", "toe_rough_m")}
        with self.assertRaises(ValueError):
            measured_reward(missing, commands, terminated, v41, 0.1)

    def test_destination_v43_pays_lift_toward_a_higher_landing(self):
        v38 = EXPERIMENTS["destination_v38"]["reward"]
        v43 = EXPERIMENTS["destination_v43"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v43).items() if asdict(v38)[k] != v}, {"landing_lift_weight"}
        )
        self.assertEqual(v43.landing_lift_weight, 0.5)
        self.assertEqual(EXPERIMENT_INPUTS["destination_v43"], "terrain_v2")
        with self.assertRaises(ValueError):
            replace(v43, landing_lift_target_m=0.0).validate()
        commands = torch.tensor(
            [[0.1, 0.0, 0.0], [0.1, 0.0, 0.0], [0.0, 0.0, 0.2], [0.0, 0.0, 0.0]]
        )
        values = telemetry(commands)
        values.update(
            gate_yaw_rate_rad_s=torch.zeros(4),
            stride_velocity_nav=torch.zeros(4, 3),
            stride_touchdown=torch.zeros(4, 6, dtype=torch.bool),
            stride_displacement_nav=torch.zeros(4, 6, 2),
        )
        values["tibia_floor_force_world_n"] = torch.zeros(4, 6, 3)
        values["scheduled_swing"] = torch.tensor([[True] * 3 + [False] * 3] * 4)
        terminated = torch.zeros(4, dtype=torch.bool)
        base, parts = measured_reward(values, commands, terminated, v38, 0.1)
        self.assertNotIn("landing_lift", parts)
        # A swinging airborne toe walking toward a 2.5 cm higher landing earns clearance up to
        # 2 cm over that landing (0.045 saturates); a landing under the 1 cm rise gate, a
        # turning row and a standing row earn nothing.
        values["toe_landing_rise_m"] = torch.full((4, 6), 0.025)
        values["toe_landing_rise_m"][1] = 0.005
        values["toe_clearance_m"] = torch.full((4, 6), 0.045)
        paid, parts = measured_reward(values, commands, terminated, v43, 0.1)
        torch.testing.assert_close(parts["landing_lift"], torch.tensor([0.5, 0.0, 0.0, 0.0]))
        torch.testing.assert_close(paid, base + parts["landing_lift"])
        # The saturation scales with the obstacle: half the clearance pays half.
        values["toe_clearance_m"] = torch.full((4, 6), 0.0225)
        _, parts = measured_reward(values, commands, terminated, v43, 0.1)
        torch.testing.assert_close(parts["landing_lift"], torch.tensor([0.25, 0.0, 0.0, 0.0]))
        # The native consolidation recipe carries the lift on top of v42 unchanged.
        v42 = EXPERIMENTS["destination_v42"]["reward"]
        v44 = EXPERIMENTS["destination_v44"]["reward"]
        self.assertEqual(
            {k for k, v in asdict(v44).items() if asdict(v42)[k] != v}, {"landing_lift_weight"}
        )
        self.assertEqual(v44.landing_lift_weight, 0.5)
        self.assertEqual(EXPERIMENT_INPUTS["destination_v44"], "terrain_v2")
        missing = {k: v for k, v in values.items() if k != "toe_landing_rise_m"}
        with self.assertRaises(ValueError):
            measured_reward(missing, commands, terminated, v43, 0.1)

    def test_the_swing_clearance_telemetry_reads_the_terrain_grid(self):
        task, native = moving_task(forward, reward=EXPERIMENTS["destination_v41"]["reward"])
        # One full control on the stock flat floor: the reward runs end to end with the term
        # priced at zero (zero roughness closes the gate).
        task.step(torch.zeros(1, 18))
        torch.testing.assert_close(task.last_components["swing_clearance"], torch.zeros(1))
        # The stock flat floor: clearance is the toe's world height over zero roughness.
        flat = task.reward_telemetry(native.commands)
        torch.testing.assert_close(flat["toe_rough_m"], torch.zeros(1, 6))
        torch.testing.assert_close(
            flat["toe_clearance_m"], flat["toe_xyz_world"].detach().cpu()[..., 2].float()
        )
        # A 5 cm riser between x = 0.1 and 0.2: a toe before it sees smooth ground, a toe on the
        # riser face sees the 5-point spread, a toe past it stands 5 cm higher on smooth ground.
        task._clearance_grid = TerrainGrid(
            SimpleNamespace(
                heights=[[0.0, 0.0, 0.05, 0.05]] * 4,
                xs=(0.0, 0.3),
                ys=(0.0, 0.3),
                resolution=0.1,
                nx=4,
                ny=4,
            )
        )
        world = torch.zeros(1, 6, 3)
        world[0, :, 0] = torch.tensor([0.05, 0.15, 0.25, 0.05, 0.15, 0.25])
        world[0, :, 1] = 0.15
        world[0, :, 2] = torch.tensor([0.02, 0.055, 0.06, 0.0, 0.025, 0.05])
        out = task._swing_clearance_telemetry({"toe_xyz_world": world})
        torch.testing.assert_close(
            out["toe_clearance_m"], torch.tensor([[0.02, 0.03, 0.01, 0.0, 0.0, 0.0]])
        )
        rough = out["toe_rough_m"]
        self.assertTrue(bool((rough[0, torch.tensor([0, 2, 3, 5])] < 1e-6).all()))
        torch.testing.assert_close(rough[0, 1], torch.tensor(0.03))
        torch.testing.assert_close(rough[0, 4], torch.tensor(0.03))

    def test_the_landing_lift_telemetry_predicts_the_touchdown_ground(self):
        task, native = moving_task(forward, reward=EXPERIMENTS["destination_v43"]["reward"])
        # One full control on the stock flat floor: the reward runs end to end, zero rise keeps
        # the gate closed, and the clearance telemetry rides along for the shared pay reference.
        task.step(torch.zeros(1, 18))
        torch.testing.assert_close(task.last_components["landing_lift"], torch.zeros(1))
        flat = task.reward_telemetry(native.commands)
        torch.testing.assert_close(flat["toe_landing_rise_m"], torch.zeros(1, 6))
        self.assertIn("toe_clearance_m", flat)
        # The 5 cm riser between x = 0.1 and 0.2, faced from x = 0.05 at heading pi/2 (a forward
        # command walks along world +x there); every toe hangs at the body's planar position.
        task._clearance_grid = TerrainGrid(
            SimpleNamespace(
                heights=[[0.0, 0.0, 0.05, 0.05]] * 4,
                xs=(0.0, 0.3),
                ys=(0.0, 0.3),
                resolution=0.1,
                nx=4,
                ny=4,
            )
        )
        native.current["root"] = pose_at(0.05, 0.15, math.pi / 2)
        world = torch.zeros(1, 6, 3)
        world[0, :, 0], world[0, :, 1], world[0, :, 2] = 0.05, 0.15, 0.02
        # Standing still, the predicted touchdown is the ground already underfoot: no rise.
        still = task._landing_lift_telemetry({"toe_xyz_world": world}, torch.zeros(1, 3))
        torch.testing.assert_close(still["toe_landing_rise_m"], torch.zeros(1, 6))
        # Walking at the riser, the prediction advances into it: some toe's landing reads ground
        # past the 1 cm gate, none reads more than the riser's 5 cm, and the values are exactly
        # the footfall touchdown prediction sampled at the ring maximum.
        command = torch.tensor([[0.1, 0.0, 0.0]])
        rise = task._landing_lift_telemetry({"toe_xyz_world": world}, command)["toe_landing_rise_m"]
        self.assertGreater(float(rise.max()), 0.01)
        self.assertLessEqual(float(rise.max()), 0.05 + 1e-6)
        self.assertTrue(bool((rise > -1e-6).all()))
        seconds = touchdown_seconds(task.gait.phase, command, "speed_v1")
        points = touchdown_points(
            native.current["root"], command, native.current["toe_body"], seconds
        )
        ring = torch.tensor(RING_XY, dtype=torch.float64)
        landing = task._clearance_grid.sample(
            points[:, :, None, 0] + ring[None, None, :, 0],
            points[:, :, None, 1] + ring[None, None, :, 1],
        ).amax(-1)
        expected = landing - task._clearance_grid.sample(world[..., 0], world[..., 1])
        torch.testing.assert_close(rise, expected)

    def test_terrain_v1_appends_zeros_without_a_terrain_and_needs_the_speed_scaled_clock(self):
        task, native = moving_task(forward, reward=EXPERIMENTS["destination_v33"]["reward"])
        env = DestinationVecEnv(
            task,
            plain,
            observation_scaling="fixed",
            gait_clock=60,
            gait_clock_scaling="speed_v1",
            destination_input="terrain_v1",
        )
        observation = env.get_observations()
        self.assertEqual(observation["policy"].shape[-1], 272)
        self.assertEqual(observation["critic"].shape[-1], 275)
        torch.testing.assert_close(observation["policy"][:, 242:], torch.zeros(1, 30))
        observation, *_ = env.step(torch.zeros(1, 18))
        torch.testing.assert_close(observation["policy"][:, 242:], torch.zeros(1, 30))
        two, _ = moving_task(forward, reward=EXPERIMENTS["destination_v35"]["reward"])
        wide = DestinationVecEnv(
            two,
            plain,
            observation_scaling="fixed",
            gait_clock=60,
            gait_clock_scaling="speed_v1",
            destination_input="terrain_v2",
        )
        observation = wide.get_observations()
        self.assertEqual(observation["policy"].shape[-1], 278)
        self.assertEqual(observation["critic"].shape[-1], 281)
        torch.testing.assert_close(observation["policy"][:, 242:], torch.zeros(1, 36))
        hidden, _ = moving_task(forward, reward=EXPERIMENTS["destination_v37"]["reward"])
        critic_only = DestinationVecEnv(
            hidden,
            plain,
            observation_scaling="fixed",
            gait_clock=60,
            gait_clock_scaling="speed_v1",
            destination_input="terrain_v2_critic",
        )
        observation = critic_only.get_observations()
        self.assertEqual(observation["policy"].shape[-1], 242)
        self.assertEqual(observation["critic"].shape[-1], 281)
        nine, _ = moving_task(forward, reward=EXPERIMENTS["destination_v9"]["reward"])
        with self.assertRaises(ValueError):
            DestinationVecEnv(
                nine,
                plain,
                observation_scaling="fixed",
                gait_clock=60,
                destination_input="terrain_v1",
            )


if __name__ == "__main__":
    unittest.main()
