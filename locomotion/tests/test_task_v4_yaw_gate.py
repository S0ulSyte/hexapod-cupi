"""The yaw-gated contact schedule: reward v4 plus one term that withholds schedule pay on a yaw error."""
import json
import math
from pathlib import Path
import tempfile
import unittest

import torch

from locomotion import task_v4, task_v4_yaw_gate as gated
from dataclasses import replace

from locomotion.task_v4_yaw_gate import REWARD_V4_YAW_GATE_CONFIG as CONFIG, measured_reward_v4_yaw_gate, yaw_gate

ALL = replace(CONFIG, yaw_gate_on="all")
from locomotion.tests.test_task_v4 import HEIGHT, LOADED, GaitDouble, telemetry

SWING_A = torch.tensor([True, False, True, False, True, False])


def scores(commands, stride_yaw, force=None, config=CONFIG):
    """Both rewards for rows whose scheduled tripod lf, lr, rm is lifted unless ``force`` says otherwise."""
    n = len(commands)
    if force is None:
        force = LOADED.expand(n, -1, -1).clone()
        force[:, [0, 2, 4], 2] = 0.
    values = telemetry(commands, tibia_floor_force_world_n=force, scheduled_swing=SWING_A.expand(n, -1),
                       gate_yaw_rate_rad_s=torch.as_tensor(stride_yaw, dtype=torch.float32))
    alive = torch.zeros(n, dtype=torch.bool)
    return (task_v4.measured_reward_v4(values, commands, alive, config, HEIGHT),
            measured_reward_v4_yaw_gate(values, commands, alive, config, HEIGHT))


class YawGateTests(unittest.TestCase):
    def test_gate_follows_the_yaw_error_relative_to_the_command_or_its_floor(self):
        commands = torch.tensor([[0., 0., .2], [0., 0., .2], [0., 0., .2], [.05, 0., 0.], [.05, 0., 0.]])
        gate = yaw_gate(torch.tensor([.2, .1, 0., 0., .065]), commands)
        torch.testing.assert_close(gate, torch.tensor([1., .75, 0., 1., 1 - .65**2]))

    def test_stepping_without_turning_earns_no_schedule_pay_on_a_turn_command(self):
        commands = torch.tensor([[0., 0., .2]]).expand(3, -1).clone()
        (v4_reward, v4_parts), (reward, parts) = scores(commands, torch.tensor([.2, .1, 0.]))
        pay = CONFIG.schedule_weight  # the scheduled tripod fully unloaded
        torch.testing.assert_close(v4_parts["gait_schedule"], torch.full((3,), pay))
        torch.testing.assert_close(parts["schedule_yaw_gate"], torch.tensor([0., -.25, -1.]) * pay)
        torch.testing.assert_close(reward - v4_reward, parts["schedule_yaw_gate"])

    def test_default_gate_leaves_translations_and_covers_turns_and_arcs(self):
        self.assertEqual(CONFIG.yaw_gate_on, "turn")
        commands = torch.tensor([[.05, 0., 0.], [-.035, .035, 0.], [0., 0., .2], [.04, 0., .15]])
        _, (_, parts) = scores(commands, torch.zeros(4))
        torch.testing.assert_close(parts["schedule_yaw_gate"], torch.tensor([0., 0., -1., -1.]) * CONFIG.schedule_weight)

    def test_heading_drift_on_a_translation_command_costs_part_of_the_pay_when_gating_all(self):
        commands = torch.tensor([[.05, 0., 0.]]).expand(2, -1).clone()
        _, (_, parts) = scores(commands, torch.tensor([0., .065]), config=ALL)
        torch.testing.assert_close(parts["schedule_yaw_gate"], torch.tensor([0., -(.65**2)]) * CONFIG.schedule_weight)
        _, (_, default) = scores(commands, torch.tensor([0., .065]))
        self.assertEqual(default["schedule_yaw_gate"].tolist(), [0., 0.])

    def test_the_gate_never_lowers_the_charge_for_lifting_stance_feet(self):
        commands = torch.tensor([[0., 0., .2]]).expand(2, -1).clone()
        force = LOADED.expand(2, -1, -1).clone()
        force[:, [1, 3, 5], 2] = 0.  # the wrong tripod lifted
        (v4_reward, v4_parts), (reward, parts) = scores(commands, torch.tensor([.2, 0.]), force)
        torch.testing.assert_close(v4_parts["gait_schedule"], torch.full((2,), -CONFIG.schedule_weight))
        self.assertEqual(parts["schedule_yaw_gate"].tolist(), [0., 0.])
        torch.testing.assert_close(reward, v4_reward)

    def test_zero_command_and_every_v4_component_are_unchanged(self):
        commands = torch.tensor([[0., 0., 0.], [.05, 0., 0.]])
        (v4_reward, v4_parts), (reward, parts) = scores(commands, torch.tensor([.3, .3]))
        self.assertEqual(float(parts["schedule_yaw_gate"][0]), 0.)
        for name, value in v4_parts.items():
            torch.testing.assert_close(parts[name], value, msg=name)
        self.assertEqual(set(parts) - set(v4_parts), {"schedule_yaw_gate"})

    def test_missing_stride_yaw_is_rejected(self):
        commands = torch.tensor([[0., 0., .2]])
        with self.assertRaisesRegex(ValueError, "gate_yaw_rate_rad_s"):
            measured_reward_v4_yaw_gate(telemetry(commands), commands, torch.zeros(1, dtype=torch.bool), CONFIG, HEIGHT)

    def test_task_averages_yaw_wobble_over_a_stride_and_keeps_drift(self):
        native = GaitDouble(2)
        task = gated.variant("forward_draw_fraction=0")(native)
        task.reset()
        native.commands[:] = torch.tensor([[.05, 0., 0.], [.05, 0., 0.]])
        task.window.restart(torch.arange(2), native.commands)
        task.gate_window.restart(torch.arange(2), native.commands)
        task.remaining_controls[:] = 1000
        heading = torch.zeros(2)
        original = native.step
        def turning(action):
            output = original(action)
            k = native.steps
            # Row 0 wobbles +-0.1 rad/s within each 60-control stride; row 1 also drifts at 0.065 rad/s.
            rate = torch.tensor([.1, .1]) * math.sin(2 * math.pi * k / 60) + torch.tensor([0., .065])
            heading.add_(rate * .02)
            pose = native.current["root"].clone()
            pose[:, 3:] = torch.stack((torch.zeros(2), torch.zeros(2), torch.sin(heading / 2), torch.cos(heading / 2)), -1)
            native.current["root"] = pose
            native.telemetry["root_pose_xyzw"] = pose.clone()
            return output
        native.step = turning
        for _ in range(120):
            task.step(torch.zeros(2, 18))
        stride = task.gate_window.update(torch.zeros(2, 1), native.commands)  # one more zero sample
        self.assertLess(abs(float(stride[0])), .01)
        self.assertAlmostEqual(float(stride[1]), .065, delta=.01)

    def test_task_applies_the_gate_and_declares_its_experimental_version(self):
        native = GaitDouble(2)
        task = gated.variant("forward_draw_fraction=0")(native)
        task.reset()
        native.commands[:] = torch.tensor([[0., 0., .2], [.05, 0., 0.]])
        task.window.restart(torch.arange(2), native.commands)
        task.remaining_controls[:] = 1000
        native.forces[:, [0, 2, 4]] = 0.  # follow the schedule without turning
        task.step(torch.zeros(2, 18))
        gate_term = task.last_components["schedule_yaw_gate"]
        self.assertLess(float(gate_term[0]), 0.)
        self.assertEqual(float(gate_term[1]), 0.)
        with tempfile.TemporaryDirectory() as directory:
            gated.TrainingTaskV4YawGate(GaitDouble(1), output_dir=directory)
            declared = json.loads((Path(directory)/"task_definition.json").read_text())
        self.assertEqual(declared["reward_version"], gated.REWARD_VERSION)
        self.assertIn("schedule_yaw_gate", declared["reward"]["formulas"])
        self.assertEqual(declared["reward"]["config"]["yaw_gate_scale_rad_s"], CONFIG.yaw_gate_scale_rad_s)
        self.assertEqual(task.status()["reward_version"], gated.REWARD_VERSION)

    def test_command_threshold_leaves_arcs_and_keeps_turns_in_place(self):
        commands = torch.tensor([[.04, 0., .15], [0., 0., .2], [0., 0., -.2]])
        _, (_, parts) = scores(commands, torch.zeros(3), config=replace(CONFIG, yaw_gate_min_command_rad_s=.18))
        torch.testing.assert_close(parts["schedule_yaw_gate"], torch.tensor([0., -1., -1.]) * CONFIG.schedule_weight)

    def test_bonus_pays_a_matched_turn_above_v4_and_nothing_without_a_turn(self):
        bonus = replace(CONFIG, yaw_gate_bonus=.5)
        commands = torch.tensor([[0., 0., .2]]).expand(4, -1).clone()
        (v4_reward, v4_parts), (reward, parts) = scores(commands, torch.tensor([.2, .2*(1-(1/3)**.5), .1, 0.]), config=bonus)
        pay = CONFIG.schedule_weight
        # gate 1 -> +50 %; gate 2/3 (the break-even 1/(1+b)) -> 0; gate 0.75 -> +12.5 %; no turn -> -100 %.
        torch.testing.assert_close(parts["schedule_yaw_gate"], torch.tensor([.5, 0., .125, -1.]) * pay, atol=1e-5, rtol=0)
        torch.testing.assert_close(reward - v4_reward, parts["schedule_yaw_gate"])
        torch.testing.assert_close(v4_parts["gait_schedule"] + parts["schedule_yaw_gate"], torch.tensor([1.5, 1., 1.125, 0.]) * pay, atol=1e-5, rtol=0)

    def test_floor_keeps_part_of_the_pay_and_still_rewards_turning_more(self):
        floor = replace(CONFIG, yaw_gate_bonus=.5, yaw_gate_floor=.5)
        commands = torch.tensor([[0., 0., .2]]).expand(3, -1).clone()
        (v4_reward, v4_parts), (_, parts) = scores(commands, torch.tensor([0., .1, .2]), config=floor)
        pay = CONFIG.schedule_weight
        # gate 0 -> half the v4 pay; gate 0.75 -> 0.5 + 1.0 * 0.75 = 1.25; gate 1 -> 1.5.
        torch.testing.assert_close(v4_parts["gait_schedule"] + parts["schedule_yaw_gate"], torch.tensor([.5, 1.25, 1.5]) * pay, atol=1e-5, rtol=0)
        self.assertEqual(CONFIG.yaw_gate_floor, 0.)
        for bad in ("yaw_gate_floor=-0.1", "yaw_gate_floor=1.5"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gated.reward_config(bad)

    def test_new_options_default_to_the_withholding_gate(self):
        self.assertEqual((CONFIG.yaw_gate_min_command_rad_s, CONFIG.yaw_gate_bonus, CONFIG.gate_window_controls), (0., 0., 60))
        for bad in ("yaw_gate_bonus=-1", "yaw_gate_min_command_rad_s=-0.1", "gate_window_controls=300"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gated.reward_config(bad)
        self.assertEqual(gated.reward_config("gate_window_controls=120").gate_window_controls, 120)

    def test_gate_off_adds_nothing(self):
        commands = torch.tensor([[0., 0., .2], [.05, 0., 0.]])
        (v4_reward, _), (reward, parts) = scores(commands, torch.zeros(2), config=replace(CONFIG, yaw_gate_on="none"))
        self.assertEqual(parts["schedule_yaw_gate"].tolist(), [0., 0.])
        torch.testing.assert_close(reward, v4_reward)

    def test_turn_draws_replace_a_share_of_moving_draws_and_keep_the_zero_quota(self):
        def draws(share, n=400):
            native = GaitDouble(n)
            task = gated.variant(f"forward_draw_fraction=0,turn_draw_fraction={share}")(native)
            task.reset()
            return native.commands.clone(), task
        base, _ = draws(0.)
        turned, task = draws(.5)
        zero = (base == 0).all(-1)
        self.assertTrue(torch.equal((turned == 0).all(-1), zero))
        pure_turn = (turned[:, :2] == 0).all(-1) & (turned[:, 2] != 0)
        share = float(pure_turn[~zero].float().mean())
        self.assertGreater(share, .45)
        self.assertEqual({round(x, 6) for x in turned[pure_turn, 2].tolist()}, {-.2, .2})
        default, _ = draws(0.)
        self.assertTrue(torch.equal(default, base))  # no extra generator draws at the default
        with self.assertRaises(ValueError):
            gated.reward_config("turn_draw_fraction=1.5")

    def test_variants_override_v4_and_gate_options(self):
        variant = gated.variant("yaw_gate_scale_rad_s=0.05,forward_draw_fraction=0")
        self.assertEqual((variant.reward_config.yaw_gate_scale_rad_s, variant.reward_config.forward_draw_fraction), (.05, 0.))
        self.assertTrue(issubclass(variant, task_v4.TrainingTaskV4))
        self.assertEqual(gated.variant("yaw_gate_on=all").reward_config.yaw_gate_on, "all")
        for bad in ("yaw_gate_scale_rad_s=0", "unknown=1", "yaw_gate_on=sometimes"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gated.reward_config(bad)


class StrideTrackingTests(unittest.TestCase):
    def test_stride_term_pays_the_kernel_of_the_stride_mean_on_moving_commands(self):
        config = replace(CONFIG, stride_tracking_weight=.3)
        commands = torch.tensor([[.05, 0., 0.], [.05, 0., 0.], [0., 0., .2], [0., 0., 0.]])
        stride = torch.tensor([[.05, 0., 0.], [.025, 0., 0.], [0., 0., .1], [.01, 0., .05]])
        values = telemetry(commands, gate_yaw_rate_rad_s=commands[:, 2].clone(), stride_velocity_nav=stride)
        alive = torch.zeros(4, dtype=torch.bool)
        v4_reward, v4_parts = task_v4.measured_reward_v4(values, commands, alive, config, HEIGHT)
        reward, parts = measured_reward_v4_yaw_gate(values, commands, alive, config, HEIGHT)
        # Full speed 1.5; half speed: linear 0.75, yaw 1; half turn: linear 1, yaw 0.75; zero command unpaid.
        torch.testing.assert_close(parts["stride_tracking"], .3 * torch.tensor([1.5, 1.25, 1.375, 0.]))
        torch.testing.assert_close(reward - v4_reward - parts["schedule_yaw_gate"], parts["stride_tracking"])
        for name, value in v4_parts.items():
            torch.testing.assert_close(parts[name], value, msg=name)

    def test_stride_term_peaks_at_the_command_and_not_above_it(self):
        config = replace(CONFIG, stride_tracking_weight=.3)
        commands = torch.tensor([[.05, 0., 0.]]).expand(3, -1).clone()
        stride = torch.tensor([[.04, 0., 0.], [.05, 0., 0.], [.06, 0., 0.]])
        values = telemetry(commands, gate_yaw_rate_rad_s=torch.zeros(3), stride_velocity_nav=stride)
        _, parts = measured_reward_v4_yaw_gate(values, commands, torch.zeros(3, dtype=torch.bool), config, HEIGHT)
        paid = parts["stride_tracking"].tolist()
        self.assertGreater(paid[1], paid[0])
        self.assertAlmostEqual(paid[0], paid[2], places=5)

    def test_default_weight_adds_no_term_and_reads_no_stride_telemetry(self):
        self.assertEqual((CONFIG.stride_tracking_weight, CONFIG.stride_window_controls), (0., 60))
        commands = torch.tensor([[.05, 0., 0.]])
        _, parts = measured_reward_v4_yaw_gate(telemetry(commands, gate_yaw_rate_rad_s=torch.zeros(1)), commands,
                                               torch.zeros(1, dtype=torch.bool), CONFIG, HEIGHT)
        self.assertNotIn("stride_tracking", parts)
        with self.assertRaisesRegex(ValueError, "stride_velocity_nav"):
            measured_reward_v4_yaw_gate(telemetry(commands, gate_yaw_rate_rad_s=torch.zeros(1)), commands,
                                        torch.zeros(1, dtype=torch.bool), replace(CONFIG, stride_tracking_weight=.3), HEIGHT)

    def test_fall_cost_covers_the_stride_term(self):
        self.assertEqual(gated.reward_config("stride_tracking_weight=0.3").stride_tracking_weight, .3)
        for bad in ("stride_tracking_weight=1", "stride_tracking_weight=-0.1", "stride_window_controls=0",
                    "stride_window_controls=300"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gated.reward_config(bad)
        config = gated.reward_config("stride_tracking_weight=1,termination_weight=30,stride_window_controls=100")
        self.assertEqual((config.stride_tracking_weight, config.stride_window_controls), (1., 100))

    def test_task_averages_velocity_over_the_stride_window_and_declares_the_term(self):
        native = GaitDouble(1)
        task = gated.variant("forward_draw_fraction=0,stride_tracking_weight=0.3")(native)
        task.reset()
        native.commands[:] = torch.tensor([[.05, 0., 0.]])
        task.window.restart(torch.arange(1), native.commands)
        task.gate_window.restart(torch.arange(1), native.commands)
        task.stride_window.restart(torch.arange(1), native.commands)
        task.remaining_controls[:] = 1000
        original, x = native.step, torch.zeros(1)
        def surging(action):
            output = original(action)
            # 0.1 m/s for the first half of each 60-control stride, still for the second: mean 0.05 m/s.
            x.add_((.1 if native.steps % 60 < 30 else 0.) * .02)
            pose = native.current["root"].clone()
            pose[:, 0], pose[:, 3:] = x, torch.tensor([0., 0., 0., 1.])
            native.current["root"] = pose
            native.telemetry["root_pose_xyzw"] = pose.clone()
            return output
        native.step = surging
        for _ in range(120):
            task.step(torch.zeros(1, 18))
        stride = task.stride_window.update(torch.tensor([[0., .1, 0.]]), native.commands)
        # The double's identity pose maps world x to the navigation frame's lateral axis; read the planar speed.
        self.assertAlmostEqual(float(torch.linalg.vector_norm(stride[0, :2])), .05, delta=.003)
        self.assertIn("stride_tracking", task.last_components)
        with tempfile.TemporaryDirectory() as directory:
            gated.variant("stride_tracking_weight=0.3")(GaitDouble(1), output_dir=directory)
            declared = json.loads((Path(directory)/"task_definition.json").read_text())
        self.assertIn("stride_tracking", declared["reward"]["formulas"])


class StrideLengthTests(unittest.TestCase):
    TOES = torch.tensor([[.2, .15], [0., .25], [-.2, .15], [.2, -.15], [0., -.25], [-.2, -.15]])

    def paid(self, commands, travel, landed, config=None):
        config = replace(CONFIG, stride_length_weight=1.) if config is None else config
        n = len(commands)
        values = telemetry(commands, gate_yaw_rate_rad_s=commands[:, 2].clone(), toe_xyz_nav=self.TOES.expand(n, -1, -1).clone(),
                           stride_touchdown=landed, stride_displacement_nav=travel)
        return measured_reward_v4_yaw_gate(values, commands, torch.zeros(n, dtype=torch.bool), config, HEIGHT)[1]

    def test_a_full_stride_pays_the_weight_and_short_backward_or_long_strides_less_or_capped(self):
        full = .05 * 1.2  # commanded speed times the 60-control schedule period
        commands = torch.tensor([[.05, 0., 0.]]).expand(5, -1).clone()
        travel = torch.zeros(5, 6, 2)
        travel[:, 0, 0] = torch.tensor([full, full / 2, -full, 2 * full, full])
        landed = torch.zeros(5, 6, dtype=torch.bool)
        landed[:4, 0] = True  # the last row's foot is still in the air
        parts = self.paid(commands, travel, landed)
        torch.testing.assert_close(parts["stride_length"], torch.tensor([1., .25, 0., 1., 0.]))

    def test_two_taps_earn_half_of_one_stride_and_turns_use_the_lever_arm(self):
        commands = torch.tensor([[.05, 0., 0.], [0., 0., .2]])
        travel = torch.zeros(2, 6, 2)
        travel[0, :2, 0] = .05 * 1.2 / 2  # two feet each land half a stride
        wanted = .2 * torch.stack((-self.TOES[:, 1], self.TOES[:, 0]), -1) * 1.2
        travel[1] = wanted  # every foot lands a full turning stride
        landed = torch.zeros(2, 6, dtype=torch.bool)
        landed[0, :2] = True
        landed[1] = True
        parts = self.paid(commands, travel, landed)
        torch.testing.assert_close(parts["stride_length"], torch.tensor([.5, 6.]))

    def test_zero_command_and_the_default_pay_nothing(self):
        travel = torch.ones(1, 6, 2)
        landed = torch.ones(1, 6, dtype=torch.bool)
        self.assertEqual(float(self.paid(torch.zeros(1, 3), travel, landed)["stride_length"][0]), 0.)
        commands = torch.tensor([[.05, 0., 0.]])
        _, parts = measured_reward_v4_yaw_gate(telemetry(commands, gate_yaw_rate_rad_s=torch.zeros(1)), commands,
                                               torch.zeros(1, dtype=torch.bool), CONFIG, HEIGHT)
        self.assertNotIn("stride_length", parts)
        with self.assertRaisesRegex(ValueError, "stride_touchdown"):
            measured_reward_v4_yaw_gate(telemetry(commands, gate_yaw_rate_rad_s=torch.zeros(1)), commands,
                                        torch.zeros(1, dtype=torch.bool), replace(CONFIG, stride_length_weight=1.), HEIGHT)

    def test_fall_cost_counts_the_average_stride_pay(self):
        # Ceiling 1.5 * 1.3 + 6 * 2 / 60 = 2.15 needs a termination weight above 21.5.
        with self.assertRaises(ValueError):
            gated.reward_config("stride_tracking_weight=0.3,stride_length_weight=2")
        config = gated.reward_config("stride_tracking_weight=0.3,stride_length_weight=2,termination_weight=21.6")
        self.assertEqual(config.stride_length_weight, 2.)
        with self.assertRaises(ValueError):
            gated.reward_config("stride_length_weight=-1")

    def test_task_measures_each_stride_from_liftoff_to_touchdown(self):
        native = GaitDouble(1)
        task = gated.variant("forward_draw_fraction=0,stride_length_weight=1")(native)
        task.reset()
        # The double's identity pose maps world x to the navigation frame's left axis.
        native.commands[:] = torch.tensor([[0., .05, 0.]])
        pose = native.current["root"].clone()
        pose[:, 3:] = torch.tensor([0., 0., 0., 1.])
        native.current["root"] = pose
        native.telemetry["root_pose_xyzw"] = pose.clone()
        task.previous_pose = pose.clone()
        for window in (task.window, task.gate_window, task.stride_window):
            window.restart(torch.arange(1), native.commands)
        task.remaining_controls[:] = 1000
        loaded = native.forces.clone()
        task.step(torch.zeros(1, 18))  # all feet planted
        native.forces[:, 0] = 0.  # foot 0 lifts off
        task.step(torch.zeros(1, 18))
        world = native.world.clone()
        world[0, 0, 0] += .03  # it travels 3 cm in world x while in the air
        native.world = world
        task.step(torch.zeros(1, 18))
        native.forces = loaded.clone()  # and lands
        task.step(torch.zeros(1, 18))
        # 3 cm against the 6 cm stride of 0.05 m/s over 1.2 s: r = 0.5 pays 0.25.
        self.assertAlmostEqual(float(task.last_components["stride_length"][0]), .25, places=4)
        self.assertFalse(bool(task.lifted[0, 0]))
        task.step(torch.zeros(1, 18))  # standing on it pays nothing more
        self.assertEqual(float(task.last_components["stride_length"][0]), 0.)
        with tempfile.TemporaryDirectory() as directory:
            gated.variant("stride_length_weight=1")(GaitDouble(1), output_dir=directory)
            declared = json.loads((Path(directory)/"task_definition.json").read_text())
        self.assertIn("stride_length", declared["reward"]["formulas"])


class HeadingTrackingTests(unittest.TestCase):
    def test_pay_follows_the_heading_error_against_its_scale(self):
        config = replace(CONFIG, heading_weight=.5)
        commands = torch.tensor([[.05, 0., 0.], [.05, 0., 0.], [0., 0., .2], [.05, 0., 0.], [0., 0., 0.]])
        error = torch.tensor([0., .125, -.25, 1., .3])
        values = telemetry(commands, gate_yaw_rate_rad_s=commands[:, 2].clone(), heading_error_rad=error)
        alive = torch.zeros(5, dtype=torch.bool)
        v4_reward, _ = task_v4.measured_reward_v4(values, commands, alive, config, HEIGHT)
        reward, parts = measured_reward_v4_yaw_gate(values, commands, alive, config, HEIGHT)
        # On target pays the weight, half the scale 3/4 of it, the scale nothing, far off the floor; rest unpaid.
        torch.testing.assert_close(parts["heading_tracking"], .5 * torch.tensor([1., .75, 0., -1., 0.]))
        torch.testing.assert_close(reward - v4_reward - parts["schedule_yaw_gate"], parts["heading_tracking"])

    def test_default_adds_nothing_and_the_option_needs_its_telemetry_and_fall_cost(self):
        self.assertEqual((CONFIG.heading_weight, CONFIG.heading_scale_rad), (0., .25))
        commands = torch.tensor([[.05, 0., 0.]])
        values = telemetry(commands, gate_yaw_rate_rad_s=torch.zeros(1))
        _, parts = measured_reward_v4_yaw_gate(values, commands, torch.zeros(1, dtype=torch.bool), CONFIG, HEIGHT)
        self.assertNotIn("heading_tracking", parts)
        with self.assertRaisesRegex(ValueError, "heading_error_rad"):
            measured_reward_v4_yaw_gate(values, commands, torch.zeros(1, dtype=torch.bool), replace(CONFIG, heading_weight=.5), HEIGHT)
        # Ceiling 1.5 + 0.5 = 2 reaches the default termination weight of 20; more needs a larger one.
        self.assertEqual(gated.reward_config("heading_weight=0.5").heading_weight, .5)
        for bad in ("heading_weight=0.6", "heading_weight=-1", "heading_scale_rad=0"):
            with self.subTest(bad=bad), self.assertRaises(ValueError):
                gated.reward_config(bad)

    def test_task_tracks_the_heading_the_command_implies_and_restarts_with_each_command(self):
        native = GaitDouble(2)
        task = gated.variant("forward_draw_fraction=0,heading_weight=0.5")(native)
        task.reset()
        native.commands[:] = torch.tensor([[.05, 0., 0.], [0., 0., .2]])
        task.remaining_controls[:] = 1000
        heading = torch.zeros(2)
        rate = torch.tensor([.05, .2])  # row 0 drifts while walking straight; row 1 turns as commanded
        original = native.step
        def turning(action):
            output = original(action)
            heading.add_(rate * .02)
            pose = native.current["root"].clone()
            pose[:, 3:] = torch.stack((torch.zeros(2), torch.zeros(2), torch.sin(heading / 2), torch.cos(heading / 2)), -1)
            native.current["root"] = pose
            native.telemetry["root_pose_xyzw"] = pose.clone()
            return output
        native.step = turning
        for _ in range(100):
            task.step(torch.zeros(2, 18))
        self.assertAlmostEqual(float(task.heading_error[0]), .1, delta=.005)  # 0.05 rad/s for 2 s
        self.assertAlmostEqual(float(task.heading_error[1]), 0., delta=.005)
        self.assertLess(float(task.last_components["heading_tracking"][0]), float(task.last_components["heading_tracking"][1]))
        native.commands[0] = torch.tensor([0., .05, 0.])  # a new command starts a new target heading
        task.step(torch.zeros(2, 18))
        self.assertAlmostEqual(float(task.heading_error[0]), .05 * .02, delta=1e-4)
        with tempfile.TemporaryDirectory() as directory:
            gated.variant("heading_weight=0.5")(GaitDouble(1), output_dir=directory)
            declared = json.loads((Path(directory)/"task_definition.json").read_text())
        self.assertIn("heading_tracking", declared["reward"]["formulas"])

class NativeSelectionTests(unittest.TestCase):
    OPTIONS = "forward_draw_fraction=0,turn_draw_fraction=0.3,gate_window_controls=120,yaw_gate_bonus=0.25"

    def test_prepare_selects_the_yaw_gate_task_for_version_four_training_alone(self):
        from locomotion.prepare import prepare
        from locomotion.tests.test_task_v4 import REMOTE
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            args = prepare(root/"gate", REMOTE, mode="train", reward_version="4", gait_clock=60,
                           reward_options=self.OPTIONS, reward_task="v4_yaw_gate")["command_args"]
            self.assertEqual(args[args.index("--reward-task")+1], "v4_yaw_gate")
            pack = json.loads((root/"gate/PACK.json").read_text())
            self.assertEqual((pack["reward_task"], pack["reward_options"]), ("v4_yaw_gate", self.OPTIONS))
            self.assertIn("locomotion/task_v4_yaw_gate.py", json.loads((root/"gate/source/FREEZE_SHA256.json").read_text()))
            plain = prepare(root/"plain", REMOTE, mode="train", reward_version="4")["command_args"]
            self.assertNotIn("--reward-task", plain)
            self.assertNotIn("reward_task", json.loads((root/"plain/PACK.json").read_text()))
            for index, bad in enumerate((dict(reward_version="2", reward_task="v4_yaw_gate"),
                                         dict(reward_version="4", reward_task="v5"))):
                with self.subTest(bad=bad), self.assertRaisesRegex(ValueError, "reward task"):
                    prepare(root/f"bad{index}", REMOTE, mode="train", **bad)

    def test_train_checks_the_task_and_its_options_before_launch(self):
        from locomotion import train
        base = ["--asset", "a", "--model", "m", "--geometry", "g", "--geometry-extrema", "e", "--stance", "s",
                "--output", "o", "--source-freeze-sha256", "f"*64, "--num-envs", "1", "--preflight-only"]
        with self.assertRaisesRegex(ValueError, "reward task"):
            train.main(base + ["--mode", "train", "--reward-version", "2", "--reward-task", "v4_yaw_gate"])
        with self.assertRaisesRegex(ValueError, "unique RewardV4YawGateConfig"):
            train.main(base + ["--mode", "train", "--reward-version", "4", "--reward-task", "v4_yaw_gate",
                               "--reward-options", "not_a_field=0"])
        # Gate options name no RewardV4Config field, so plain v4 refuses them.
        with self.assertRaisesRegex(ValueError, "unique RewardV4Config"):
            train.main(base + ["--mode", "train", "--reward-version", "4", "--reward-options", "turn_draw_fraction=0.3"])


if __name__ == "__main__":
    unittest.main()
