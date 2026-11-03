"""Check the per-episode dynamics randomization and its bitwise-inert default path."""

import unittest

import numpy as np
import torch

from locomotion.robots.hexapod import MASS_KG
from locomotion.surrogate.env import (
    ContactModel,
    DynamicsRandomization,
    SurrogateConfig,
    SurrogateEnv,
    randomization_overrides,
)

WIDE = DynamicsRandomization(
    friction_scale=(0.6, 1.4),
    mass_scale=(0.8, 1.2),
    motor_scale=(0.8, 1.2),
    kd_scale=(0.8, 1.2),
    action_delay_controls=1,
    sag_per_minute=(0.0, 0.3),
    sag_floor=0.8,
)
SOFT_CONTACT = ContactModel(touchdown_overshoot=False)


def build(num_envs, randomization, threads=0, contact=None, seed=20260917):
    return SurrogateEnv(
        SurrogateConfig(num_envs=num_envs, seed=seed, episode_seconds=4.0),
        threads=threads,
        contact=contact,
        randomization=randomization,
    )


def roll(env, controls=25, std=0.3, seed=7):
    """Step seeded Gaussian actions and return the physics state after the last control."""
    generator = torch.Generator().manual_seed(seed)
    for _ in range(controls):
        output = env.step(std * torch.randn(3, 18, generator=generator)[: env.num_envs])
        done = output["terminated"] | output["truncated"]
        if bool(done.any()):
            env.reset(done.nonzero().flatten())
    return env._state[:, 1:].copy()


class RandomizationTests(unittest.TestCase):
    def test_validation_and_overrides(self):
        model = randomization_overrides(["motor_scale=(0.9, 1.1)", "action_delay_controls=1"])
        self.assertEqual((model.motor_scale, model.action_delay_controls), ((0.9, 1.1), 1))
        self.assertEqual(randomization_overrides([]), DynamicsRandomization())
        for bad in (["unknown_field=1"], ["motor_scale"]):
            with self.assertRaises(ValueError):
                randomization_overrides(bad)
        for bad in (
            dict(mass_scale=(0.4, 1.0)),
            dict(motor_scale=(1.2, 0.8)),
            dict(friction_scale=(0.1, 1.0)),
            dict(action_delay_controls=3),
            dict(action_delay_controls=1.0),
            dict(sag_per_minute=(0.0, 0.6)),
            dict(sag_floor=0.4),
        ):
            with self.assertRaises(ValueError):
                DynamicsRandomization(**bad)
        # The touchdown correction reads one shared friction, so randomizing it needs the model off.
        with self.assertRaises(ValueError):
            build(1, DynamicsRandomization(friction_scale=(0.8, 1.2)))
        build(1, DynamicsRandomization(friction_scale=(0.8, 1.2)), contact=SOFT_CONTACT).close()

    def test_inert_ranges_match_the_default_trajectory_bitwise(self):
        plain = build(3, None)
        inert = build(3, DynamicsRandomization())
        try:
            self.assertTrue(np.array_equal(roll(plain), roll(inert)))
        finally:
            plain.close()
            inert.close()

    def test_draws_are_reproducible_and_batch_independent(self):
        first = build(3, WIDE, contact=SOFT_CONTACT)
        second = build(3, WIDE, contact=SOFT_CONTACT)
        alone = build(1, WIDE, contact=SOFT_CONTACT)
        threaded = build(3, WIDE, contact=SOFT_CONTACT, threads=2)
        try:
            for name, values in first.randomization_draws.items():
                self.assertTrue(np.array_equal(values, second.randomization_draws[name]), name)
                self.assertEqual(values[0], alone.randomization_draws[name][0], name)
            state = roll(first)
            self.assertTrue(np.array_equal(state, roll(threaded)))
            self.assertTrue(np.array_equal(state[:1], roll(alone)))
            # The draws moved off their defaults at all (the ranges are wide).
            self.assertNotEqual(float(first.randomization_draws["mass_scale"].sum()), 3.0)
        finally:
            for env in (first, second, alone, threaded):
                env.close()

    def test_mass_and_friction_reach_the_replica_models_per_reset(self):
        env = build(3, WIDE, contact=SOFT_CONTACT)
        try:
            draws = env.randomization_draws
            for row in range(3):
                self.assertAlmostEqual(
                    float(env._models[row].body_mass.sum()),
                    draws["mass_scale"][row] * MASS_KG,
                    delta=1e-9,
                )
                self.assertAlmostEqual(
                    float(env._models[row].geom_friction[0, 0]),
                    draws["friction_scale"][row] * env.contact_model.friction[0],
                    delta=1e-12,
                )
            before = {row: env._models[row].body_mass.copy() for row in range(3)}
            kept = draws["mass_scale"].copy()
            env.reset([1])
            self.assertTrue(np.array_equal(env._models[0].body_mass, before[0]))
            self.assertTrue(np.array_equal(env._models[2].body_mass, before[2]))
            self.assertNotEqual(draws["mass_scale"][1], kept[1])
            self.assertEqual(draws["mass_scale"][0], kept[0])
        finally:
            env.close()

    def test_motor_scale_caps_the_applied_torque_exactly(self):
        env = build(1, DynamicsRandomization(motor_scale=(0.5, 0.5)))
        try:
            # A large position error at rest requests far over the scaled ceiling 0.5 * 1.6 N m.
            env._state[:, 1 + env._qpos_columns] = (env.neutral - 0.2).numpy().astype(np.float64)
            env.step(torch.ones(1, 18))
            applied = env.telemetry["applied_torque_abs_max_400hz"]
            requested = env.telemetry["requested_torque_abs_max_400hz"]
            self.assertGreater(float(requested.max()), 1.6)
            self.assertAlmostEqual(float(applied.max()), 0.5 * 1.6, places=6)
        finally:
            env.close()

    def test_sag_drains_the_ceiling_toward_its_floor(self):
        env = build(1, DynamicsRandomization(sag_per_minute=(0.3, 0.3), sag_floor=0.9))
        try:
            rate = 0.3 * env.cfg.control_dt / 60.0
            self.assertAlmostEqual(float(env._sag_rate[0]), rate, places=9)
            for _ in range(50):
                env.step(torch.zeros(1, 18))
            self.assertAlmostEqual(float(env._sag_scale[0]), 1.0 - 50 * rate, places=5)
            env._sag_scale[:] = 0.9 + 0.5 * rate
            env.step(torch.zeros(1, 18))
            env.step(torch.zeros(1, 18))
            self.assertAlmostEqual(float(env._sag_scale[0]), 0.9, places=6)
            env.reset([0])
            self.assertEqual(float(env._sag_scale[0]), 1.0)
        finally:
            env.close()

    def test_action_delay_holds_the_previous_action_one_control(self):
        delayed = build(2, DynamicsRandomization(action_delay_controls=1))
        plain = build(2, None)
        try:
            delayed._delay_steps[:] = 1
            first = 0.3 * torch.ones(2, 18)
            second = -0.3 * torch.ones(2, 18)
            # The first control of an episode is undelayed; afterwards the servo runs one behind.
            delayed.step(first)
            delayed.step(second)
            plain.step(first)
            plain.step(first)
            self.assertTrue(np.array_equal(delayed._state, plain._state))
            delayed.step(second)
            plain.step(second)
            self.assertTrue(np.array_equal(delayed._state, plain._state))
        finally:
            delayed.close()
            plain.close()


if __name__ == "__main__":
    unittest.main()
