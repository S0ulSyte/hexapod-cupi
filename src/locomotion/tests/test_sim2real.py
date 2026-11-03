"""Check the compensated actor around a frozen core: equality at start, bounds and reload."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path

import torch
from tensordict import TensorDict

from locomotion.rl.ppo import load_actor
from locomotion.rl.sim2real import (
    MODULATED_COLUMNS,
    CompensatedActor,
    WarmStartedCritic,
)

OBS_GROUPS = {"actor": ["policy"], "critic": ["critic"]}
CORE_ACTOR = {
    "class_name": "rsl_rl.models:MLPModel",
    "hidden_dims": [64, 32],
    "activation": "elu",
    "obs_normalization": False,
    "distribution_cfg": {
        "class_name": "locomotion.rl.ppo:BoundedMeanGaussian",
        "init_std": 0.05,
        "std_type": "log",
    },
}


def observations(rows=4, generator=None):
    generator = torch.Generator().manual_seed(11) if generator is None else generator
    return TensorDict(
        {
            "policy": torch.randn(rows, 278, generator=generator),
            "critic": torch.randn(rows, 281, generator=generator),
        },
        batch_size=[rows],
    )


class SimToRealTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        from rsl_rl.models import MLPModel

        torch.manual_seed(3)
        cls.obs = observations()
        actor_settings = copy.deepcopy(CORE_ACTOR)
        actor_settings.pop("class_name")
        cls.core_actor = MLPModel(cls.obs, OBS_GROUPS, "actor", 18, **actor_settings)
        cls.core_critic = MLPModel(
            cls.obs, OBS_GROUPS, "critic", 1, hidden_dims=[64, 32], activation="elu"
        )
        cls.tmp = tempfile.TemporaryDirectory()
        cls.checkpoint_path = Path(cls.tmp.name) / "core.pt"
        torch.save(
            {
                "actor_state_dict": cls.core_actor.state_dict(),
                "critic_state_dict": cls.core_critic.state_dict(),
            },
            cls.checkpoint_path,
        )
        cls.checkpoint_sha = hashlib.sha256(cls.checkpoint_path.read_bytes()).hexdigest()

    @classmethod
    def tearDownClass(cls):
        cls.tmp.cleanup()

    def core_settings(self):
        return {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_sha256": self.checkpoint_sha,
            "actor": copy.deepcopy(CORE_ACTOR),
        }

    def actor_settings(self):
        return dict(
            core=self.core_settings(),
            encoder_hidden=(32, 16),
            latent_dim=4,
            hidden_dims=[64, 32],
            activation="elu",
            obs_normalization=False,
            distribution_cfg={
                "class_name": "locomotion.rl.ppo:BoundedMeanGaussian",
                "init_std": 0.05,
                "std_type": "log",
            },
        )

    def composite(self):
        return CompensatedActor(self.obs, OBS_GROUPS, "actor", 18, **self.actor_settings())

    def test_the_composite_matches_its_frozen_core_at_initialization(self):
        composite = self.composite()
        self.assertTrue(torch.equal(composite(self.obs), self.core_actor(self.obs)))
        latent = composite.dynamics_latent(self.obs)
        self.assertEqual(tuple(latent.shape), (4, 4))

    def test_only_the_compensator_trains(self):
        composite = self.composite()
        composite(self.obs).sum().backward()
        for name, parameter in composite.mlp.core.named_parameters():
            self.assertFalse(parameter.requires_grad, name)
            self.assertIsNone(parameter.grad, name)
        for module in (composite.mlp.gain_head, composite.mlp.bias_head, composite.mlp.delta_head):
            for parameter in module.parameters():
                self.assertTrue(parameter.requires_grad)
                self.assertIsNotNone(parameter.grad)

    def test_bounds_hold_and_untouched_columns_pass_through(self):
        composite = self.composite()
        mean = composite.mlp
        with torch.no_grad():
            for head in (mean.gain_head, mean.bias_head, mean.delta_head):
                head.weight.normal_(0.0, 10.0)
                head.bias.normal_(0.0, 10.0)
        policy = self.obs["policy"]
        latent = mean.latent(policy)
        gains = 1.0 + mean.gain_limit * torch.tanh(mean.gain_head(latent))
        biases = mean.bias_limit * torch.tanh(mean.bias_head(latent))
        self.assertTrue(bool((gains >= 0.8).all() and (gains <= 1.2).all()))
        modulated = policy.clone()
        modulated[..., mean.columns] = (
            policy[..., mean.columns] * gains[..., mean.slots] + biases[..., mean.slots]
        )
        untouched = [i for i in range(278) if i not in set(MODULATED_COLUMNS)]
        self.assertTrue(torch.equal(modulated[..., untouched], policy[..., untouched]))
        delta = mean.delta_limit * torch.tanh(mean.delta_head(latent))
        self.assertTrue(bool((delta.abs() <= mean.delta_limit).all()))
        expected = torch.tanh(mean.core(modulated) + delta)
        self.assertTrue(torch.equal(composite(self.obs), expected))

    def test_load_actor_rebuilds_the_recorded_composite(self):
        composite = self.composite()
        with torch.no_grad():
            for parameter in composite.mlp.encoder.parameters():
                parameter.normal_(0.0, 0.5)
            composite.mlp.delta_head.bias.normal_(0.0, 2.0)
        config = {
            "actor": {"class_name": "locomotion.rl.sim2real:CompensatedActor"}
            | self.actor_settings(),
            "obs_groups": OBS_GROUPS,
        }
        rebuilt, _ = load_actor(
            {"actor_state_dict": composite.state_dict()}, config, self.obs, "cpu"
        )
        self.assertTrue(torch.equal(rebuilt(self.obs), composite(self.obs)))

    def test_core_verification_failures_raise(self):
        settings = self.actor_settings()
        settings["core"]["checkpoint_sha256"] = "0" * 64
        with self.assertRaises(ValueError):
            CompensatedActor(self.obs, OBS_GROUPS, "actor", 18, **settings)
        settings = self.actor_settings()
        settings["core"]["checkpoint"] = str(self.checkpoint_path) + ".missing"
        with self.assertRaises(ValueError):
            CompensatedActor(self.obs, OBS_GROUPS, "actor", 18, **settings)
        settings = self.actor_settings()
        settings["core"]["actor"]["obs_normalization"] = True
        with self.assertRaises(ValueError):
            CompensatedActor(self.obs, OBS_GROUPS, "actor", 18, **settings)
        settings = self.actor_settings()
        settings.pop("core")
        with self.assertRaises(ValueError):
            CompensatedActor(self.obs, OBS_GROUPS, "actor", 18, **settings)
        with self.assertRaises(ValueError):
            CompensatedActor(
                self.obs, OBS_GROUPS, "actor", 18, **self.actor_settings() | {"latent_dim": 0}
            )
        with self.assertRaises(ValueError):
            CompensatedActor(
                self.obs,
                OBS_GROUPS,
                "actor",
                18,
                **self.actor_settings() | {"observation_gain_limit": 1.5},
            )

    def test_warm_started_critic_matches_the_core_critic_then_trains(self):
        critic = WarmStartedCritic(
            self.obs,
            OBS_GROUPS,
            "critic",
            1,
            core={
                "checkpoint": str(self.checkpoint_path),
                "checkpoint_sha256": self.checkpoint_sha,
            },
            hidden_dims=[64, 32],
            activation="elu",
        )
        self.assertTrue(torch.equal(critic(self.obs), self.core_critic(self.obs)))
        self.assertTrue(all(p.requires_grad for p in critic.parameters()))


if __name__ == "__main__":
    unittest.main()
