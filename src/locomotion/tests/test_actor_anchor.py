"""Check the actor anchor: pull math, exclusions and validation."""

import unittest
from types import SimpleNamespace

import torch
from tensordict import TensorDict

from locomotion.rl.ppo import ActorAnchor


def small_actor():
    from rsl_rl.models import MLPModel

    obs = TensorDict({"policy": torch.randn(2, 8)}, batch_size=[2])
    return MLPModel(
        obs,
        {"actor": ["policy"]},
        "actor",
        3,
        hidden_dims=[4],
        activation="elu",
        distribution_cfg={
            "class_name": "locomotion.rl.ppo:BoundedMeanGaussian",
            "init_std": 0.05,
            "std_type": "log",
        },
    )


class ActorAnchorTests(unittest.TestCase):
    def test_the_anchor_removes_a_fraction_of_the_displacement(self):
        torch.manual_seed(2)
        actor = small_actor()
        algorithm = SimpleNamespace(actor=actor, update=lambda: {"ok": 1.0})
        anchor = ActorAnchor(algorithm, 0.25)
        init = {name: p.detach().clone() for name, p in actor.named_parameters()}
        with torch.no_grad():
            for name, parameter in actor.named_parameters():
                parameter.add_(1.0)
        self.assertEqual(anchor.update(), {"ok": 1.0})
        for name, parameter in actor.named_parameters():
            if "distribution" in name:
                # The deviation schedule owns the distribution parameters: never pulled.
                self.assertTrue(torch.allclose(parameter, init[name] + 1.0))
            else:
                self.assertTrue(torch.allclose(parameter, init[name] + 0.75), name)
        # Repeated updates decay the displacement geometrically toward the init.
        anchor.update()
        for name, parameter in actor.named_parameters():
            if "distribution" not in name:
                self.assertTrue(torch.allclose(parameter, init[name] + 0.5625), name)

    def test_the_anchor_validates_its_weight_and_parameters(self):
        actor = small_actor()
        algorithm = SimpleNamespace(actor=actor, update=lambda: {})
        for bad in (0.0, 1.0, -0.1, 2.0):
            with self.assertRaises(ValueError):
                ActorAnchor(algorithm, bad)
        for parameter in actor.parameters():
            parameter.requires_grad_(False)
        with self.assertRaises(ValueError):
            ActorAnchor(algorithm, 0.1)


if __name__ == "__main__":
    unittest.main()
