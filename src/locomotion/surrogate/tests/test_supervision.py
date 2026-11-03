"""Check the latent identification supervision and its wrapper buffer."""

import copy
import hashlib
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from tensordict import TensorDict

import locomotion.omniwalk.task as task_module
from locomotion.omniwalk.destination_task import experiment_task
from locomotion.rl.sim2real import (
    TARGET_FIELDS,
    CompensatedActor,
    LatentSupervision,
    Sim2RealVecEnv,
)
from locomotion.surrogate.env import (
    ContactModel,
    DynamicsRandomization,
    SurrogateConfig,
    SurrogateEnv,
)

WIDE = DynamicsRandomization(
    friction_scale=(0.6, 1.4),
    mass_scale=(0.9, 1.1),
    motor_scale=(0.9, 1.1),
    kd_scale=(0.8, 1.2),
    action_delay_controls=1,
    sag_per_minute=(0.0, 0.1),
    sag_floor=0.85,
)


def composite_actor(tmp):
    """A small compensated actor around a freshly written core checkpoint."""
    from rsl_rl.models import MLPModel

    obs = TensorDict({"policy": torch.randn(4, 278), "critic": torch.randn(4, 281)}, batch_size=[4])
    groups = {"actor": ["policy"], "critic": ["critic"]}
    distribution = {
        "class_name": "locomotion.rl.ppo:BoundedMeanGaussian",
        "init_std": 0.05,
        "std_type": "log",
    }
    core = MLPModel(
        obs,
        groups,
        "actor",
        18,
        hidden_dims=[32],
        activation="elu",
        obs_normalization=False,
        distribution_cfg=copy.deepcopy(distribution),
    )
    path = Path(tmp) / "core.pt"
    torch.save({"actor_state_dict": core.state_dict()}, path)
    settings = {
        "core": {
            "checkpoint": str(path),
            "checkpoint_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "actor": {
                "class_name": "rsl_rl.models:MLPModel",
                "hidden_dims": [32],
                "activation": "elu",
                "obs_normalization": False,
                "distribution_cfg": copy.deepcopy(distribution),
            },
        },
        "encoder_hidden": (32,),
        "latent_dim": 6,
        "hidden_dims": [32],
        "activation": "elu",
        "obs_normalization": False,
        "distribution_cfg": copy.deepcopy(distribution),
    }
    return CompensatedActor(obs, groups, "actor", 18, **settings)


class LatentSupervisionTests(unittest.TestCase):
    def test_supervision_makes_the_latent_identify_synthetic_draws(self):
        torch.manual_seed(5)
        with tempfile.TemporaryDirectory() as tmp:
            actor = composite_actor(tmp)
            observation = torch.randn(512, 278)
            # The true draws are a fixed linear readout of six observation columns, scaled into
            # each field's range, so an encoder that identifies them exists.
            mixing = torch.randn(6, 6) * 0.5
            source = torch.tanh(observation[:, :6] @ mixing)
            lows = torch.tensor(
                [
                    WIDE.friction_scale[0],
                    WIDE.mass_scale[0],
                    WIDE.motor_scale[0],
                    WIDE.kd_scale[0],
                    0.0,
                    WIDE.sag_per_minute[0],
                ]
            )
            highs = torch.tensor(
                [
                    WIDE.friction_scale[1],
                    WIDE.mass_scale[1],
                    WIDE.motor_scale[1],
                    WIDE.kd_scale[1],
                    1.0,
                    WIDE.sag_per_minute[1],
                ]
            )
            draws = lows + (source + 1) / 2 * (highs - lows)
            wrapper = SimpleNamespace(supervision_batch=lambda: (observation, draws))
            algorithm = SimpleNamespace(actor=actor, update=lambda: {})
            supervision = LatentSupervision(algorithm, wrapper, WIDE, rate=3e-3, steps=4)
            supervision.update()
            first = supervision.last_loss
            for _ in range(150):
                supervision.update()
            self.assertLess(supervision.last_loss, first / 5)
            # The information now lives in the latent itself: an offline linear fit recovers it.
            with torch.no_grad():
                latent = actor.dynamics_latent(observation).numpy()
            design = np.concatenate([latent, np.ones((len(latent), 1))], 1)
            coefficients, *_ = np.linalg.lstsq(design, draws.numpy(), rcond=None)
            residual = draws.numpy() - design @ coefficients
            quality = 1 - residual.var(0) / draws.numpy().var(0)
            self.assertGreater(float(quality.min()), 0.5)

    def test_supervision_validates_its_pieces(self):
        from rsl_rl.models import MLPModel

        obs = TensorDict(
            {"policy": torch.randn(2, 278), "critic": torch.randn(2, 281)}, batch_size=[2]
        )
        groups = {"actor": ["policy"], "critic": ["critic"]}
        plain = MLPModel(obs, groups, "actor", 18, hidden_dims=[16], activation="elu")
        wrapper = SimpleNamespace(supervision_batch=lambda: None)
        with self.assertRaises(ValueError):
            LatentSupervision(SimpleNamespace(actor=plain, update=lambda: {}), wrapper, WIDE, 1e-3)
        with tempfile.TemporaryDirectory() as tmp:
            actor = composite_actor(tmp)
            algorithm = SimpleNamespace(actor=actor, update=lambda: {})
            with self.assertRaises(ValueError):
                LatentSupervision(algorithm, SimpleNamespace(), WIDE, 1e-3)
            with self.assertRaises(ValueError):
                LatentSupervision(algorithm, wrapper, None, 1e-3)
            supervision = LatentSupervision(algorithm, wrapper, WIDE, 1e-3)
            self.assertEqual(supervision.update(), {})
            self.assertIsNone(supervision.last_loss)

    def test_the_wrapper_buffers_observations_with_their_draws(self):
        env = SurrogateEnv(
            SurrogateConfig(num_envs=2, seed=20260917, episode_seconds=2.0),
            threads=0,
            contact=ContactModel(touchdown_overshoot=False),
            randomization=WIDE,
        )
        try:
            task = experiment_task("destination_v35")(
                env, task_module.TaskConfig(seed=20260917), None
            )
            wrapped = Sim2RealVecEnv(
                task,
                destination_input="terrain_v2",
                gait_clock_scaling="speed_v1",
                gait_clock=60,
                observation_scaling="fixed",
                command_segments="bootstrap",
                supervision_rows=64,
            )
            self.assertIsNone(wrapped.supervision_batch())
            wrapped.get_observations()
            for _ in range(3):
                wrapped.step(torch.zeros(2, 18))
            observation, draws = wrapped.supervision_batch()
            self.assertEqual((observation.shape[-1], draws.shape[-1]), (278, len(TARGET_FIELDS)))
            self.assertEqual(len(observation), len(draws))
            expected = np.stack(
                [env.randomization_draws[name] for name in TARGET_FIELDS], -1
            ).astype(np.float32)
            self.assertTrue(np.array_equal(draws[-2:].numpy(), expected))
            for _ in range(40):
                wrapped.step(torch.zeros(2, 18))
            kept = len(wrapped.supervision_batch()[0])
            self.assertLessEqual(kept, 64 + 2)
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
