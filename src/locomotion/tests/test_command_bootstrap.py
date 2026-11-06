"""Check command return targets through RSL-RL, including mixed episode endings."""

import io
import unittest
from contextlib import redirect_stdout
from types import SimpleNamespace
from unittest.mock import patch

import torch
from rsl_rl.algorithms import PPO

import locomotion.omniwalk.agents.rsl_rl_ppo_cfg as ppo_cfg
import locomotion.rl.wrapper as wrapper


class MixedTask:
    num_envs, num_actions, device = 4, 18, "cpu"
    cfg = SimpleNamespace(episode_seconds=20.0, control_dt=0.02, declaration=lambda: {})

    def __init__(self):
        self.commands = torch.tensor([[0.05, 0.0, 0.0]] * 4)
        self.episode_steps = torch.zeros(4, dtype=torch.long)
        self.position = torch.ones(4)
        self.resets = []

    def declaration(self):
        return {}

    def output(self):
        obs = torch.zeros(4, 231)
        obs[:, 0] = self.position
        obs[:, 210:213] = self.commands
        return {"obs": obs, "critic": torch.cat((obs, torch.zeros(4, 3)), -1)}

    def reset(self, selected=None):
        selected = torch.arange(4) if selected is None else selected
        self.resets.append(selected.tolist())
        self.position[selected] = 1.0
        self.episode_steps[selected] = 0
        return self.output()

    def step(self, action):
        self.position += 6.0
        self.episode_steps += 1
        self.commands[1:] = 0.0
        return {
            **self.output(),
            "reward": torch.ones(4),
            "terminated": torch.tensor([False, False, True, False]),
            "truncated": torch.tensor([False, False, True, True]),
        }


class CommandBootstrapTests(unittest.TestCase):
    def test_post_action_value_holds_old_command_and_does_not_double_bootstrap(self):
        for ceiling in (None, 3e-4):
            with self.subTest(ceiling=ceiling):
                task = MixedTask()
                wrapped = wrapper.OmniVecEnv(
                    task,
                    command_segments="bootstrap",
                    observation_scaling="fixed",
                    gait_clock=60,
                    velocity_noise=0.5,
                )
                config = ppo_cfg.ppo_config(
                    7,
                    command_segments="bootstrap",
                    learning_rate_max=ceiling,
                    observation_normalization="none",
                )
                config.pop("environment_wrapper")
                config.update(num_steps_per_env=1, multi_gpu=None)
                for role in ("actor", "critic"):
                    config[role]["hidden_dims"] = [8]
                obs = wrapped.get_observations()
                with redirect_stdout(io.StringIO()):
                    alg = PPO.construct_algorithm(obs, wrapped, config, "cpu")

                def value(observation):
                    critic = observation["critic"]
                    return critic[:, :1] + 100 * critic[:, 210:211]

                with patch.object(alg.critic, "forward", side_effect=value), torch.no_grad():
                    action = alg.act(obs)
                    next_obs, rewards, done, extras = wrapped.step(action)
                    rng = torch.random.get_rng_state().clone()
                    alg.process_env_step(next_obs, rewards, done, extras)
                torch.testing.assert_close(torch.random.get_rng_state(), rng)
                # Before: 2 + 100 = 102. After under the held command: 14 + 100 = 114.
                torch.testing.assert_close(
                    alg.storage.rewards[0, :, 0],
                    torch.tensor([1.0, 1.0 + 0.99 * 114.0, 1.0, 1.0 + 0.99 * 102.0]),
                )
                torch.testing.assert_close(alg.storage.values[0, :, 0], torch.full((4,), 102.0))
                self.assertEqual(done.tolist(), [0, 1, 1, 1])
                self.assertEqual(task.resets, [[0, 1, 2, 3], [2, 3]])
                self.assertEqual(float(next_obs["critic"][1, 210]), 0.0)
                self.assertEqual(next_obs["critic"][1, -2:].tolist(), [0.0, 0.0])
                terminal = extras["command_bootstrap_observation"]["critic"]
                torch.testing.assert_close(
                    terminal[1, -2:],
                    wrapper.clock_features(torch.tensor([1]), 60, torch.tensor([[0.05, 0.0, 0.0]]))[
                        0
                    ],
                )


class VaryingTask(MixedTask):
    """Row 1's command evolves within its run; row 3's run truly switches; row 2 terminates."""

    def step(self, action):
        self.position += 6.0
        self.episode_steps += 1
        self.commands[1] = torch.tensor([0.05, 0.01, 0.0])
        self.commands[3] = 0.0
        return {
            **self.output(),
            "reward": torch.ones(4),
            "terminated": torch.tensor([False, False, True, False]),
            "truncated": torch.tensor([False, False, False, False]),
            "command_switched": torch.tensor([False, False, False, True]),
        }


class VaryingCommandSwitchTests(unittest.TestCase):
    def test_published_switch_flag_overrides_value_comparison(self):
        task = VaryingTask()
        wrapped = wrapper.OmniVecEnv(
            task,
            command_segments="bootstrap",
            observation_scaling="fixed",
            gait_clock=60,
            velocity_noise=0.5,
        )
        wrapped.get_observations()
        _, _, done, extras = wrapped.step(torch.zeros(4, 18))
        self.assertEqual(done.tolist(), [0, 0, 1, 1])
        self.assertEqual(extras["command_time_outs"].tolist(), [False, False, False, True])
        self.assertEqual(extras["time_outs"].tolist(), [False, False, False, True])
        self.assertIn("command_bootstrap_observation", extras)


if __name__ == "__main__":
    unittest.main()
