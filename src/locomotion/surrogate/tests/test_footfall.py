"""terrain_v1 on the surrogate: the footfall inputs read the terrain and evaluation rebuilds them."""

import unittest

import torch

from locomotion.omniwalk import task as task_module
from locomotion.omniwalk.destination_task import experiment_task
from locomotion.rl.destination_wrapper import DestinationVecEnv
from locomotion.surrogate.env import ContactModel, SurrogateConfig, SurrogateEnv
from locomotion.surrogate.terrain import Terrain

OPTIONS = {
    "observation_scaling": "fixed",
    "command_segments": "bootstrap",
    "gait_clock": 60,
    "action_smoothing": "mean2",
    "destination_input": "terrain_v1",
    "gait_clock_scaling": "speed_v1",
}
# A box the front toes stand on and every forward touchdown lands on; the rear toes stay on the floor.
BOX = {
    "size_m": [3, 3],
    "resolution_m": 0.02,
    "features": [{"type": "box", "center": [0.55, 0], "size": [1.0, 3.0], "height": 0.08}],
}


def build(terrain):
    env = SurrogateEnv(
        SurrogateConfig(num_envs=1, seed=20260917),
        threads=0,
        contact=ContactModel(touchdown_overshoot=False),
        terrain=terrain,
    )
    task = experiment_task("destination_v33")(env, task_module.TaskConfig(seed=20260917), None)
    wrapped = DestinationVecEnv(task, **OPTIONS)
    env.commands[:] = torch.tensor([[0.08, 0.0, 0.0]])
    task.remaining_controls[:] = 5000
    return env, task, wrapped


class FootfallTests(unittest.TestCase):
    def test_the_lookahead_reads_the_ground_ahead_relative_to_the_support(self):
        env, task, wrapped = build(Terrain(BOX))
        try:
            for _ in range(5):
                observation, *_ = wrapped.step(torch.zeros(1, 18))
            block = observation["policy"][0, 242:272].reshape(6, 5)
            front, rear = block[[0, 3]], block[[2, 5]]
            # The 8 cm box is 0.8 of the 0.1 m input scale above the floor the rear toes stand on.
            self.assertGreater(float(front.mean() - rear.mean()), 0.4)
            self.assertGreater(float(front.max()), 0.2)
            self.assertLess(float(rear.min()), -0.05)
        finally:
            env.close()

    def test_evaluation_rebuilds_the_training_inputs(self):
        from locomotion.omniwalk.mdp.footfall import TerrainV1Tracker

        env, task, wrapped = build(Terrain(BOX))
        tracker = TerrainV1Tracker(1, "cpu", env=env, scaling="speed_v1")
        try:
            for control in range(25):
                observation = wrapped.get_observations()
                expected = tracker.inputs(
                    env.current["root"], env.commands.clone(), env.episode_steps
                )
                torch.testing.assert_close(
                    observation["policy"][:, 233:272],
                    expected,
                    atol=1e-4,
                    rtol=0,
                    msg=f"control {control}",
                )
                wrapped.step(torch.zeros(1, 18))
        finally:
            env.close()


if __name__ == "__main__":
    unittest.main()
