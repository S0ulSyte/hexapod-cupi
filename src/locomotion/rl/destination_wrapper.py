"""Give the actor and the critic the gap to each command's destination (experiment ``destination_v1``).

``DestinationVecEnv`` is ``OmniVecEnv`` with five inputs after its own: the planar gap to the pose the
command would have reached, in the body's navigation axes over 0.1 m, the sine and cosine of the heading
gap, and the controls since the command began over 250 (``omniwalk/mdp/destination.py``). A command that
has not run a control reads the destination at the body. It needs a ``DestinationTrainingTask``;
``ppo_config`` records ``destination_input="ghost"``, so evaluation rebuilds the same inputs.
``terrain_v1`` appends the 30 footfall-lookahead values of ``omniwalk/mdp/footfall.py`` after the
nine ``ghost_v11`` values (zeros without a terrain, so flat suites and the native robot stay valid).
"""

import torch

from locomotion.omniwalk.mdp.destination import (
    NEUTRAL_FEATURES,
    destination_features,
    ghost_v10_features,
    planar_pose,
)
from locomotion.rl.wrapper import OmniVecEnv


class DestinationVecEnv(OmniVecEnv):
    """``OmniVecEnv`` with the five destination inputs appended to the actor and critic inputs."""

    def __init__(
        self,
        task,
        tensor_dict=None,
        *,
        destination_input="ghost",
        gait_clock_scaling=None,
        **options,
    ):
        if destination_input not in ("ghost", "ghost_v10", "ghost_v11", "terrain_v1"):
            raise ValueError(
                "DestinationVecEnv applies the ghost, ghost_v10, ghost_v11 or terrain_v1 input"
            )
        if not hasattr(task, "destination"):
            raise ValueError("The destination inputs need a DestinationTrainingTask")
        # A task's speed-scaled clock must be recorded so that evaluation rebuilds the same phase.
        # Evaluation rebuilds that phase itself, so a recorded scaling may meet a task without one.
        kept = getattr(task.reward_config, "gait_clock_scaling", None)
        if (kept is not None and kept != gait_clock_scaling) or (
            gait_clock_scaling is not None and not options.get("gait_clock")
        ):
            raise ValueError(
                "The gait_clock_scaling option must record the task's and needs the gait clock"
            )
        self.destination_input = destination_input
        self.footfall = None
        if destination_input == "terrain_v1":
            from locomotion.omniwalk.mdp.footfall import FootfallInputs

            # The recorded option names the clock; a task may carry none (the flat suites
            # evaluate every checkpoint under the v9 task, whose reward predates the clock).
            self.footfall = FootfallInputs(task, gait_clock_scaling or kept)
        super().__init__(task, tensor_dict, **options)

    def destination_inputs(self, pending, command=None):
        """The task's inputs; rows in ``pending`` read a command that has not run a control yet: the neutral
        values for ``ghost``, and a destination at the body for ``ghost_v10`` and ``ghost_v11``. ``command`` defaults to the
        task's current commands."""
        task = self.task
        if self.destination_input in ("ghost_v10", "ghost_v11", "terrain_v1"):
            pending = pending[:, None].to(task.destination.device)
            here = planar_pose(task.current["root"]).to(task.destination)
            ghost = ghost_v10_features(
                task.current["root"],
                task.control_start_pose,
                torch.where(pending, here, task.destination),
                torch.where(
                    pending[:, 0], torch.zeros_like(task.command_elapsed), task.command_elapsed
                ),
                task.commands if command is None else command,
                along_track=self.destination_input in ("ghost_v11", "terrain_v1"),
            )
            if self.footfall is None:
                return ghost
            # terrain_v1: the nine ghost_v11 values, then the 30 footfall-lookahead values.
            return torch.cat((ghost, self.footfall.features(command).to(ghost)), -1)
        current = destination_features(task.current["root"], task.destination, task.command_elapsed)
        neutral = torch.tensor(NEUTRAL_FEATURES, device=current.device).expand_as(current)
        return torch.where(pending[:, None].to(current.device), neutral, current)

    def observations(self, output):
        base = super().observations(output)
        extra = self.destination_inputs(self.task.command_started)
        policy = torch.cat((base["policy"], extra.to(base["policy"])), -1)
        critic = torch.cat((base["critic"], extra.to(base["critic"])), -1)
        return self.tensor_dict({"policy": policy, "critic": critic}, batch_size=[self.num_envs])

    def step(self, action):
        held_command = self.task.commands.clone()
        observation, rewards, done, extras = super().step(action)
        bootstrap = extras.get("command_bootstrap_observation")
        if bootstrap is not None:
            # A switched row has not reset: its pose is post-action and the task moves its destination
            # and time to the new command only at the next control, so these are the held command's.
            held = self.destination_inputs(
                torch.zeros_like(self.task.command_started), held_command
            )
            critic = torch.cat((bootstrap["critic"], held.to(bootstrap["critic"])), -1)
            extras["command_bootstrap_observation"] = self.tensor_dict(
                {"critic": critic}, batch_size=[self.num_envs]
            )
        return observation, rewards, done, extras
