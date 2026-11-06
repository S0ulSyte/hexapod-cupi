"""Wrap the walking task as an RSL-RL vector environment with input scales and a gait clock."""

import math

import torch
from tensordict import TensorDict

# Fixed input scales, selected with observation_scaling='fixed'. Each proprioception frame holds
# body rates (rad/s), projected gravity, joint offsets (rad) and joint rates (rad/s); the command
# follows at columns 210:213 and the executed-target feature at 213:231. The scales bring each
# group to order one at this robot's commands (0.05 m/s, 0.2 rad/s) and action range (0.35 rad).
FRAME_SCALES = (2.0,) * 3 + (1.0,) * 3 + (1 / 0.35,) * 18 + (0.5,) * 18
COMMAND_SCALES = (20.0, 20.0, 5.0)
ACTOR_SCALES = FRAME_SCALES * 5 + COMMAND_SCALES + (1.0,) * 18
CRITIC_SCALES = ACTOR_SCALES + (20.0,) * 3


def scale_observation(value, scaling, *, critic=False):
    """Apply the declared fixed scales to a 231-value actor or 234-value critic observation."""
    if scaling == "none":
        return value
    if scaling != "fixed":
        raise ValueError("Observation scaling must be none or fixed")
    scales = CRITIC_SCALES if critic else ACTOR_SCALES
    if value.shape[-1] != len(scales):
        raise ValueError("Observation width differs from the declared scales")
    return value * torch.as_tensor(scales, dtype=value.dtype, device=value.device)


def clock_features(episode_steps, period, command, cycles=None):
    """Sine and cosine of the gait phase while a motion command is held; zeros under a zero command.
    ``cycles``, a speed-scaled phase in cycles (``omniwalk/mdp/gait.py``), replaces the control count."""
    if cycles is None:
        phase = 2 * math.pi * (episode_steps.to(torch.float32) % period) / period
    else:
        phase = (2 * math.pi * torch.remainder(cycles, 1.0)).to(torch.float32)
    moving = (command != 0).any(-1, keepdim=True).to(torch.float32)
    return torch.stack((torch.sin(phase), torch.cos(phase)), -1) * moving.to(phase.device)


def smoothed_action(action, previous, episode_steps):
    """Mean of this action and the previous one; an episode's first control pairs with the neutral action."""
    fresh = (episode_steps == 0)[:, None].to(action.device)
    return 0.5 * (action + torch.where(fresh, torch.zeros_like(action), previous))


class OmniVecEnv:
    """Preserve terminal rewards and reset selected replicas before the next action."""

    def __init__(
        self,
        task,
        tensor_dict=None,
        *,
        observation_scaling="none",
        command_segments="continuous",
        gait_clock=0,
        action_smoothing="none",
        velocity_noise=0.0,
    ):
        if tensor_dict is None:
            tensor_dict = TensorDict
        if observation_scaling not in ("none", "fixed") or command_segments not in (
            "continuous",
            "bootstrap",
        ):
            raise ValueError("Unknown observation scaling or command segment option")
        if action_smoothing not in ("none", "mean2"):
            raise ValueError("Action smoothing must be none or mean2")
        self.action_smoothing, self.velocity_noise = action_smoothing, velocity_noise
        self.previous_action = torch.zeros(task.num_envs, 18, device=task.device)
        self.task, self.tensor_dict = task, tensor_dict
        self.observation_scaling, self.command_segments = observation_scaling, command_segments
        self.gait_clock = gait_clock
        self.num_envs, self.num_actions, self.device = task.num_envs, 18, task.device
        self.max_episode_length = round(task.cfg.episode_seconds / task.cfg.control_dt)
        self.cfg = {
            "physics": task.cfg.declaration(),
            "task": task.declaration(),
            "algorithm": "RSL-RL PPO",
            "motion_prior": False,
        }
        self.current = task.reset()

    @property
    def episode_length_buf(self):
        return self.task.episode_steps

    def observations(self, output):
        policy = scale_observation(output["obs"], self.observation_scaling)
        critic = scale_observation(output["critic"], self.observation_scaling, critic=True)
        if self.velocity_noise:
            # The actor alone reads noisy joint velocities; evaluation and the critic read the measured values.
            scale = FRAME_SCALES[-1] if self.observation_scaling == "fixed" else 1.0
            policy = policy.clone()
            for frame in range(5):
                block = policy[:, 42 * frame + 24 : 42 * frame + 42]
                block += torch.randn_like(block) * self.velocity_noise * scale
        if self.gait_clock:
            # The phase restarts with each episode, so evaluation can rebuild it from the commands seen.
            clock = self.clock(output["obs"][:, 210:213]).to(policy)
            policy, critic = torch.cat((policy, clock), -1), torch.cat((critic, clock), -1)
        return self.tensor_dict({"policy": policy, "critic": critic}, batch_size=[self.num_envs])

    def clock(self, command):
        """The clock inputs: the task's speed-scaled phase when it keeps one, else the control count."""
        cycles = getattr(self.task, "gait_clock_phase", None)
        return clock_features(self.task.episode_steps, self.gait_clock, command, cycles)

    def get_observations(self):
        return self.observations(self.current)

    def step(self, action):
        held = self.task.commands.clone() if self.command_segments == "bootstrap" else None
        if self.action_smoothing == "mean2":
            # PPO keeps the raw sample; the environment receives the two-control mean.
            raw = action.detach().clone()
            action = smoothed_action(
                action, self.previous_action.to(action), self.task.episode_steps
            )
            self.previous_action = raw
        output = self.task.step(action)
        rewards = output["reward"].clone()
        done = output["terminated"] | output["truncated"]
        # A physical termination overrides a coincident episode timeout.
        extras = {"time_outs": output["truncated"] & ~output["terminated"]}
        if held is not None:
            # A task whose commands evolve within a run publishes the true resample flag;
            # comparing values cannot tell that evolution from a drawn switch.
            switched = output.get("command_switched")
            switched = (
                (output["obs"][:, 210:213] != held).any(-1) if switched is None else switched
            ) & ~done
            if bool(switched.any()):
                # Capture the post-action state under the held command before any selected reset.
                critic = output["critic"].clone()
                critic[:, 210:213] = held
                critic = scale_observation(critic, self.observation_scaling, critic=True)
                if self.gait_clock:
                    critic = torch.cat((critic, self.clock(held).to(critic)), -1)
                extras["command_time_outs"] = switched
                extras["command_bootstrap_observation"] = self.tensor_dict(
                    {"critic": critic}, batch_size=[self.num_envs]
                )
            extras["time_outs"] = extras["time_outs"] | switched
        selected = done.nonzero(as_tuple=False).flatten()
        self.current = self.task.reset(selected) if len(selected) else output
        if held is not None:
            done = done | switched
        return self.get_observations(), rewards, done.long(), extras
