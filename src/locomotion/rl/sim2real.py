"""Train a bounded input/output compensator around a frozen walking policy (sim2real design).

The composite is an ordinary RSL-RL actor: ``ppo_config["actor"]["class_name"]`` selects
``CompensatedActor``, so the stock learner, checkpoint format and evaluator need no changes,
and PPO optimizes the compensator alone because every core parameter keeps
``requires_grad False``. The compensator encodes the actor observation into a small dynamics
latent and emits bounded per-group gains and biases on the proprioception and executed-target
columns plus a bounded pre-tanh delta on the core's action. It never touches the command,
clock, ghost or terrain columns, so the task the core is asked to do stays honest and the
same reward judges the composite. Its heads start at zero: at initialization the composite
computes bitwise what the frozen core computes, so the core's benchmarks are the starting
point rather than a hope. The latent exists to be read back out: a linear regression from it
to the drawn dynamics (fit offline by the readout suite) turns the compensator into a
measurement instrument, and a drawn world it compensates well is one it has identified.
"""

import copy
import hashlib
from pathlib import Path

import numpy as np
import torch
from rsl_rl.models import MLPModel
from rsl_rl.utils import resolve_callable
from torch import nn

from locomotion.rl.destination_wrapper import DestinationVecEnv

# The actor observation starts with five 42-value proprioception frames and, at 210:213 and
# 213:231, the command and the executed-target feature (rl/wrapper.py records the layout).
# The compensator modulates the frames and the executed target through 60 gain/bias slots:
# one slot per frame column, shared across the five frames, and one per executed-target
# column. Everything after column 231 (clock, ghost, terrain) passes through untouched.
FRAME_VALUES = 42
FRAME_COUNT = 5
COMMAND_SLICE = (210, 213)
EXECUTED_SLICE = (213, 231)
MODULATED_COLUMNS = tuple(range(FRAME_COUNT * FRAME_VALUES)) + tuple(range(*EXECUTED_SLICE))
MODULATION_SLOTS = tuple(
    column % FRAME_VALUES for column in range(FRAME_COUNT * FRAME_VALUES)
) + tuple(range(FRAME_VALUES, FRAME_VALUES + EXECUTED_SLICE[1] - EXECUTED_SLICE[0]))
SLOT_COUNT = FRAME_VALUES + EXECUTED_SLICE[1] - EXECUTED_SLICE[0]


def _verified_checkpoint(settings):
    """The torch checkpoint at ``settings['checkpoint']``, required to match its recorded sha256."""
    path = Path(settings["checkpoint"])
    if not path.is_file():
        raise ValueError("The frozen core checkpoint is missing: " + str(path))
    data = path.read_bytes()
    digest = hashlib.sha256(data).hexdigest()
    if digest != settings["checkpoint_sha256"]:
        raise ValueError("The frozen core checkpoint bytes differ from the recorded sha256")
    return torch.load(path, weights_only=False, map_location="cpu")


def _core_actor(core, obs, obs_groups, obs_set, output_dim):
    """The recorded core actor, rebuilt from its declaration's actor settings and frozen."""
    settings = copy.deepcopy(core)
    actor_settings = settings.pop("actor")
    checkpoint = _verified_checkpoint(settings)
    extra = set(settings) - {"checkpoint", "checkpoint_sha256"}
    if extra:
        raise ValueError("Unknown core settings: " + ", ".join(sorted(extra)))
    if actor_settings.get("obs_normalization", False):
        raise ValueError("The compensator modulates raw inputs; the core must not normalize them")
    model_class = resolve_callable(actor_settings.pop("class_name"))
    model = model_class(obs, obs_groups, obs_set, output_dim, **actor_settings)
    model.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    return model


class _CompensatedMean(nn.Module):
    """The composite pre-distribution output: the frozen core on modulated inputs plus a delta."""

    def __init__(self, core_mlp, encoder, limits, output_dim, latent_dim):
        super().__init__()
        self.core = core_mlp
        self.encoder = encoder
        self.gain_head = nn.Linear(latent_dim, SLOT_COUNT)
        self.bias_head = nn.Linear(latent_dim, SLOT_COUNT)
        self.delta_head = nn.Linear(latent_dim, output_dim)
        # Zero heads make the composite compute exactly what the core does at initialization.
        for head in (self.gain_head, self.bias_head, self.delta_head):
            nn.init.zeros_(head.weight)
            nn.init.zeros_(head.bias)
        self.gain_limit, self.bias_limit, self.delta_limit = limits
        self.register_buffer("columns", torch.tensor(MODULATED_COLUMNS), persistent=False)
        self.register_buffer("slots", torch.tensor(MODULATION_SLOTS), persistent=False)

    def latent(self, observation):
        return self.encoder(observation)

    def forward(self, observation):
        latent = self.encoder(observation)
        gains = 1.0 + self.gain_limit * torch.tanh(self.gain_head(latent))
        biases = self.bias_limit * torch.tanh(self.bias_head(latent))
        modulated = observation.clone()
        modulated[..., self.columns] = (
            observation[..., self.columns] * gains[..., self.slots] + biases[..., self.slots]
        )
        return self.core(modulated) + self.delta_limit * torch.tanh(self.delta_head(latent))


class CompensatedActor(MLPModel):
    """``MLPModel`` whose mean is a frozen core on compensator-modulated inputs plus a delta.

    ``core`` records the frozen checkpoint (path and sha256) and the actor settings of its own
    declaration, so evaluation rebuilds the composite from the recorded ``ppo_config`` alone and
    the checkpoint's ``actor_state_dict`` then restores both halves through the stock strict load.
    """

    def __init__(
        self,
        obs,
        obs_groups,
        obs_set,
        output_dim,
        core=None,
        encoder_hidden=(128, 64),
        latent_dim=8,
        observation_gain_limit=0.2,
        observation_bias_limit=0.25,
        action_delta_limit=0.25,
        **settings,
    ):
        if core is None:
            raise ValueError("CompensatedActor needs the frozen core settings")
        limits = (observation_gain_limit, observation_bias_limit, action_delta_limit)
        if not all(type(limit) is float and 0.0 < limit <= 1.0 for limit in limits):
            raise ValueError("The gain, bias and delta limits are floats in (0, 1]")
        if type(latent_dim) is not int or not 1 <= latent_dim <= 64:
            raise ValueError("The dynamics latent holds 1 to 64 values")
        super().__init__(obs, obs_groups, obs_set, output_dim, **settings)
        if self.obs_normalization:
            raise ValueError(
                "The compensator modulates raw inputs; use observation_normalization none"
            )
        if self.obs_dim < EXECUTED_SLICE[1]:
            raise ValueError("The actor observation ends before the executed-target feature")
        frozen = _core_actor(core, obs, obs_groups, obs_set, output_dim)
        if frozen.obs_dim != self.obs_dim:
            raise ValueError("The frozen core reads a different observation width")
        encoder, width = [], self.obs_dim
        for size in encoder_hidden:
            encoder.extend((nn.Linear(width, int(size)), nn.ELU()))
            width = int(size)
        encoder.append(nn.Linear(width, latent_dim))
        # The composite replaces the stock trunk; only the compensator's parameters require grad.
        self.mlp = _CompensatedMean(
            frozen.mlp, nn.Sequential(*encoder), limits, output_dim, latent_dim
        )

    def dynamics_latent(self, obs):
        """The compensator's latent for readout measurements, from the observation dict or tensor."""
        latent = obs if torch.is_tensor(obs) else self.get_latent(obs)
        return self.mlp.latent(latent)


class WarmStartedCritic(MLPModel):
    """``MLPModel`` critic that starts from the core checkpoint's critic and keeps training.

    The compensated actor barely moves until the value function is sane (its heads start at
    zero and the rate schedule is KL-adaptive), so starting the critic at the core's value
    estimate removes the wasted updates a fresh critic would spend rediscovering it.
    """

    def __init__(self, obs, obs_groups, obs_set, output_dim, core=None, **settings):
        if core is None:
            raise ValueError("WarmStartedCritic needs the frozen core settings")
        super().__init__(obs, obs_groups, obs_set, output_dim, **settings)
        extra = set(core) - {"checkpoint", "checkpoint_sha256"}
        if extra:
            raise ValueError("Unknown core settings: " + ", ".join(sorted(extra)))
        checkpoint = _verified_checkpoint(core)
        self.load_state_dict(checkpoint["critic_state_dict"], strict=True)


# The identification targets, in the order the wrapper stacks them and the readout reads them.
TARGET_FIELDS = (
    "friction_scale",
    "mass_scale",
    "motor_scale",
    "kd_scale",
    "action_delay_controls",
    "sag_per_minute",
)


class Sim2RealVecEnv(DestinationVecEnv):
    """``DestinationVecEnv`` that pairs each policy observation with the episode's dynamics draws.

    Under a randomized environment it keeps a bounded buffer of (observation, draw) rows for
    ``LatentSupervision``; without randomization the buffer stays empty and the wrapper behaves
    exactly like its parent, so evaluation rebuilds recorded runs unchanged.
    """

    def __init__(self, task, tensor_dict=None, *, supervision_rows=8192, **options):
        super().__init__(task, tensor_dict, **options)
        self.supervision_rows = int(supervision_rows)
        self._supervision_obs, self._supervision_draws = [], []

    def observations(self, output):
        observation = super().observations(output)
        draws = getattr(self.task.env, "randomization_draws", None)
        if draws is not None:
            target = torch.from_numpy(
                np.stack([draws[name] for name in TARGET_FIELDS], -1).astype(np.float32)
            )
            self._supervision_obs.append(observation["policy"].detach().clone())
            self._supervision_draws.append(target)
            kept = sum(len(rows) for rows in self._supervision_obs)
            while kept - len(self._supervision_obs[0]) >= self.supervision_rows:
                kept -= len(self._supervision_obs.pop(0))
                self._supervision_draws.pop(0)
        return observation

    def supervision_batch(self):
        """Every buffered (observation, draw) row, or ``None`` while the buffer is empty."""
        if not self._supervision_obs:
            return None
        return torch.cat(self._supervision_obs), torch.cat(self._supervision_draws)


class LatentSupervision:
    """Teach the compensator's latent to identify the drawn dynamics (RMA-style supervision).

    The task reward is untouched: after each PPO update this runs a few supervised steps on the
    compensator's encoder and a readout head alone, regressing the wrapper's buffered draws
    (scaled to [-1, 1] over their declared ranges) from the latent. The readout head lives here,
    not in the checkpoint; the recorded instrument stays the offline held-out regression of the
    readout suite — this loss only makes the information exist.
    """

    def __init__(self, algorithm, wrapper, randomization, rate, steps=4):
        if not hasattr(algorithm.actor, "dynamics_latent"):
            raise ValueError("Latent supervision needs a CompensatedActor")
        if not hasattr(wrapper, "supervision_batch"):
            raise ValueError("Latent supervision needs the Sim2RealVecEnv wrapper")
        if randomization is None:
            raise ValueError("Latent supervision needs a randomized environment")
        self.actor, self.wrapper, self.steps = algorithm.actor, wrapper, int(steps)
        centers, halves = [], []
        for name in TARGET_FIELDS:
            value = getattr(randomization, name)
            low, high = (value, value) if isinstance(value, int) else value
            centers.append((high + low) / 2.0)
            # A degenerate range contributes nothing: its rows scale to zero error weight.
            halves.append((high - low) / 2.0 or float("inf"))
        self.center = torch.tensor(centers)
        self.half = torch.tensor(halves)
        latent_dim = self.actor.mlp.encoder[-1].out_features
        self.readout = nn.Linear(latent_dim, len(TARGET_FIELDS))
        self.optimizer = torch.optim.Adam(
            [*self.actor.mlp.encoder.parameters(), *self.readout.parameters()], lr=float(rate)
        )
        self.inner_update = algorithm.update
        self.last_loss = None

    def update(self):
        losses = self.inner_update()
        batch = self.wrapper.supervision_batch()
        if batch is not None:
            observation, draws = batch
            target = (draws - self.center) / self.half
            for _ in range(self.steps):
                self.optimizer.zero_grad()
                loss = nn.functional.mse_loss(
                    self.readout(self.actor.dynamics_latent(observation)), target
                )
                loss.backward()
                self.optimizer.step()
            self.last_loss = float(loss.detach())
        return losses
