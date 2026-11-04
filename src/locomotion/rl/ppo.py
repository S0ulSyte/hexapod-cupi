"""Extend stock RSL-RL PPO with command bootstraps, a rate cap, bounded means and a deviation schedule."""

import copy
import math

import torch
from rsl_rl.algorithms import PPO
from rsl_rl.modules.distribution import GaussianDistribution
from rsl_rl.utils import resolve_callable
from torch import nn


class CommandBootstrapPPO(PPO):
    """Value a completed command hold at its post-action state with the held command."""

    def process_env_step(self, obs, rewards, dones, extras):
        switched = extras.get("command_time_outs")
        if switched is not None:
            with torch.no_grad():
                values = self.critic(extras["command_bootstrap_observation"]).squeeze(-1)
            rewards = rewards + self.gamma * values * switched.to(values)
            # Stock PPO uses pre-action values for timeouts. Exclude the holds handled above.
            extras = {**extras, "time_outs": extras["time_outs"] & ~switched}
        super().process_env_step(obs, rewards, dones, extras)


class CappedRatePPO(CommandBootstrapPPO):
    """Stock PPO whose learning rate never exceeds ``learning_rate_max`` and, when ``learning_rate_min``
    is given, never falls below it (the stock adaptive schedule's own floor is 1e-5)."""

    def __init__(self, *args, learning_rate_max, learning_rate_min=None, **kwargs):
        self.learning_rate_max = float(learning_rate_max)
        self.learning_rate_min = None if learning_rate_min is None else float(learning_rate_min)
        if self.learning_rate_min is not None and not (
            0 < self.learning_rate_min <= self.learning_rate_max
        ):
            raise ValueError("The learning-rate floor lies above zero and at most the ceiling")
        super().__init__(*args, **kwargs)
        for group in self.optimizer.param_groups:
            group["lr"] = self.learning_rate

    @property
    def learning_rate(self):
        return self._learning_rate

    @learning_rate.setter
    def learning_rate(self, value):
        value = min(float(value), self.learning_rate_max)
        if self.learning_rate_min is not None:
            value = max(value, self.learning_rate_min)
        self._learning_rate = value


# The environment clips sampled actions before its physical target limits. PPO stores those raw
# samples and evaluates their Gaussian likelihoods. Only the mean uses tanh; this distribution
# does not squash sampled actions.
class BoundedMeanGaussian(GaussianDistribution):
    """Keep the policy mean in [-1, 1] for training and inference."""

    def update(self, mlp_output):
        super().update(torch.tanh(mlp_output))

    def deterministic_output(self, mlp_output):
        return torch.tanh(mlp_output)

    def as_deterministic_output_module(self):
        return nn.Tanh()


class CorrelatedBoundedMeanGaussian(BoundedMeanGaussian):
    """Bounded mean whose exploration noise persists across controls."""

    def __init__(self, output_dim, init_std=1.0, std_type="scalar", noise_correlation=0.0):
        super().__init__(output_dim, init_std=init_std, std_type=std_type)
        if not 0 <= noise_correlation < 1:
            raise ValueError("Noise correlation must lie in [0, 1)")
        self.noise_correlation = float(noise_correlation)
        self._noise = None

    def sample(self):
        mean, std = self.mean, self.std
        white = torch.randn_like(mean)
        if self._noise is None or self._noise.shape != white.shape:
            self._noise = white
        else:
            keep = self.noise_correlation
            self._noise = keep * self._noise + (1 - keep * keep) ** 0.5 * white
        return mean + std * self._noise


def load_actor(checkpoint, config, observations, device):
    """Load the recorded stock actor without constructing a learner."""
    settings = copy.deepcopy(config["actor"])
    model = resolve_callable(settings.pop("class_name"))
    target = torch.device(device)
    devices = (
        [target.index if target.index is not None else torch.cuda.current_device()]
        if target.type == "cuda"
        else []
    )
    with torch.random.fork_rng(devices=devices):
        actor = model(observations, config["obs_groups"], "actor", 18, **settings).to(device)
    actor.load_state_dict(checkpoint["actor_state_dict"], strict=True)
    actor.eval()
    return actor, checkpoint.get("infos")


class DeviationSchedule:
    """Hold the action deviation on a declared linear schedule; PPO no longer learns it."""

    def __init__(self, algorithm, start, final, updates):
        self.algorithm, self.start, self.final, self.updates = (
            algorithm,
            float(start),
            float(final),
            int(updates),
        )
        self.parameter = algorithm.actor.distribution.log_std_param
        self.parameter.requires_grad_(False)
        self.inner_update = algorithm.update
        self.completed = 0
        self.apply()

    def value(self):
        return self.start + (self.final - self.start) * min(1.0, self.completed / self.updates)

    def apply(self):
        with torch.no_grad():
            self.parameter.fill_(math.log(self.value()))

    def update(self):
        losses = self.inner_update()
        self.completed += 1
        self.apply()
        return losses


class ActorAnchor:
    """Pull the actor toward its initialization after each update (decoupled from the optimizer).

    Long surrogate continuations drift away from a native init's precision — the recorded
    quiet/formal/omni tax — while the levers that matter ride strong reward gradients. The
    anchor removes a fraction of each parameter's displacement from its post-initialization
    value after every update, so weak-gradient directions decay back to the init while strong
    gradients hold their ground at a displacement of roughly step/weight. Distribution
    parameters are excluded: the deviation schedule owns them.
    """

    def __init__(self, algorithm, weight):
        weight = float(weight)
        if not 0 < weight < 1:
            raise ValueError("The anchor weight is a fraction in (0, 1)")
        self.weight = weight
        self.anchored = [
            (parameter, parameter.detach().clone())
            for name, parameter in algorithm.actor.named_parameters()
            if "distribution" not in name and parameter.requires_grad
        ]
        if not self.anchored:
            raise ValueError("The actor anchor found no trainable non-distribution parameters")
        self.inner_update = algorithm.update

    def update(self):
        losses = self.inner_update()
        with torch.no_grad():
            for parameter, init in self.anchored:
                parameter.add_(init - parameter, alpha=self.weight)
        return losses


# An empirical normalizer's statistics for an added input start as RSL-RL starts them: mean 0, variance and
# deviation 1, so the input passes unscaled until the normalizer updates; a zero deviation would divide by eps.
NORMALIZER_FILL = {"_mean": 0.0, "_var": 1.0, "_std": 1.0}


def widen_inputs(loaded, algorithm):
    """Give a checkpoint's first layers zero weights and zero optimizer moments for inputs a wrapper added.

    A weight whose input width grew keeps its columns and gains zero columns at the end, so the network
    computes what it computed before the new inputs existed. An observation normalizer's mean gains zeros
    and its variance and deviation gain ones. Every other shape must match. The new columns' Adam moments
    start at zero but share the weight's step count, so Adam's bias correction no longer offsets them:
    their first optimizer step is about 3.2 times the learning rate and, under a steady gradient, their
    step peaks near 6.5 times by the tenth optimizer step and is still about twice it after 300. Returns
    ``[name, old inputs, new inputs]`` for each widened weight.
    """
    widened, grown = [], {}
    for key, model in (
        ("actor_state_dict", algorithm.actor),
        ("critic_state_dict", algorithm.critic),
    ):
        target = model.state_dict()
        for name, value in list(loaded[key].items()):
            if name not in target:
                raise ValueError(
                    f"Checkpoint {key}.{name} has no counterpart in the new network; "
                    "the observation normalization may differ"
                )
            want = target[name].shape
            if value.shape == want:
                continue
            if value.ndim != 2 or value.shape[0] != want[0] or value.shape[1] > want[1]:
                raise ValueError(
                    f"Checkpoint {key}.{name} {tuple(value.shape)} cannot start a {tuple(want)} weight"
                )
            fill = (
                NORMALIZER_FILL.get(name.rsplit(".", 1)[-1], 0.0) if "normalizer" in name else 0.0
            )
            extra = value.new_full((want[0], want[1] - value.shape[1]), fill)
            loaded[key][name] = torch.cat((value, extra), 1)
            if "normalizer" in name:
                continue
            widened.append([f"{key[: -len('_state_dict')]}.{name}", value.shape[1], want[1]])
            grown[tuple(value.shape)] = want
    params = [p for group in algorithm.optimizer.param_groups for p in group["params"]]
    for index, state in loaded["optimizer_state_dict"]["state"].items():
        for moment in ("exp_avg", "exp_avg_sq"):
            value = state.get(moment)
            if value is None or value.shape == params[index].shape:
                continue
            if grown.get(tuple(value.shape)) != params[index].shape:
                raise ValueError(
                    f"Optimizer state {index} {tuple(value.shape)} differs from its parameter"
                )
            extra = value.new_zeros(value.shape[0], params[index].shape[1] - value.shape[1])
            state[moment] = torch.cat((value, extra), 1)
    return widened


def initialize_learner(algorithm, loaded, configured_rate, rate="configured"):
    """Start a new run from a checkpoint's actor, critic and optimizer state, widened for added inputs.

    ``rate="configured"`` restarts an adaptive schedule at the configured rate, as a new run does;
    ``"checkpoint"`` continues from the rate the checkpoint's optimizer held. Returns the widened
    weights and the starting rate.
    """
    if rate not in ("configured", "checkpoint"):
        raise ValueError("The starting rate is configured or checkpoint")
    widened = widen_inputs(loaded, algorithm)
    algorithm.load(
        loaded, {"actor": True, "critic": True, "optimizer": True, "iteration": False}, True
    )
    start = configured_rate if rate == "configured" else algorithm.optimizer.param_groups[0]["lr"]
    algorithm.learning_rate = start
    for group in algorithm.optimizer.param_groups:
        group["lr"] = algorithm.learning_rate
    return widened, algorithm.learning_rate
