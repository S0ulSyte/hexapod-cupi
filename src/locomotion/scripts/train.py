"""Train PPO or AMP on the admitted canonical simulator."""

from __future__ import annotations

import argparse
import copy
import json
import os
import time
from functools import partial
from pathlib import Path

from locomotion.omniwalk.mdp.commands import GAIT_CLOCK_CONTROLS, TaskConfig
from locomotion.scripts.cli import (
    ConfigRecord,
    add_app_arguments,
    add_common_arguments,
    add_learner_arguments,
    learner_config,
    resolve_options,
    run,
    save,
    scalars,
    sha,
)
from locomotion.spark.validation import collected_transitions, decay_updates, finish_training_update


def parse_args(argv=None):
    """Parse the command line and resolve the recipe and the training gait-clock default."""
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    add_common_arguments(parser)
    add_learner_arguments(parser)
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Checkpoint whose actor, critic and optimizer state start this run; inputs that the "
        "wrapper adds start at zero weight. The update count and configuration are this run's own.",
    )
    parser.add_argument("--initialize-from-sha256")
    parser.add_argument("--initialize-from-declaration-sha256")
    parser.add_argument(
        "--initialize-rate",
        choices=["configured", "checkpoint"],
        default="configured",
        help="Restart the adaptive learning rate at its configured value or continue the checkpoint's.",
    )
    parser.add_argument(
        "--anchor-weight",
        type=float,
        help="Fraction in (0, 1) of each actor parameter's displacement from its loaded start "
        "removed after every update (rl/ppo.py ActorAnchor), as the surrogate trainer applies it.",
    )
    add_app_arguments(parser, argv)
    args = resolve_options(parser.parse_args(argv), "train")
    starting = (
        args.initialize_from,
        args.initialize_from_sha256,
        args.initialize_from_declaration_sha256,
    )
    if len({value is None for value in starting}) != 1:
        raise ValueError("A starting checkpoint needs its SHA-256 and its declaration's")
    return args


def require_gait_clock(args):
    """Reject an MLP policy without the gait clock that the active reward reads."""
    if args.networks == "mlp" and args.gait_clock != GAIT_CLOCK_CONTROLS:
        raise ValueError(f"The active reward requires the {GAIT_CLOCK_CONTROLS}-control gait clock")


def training_wrapper(env, args, task):
    """Wrap the task for RSL-RL; AMP adds collision capture and the discriminator configuration."""
    from locomotion.rl.wrapper import OmniVecEnv

    wrapper, options = OmniVecEnv, {}
    if args.experiment is not None:
        from locomotion.omniwalk.agents.rsl_rl_ppo_cfg import EXPERIMENT_CLOCKS, EXPERIMENT_INPUTS
        from locomotion.rl.destination_wrapper import DestinationVecEnv

        wrapper = DestinationVecEnv
        options = {"destination_input": EXPERIMENT_INPUTS[args.experiment]}
        if args.experiment in EXPERIMENT_CLOCKS:
            options["gait_clock_scaling"] = EXPERIMENT_CLOCKS[args.experiment]
    if args.learner == "amp":
        from locomotion.amp.amp import AMPConfig, AMPVecEnv, CollisionCapture

        wrapper = AMPVecEnv
        options = {"collision": CollisionCapture(env), "amp_config": AMPConfig().resolve(args.seed)}
    return wrapper(
        task,
        **options,
        observation_scaling=args.observation_scaling,
        command_segments=args.command_segments,
        gait_clock=args.gait_clock,
        action_smoothing=args.action_smoothing,
        velocity_noise=args.velocity_noise,
    )


def write_checkpoint(runner, args, config, identity, state, num_envs, update):
    """Save the runner with its identity, write the JSON declaration and name the checkpoint in state.json."""
    path = args.output / f"checkpoint_update{update:06d}.pt"
    runner.save(str(path), infos={"identity": identity, "updates": update})
    save(
        path.with_suffix(".json"),
        {
            "identity": identity,
            "updates": update,
            "checkpoint_sha256": sha(path),
            "transitions": collected_transitions(update, config, num_envs),
        },
    )
    state["checkpoint"] = str(path)
    state["checkpoint_sha256"] = sha(path)
    save(args.output / "state.json", state)


def metrics_row(values, runner, env, metrics, diagnostics, config):
    """Return the metrics.jsonl row for the values RSL-RL logs after one update."""
    from locomotion.evaluation.ppo_diagnostics import action_metrics

    update = values["it"] + 1
    return {
        "update": update,
        "transitions": collected_transitions(update, config, env.num_envs),
        "collection_seconds": values["collect_time"],
        "learning_seconds": values["learn_time"],
        "loss": values["loss_dict"],
        "learning_rate": values["learning_rate"],
        "mean_action_std": float(values["action_std"].mean()),
        "actions": action_metrics(runner.alg.storage, env.joint_names),
        "task": metrics.status(reset_interval=True),
        "policy_update": diagnostics.latest,
    }


def name_wandb_run(args, identity):
    """Name the W&B run after the seed and the frozen source; RSL-RL names it after its log directory."""
    import wandb

    wandb.run.name = f"seed{args.seed}-{args.source_freeze_sha256[:12]}"
    wandb.config.update({"identity": identity}, allow_val_change=True)


def start_from_checkpoint(runner, args, config, identity):
    """Load the starting checkpoint's networks and optimizer, widened for added inputs, and record them."""
    import torch

    from locomotion.evaluation.checkpoint import (
        checkpoint_reward_version,
        require_embedded_declaration,
        require_starting_identity,
    )
    from locomotion.rl.ppo import initialize_learner

    declaration = args.initialize_from.with_suffix(".json")
    if sha(args.initialize_from) != args.initialize_from_sha256:
        raise ValueError("Starting checkpoint bytes differ")
    if sha(declaration) != args.initialize_from_declaration_sha256:
        raise ValueError("Starting checkpoint declaration bytes differ")
    record = json.loads(declaration.read_text())
    if record["checkpoint_sha256"] != args.initialize_from_sha256:
        raise ValueError("Starting checkpoint declaration names other bytes")
    require_starting_identity(record, identity)
    loaded = torch.load(str(args.initialize_from), weights_only=False, map_location=args.device)
    require_embedded_declaration(loaded.get("infos"), record)
    widened, rate = initialize_learner(
        runner.alg, loaded, config["algorithm"]["learning_rate"], args.initialize_rate
    )
    identity["initialized_from"] = {
        "checkpoint_sha256": args.initialize_from_sha256,
        "declaration_sha256": args.initialize_from_declaration_sha256,
        "experiment": record["identity"].get("experiment"),
        "reward_version": checkpoint_reward_version(record),
        "reward_task": record["identity"].get("reward_task"),
        "updates": record["updates"],
        "physics_config_equal": record["identity"].get("physics_config")
        == identity["physics_config"],
        "loaded": ["actor", "critic", "optimizer"],
        "widened_inputs": widened,
        "learning_rate": {"value": rate, "source": args.initialize_rate},
    }


def training_runner(env, args, config, identity, state, started):
    """Build the RSL-RL runner with update diagnostics, load capture and a record after each update.

    Return the runner and the load accumulator.
    """
    from rsl_rl.runners import OnPolicyRunner

    from locomotion.evaluation.loads import TrainingLoads
    from locomotion.evaluation.ppo_diagnostics import UpdateDiagnostics
    from locomotion.evaluation.task_metrics import TaskMetrics
    from locomotion.omniwalk.task import TrainingTask
    from locomotion.rl.ppo import DeviationSchedule

    if args.experiment is None:
        task = TrainingTask(env, TaskConfig(seed=args.seed), args.output / "task")
    else:
        from locomotion.omniwalk.destination_task import EXPERIMENTS, DestinationTrainingTask

        settings = EXPERIMENTS[args.experiment]
        if args.episode_seconds != settings["episode_seconds"]:
            raise ValueError(f"{args.experiment} trains {settings['episode_seconds']} s episodes")
        task = DestinationTrainingTask(
            env, TaskConfig(seed=args.seed), args.output / "task", settings["reward"]
        )
    metrics = TaskMetrics(task)
    wrapped = training_wrapper(env, args, task)
    if args.networks == "mlp" and args.gait_clock != task.reward_config.schedule_period_controls:
        raise ValueError("The contact-schedule reward needs a gait clock of the same period")
    runner_config = copy.deepcopy(config)
    # The wrapper consumes its own options; the runner receives the stock keys.
    runner_config.pop("environment_wrapper", None)
    runner_config.pop("exploration", None)
    if args.logger == "wandb":
        # The logger choice stays out of ppo_config, which checkpoint loading compares.
        # Copy saved checkpoints: container paths do not exist where offline runs sync.
        os.environ.update(
            WANDB_DIR=str(args.output), WANDB_MODE=args.wandb_mode, WANDB_SYMLINK="false"
        )
        runner_config.update(logger="wandb", wandb_project=args.wandb_project)
        wrapped.cfg = ConfigRecord(wrapped.cfg)
    runner = OnPolicyRunner(
        wrapped, runner_config, str(args.output / "learner"), device=args.device
    )
    if args.initialize_from is not None:
        start_from_checkpoint(runner, args, config, identity)
    diagnostics = UpdateDiagnostics(runner.alg)
    runner.alg.update = diagnostics.update
    if args.action_std_final is not None:
        # The schedule wraps the diagnostic update, so each record reads the deviation it trained with.
        runner.alg.update = DeviationSchedule(
            runner.alg,
            args.action_std,
            args.action_std_final,
            decay_updates(args.updates, args.action_std_decay_updates),
        ).update
    if args.anchor_weight:
        from locomotion.rl.ppo import ActorAnchor

        # Installed after --initialize-from, so the anchor holds the widened, loaded start.
        runner.alg.update = ActorAnchor(runner.alg, args.anchor_weight).update
    identity["anchor_weight"] = args.anchor_weight or None
    loads = TrainingLoads(env)
    if args.learner == "amp":
        wrapped.collision.inner = loads
        env.capture = wrapped.collision
    else:
        env.capture = loads
    checkpoint = partial(write_checkpoint, runner, args, config, identity, state, env.num_envs)
    log_update = runner.logger.log

    def log(**values):
        log_update(**values)
        row = metrics_row(values, runner, env, metrics, diagnostics, config)
        with (args.output / "metrics.jsonl").open("a") as stream:
            stream.write(json.dumps(row, allow_nan=False) + "\n")
        if runner.logger.writer is not None:
            for tag, value in scalars(row["task"], "Task").items():
                runner.logger.writer.add_scalar(tag, value, values["it"])
        if args.logger == "wandb" and row["update"] == 1:
            name_wandb_run(args, identity)
        state.update(
            updates=row["update"],
            transitions=row["transitions"],
            wall_seconds=time.monotonic() - started,
        )
        save(args.output / "state.json", state)
        finish_training_update(
            row["update"],
            args.updates,
            time.monotonic() - started,
            args.max_wall_seconds,
            checkpoint,
            lambda: save(args.output / "force_metrics.json", loads.report()),
        )

    runner.logger.log = log
    return runner, loads


def learn(env, session):
    """Run the requested updates and save the loads; run() saves them again after any outcome."""
    args = session.args
    config = learner_config(args, session.identity, session.state)
    runner, session.loads = training_runner(
        env, args, config, session.identity, session.state, session.started
    )
    runner.learn(args.updates, init_at_random_ep_len=False)
    save(args.output / "force_metrics.json", session.loads.report())


def main(argv=None):
    args = parse_args(argv)
    require_gait_clock(args)
    return run(args, "train", learn)


if __name__ == "__main__":
    raise SystemExit(main())
