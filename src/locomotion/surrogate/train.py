"""Train PPO on the MuJoCo surrogate through the training path of ``locomotion/scripts/train.py``."""

from __future__ import annotations

import argparse
import contextlib
import copy
import importlib
import importlib.metadata
import importlib.util
import inspect
import io
import json
import os
import sys
import time
import traceback
from dataclasses import asdict, replace
from pathlib import Path

from locomotion.omniwalk.mdp.commands import GAIT_CLOCK_CONTROLS
from locomotion.spark.sources import runtime_paths, source_files

PACKAGE = Path(__file__).resolve().parent


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def resolve(spec):
    """``module:Attribute`` or ``module:factory(text)``; the result must be a class."""
    module, separator, name = spec.partition(":")
    if not separator:
        raise ValueError("Expected module:Class, got " + spec)
    name, call, argument = name.partition("(")
    if call and not argument.endswith(")"):
        raise ValueError("Expected module:factory(text), got " + spec)
    value = importlib.import_module(module)
    for part in name.split("."):
        value = getattr(value, part)
    if call:
        value = value(argument[:-1].strip().strip("'\""))
    if not inspect.isclass(value):
        raise ValueError("The specification does not resolve to a class: " + spec)
    return value


MAX_THREADS = 6


def thread_count(value):
    """The thread count of one process. Each tool accepts 1 to ``MAX_THREADS``."""
    if not 1 <= value <= MAX_THREADS:
        raise ValueError(f"Use 1 to {MAX_THREADS} threads per process")
    return value


@contextlib.contextmanager
def process_threads(count):
    """Run a block with ``count`` torch threads, then restore the caller's thread settings."""
    thread_count(count)
    previous_variable = os.environ.get("OMP_NUM_THREADS")
    os.environ.setdefault("OMP_NUM_THREADS", str(count))
    import torch

    previous_count = torch.get_num_threads()
    torch.set_num_threads(count)
    try:
        yield
    finally:
        torch.set_num_threads(previous_count)
        if previous_variable is None:
            os.environ.pop("OMP_NUM_THREADS", None)


WRAPPER_DEFAULTS = {
    "observation_scaling": "none",
    "command_segments": "continuous",
    "gait_clock": 0,
    "action_smoothing": "none",
    "velocity_noise": 0.0,
}


def wrapper_options(config):
    """The wrapper keywords of a PPO configuration: the two original options at their recorded or
    default values, and every other key the configuration records (``rsl_rl_ppo_cfg.ppo_config`` omits defaults)."""
    return {
        "observation_scaling": "none",
        "command_segments": "continuous",
        **config.get("environment_wrapper", {}),
    }


def build_wrapper(wrapper_class, task, options):
    """Construct the RSL-RL wrapper with its recorded keywords, as ``locomotion/scripts/train.py`` does.

    A constructor that lacks a recorded keyword is an error unless the keyword holds its default,
    so a substitute wrapper with the plain ``(task)`` constructor works at the default options.
    """
    parameters = inspect.signature(wrapper_class.__init__).parameters
    open_keywords = any(p.kind is inspect.Parameter.VAR_KEYWORD for p in parameters.values())
    accepted = {key: value for key, value in options.items() if open_keywords or key in parameters}
    refused = {
        key: value
        for key, value in options.items()
        if key not in accepted and value != WRAPPER_DEFAULTS.get(key)
    }
    if refused:
        raise ValueError(
            f"{wrapper_class.__name__} does not accept the wrapper options {sorted(refused)}"
        )
    return wrapper_class(task, **accepted)


def learner_configuration(config_module, seed, options):
    """``rsl_rl_ppo_cfg.ppo_config`` with the learner options; an option the checked-out revision lacks must hold its default."""
    accepted = inspect.signature(config_module.ppo_config).parameters
    defaults = {
        "observation_scaling": "none",
        "command_segments": "continuous",
        "learning_rate_max": None,
        "action_std": 0.15,
        "action_noise_correlation": 0.0,
        "action_std_final": None,
        "gait_clock": 0,
        "action_smoothing": "none",
        "velocity_noise": 0.0,
    }
    missing = [
        key for key, value in options.items() if key not in accepted and value != defaults.get(key)
    ]
    if missing:
        raise ValueError(
            f"locomotion.omniwalk.agents.rsl_rl_ppo_cfg.ppo_config at this revision has no option {missing}"
        )
    return config_module.ppo_config(
        seed, **{key: value for key, value in options.items() if key in accepted}
    )


def check_gait_schedule(task, config):
    """The contact-schedule reward reads the phase that the gait clock shows the policy (``locomotion/scripts/train.py``)."""
    reward = getattr(task, "reward_config", None)
    if getattr(reward, "schedule_weight", 0) and (
        config.get("environment_wrapper", {}).get("gait_clock", 0)
        != getattr(reward, "schedule_period_controls", None)
    ):
        raise ValueError(
            "The contact-schedule reward needs a gait clock of the same period: pass --gait-clock "
            f"{getattr(reward, 'schedule_period_controls', None)} or set schedule_weight=0 in the task"
        )


def check_command_bootstrap(options, config):
    """The bootstrap wrapper needs the algorithm that values a finished hold at its post-action state.

    Stock PPO reads ``time_outs`` alone and adds the value saved before the action, so that pair
    trains the earlier target under a ``command_segments: bootstrap`` record.
    """
    if options.get("command_segments") != "bootstrap":
        return
    from rsl_rl.utils import resolve_callable

    from locomotion.rl.ppo import CommandBootstrapPPO

    algorithm = resolve_callable(config["algorithm"]["class_name"])
    if not (inspect.isclass(algorithm) and issubclass(algorithm, CommandBootstrapPPO)):
        raise ValueError(
            "The bootstrap wrapper option needs an algorithm class derived from "
            "locomotion.rl.ppo:CommandBootstrapPPO"
        )


def external_class_files(task_class, wrapper_class, config):
    """Source hashes of the selected classes that ``source_files`` and ``surrogate_files`` omit."""
    import sysconfig

    from rsl_rl.utils import resolve_callable

    from locomotion.surrogate.env import ROOT, sha

    def named(value):
        if isinstance(value, dict):
            for key, item in value.items():
                if key == "class_name" and isinstance(item, str):
                    yield resolve_callable(item)
                else:
                    yield from named(item)

    hashed = {
        path.resolve() for path in (*runtime_paths(ROOT / "src/locomotion"), *PACKAGE.glob("*.py"))
    }
    installed = [
        Path(sysconfig.get_path(name)).resolve()
        for name in ("stdlib", "platstdlib", "purelib", "platlib")
    ]
    files = {}
    for selected in (task_class, wrapper_class, *named(config)):
        for part in inspect.getmro(selected) if inspect.isclass(selected) else (selected,):
            try:
                source = inspect.getsourcefile(part)
            except TypeError:  # A built-in class has no source file.
                continue
            path = Path(source).resolve() if source else None
            if (
                path is not None
                and path.is_file()
                and path not in hashed
                and not any(root in path.parents for root in installed)
            ):
                files[part.__module__] = sha(path)
    return dict(sorted(files.items()))


def merge(base, override):
    """Recursive dictionary update for --ppo-config-overrides."""
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(base.get(key), dict):
            merge(base[key], value)
        else:
            base[key] = value
    return base


def resume_record(path, identity):
    """Check the checkpoint bytes and training contract before loading learner state."""
    import torch

    from locomotion.surrogate.env import sha

    path = Path(path)
    identity = json.loads(json.dumps(identity, allow_nan=False))
    record = json.loads(path.with_suffix(".json").read_text())
    if record.get("checkpoint_sha256") != sha(path):
        raise ValueError("Resume checkpoint hash differs from its record")
    parent = record["identity"]
    # Run location and execution threads do not change the training contract; a run's starting checkpoint
    # is its own history, which the resumed run carries forward from the record.
    excluded = {
        "repository",
        "threads",
        "scope",
        "resume",
        "exploration_schedule_updates",
        "initialized_from",
    }
    different = sorted(
        key
        for key in (parent.keys() | identity.keys()) - excluded
        if parent.get(key) != identity.get(key)
    )
    if different:
        raise ValueError("Resume configuration differs: " + ", ".join(different))
    updates = record.get("updates")
    if type(updates) is not int or updates < 1:
        raise ValueError("Resume checkpoint needs a positive completed update count")
    transitions = updates * parent["ppo_config"]["num_steps_per_env"] * parent["config"]["num_envs"]
    if record.get("transitions") != transitions:
        raise ValueError("Resume transition count differs from its training contract")
    if identity["action_std_final"] is not None:
        horizon = parent.get("exploration_schedule_updates")
        if type(horizon) is not int or horizon < 1:
            raise ValueError("Resume checkpoint needs the original exploration schedule horizon")
    stored = torch.load(path, map_location="cpu", weights_only=False)
    if json.loads(json.dumps(stored.get("infos"), allow_nan=False)) != {
        "identity": parent,
        "updates": updates,
    }:
        raise ValueError("Resume checkpoint metadata differs from its record")
    return record


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--output", type=Path, required=True, help="New run directory.")
    parser.add_argument(
        "--task",
        default="locomotion.omniwalk.task:TrainingTask",
        help="module:Class or module:factory(text) that gives a class with the TrainingTask "
        "interface: Class(env, TaskConfig, output_dir).",
    )
    parser.add_argument(
        "--wrapper",
        default="locomotion.rl.wrapper:OmniVecEnv",
        help="module:Class that wraps the task for RSL-RL.",
    )
    parser.add_argument(
        "--ppo-config-overrides",
        default="{}",
        help="JSON merged into rsl_rl_ppo_cfg.ppo_config(...), or @file.json.",
    )
    parser.add_argument(
        "--distribution",
        help="Actor distribution class_name (RSL-RL name or module:Class); overrides --action-mean.",
    )
    parser.add_argument("--action-mean", choices=["unbounded", "tanh"], default="unbounded")
    parser.add_argument(
        "--observation-normalization", choices=["empirical", "none"], default="empirical"
    )
    parser.add_argument(
        "--observation-scaling",
        choices=["none", "fixed"],
        default="none",
        help="Pass raw observations or apply the fixed input scales declared in rl/wrapper.py.",
    )
    parser.add_argument(
        "--command-segments",
        choices=["continuous", "bootstrap"],
        default="continuous",
        help="Carry returns across command changes or bootstrap the return at each change.",
    )
    parser.add_argument(
        "--learning-rate-max",
        type=float,
        help="Ceiling for the adaptive learning rate; the stock schedule allows 1e-2.",
    )
    parser.add_argument(
        "--action-std",
        type=float,
        default=0.15,
        help="Initial standard deviation of the Gaussian action distribution.",
    )
    parser.add_argument(
        "--gait-clock",
        type=int,
        default=GAIT_CLOCK_CONTROLS,
        help="Append the sine and cosine of a gait phase with this period in controls; 0 adds none.",
    )
    parser.add_argument(
        "--action-std-final",
        type=float,
        help="Lower the action deviation on a linear schedule to this value; PPO then does not learn it.",
    )
    parser.add_argument(
        "--action-noise-correlation",
        type=float,
        default=0.0,
        help="Share of each exploration noise sample carried to the next control (needs the tanh mean).",
    )
    parser.add_argument(
        "--action-smoothing",
        choices=["none", "mean2"],
        default="none",
        help="mean2 sends the environment the mean of each action and the previous one.",
    )
    parser.add_argument(
        "--velocity-noise",
        type=float,
        default=0.0,
        help="Standard deviation in rad/s of the noise on the actor's joint-velocity inputs.",
    )
    parser.add_argument(
        "--updates",
        type=int,
        default=2000,
        help="PPO updates to run. With --resume this counts updates after the checkpoint.",
    )
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument(
        "--threads",
        type=int,
        default=2,
        help="Physics and torch threads for this process (1 to 6).",
    )
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument(
        "--terrain",
        type=Path,
        help="Terrain spec (JSON, surrogate/terrain.py) to train on: the floor becomes its height field, the "
        "touchdown model is off, and every reset starts at a random place and heading on it.",
    )
    parser.add_argument(
        "--spawn-margin",
        type=float,
        default=3.5,
        help="With --terrain, the least distance from a start to the terrain's edge, in metres.",
    )
    parser.add_argument("--checkpoint-interval", type=int, default=50)
    parser.add_argument("--max-wall-seconds", type=float, default=86400.0)
    parser.add_argument(
        "--contact",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="ContactModel override for sensitivity runs, e.g. noslip_iterations=0.",
    )
    parser.add_argument(
        "--randomize",
        action="append",
        default=[],
        metavar="FIELD=VALUE",
        help="DynamicsRandomization override for sim2real robustness runs, e.g. "
        "motor_scale=(0.9,1.1); without any the canonical dynamics stay bitwise unchanged.",
    )
    parser.add_argument(
        "--latent-supervision",
        type=float,
        default=0.0,
        help="Learning rate of the identification loss on the compensator's latent (the task "
        "reward is untouched); 0 keeps the stock learner. Needs --randomize and the "
        "locomotion.rl.sim2real:Sim2RealVecEnv wrapper.",
    )
    parser.add_argument(
        "--resume", type=Path, help="Checkpoint to continue from (same configuration)."
    )
    parser.add_argument(
        "--initialize-from",
        type=Path,
        help="Checkpoint whose actor, critic and optimizer state start a new run; inputs that the "
        "wrapper adds start at zero weight. The update count and configuration are this run's own.",
    )
    parser.add_argument(
        "--initialize-rate",
        choices=["configured", "checkpoint"],
        default="configured",
        help="Restart the adaptive learning rate at its configured value or continue the checkpoint's.",
    )
    parser.add_argument(
        "--no-loads", action="store_true", help="Skip the 400 Hz TrainingLoads accumulator."
    )
    parser.add_argument(
        "--no-update-diagnostics",
        action="store_true",
        help="Skip UpdateDiagnostics (needed for non-Gaussian actors).",
    )
    parser.add_argument("--verbose", action="store_true", help="Print RSL-RL's per-update table.")
    return parser


def main(argv=None):
    args = build_parser().parse_args(argv)
    thread_count(args.threads)
    if args.updates < 1 or args.num_envs < 1 or args.seed < 0:
        raise ValueError("Positive updates and replicas and a nonnegative seed are required")
    with process_threads(args.threads):
        return run(args, argv)


def run(args, argv=None):
    """One training run for parsed arguments, under the thread settings that ``main`` installed."""
    import mujoco
    import numpy as np
    import torch
    from rsl_rl.runners import OnPolicyRunner

    import locomotion.evaluation.ppo_diagnostics as diagnostics_module
    import locomotion.omniwalk.agents.rsl_rl_ppo_cfg as ppo_cfg
    import locomotion.omniwalk.task as task_module
    import locomotion.rl.ppo as algorithms
    from locomotion.evaluation.task_metrics import TaskMetrics, scalars
    from locomotion.surrogate.env import (
        DEFAULT_GEOMETRY,
        DEFAULT_MODEL,
        DEFAULT_STANCE,
        ROOT,
        SurrogateConfig,
        SurrogateEnv,
        contact_overrides,
        randomization_overrides,
        repository_state,
        sha,
    )

    version = importlib.metadata.version("rsl-rl-lib")
    if version != "5.0.1":
        raise ValueError("RSL-RL version differs: " + version)
    overrides = args.ppo_config_overrides
    overrides = json.loads(
        Path(overrides[1:]).read_text() if overrides.startswith("@") else overrides
    )
    contact = contact_overrides(args.contact)
    randomization = randomization_overrides(args.randomize) if args.randomize else None
    terrain = None
    if args.terrain is not None:
        from locomotion.surrogate.terrain import Terrain

        terrain = Terrain(json.loads(args.terrain.read_text()))
        # The touchdown model assumes the flat floor.
        contact = replace(contact, touchdown_overshoot=False)
    task_class, wrapper_class = resolve(args.task), resolve(args.wrapper)
    config = learner_configuration(
        ppo_cfg,
        args.seed,
        dict(
            action_mean=args.action_mean,
            observation_normalization=args.observation_normalization,
            observation_scaling=args.observation_scaling,
            command_segments=args.command_segments,
            learning_rate_max=args.learning_rate_max,
            action_std=args.action_std,
            action_noise_correlation=args.action_noise_correlation,
            action_std_final=args.action_std_final,
            gait_clock=args.gait_clock,
            action_smoothing=args.action_smoothing,
            velocity_noise=args.velocity_noise,
        ),
    )
    if args.distribution:
        config["actor"]["distribution_cfg"]["class_name"] = args.distribution
    merge(config, overrides)
    # The wrapper reads its options from the merged configuration, so an override reaches it too.
    options = wrapper_options(config)
    args.output.mkdir(parents=True, exist_ok=False)
    cfg = SurrogateConfig(
        num_envs=args.num_envs, seed=args.seed, episode_seconds=args.episode_seconds
    )
    identity = {
        "schema": "hexapod_surrogate_ppo_v1",
        "surrogate": True,
        "backend": "mujoco " + mujoco.__version__,
        "source_files": source_files(ROOT / "src/locomotion"),
        "surrogate_files": {p.name: sha(p) for p in sorted(PACKAGE.glob("*.py"))},
        "model_sha256": sha(DEFAULT_MODEL),
        "stance_sha256": sha(DEFAULT_STANCE),
        "geometry_sha256": sha(DEFAULT_GEOMETRY),
        "config": cfg.declaration(),
        "contact_model": asdict(contact),
        "randomization": None if randomization is None else asdict(randomization),
        "latent_supervision": args.latent_supervision or None,
        "terrain": None
        if terrain is None
        else {
            "spec": terrain.spec,
            "spec_sha256": sha(args.terrain),
            "spawn_margin_m": args.spawn_margin,
        },
        "seed": args.seed,
        "task": args.task,
        "wrapper": args.wrapper,
        "action_mean": args.action_mean,
        "observation_normalization": args.observation_normalization,
        "distribution": args.distribution,
        "observation_scaling": options.get("observation_scaling", "none"),
        "command_segments": options.get("command_segments", "continuous"),
        "gait_clock": options.get("gait_clock", 0),
        "action_smoothing": options.get("action_smoothing", "none"),
        "velocity_noise": options.get("velocity_noise", 0.0),
        "learning_rate_max": args.learning_rate_max,
        "action_std": args.action_std,
        "action_std_final": config.get("exploration", {}).get("action_std_final"),
        "exploration_schedule_updates": args.updates if config.get("exploration") else None,
        "action_noise_correlation": args.action_noise_correlation,
        "repository": repository_state(),
        "ppo_config_overrides": overrides,
        "rsl_rl_required_version": "5.0.1",
        "ppo_config": config,
        "motion_prior": False,
        "learner": "ppo",
        "networks": "mlp",
        "threads": args.threads,
        "scope": "CPU MuJoCo surrogate; design evidence. Native runs are the only acceptance evidence.",
    }
    state = {
        "mode": "train",
        "status": "initializing",
        "identity": identity,
        "errors": [],
        "stage2_complete": False,
        "argv": list(sys.argv[1:] if argv is None else argv),
    }
    save(args.output / "state.json", state)
    save(args.output / "ppo_config.json", config)
    started = time.monotonic()
    loads = None
    try:
        torch.manual_seed(args.seed)
        np.random.seed(args.seed)
        metadata = json.loads(DEFAULT_STANCE.read_text())
        env = SurrogateEnv(
            cfg,
            None,
            DEFAULT_MODEL,
            DEFAULT_GEOMETRY,
            args.output / "native",
            reference_metadata=metadata,
            threads=args.threads,
            contact=contact,
            substep_state=False,
            terrain=terrain,
            spawn=None if terrain is None else terrain.spawner(args.spawn_margin, args.seed),
            randomization=randomization,
        )
        task = task_class(env, task_module.TaskConfig(seed=args.seed), args.output / "task")
        # A task class outside TrainingTask may report its own status.
        reporter = (
            TaskMetrics(task)
            if isinstance(task, task_module.TrainingTask)
            else task
            if hasattr(task, "status")
            else None
        )
        identity["task_definition"] = task.declaration()
        identity["external_class_files"] = external_class_files(task_class, wrapper_class, config)
        check_gait_schedule(task, config)
        check_command_bootstrap(options, config)
        if args.initialize_from is not None and args.resume is not None:
            raise ValueError(
                "--initialize-from starts a new run; it cannot be combined with --resume"
            )
        parent = resume_record(args.resume, identity) if args.resume is not None else None
        if parent is not None:
            identity["exploration_schedule_updates"] = parent["identity"].get(
                "exploration_schedule_updates"
            )
            if "initialized_from" in parent["identity"]:
                identity["initialized_from"] = parent["identity"]["initialized_from"]
            identity["resume"] = {
                "checkpoint_sha256": parent["checkpoint_sha256"],
                "record_sha256": sha(args.resume.with_suffix(".json")),
                "identity": parent["identity"],
                "updates": parent["updates"],
                "transitions": parent["transitions"],
                "state_scope": "Learner and optimizer resume; simulator and random generators restart from the seed.",
            }
        wrapped = build_wrapper(wrapper_class, task, options)
        runner_config = copy.deepcopy(config)
        # The wrapper consumes its own options; the runner receives the stock keys.
        runner_config.pop("environment_wrapper", None)
        runner_config.pop("exploration", None)
        log_dir = str(args.output / "learner") if importlib.util.find_spec("tensorboard") else None
        runner = OnPolicyRunner(wrapped, runner_config, log_dir, device="cpu")
        if args.resume is not None:
            infos = runner.load(str(args.resume), strict=True, map_location="cpu")
            if (
                json.loads(json.dumps(infos, allow_nan=False))
                != {"identity": parent["identity"], "updates": parent["updates"]}
                or sha(args.resume) != parent["checkpoint_sha256"]
            ):
                raise ValueError("Resume checkpoint changed during loading")
            # RSL-RL stores the last iteration index; continue update numbering after it.
            runner.current_learning_iteration = int(infos["updates"])
            # RSL-RL restores optimizer groups but leaves the adaptive schedule's rate at its initial value.
            runner.alg.learning_rate = runner.alg.optimizer.param_groups[0]["lr"]
        if args.initialize_from is not None:
            from locomotion.rl.ppo import initialize_learner

            loaded = torch.load(str(args.initialize_from), weights_only=False, map_location="cpu")
            widened, rate = initialize_learner(
                runner.alg, loaded, config["algorithm"]["learning_rate"], args.initialize_rate
            )
            identity["initialized_from"] = {
                "checkpoint_sha256": sha(args.initialize_from),
                "loaded": ["actor", "critic", "optimizer"],
                "widened_inputs": widened,
                "learning_rate": {"value": rate, "source": args.initialize_rate},
            }
        first = runner.current_learning_iteration
        state.update(status="running", tensorboard=log_dir is not None)
        save(args.output / "state.json", state)
        diagnostics = None
        if not args.no_update_diagnostics:
            diagnostics = diagnostics_module.UpdateDiagnostics(runner.alg)
            runner.alg.update = diagnostics.update
        final_std = config.get("exploration", {}).get("action_std_final")
        if final_std is not None:
            # Keep the original schedule horizon when adding updates to a resumed run.
            schedule = algorithms.DeviationSchedule(
                runner.alg,
                config["actor"]["distribution_cfg"]["init_std"],
                final_std,
                identity["exploration_schedule_updates"],
            )
            schedule.completed = first
            schedule.apply()
            runner.alg.update = schedule.update
        if args.latent_supervision:
            from locomotion.rl.sim2real import LatentSupervision

            supervision = LatentSupervision(
                runner.alg, wrapped, randomization, args.latent_supervision
            )
            runner.alg.update = supervision.update

        def checkpoint(update):
            path = args.output / f"checkpoint_update{update:06d}.pt"
            runner.save(str(path), infos={"identity": identity, "updates": update})
            save(
                path.with_suffix(".json"),
                {
                    "identity": identity,
                    "updates": update,
                    "checkpoint_sha256": sha(path),
                    "transitions": update * config["num_steps_per_env"] * env.num_envs,
                },
            )
            state["checkpoint"] = str(path)
            state["checkpoint_sha256"] = sha(path)
            save(args.output / "state.json", state)

        if not args.no_loads:
            from locomotion.evaluation.loads import TrainingLoads

            loads = TrainingLoads(env)
            loads.uses_state = (
                False  # TrainingLoads reads forces and torques; skip substep state builds.
            )
            env.capture = loads

        def save_loads():
            if loads is not None:
                save(args.output / "force_metrics.json", loads.report())

        original_log = runner.logger.log
        target = first + args.updates

        def log(**values):
            if args.verbose:
                original_log(**values)
            else:
                with contextlib.redirect_stdout(io.StringIO()):
                    original_log(**values)
            update = values["it"] + 1
            status = reporter.status(reset_interval=True) if reporter is not None else None
            row = {
                "update": update,
                "transitions": update * config["num_steps_per_env"] * env.num_envs,
                "collection_seconds": values["collect_time"],
                "learning_seconds": values["learn_time"],
                "loss": values["loss_dict"],
                "learning_rate": values["learning_rate"],
                "mean_action_std": float(values["action_std"].detach().mean()),
                "actions": diagnostics_module.action_metrics(runner.alg.storage, env.joint_names),
                "task": status,
            }
            if diagnostics is not None:
                row["policy_update"] = diagnostics.latest
            with (args.output / "metrics.jsonl").open("a") as stream:
                stream.write(json.dumps(row, allow_nan=False) + "\n")
            if runner.logger.writer is not None and status is not None:
                for tag, value in scalars(status, "Task").items():
                    runner.logger.writer.add_scalar(tag, value, values["it"])
            state.update(
                updates=update,
                transitions=row["transitions"],
                wall_seconds=time.monotonic() - started,
                speed_clamp_events=env.speed_clamp_events,
                peak_joint_speed_rad_s=env.peak_joint_speed,
                touchdown_events=env.touchdown_events,
                touchdown_overshoot_events=env.overshoot_events,
            )
            if update % 10 == 0 or update == target:
                save(args.output / "state.json", state)
                interval = (status or {}).get("interval_metrics", {})
                print(
                    json.dumps(
                        {
                            "update": update,
                            "reward": interval.get("task_reward_mean"),
                            "command_speed": interval.get(
                                "moving_signed_command_direction_speed_mps"
                            ),
                            "std": row["mean_action_std"],
                            "collect_s": round(values["collect_time"], 3),
                            "learn_s": round(values["learn_time"], 3),
                            "wall_s": round(time.monotonic() - started, 1),
                        }
                    ),
                    flush=True,
                )
            expired = time.monotonic() - started >= args.max_wall_seconds
            if update % args.checkpoint_interval == 0 or update == target or expired:
                checkpoint(update)
                save_loads()
            if expired and update < target:
                raise TimeoutError("Training allocation ended after a complete PPO update")

        runner.logger.log = log
        runner.learn(args.updates, init_at_random_ep_len=False)
        save_loads()
        state["status"] = "completed"
    except BaseException as error:
        state["status"] = "failed"
        state["errors"].append(repr(error))
        state["traceback"] = traceback.format_exc()
        print(state["traceback"], flush=True)
    finally:
        state["wall_seconds"] = time.monotonic() - started
        save(args.output / "state.json", state)
    return 0 if state["status"] == "completed" else 1


if __name__ == "__main__":
    raise SystemExit(main())
