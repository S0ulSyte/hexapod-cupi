"""Share the arguments, validation, identity and run lifecycle of the native locomotion scripts.

train.py trains, play.py evaluates a checkpoint and diagnose.py runs the standing diagnostic or the
noise probe, as in Isaac Lab's scripts/reinforcement_learning/rsl_rl; this module takes the role of
its cli_args.py.
"""

from __future__ import annotations

import argparse
import faulthandler
import hashlib
import importlib.metadata
import importlib.util
import json
import os
import re
import sys
import time
import traceback
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from locomotion.evaluation.admission import require_admission, verify_native_recipe, verify_scene
from locomotion.evaluation.camera import prepare_policy_scene
from locomotion.evaluation.checkpoint import (
    DECLARATION_SHA,
    checkpoint_reward_version,
    physics_config,
    read_compatibility_receipt,
    require_checkpoint_identity,
    require_compatibility_identity,
)
from locomotion.evaluation.task_metrics import scalars as scalars
from locomotion.omniwalk.agents.rsl_rl_ppo_cfg import (
    EXPERIMENT_CLOCKS,
    EXPERIMENT_INPUTS,
    RECIPES,
    ppo_config,
)
from locomotion.omniwalk.env_cfg import EnvConfig
from locomotion.omniwalk.mdp.commands import GAIT_CLOCK_CONTROLS
from locomotion.robots.hexapod import MODEL_SHA256, USD_SHA256, verify_assets
from locomotion.spark.sources import physics_files, source_files, verify_runtime
from locomotion.spark.validation import (
    ALLOCATION_PROFILES,
    LEARNER_OPTION_DEFAULTS,
    OPTION_DEFAULTS,
    resolve_learner_options,
    update_limit,
    validate_deadline,
    validate_decay,
)

SOURCE = Path(__file__).resolve().parents[1]


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def print_progress(value):
    print(json.dumps(value), flush=True)


class ConfigRecord(dict):
    """Give RSL-RL's W&B writer the to_dict() it expects from an environment config."""

    def to_dict(self):
        return dict(self)


@dataclass
class Session:
    """The records that run() gives a mode body; a body that sets loads gets force_metrics.json after any outcome."""

    args: argparse.Namespace
    identity: dict
    state: dict
    compatibility: dict | None
    checkpoint_record: dict | None
    started: float
    loads: object = None


def add_common_arguments(parser):
    """Add the inputs, allocation, checkpoint and logger arguments of every script."""
    for name in ("asset", "model", "geometry", "geometry-extrema", "stance", "output"):
        parser.add_argument("--" + name, type=Path, required=True)
    parser.add_argument("--standing-admission", type=Path)
    parser.add_argument("--source-freeze-sha256", required=True)
    parser.add_argument(
        "--num-envs",
        type=int,
        choices=[1, 32, 128, 512, 1024],
        required=True,
        help="Replica count. Training and probes accept 128 and 512; diagnostics and profiles accept each listed count.",
    )
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument(
        "--extended-updates",
        action="store_true",
        help="Accept up to 5000 training updates; without this opt-in training stops at 2000.",
    )
    parser.add_argument("--max-wall-seconds", type=float, default=6200.0)
    parser.add_argument("--allocation-profile", choices=ALLOCATION_PROFILES, default="standard")
    parser.add_argument("--eval-scope", choices=["focus", "probes", "full"], default="focus")
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha256")
    parser.add_argument("--checkpoint-compatibility", type=Path)
    parser.add_argument("--checkpoint-compatibility-sha256")
    parser.add_argument("--preflight-only", action="store_true")
    parser.add_argument("--logger", choices=["tensorboard", "wandb"], default="tensorboard")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-mode", choices=["offline", "online"], default="offline")
    parser.add_argument(
        "--episode-seconds",
        type=float,
        default=20.0,
        help="Training episode length; 20 s by default. Shorter episodes bound how far a robot walks.",
    )
    parser.add_argument(
        "--video-case",
        help="Learning probe to record on video; the first selected probe by default.",
    )
    parser.add_argument(
        "--experiment",
        choices=sorted(EXPERIMENT_INPUTS),
        help="A reviewed experiment's task, reward and inputs (omniwalk/destination_task.py); "
        "training and evaluation of its checkpoints name it.",
    )


def add_learner_arguments(parser):
    """Add the learner and recipe arguments with the option flags that replace recipe values."""
    parser.add_argument(
        "--terrain",
        type=Path,
        help="Terrain spec (JSON, locomotion/surrogate/terrain.py) tiled under every environment; "
        "the floor becomes its height field and the env exposes the grid to terrain inputs.",
    )
    parser.add_argument(
        "--terrain-mesh-resolution",
        type=float,
        default=0.08,
        help="Collision-mesh decimation in metres; the input grid keeps the spec's resolution.",
    )
    parser.add_argument(
        "--terrain-spawn",
        action="store_true",
        help="Scatter each reset inside its environment's cell with a random heading.",
    )
    parser.add_argument("--learner", choices=["ppo", "amp"], default="ppo")
    parser.add_argument("--networks", choices=["mlp", "paper"], default="mlp")
    parser.add_argument(
        "--recipe",
        choices=[*RECIPES, "none"],
        default="c48_long",
        help="Learner options of PPO training and evaluation (%(default)s by default) from omniwalk/agents/"
        "rsl_rl_ppo_cfg.py; option flags replace its values. Other runs and the recipe none keep the defaults.",
    )
    parser.add_argument(
        "--action-mean",
        choices=["unbounded", "tanh"],
        help="Use the historical Gaussian mean or bound the mean with tanh.",
    )
    parser.add_argument(
        "--observation-normalization",
        choices=["empirical", "none"],
        help="Use running observation statistics or retain the raw PPO observations.",
    )
    parser.add_argument(
        "--observation-scaling",
        choices=["none", "fixed"],
        help="Pass raw observations or apply the fixed input scales declared in wrapper.py.",
    )
    parser.add_argument(
        "--command-segments",
        choices=["continuous", "bootstrap"],
        help="Carry returns across command changes or bootstrap the return at each change.",
    )
    parser.add_argument(
        "--learning-rate-max",
        type=float,
        help="Ceiling for the adaptive learning rate; the stock schedule allows 1e-2.",
    )
    parser.add_argument(
        "--learning-rate-min",
        type=float,
        help="Floor for the adaptive learning rate above the stock schedule's 1e-5; needs the ceiling.",
    )
    parser.add_argument(
        "--action-std",
        type=float,
        help="Initial standard deviation of the Gaussian action distribution.",
    )
    parser.add_argument(
        "--action-smoothing",
        choices=["none", "mean2"],
        help="mean2 sends the mean of each action and the previous one to the environment.",
    )
    parser.add_argument(
        "--velocity-noise",
        type=float,
        help="Standard deviation in rad/s of Gaussian noise on the actor joint-velocity inputs in training.",
    )
    parser.add_argument(
        "--gait-clock",
        type=int,
        help=f"Append gait phase with this period in controls; without a recipe value, training "
        f"defaults to {GAIT_CLOCK_CONTROLS}, other modes to 0.",
    )
    parser.add_argument(
        "--action-std-final",
        type=float,
        help="Lower the action deviation linearly to this value over the run; PPO then does not learn it.",
    )
    parser.add_argument(
        "--action-std-decay-updates",
        type=int,
        help="Reach the final action deviation after this many updates and hold it; the run length by default.",
    )
    parser.add_argument(
        "--action-noise-correlation",
        type=float,
        help="Share of each exploration noise sample carried to the next control (tanh mean only).",
    )
    parser.add_argument(
        "--steps-per-env",
        type=int,
        help="Controls each robot collects per PPO update, a multiple of 6 from 12 to 96; "
        "mini-batches scale with it.",
    )


def add_app_arguments(parser, argv=None):
    """Add the Isaac Lab AppLauncher arguments.

    A preflight or help run declares the two launcher options itself, so it needs no Isaac installation.
    """
    if any(
        flag in (argv if argv is not None else sys.argv[1:])
        for flag in ("--preflight-only", "--help", "-h")
    ):
        parser.add_argument("--headless", action="store_true")
        parser.add_argument("--device", default="cuda:0")
    else:
        from isaaclab.app import AppLauncher

        AppLauncher.add_app_launcher_args(parser)


def resolve_options(args, mode):
    """Resolve the recipe and the gait-clock default of the mode into the learner options of parsed arguments."""
    vars(args).update(
        resolve_learner_options(
            args.recipe,
            mode,
            args.learner,
            args.networks,
            {key: getattr(args, key) for key in OPTION_DEFAULTS},
        )
    )
    return args


def learner_options(args):
    """Return the learner options that ppo_config records, in its keyword order."""
    return dict(
        observation_scaling=args.observation_scaling,
        command_segments=args.command_segments,
        learning_rate_max=args.learning_rate_max,
        learning_rate_min=args.learning_rate_min,
        action_std=args.action_std,
        action_noise_correlation=args.action_noise_correlation,
        action_std_final=args.action_std_final,
        gait_clock=args.gait_clock,
        action_smoothing=args.action_smoothing,
        velocity_noise=args.velocity_noise,
        action_std_decay_updates=args.action_std_decay_updates,
        steps_per_env=args.steps_per_env,
    )


def validate(args):
    """Reject option combinations, changed assets and a changed frozen source.

    Return the compatibility receipt, or None when the run names no receipt.
    """
    if (args.networks == "paper" and args.learner != "amp") or (
        args.learner == "amp" and args.mode not in ("train", "evaluate")
    ):
        raise ValueError("The paper networks need the AMP learner for training or evaluation")
    compatibility = None
    if (
        args.checkpoint_compatibility is not None
        or args.checkpoint_compatibility_sha256 is not None
    ):
        if (
            args.mode != "evaluate"
            or args.eval_scope != "probes"
            or args.num_envs != 1
            or args.checkpoint_compatibility is None
            or args.checkpoint_compatibility_sha256 is None
        ):
            raise ValueError("Compatibility requires evaluation and both receipt arguments")
        compatibility = read_compatibility_receipt(
            args.checkpoint_compatibility, args.checkpoint_compatibility_sha256
        )
    if args.logger == "wandb":
        if (
            args.mode != "train"
            or not re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", args.wandb_project or "")
            or importlib.util.find_spec("wandb") is None
            or (args.wandb_mode == "online" and not os.environ.get("WANDB_API_KEY"))
        ):
            raise ValueError(
                "W&B logging needs train mode, a project name, the wandb package "
                "and WANDB_API_KEY when online"
            )
    elif args.wandb_project is not None or args.wandb_mode != "offline":
        raise ValueError("W&B options require --logger wandb")
    if args.action_mean != "unbounded" and args.mode not in ("train", "evaluate"):
        raise ValueError("Bounded action means apply to training and evaluation")
    if args.observation_normalization != "empirical" and args.mode not in ("train", "evaluate"):
        raise ValueError("Observation normalization selection requires PPO training or evaluation")
    if learner_options(args) != LEARNER_OPTION_DEFAULTS and args.mode not in ("train", "evaluate"):
        raise ValueError(
            "Observation scaling, command segments, the rate ceiling, the action deviation and "
            "the noise correlation require PPO training or evaluation"
        )
    if not 12 <= args.steps_per_env <= 96 or args.steps_per_env % 6:
        raise ValueError("The rollout length is a multiple of 6 from 12 to 96 controls per robot")
    if args.experiment is not None and (
        args.mode not in ("train", "evaluate") or args.learner != "ppo"
    ):
        raise ValueError("An experiment applies to PPO training and evaluation")
    if args.video_case is not None and (args.mode != "evaluate" or args.eval_scope == "full"):
        raise ValueError("A video case applies to a learning-probe evaluation")
    if args.episode_seconds != 20.0 and (
        args.mode != "train" or not 5.0 <= args.episode_seconds <= 32.0
    ):
        # 32 s, the evaluator's longest probe, as locomotion.spark.prepare allows.
        raise ValueError("Episode length selection applies to training, between 5 and 32 seconds")
    if args.terrain is not None:
        if args.mode != "train" or args.learner != "ppo":
            raise ValueError("A terrain spec applies to PPO training")
        if not args.terrain.is_file():
            raise ValueError("The terrain spec file does not exist")
    if args.terrain_spawn and args.terrain is None:
        raise ValueError("Terrain spawn randomization needs a terrain spec")
    if args.action_std_decay_updates is not None and args.action_std_final is None:
        raise ValueError("The deviation decay length needs a final action deviation")
    validate_decay(args.mode, args.updates, args.action_std_decay_updates)
    limit = update_limit(args.mode, args.extended_updates)
    verify_assets(args.asset, args.model)
    validate_deadline(args.mode, args.allocation_profile, args.max_wall_seconds)
    if (
        not args.headless
        or args.device != "cuda:0"
        or not 1 <= args.updates <= limit
        or args.seed < 0
        or (
            args.mode in ("train", "probe")
            and (args.num_envs not in (128, 512) or args.checkpoint is not None)
        )
        or (args.mode == "evaluate" and (args.num_envs != 1 or args.checkpoint is None))
    ):
        raise ValueError(
            "A bounded scratch training, a 128-replica probe or single-replica evaluation is required"
        )
    if sha(SOURCE.parent / "FREEZE_SHA256.json") != args.source_freeze_sha256:
        raise ValueError("Frozen source identity differs")
    return compatibility


def environment_config(args):
    """Build the environment configuration for the mode; full evaluation uses the evaluation configuration."""
    from locomotion.evaluation.runner import EvaluationEnvConfig

    config_class = (
        EvaluationEnvConfig if args.mode == "evaluate" and args.eval_scope == "full" else EnvConfig
    )
    if args.mode == "evaluate":
        episode_seconds = 90.0 if args.eval_scope == "full" else 60.0
    else:
        episode_seconds = 60.0 if args.mode == "diagnostic" else args.episode_seconds
    terrain = {}
    if getattr(args, "terrain", None) is not None:
        terrain = {
            "terrain_spec_path": str(args.terrain),
            "terrain_mesh_resolution_m": args.terrain_mesh_resolution,
            "terrain_spawn": bool(args.terrain_spawn),
        }
    return config_class(
        num_envs=args.num_envs,
        seed=args.seed,
        record_motion_features=args.mode == "evaluate" or args.learner == "amp",
        render=args.mode == "evaluate",
        episode_seconds=episode_seconds,
        device=args.device,
        **terrain,
    )


def build_identity(args, cfg, metadata):
    """Record the sources, inputs, physics and learner that the run and its checkpoints bind.

    Every script records the hash of scripts/train.py, so a checkpoint and its evaluation bind the same training entry.
    """
    identity = {
        "schema": "hexapod_locomotion_ppo_v2",
        "source_files": source_files(SOURCE),
        "runtime_sources": verify_runtime(SOURCE.parent),
        "model_sha256": MODEL_SHA256,
        "usd_sha256": USD_SHA256,
        "stance_sha256": sha(args.stance),
        "geometry_sha256": sha(args.geometry),
        "geometry_extrema_sha256": sha(args.geometry_extrema),
        "config": cfg.declaration(),
        "motion_prior": args.learner == "amp",
        "behavior_cloning": False,
        "seed": args.seed,
        "rsl_rl_required_version": "5.0.1",
        "adapter_sha256": sha(SOURCE / "rl/wrapper.py"),
        "entry_sha256": sha(SOURCE / "scripts/train.py"),
        "reward_version": "4",
        "reward_task": "v4_yaw_gate",
        "learner": args.learner,
        "networks": args.networks,
    }
    if args.experiment is not None:
        from locomotion.omniwalk.destination_task import REWARD_VERSION

        # A named experiment trains its own reward, not canonical version 4.
        identity["experiment"] = args.experiment
        identity["reward_version"] = REWARD_VERSION
        identity["reward_task"] = args.experiment
    if getattr(args, "terrain", None) is not None:
        identity["terrain"] = {
            "spec_path": str(args.terrain),
            "spec_sha256": sha(args.terrain),
            "mesh_resolution_m": args.terrain_mesh_resolution,
            "spawn": bool(args.terrain_spawn),
        }
    if getattr(args, "initialize_from", None) is not None:
        identity["initialized_from"] = {
            "checkpoint_sha256": args.initialize_from_sha256,
            "declaration_sha256": args.initialize_from_declaration_sha256,
        }
    if args.extended_updates:
        # The exploration decay length enters ppo_config; the opt-in records the requested run length beside it.
        identity["training_updates"] = args.updates
    if args.learner == "amp":
        from locomotion.amp.amp import AMPConfig, load_demonstrations

        identity["learner_source_files"] = {
            name: sha(SOURCE / name)
            for name in ("amp/features.py", "amp/discriminator.py", "amp/amp.py", "amp/networks.py")
        }
        identity["amp_dataset"] = load_demonstrations(SOURCE.parent / AMPConfig().dataset)[1]
    identity["physics_source_files"] = physics_files(identity["source_files"])
    identity["physics_config"] = physics_config(cfg, metadata)
    if args.mode != "diagnostic":
        identity["standing_admission"] = require_admission(args.standing_admission, identity, cfg)
    return identity


def bind_checkpoint(args, identity, compatibility):
    """Check the checkpoint bytes and declaration, then record the reward it learned from.

    Return the checkpoint declaration.
    """
    if sha(args.checkpoint) != args.checkpoint_sha256:
        raise ValueError("Checkpoint bytes differ")
    record = json.loads(args.checkpoint.with_suffix(".json").read_text())
    if record["checkpoint_sha256"] != args.checkpoint_sha256:
        raise ValueError("Checkpoint declaration differs")
    # The experiment decides the task, the wrapper and the inputs; it must be the one the checkpoint trained.
    if record["identity"].get("experiment") != identity.get("experiment"):
        raise ValueError(
            f"Checkpoint experiment {record['identity'].get('experiment')} differs from the requested "
            f"{identity.get('experiment')}"
        )
    if compatibility is None:
        require_checkpoint_identity(record, identity)
    else:
        if sha(args.checkpoint.with_suffix(".json")) != DECLARATION_SHA:
            raise ValueError("Compatibility checkpoint declaration differs")
        require_compatibility_identity(compatibility, record, identity)
    # Evaluation computes no training reward; record the one the checkpoint learned from.
    identity["reward_version"] = checkpoint_reward_version(record)
    for key in ("reward_task", "reward_options"):
        identity.pop(key, None)
        if key in record["identity"]:
            identity[key] = record["identity"][key]
    identity["behavior_cloning"] = record["identity"].get("behavior_cloning", False)
    if "bc" in record["identity"]:
        identity["bc"] = record["identity"]["bc"]
    return record


def initial_state(args, identity, compatibility, checkpoint_record):
    """Return the first state.json record of the run."""
    state = {
        "mode": args.mode,
        "status": "initializing",
        "identity": identity,
        "errors": [],
        "stage2_complete": False,
        "runtime_binding": {"runtime_tree_sha256": args.source_freeze_sha256},
        "allocation": {
            "profile": args.allocation_profile,
            "max_wall_seconds": args.max_wall_seconds,
        },
        "experiment_logger": {
            "backend": args.logger,
            "wandb_project": args.wandb_project,
            "wandb_mode": args.wandb_mode if args.logger == "wandb" else None,
        },
    }
    if compatibility is not None:
        state["checkpoint_compatibility"] = {
            "receipt_sha256": args.checkpoint_compatibility_sha256,
            "original_identity": checkpoint_record["identity"],
            "checkpoint_sha256": args.checkpoint_sha256,
        }
    return state


def launch_app(args):
    """Start Isaac Sim; a stalled start writes the thread stacks to the output every 45 s."""
    from isaaclab.app import AppLauncher

    args.enable_cameras = args.mode == "evaluate"
    args.headless_explicit = False
    with (args.output / "startup_tracebacks.log").open("w") as stream:
        faulthandler.enable(file=stream)
        faulthandler.dump_traceback_later(45.0, repeat=True, file=stream)
        try:
            return AppLauncher(args).app
        finally:
            faulthandler.cancel_dump_traceback_later()
            faulthandler.disable()


def create_env(args, cfg, metadata):
    """Seed torch and NumPy, build the native environment and check its scene; evaluation adds the policy camera scene."""
    import torch

    from locomotion.omniwalk.adapter import OmniEnv

    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    env = OmniEnv(
        cfg,
        args.asset,
        args.model,
        args.geometry,
        args.output / "native",
        reference_metadata=metadata,
        verify=verify_scene,
    )
    if args.mode == "evaluate":
        prepare_policy_scene(env)
    return env


def learner_config(args, identity, state):
    """Check RSL-RL, write ppo_config.json and record the upstream sources the learner runs."""
    import rsl_rl

    version = importlib.metadata.version("rsl-rl-lib")
    if version != "5.0.1":
        raise ValueError("RSL-RL version differs: " + version)
    factory, options = ppo_config, {}
    if args.learner == "amp":
        from locomotion.amp.amp import amp_ppo_config

        factory, options = amp_ppo_config, {"networks": args.networks}
    if args.experiment is not None:
        options["destination_input"] = EXPERIMENT_INPUTS[args.experiment]
        if args.experiment in EXPERIMENT_CLOCKS:
            options["gait_clock_scaling"] = EXPERIMENT_CLOCKS[args.experiment]
    config = factory(
        args.seed,
        **options,
        action_mean=args.action_mean,
        observation_normalization=args.observation_normalization,
        **learner_options(args),
    )
    save(args.output / "ppo_config.json", config)
    upstream = Path(rsl_rl.__file__).parent
    identity["upstream_source_files"] = {
        name: sha(upstream / name)
        for name in (
            "runners/on_policy_runner.py",
            "algorithms/ppo.py",
            "models/mlp_model.py",
            "storage/rollout_storage.py",
        )
    }
    identity["ppo_config"] = config
    state["status"] = "running"
    save(args.output / "state.json", state)
    return config


def run(args, mode, body):
    """Run one native mode and return the exit code.

    Validate the run and bind its identity; a preflight prints the identity and stops. Otherwise call
    body(env, session) on the native environment between state.json records.
    """
    args.mode = mode
    compatibility = validate(args)
    metadata = json.loads(args.stance.read_text())
    cfg = environment_config(args)
    identity = build_identity(args, cfg, metadata)
    checkpoint_record = (
        bind_checkpoint(args, identity, compatibility) if mode == "evaluate" else None
    )
    if args.preflight_only:
        print(json.dumps(identity, indent=2))
        return 0
    args.output.mkdir(parents=True, exist_ok=False)
    state = initial_state(args, identity, compatibility, checkpoint_record)
    save(args.output / "state.json", state)
    app = None
    session = Session(args, identity, state, compatibility, checkpoint_record, time.monotonic())
    try:
        app = launch_app(args)
        print("REFERENCE_SCREEN_APP_READY", flush=True)
        env = create_env(args, cfg, metadata)
        body(env, session)
        verify_native_recipe(env, "after_controlled_steps")
        state["status"] = "completed"
    except BaseException as error:
        state["status"] = "failed"
        state["errors"].append(repr(error))
        state["traceback"] = traceback.format_exc()
        print(state["traceback"], flush=True)
    finally:
        if session.loads is not None:
            save(args.output / "force_metrics.json", session.loads.report())
        state["wall_seconds"] = time.monotonic() - session.started
        save(args.output / "state.json", state)
        if app is not None:
            app.close()
    return 0 if state["status"] == "completed" else 1
