"""Copy a named kernel and explicit inputs into a fresh native allocation."""

from __future__ import annotations

import argparse
import json
import re
import shutil
from pathlib import Path

from locomotion.omniwalk.agents.rsl_rl_ppo_cfg import (
    EXPERIMENT_CLOCKS,
    EXPERIMENT_INPUTS,
    RECIPES,
    ppo_config,
)
from locomotion.omniwalk.mdp.commands import GAIT_CLOCK_CONTROLS
from locomotion.robots.hexapod import sha
from locomotion.spark.safety import HOME_ROOT, RUN_ROOTS, within_roots
from locomotion.spark.sources import runtime_paths, write_runtime
from locomotion.spark.validation import (
    ALLOCATION_PROFILES,
    CLEANUP_MARGIN_SECONDS,
    ENTRY_POINTS,
    LEARNER_OPTION_DEFAULTS,
    resolve_learner_options,
    update_limit,
    validate_deadline,
    validate_decay,
)

ROOT = Path(__file__).resolve().parents[3]
# Standing diagnostics accept each count; training and the noise probe require admitted counts.
REPLICA_COUNTS = (1, 32, 128, 512, 1024)
# Counts with standing admission and a program-lead budget for training and the noise probe.
TRAINING_COUNTS = (128, 512)
REMOTE_ROOT = HOME_ROOT


def save(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def prepare(
    output,
    remote_root,
    *,
    mode="train",
    updates=512,
    seed=20260917,
    num_envs=None,
    inputs=None,
    eval_scope="focus",
    checkpoint=None,
    checkpoint_sha=None,
    checkpoint_declaration_sha=None,
    checkpoint_compatibility=None,
    checkpoint_compatibility_sha256=None,
    logger="tensorboard",
    wandb_project=None,
    wandb_mode="offline",
    action_mean=None,
    learner="ppo",
    networks="mlp",
    recipe="c48_long",
    observation_normalization=None,
    observation_scaling=None,
    command_segments=None,
    learning_rate_max=None,
    learning_rate_min=None,
    action_std=None,
    action_noise_correlation=None,
    action_std_final=None,
    gait_clock=None,
    action_smoothing=None,
    velocity_noise=None,
    episode_seconds=20.0,
    video_case=None,
    extended_updates=False,
    action_std_decay_updates=None,
    steps_per_env=None,
    anchor_weight=None,
    experiment=None,
    terrain=None,
    terrain_sha=None,
    terrain_mesh_resolution=None,
    terrain_spawn=False,
    initialize_from=None,
    initialize_from_sha=None,
    initialize_from_declaration_sha=None,
    initialize_rate="configured",
    allocation_profile="standard",
    max_wall_seconds=6200,
    root=ROOT,
):
    """Write a native allocation pack. PPO training and evaluation start from the named recipe; a learner
    option that is not None replaces the recipe value."""
    output, remote_root, root = map(Path, (output, remote_root, root))
    tree = root / "src"
    if (
        learner not in ("ppo", "amp")
        or networks not in ("mlp", "paper")
        or (networks == "paper" and learner != "amp")
        or (learner == "amp" and mode not in ("train", "evaluate"))
    ):
        raise ValueError("The paper networks need the AMP learner for training or evaluation")
    if recipe not in (*RECIPES, "none"):
        raise ValueError("Unknown learner recipe")
    if experiment is not None and (
        experiment not in EXPERIMENT_INPUTS or mode not in ("train", "evaluate") or learner != "ppo"
    ):
        raise ValueError("A named experiment applies to PPO training and evaluation")
    if terrain is not None:
        terrain = Path(terrain)
        if mode != "train" or learner != "ppo":
            raise ValueError("A terrain spec applies to PPO training")
        # The spec is a remote run-tree path at launch time, validated like a checkpoint: by
        # allowed root and hash shape, never by a local stat.
        if (
            not within_roots(terrain, RUN_ROOTS)
            or ".." in terrain.parts
            or terrain_sha is None
            or len(terrain_sha) != 64
            or set(terrain_sha) - set("0123456789abcdef")
        ):
            raise ValueError("Invalid terrain spec identity")
    elif terrain_spawn or terrain_sha is not None or terrain_mesh_resolution is not None:
        raise ValueError("Terrain options need a terrain spec")
    learner_options = resolve_learner_options(
        recipe,
        mode,
        learner,
        networks,
        dict(
            action_mean=action_mean,
            observation_normalization=observation_normalization,
            observation_scaling=observation_scaling,
            command_segments=command_segments,
            learning_rate_max=learning_rate_max,
            learning_rate_min=learning_rate_min,
            action_std=action_std,
            action_noise_correlation=action_noise_correlation,
            action_std_final=action_std_final,
            gait_clock=gait_clock,
            action_smoothing=action_smoothing,
            velocity_noise=velocity_noise,
            action_std_decay_updates=action_std_decay_updates,
            steps_per_env=steps_per_env,
        ),
    )
    action_mean, observation_normalization = (
        learner_options.pop("action_mean"),
        learner_options.pop("observation_normalization"),
    )
    gait_clock = learner_options["gait_clock"]
    if mode == "train" and networks == "mlp" and gait_clock != GAIT_CLOCK_CONTROLS:
        raise ValueError(f"The active reward requires the {GAIT_CLOCK_CONTROLS}-control gait clock")
    validate_deadline(mode, allocation_profile, max_wall_seconds)
    if type(max_wall_seconds) is not int:
        raise ValueError("Preparation requires an integer native deadline")
    if type(extended_updates) is not bool:
        raise ValueError("Extended updates is an explicit opt-in")
    limit = update_limit(mode, extended_updates)
    if (
        not within_roots(remote_root, RUN_ROOTS)
        or ".." in remote_root.parts
        or mode not in ("diagnostic", "train", "evaluate", "probe", "replay")
        or type(updates) is not int
        or not 1 <= updates <= limit
        or type(seed) is not int
        or seed < 0
    ):
        raise ValueError("Invalid native allocation")
    num_envs = (128 if mode in ("train", "probe") else 1) if num_envs is None else num_envs
    if (
        num_envs not in REPLICA_COUNTS
        or (mode in ("evaluate", "replay") and num_envs != 1)
        or (mode in ("train", "probe") and num_envs not in TRAINING_COUNTS)
        or eval_scope not in ("focus", "probes", "full")
    ):
        raise ValueError("Invalid replica count for this mode")
    wandb_valid = (
        mode == "train"
        and wandb_mode in ("offline", "online")
        and re.fullmatch(r"[A-Za-z0-9_.-]{1,128}", wandb_project or "") is not None
    )
    if (
        (logger == "wandb" and not wandb_valid)
        or (logger == "tensorboard" and (wandb_project is not None or wandb_mode != "offline"))
        or logger not in ("tensorboard", "wandb")
    ):
        raise ValueError("W&B logging needs train mode, a project name and offline or online mode")
    if action_mean not in ("unbounded", "tanh") or (
        action_mean != "unbounded" and mode not in ("train", "evaluate")
    ):
        raise ValueError("Bounded action means apply to training and evaluation")
    if observation_normalization not in ("empirical", "none") or (
        observation_normalization != "empirical" and (mode not in ("train", "evaluate"))
    ):
        raise ValueError("Observation normalization selection requires PPO training or evaluation")
    selected_options = {
        key: value
        for key, value in learner_options.items()
        if value != LEARNER_OPTION_DEFAULTS[key]
    }
    try:
        ppo_config(
            seed,
            action_mean=action_mean,
            observation_normalization=observation_normalization,
            **learner_options,
            **({} if experiment is None else {"destination_input": EXPERIMENT_INPUTS[experiment]}),
            **(
                {"gait_clock_scaling": EXPERIMENT_CLOCKS[experiment]}
                if experiment in EXPERIMENT_CLOCKS
                else {}
            ),
        )
    except ValueError as error:
        raise ValueError(
            "Observation scaling, command segments, the rate ceiling, the action deviation and "
            "the noise correlation require PPO training or evaluation"
        ) from error
    if selected_options and (mode not in ("train", "evaluate")):
        raise ValueError(
            "Observation scaling, command segments, the rate ceiling, the action deviation and "
            "the noise correlation require PPO training or evaluation"
        )
    if networks == "paper" and (
        gait_clock
        or learner_options["observation_scaling"] != "none"
        or learner_options["velocity_noise"]
    ):
        raise ValueError(
            "Paper networks require their recorded raw observation groups without gait-clock inputs"
        )
    validate_decay(mode, updates, learner_options["action_std_decay_updates"])
    if video_case is not None and (
        mode != "evaluate" or eval_scope == "full" or not video_case.startswith("learning:")
    ):
        raise ValueError("A video case applies to a learning-probe evaluation")
    if type(episode_seconds) is not float or (
        episode_seconds != 20.0 and (mode != "train" or not 5.0 <= episode_seconds <= 32.0)
    ):
        # 32 s, the evaluator's longest probe, lets destination_v28 and later train its long stands.
        raise ValueError("Episode length selection applies to training, between 5 and 32 seconds")
    if initialize_from is not None and (mode != "train" or learner != "ppo"):
        raise ValueError("A starting checkpoint applies to PPO training")
    if anchor_weight is not None and (
        mode != "train"
        or learner != "ppo"
        or type(anchor_weight) is not float
        or not 0 < anchor_weight < 1
    ):
        raise ValueError("The anchor weight applies to PPO training as a fraction in (0, 1)")
    starting = (initialize_from, initialize_from_sha, initialize_from_declaration_sha)
    if len({value is None for value in starting}) != 1 or initialize_rate not in (
        "configured",
        "checkpoint",
    ):
        raise ValueError(
            "A starting checkpoint needs its two file hashes and a configured or checkpoint rate"
        )
    # Check the checkpoint paths and hashes before anything is written, so a refusal leaves no pack.
    if initialize_from is not None:
        initialize_from = Path(initialize_from)
        if (
            not within_roots(initialize_from, RUN_ROOTS)
            or ".." in initialize_from.parts
            or any(
                len(h) != 64 or set(h) - set("0123456789abcdef")
                for h in (initialize_from_sha, initialize_from_declaration_sha)
            )
        ):
            raise ValueError("Invalid starting checkpoint identity")
    supplied = (
        checkpoint is not None,
        checkpoint_sha is not None,
        checkpoint_declaration_sha is not None,
    )
    if (mode == "evaluate" and not all(supplied)) or (mode != "evaluate" and any(supplied)):
        raise ValueError("Evaluation requires a checkpoint and its two file hashes")
    if checkpoint is not None:
        checkpoint = Path(checkpoint)
        if (
            not within_roots(checkpoint, RUN_ROOTS)
            or ".." in checkpoint.parts
            or any(
                len(h) != 64 or set(h) - set("0123456789abcdef")
                for h in (checkpoint_sha, checkpoint_declaration_sha)
            )
        ):
            raise ValueError("Invalid checkpoint identity")
    compatibility = (
        checkpoint_compatibility is not None,
        checkpoint_compatibility_sha256 is not None,
    )
    if any(compatibility) and (
        mode != "evaluate" or eval_scope != "probes" or not all(compatibility)
    ):
        raise ValueError("Compatibility requires evaluation and both receipt arguments")
    if all(compatibility):
        receipt = Path(checkpoint_compatibility)
        if (
            not within_roots(receipt, RUN_ROOTS)
            or ".." in receipt.parts
            or len(checkpoint_compatibility_sha256) != 64
            or set(checkpoint_compatibility_sha256) - set("0123456789abcdef")
        ):
            raise ValueError("Invalid checkpoint compatibility path or hash")
    inputs = root / "configs/locomotion_spark.json" if inputs is None else Path(inputs)
    declared = json.loads(inputs.read_text())
    for key in ("asset", "geometry_source", "stance"):
        path = Path(declared[key])
        if not path.is_absolute() or ".." in path.parts:
            raise ValueError("Inputs must use explicit absolute paths")
    if declared["stance"] not in declared["input_files"]:
        raise ValueError("The stance file needs a recorded hash")
    output.mkdir(parents=True, exist_ok=False)
    source = output / "source"
    package = source / "locomotion"
    package.mkdir(parents=True)
    for path in runtime_paths(tree / "locomotion"):
        destination = package / path.relative_to(tree / "locomotion")
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    prior_sources = (
        (
            "__init__.py",
            "amp.py",
            "discriminator.py",
            "networks.py",
            "dataset.py",
            "processing_sources_v2.json",
        )
        if learner == "amp"
        else (("__init__.py", "replay_native.py") if mode == "replay" else ())
    )
    if prior_sources:
        prior_package = package / "amp"
        prior_package.mkdir(exist_ok=True)
        for name in prior_sources:
            shutil.copy2(tree / "locomotion/amp" / name, prior_package / name)
    if learner == "amp":
        bank = Path("locomotion/priors/datasets/amp_demonstrations_001")
        shutil.copytree(tree / bank, source / bank)
    contracts = source / "contracts"
    contracts.mkdir()
    for name in ("__init__.py", "release.py"):
        shutil.copy2(tree / "contracts" / name, contracts / name)
    shim = source / "triton"
    shim.mkdir()
    (shim / "__init__.py").write_text(
        '"""Shadow the bundled triton, whose native library faults on import on this image.\n'
        "\n"
        "torch probes triton with ``has_triton_package`` and treats ImportError as triton\n"
        "absent, the normal aarch64 path, so raising here keeps lazy ``torch._dynamo``\n"
        "imports (tensordict reaches one through ``torch.compiler.allow_in_graph``) from\n"
        "loading the faulting library (terrain_isaac_smoke_20261031_007/_008, probe_002).\n"
        '"""\n'
        "\n"
        'raise ImportError("triton disabled: the bundled native library faults on this image")\n'
    )
    write_runtime(source)
    save(
        source / "FREEZE_SHA256.json",
        {
            p.relative_to(source).as_posix(): sha(p)
            for p in sorted(source.rglob("*"))
            if p.is_file()
        },
    )
    freeze = sha(source / "FREEZE_SHA256.json")
    module, selection = ENTRY_POINTS[mode]
    binding = {
        "schema": "hexapod_locomotion_launch_v1",
        "root_review_complete": False,
        "module": module,
        "mode": mode,
        "source": str(remote_root / "source"),
        "output": str(remote_root / "run"),
        "asset": declared["asset"],
        "geometry_source": declared["geometry_source"],
        "prior": str(Path(declared["stance"]).parent),
        "input_files": declared["input_files"],
        "extra_mounts": declared.get("extra_mounts", []),
        "source_freeze_sha256": freeze,
        "allocation_profile": allocation_profile,
        "max_wall_seconds": max_wall_seconds,
        "max_seconds": max_wall_seconds + CLEANUP_MARGIN_SECONDS,
        "stage2_complete": False,
        "physical_admission": False,
        "command_args": [
            *selection,
            "--asset",
            "/asset",
            "--model",
            "/asset/source/model.json",
            "--geometry",
            "/geometry_source/geometry/geometry.json",
            "--geometry-extrema",
            "/geometry_source/geometry/geometry_extrema.npz",
            "--stance",
            "/prior/" + Path(declared["stance"]).name,
            "--num-envs",
            str(num_envs),
            "--source-freeze-sha256",
            freeze,
            "--max-wall-seconds",
            str(max_wall_seconds),
            "--headless",
            "--device",
            "cuda:0",
        ],
    }
    if learner != "ppo":
        binding["command_args"] += ["--learner", learner, "--networks", networks]
    if allocation_profile != "standard":
        binding["command_args"] += ["--allocation-profile", allocation_profile]
    if mode != "diagnostic":
        binding["command_args"] += ["--standing-admission", "/admission/admission.json"]
    if mode in ("train", "evaluate"):
        binding["command_args"] += ["--updates", str(updates), "--seed", str(seed)]
    if extended_updates:
        binding["command_args"] += ["--extended-updates"]
    if mode == "probe":
        binding["command_args"] += ["--seed", str(seed)]
    if mode in ("train", "evaluate") and learner == "ppo":
        # The flags below name each option that differs from its default, so the trainer applies no recipe.
        binding["command_args"] += ["--recipe", "none"]
    if action_mean != "unbounded":
        binding["command_args"] += ["--action-mean", action_mean]
    if observation_normalization != "empirical":
        binding["command_args"] += ["--observation-normalization", observation_normalization]
    for key, value in selected_options.items():
        binding["command_args"] += ["--" + key.replace("_", "-"), str(value)]
    if logger == "wandb":
        binding["command_args"] += [
            "--logger",
            "wandb",
            "--wandb-project",
            wandb_project,
            "--wandb-mode",
            wandb_mode,
        ]
    if mode == "evaluate":
        binding["command_args"] += ["--eval-scope", eval_scope]
    if video_case is not None:
        binding["command_args"] += ["--video-case", video_case]
    if episode_seconds != 20.0:
        binding["command_args"] += ["--episode-seconds", str(episode_seconds)]
    if anchor_weight is not None:
        binding["command_args"] += ["--anchor-weight", str(anchor_weight)]
    if checkpoint is not None:
        binding["extra_mounts"].append([str(checkpoint.parent), "/checkpoint"])
        binding["input_files"][str(checkpoint)] = checkpoint_sha
        binding["input_files"][str(checkpoint.with_suffix(".json"))] = checkpoint_declaration_sha
        binding["command_args"] += [
            "--checkpoint",
            "/checkpoint/" + checkpoint.name,
            "--checkpoint-sha256",
            checkpoint_sha,
        ]
    if experiment is not None:
        binding["command_args"] += ["--experiment", experiment]
    if terrain is not None:
        # The spec takes its own read-only mount, like a checkpoint, and the launcher verifies
        # its recorded hash before execution.
        binding["extra_mounts"].append([str(terrain.parent), "/terrain"])
        binding["input_files"][str(terrain)] = terrain_sha
        binding["command_args"] += ["--terrain", "/terrain/" + terrain.name]
        if terrain_mesh_resolution is not None:
            binding["command_args"] += [
                "--terrain-mesh-resolution",
                str(terrain_mesh_resolution),
            ]
        if terrain_spawn:
            binding["command_args"] += ["--terrain-spawn"]
    if initialize_from is not None:
        # Training mounts no evaluation checkpoint, so the starting checkpoint and its declaration take
        # the launcher's read-only /checkpoint mount.
        binding["extra_mounts"].append([str(initialize_from.parent), "/checkpoint"])
        binding["input_files"][str(initialize_from)] = initialize_from_sha
        binding["input_files"][str(initialize_from.with_suffix(".json"))] = (
            initialize_from_declaration_sha
        )
        binding["command_args"] += [
            "--initialize-from",
            "/checkpoint/" + initialize_from.name,
            "--initialize-from-sha256",
            initialize_from_sha,
            "--initialize-from-declaration-sha256",
            initialize_from_declaration_sha,
        ]
        if initialize_rate != "configured":
            binding["command_args"] += ["--initialize-rate", initialize_rate]
    if all(compatibility):
        binding["extra_mounts"].append([str(receipt.parent), "/compatibility"])
        binding["input_files"][str(receipt)] = checkpoint_compatibility_sha256
        binding["command_args"] += [
            "--checkpoint-compatibility",
            "/compatibility/" + receipt.name,
            "--checkpoint-compatibility-sha256",
            checkpoint_compatibility_sha256,
        ]
    save(output / "binding.json", binding)
    reward_version = "4"
    if experiment is not None:
        from locomotion.omniwalk.destination_task import REWARD_VERSION as reward_version
    save(
        output / "PACK.json",
        {
            "remote_root": str(remote_root),
            "source_freeze_sha256": freeze,
            "binding_sha256": sha(output / "binding.json"),
            "mode": mode,
            "seed": seed,
            "reward_version": reward_version,
            "reward_task": "v4_yaw_gate" if experiment is None else experiment,
            "action_mean": action_mean,
            "learner": learner,
            "networks": networks,
            "observation_normalization": observation_normalization,
            **selected_options,
            **({} if video_case is None else {"video_case": video_case}),
            **({} if episode_seconds == 20.0 else {"episode_seconds": episode_seconds}),
            **({} if anchor_weight is None else {"anchor_weight": anchor_weight}),
            **({"extended_updates": True} if extended_updates else {}),
            **({} if experiment is None else {"experiment": experiment}),
            **(
                {}
                if initialize_from is None
                else {
                    "initialize_from": str(initialize_from),
                    "initialize_from_sha256": initialize_from_sha,
                    "initialize_from_declaration_sha256": initialize_from_declaration_sha,
                    "initialize_rate": initialize_rate,
                }
            ),
            "allocation_profile": allocation_profile,
            "max_wall_seconds": max_wall_seconds,
            "updates": updates,
            "stage2_complete": False,
            "files": {
                p.relative_to(output).as_posix(): sha(p)
                for p in sorted(output.rglob("*"))
                if p.is_file()
            },
        },
    )
    return binding


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--remote-root", required=True)
    parser.add_argument("--inputs", type=Path)
    parser.add_argument(
        "--mode",
        choices=("diagnostic", "train", "evaluate", "probe", "replay"),
        required=True,
        help="probe runs the noise probe (diagnose.py --noise-probe); diagnostic runs the standing "
        "diagnostic.",
    )
    parser.add_argument("--num-envs", type=int)
    parser.add_argument("--eval-scope", choices=["focus", "probes", "full"], default="focus")
    parser.add_argument("--updates", type=int, default=512)
    parser.add_argument(
        "--extended-updates",
        action="store_true",
        help="Accept up to 5000 training updates; without this opt-in the limit stays 2000.",
    )
    parser.add_argument("--seed", type=int, default=20260917)
    parser.add_argument("--checkpoint", type=Path)
    parser.add_argument("--checkpoint-sha")
    parser.add_argument("--checkpoint-declaration-sha")
    parser.add_argument("--checkpoint-compatibility", type=Path)
    parser.add_argument("--checkpoint-compatibility-sha256")
    parser.add_argument("--logger", choices=["tensorboard", "wandb"], default="tensorboard")
    parser.add_argument("--wandb-project")
    parser.add_argument("--wandb-mode", choices=["offline", "online"], default="offline")
    parser.add_argument("--learner", choices=["ppo", "amp"], default="ppo")
    parser.add_argument("--networks", choices=["mlp", "paper"], default="mlp")
    parser.add_argument(
        "--recipe",
        choices=[*RECIPES, "none"],
        default="c48_long",
        help="Learner options of PPO training and evaluation (%(default)s by default) from omniwalk/agents/"
        "rsl_rl_ppo_cfg.py; option flags replace its values. Other runs and the recipe none keep the defaults.",
    )
    parser.add_argument("--action-mean", choices=["unbounded", "tanh"])
    parser.add_argument("--observation-normalization", choices=["empirical", "none"])
    parser.add_argument("--observation-scaling", choices=["none", "fixed"])
    parser.add_argument("--command-segments", choices=["continuous", "bootstrap"])
    parser.add_argument("--learning-rate-max", type=float)
    parser.add_argument("--learning-rate-min", type=float)
    parser.add_argument("--action-std", type=float)
    parser.add_argument("--action-noise-correlation", type=float)
    parser.add_argument("--action-std-final", type=float)
    parser.add_argument("--action-std-decay-updates", type=int)
    parser.add_argument("--steps-per-env", type=int)
    parser.add_argument("--anchor-weight", type=float)
    parser.add_argument("--gait-clock", type=int)
    parser.add_argument("--action-smoothing", choices=["none", "mean2"])
    parser.add_argument("--velocity-noise", type=float)
    parser.add_argument("--experiment", choices=sorted(EXPERIMENT_INPUTS))
    parser.add_argument("--terrain", type=Path)
    parser.add_argument("--terrain-sha")
    parser.add_argument("--terrain-mesh-resolution", type=float)
    parser.add_argument("--terrain-spawn", action="store_true")
    parser.add_argument("--initialize-from", type=Path)
    parser.add_argument("--initialize-from-sha")
    parser.add_argument("--initialize-from-declaration-sha")
    parser.add_argument(
        "--initialize-rate", choices=["configured", "checkpoint"], default="configured"
    )
    parser.add_argument("--episode-seconds", type=float, default=20.0)
    parser.add_argument("--video-case")
    parser.add_argument("--allocation-profile", choices=ALLOCATION_PROFILES, default="standard")
    parser.add_argument("--max-wall-seconds", type=int, default=6200)
    args = parser.parse_args()
    print(json.dumps(prepare(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
