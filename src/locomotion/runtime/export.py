"""Export a ``terrain_v2_odom`` checkpoint and the approved robot model to a runtime bundle.

The bundle is a directory with ``bundle.json`` (identity, robot constants, input and action
constants) and ``weights.npz`` (the actor MLP), and the json records the sha256 of the weights.
Every constant is read from the module or file that training and evaluation read, so the
numpy runtime carries no copied numbers of its own. Only this module needs torch:

    uv run python -m locomotion.runtime.export --checkpoint CKPT.pt --output DIR
"""

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

import numpy as np

SCHEMA = "hexapod_runtime_bundle_v1"
ROOT = Path(__file__).resolve().parents[3]
ROBOT = ROOT / "robot/hexapod_mkii_updated_v1"
MODEL = ROBOT / "model_rs05_mass_corrected.json"
STANCE = ROBOT / "stance.json"
# The one actor layout the runtime rebuilds; any other recorded option is refused.
REQUIRED_WRAPPER = {
    "destination_input": "terrain_v2_odom",
    "observation_scaling": "fixed",
    "gait_clock_scaling": "speed_v1",
    "action_smoothing": "mean2",
}
REQUIRED_ACTOR = {
    "class_name": "MLPModel",
    "activation": "elu",
    "obs_normalization": False,
}
BOUNDED_DISTRIBUTION = "locomotion.rl.ppo:BoundedMeanGaussian"


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _source_revision():
    try:
        return subprocess.run(
            ["git", "-C", str(ROOT), "rev-parse", "HEAD"],
            capture_output=True,
            text=True,
            check=True,
        ).stdout.strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _checked_identity(checkpoint):
    identity = (checkpoint.get("infos") or {}).get("identity") or {}
    config = identity.get("ppo_config") or {}
    wrapper = config.get("environment_wrapper") or {}
    for key, value in REQUIRED_WRAPPER.items():
        if wrapper.get(key) != value:
            raise ValueError(
                f"The runtime rebuilds {key}={value!r}; checkpoint has {wrapper.get(key)!r}"
            )
    actor = config.get("actor") or {}
    for key, value in REQUIRED_ACTOR.items():
        if actor.get(key) != value:
            raise ValueError(f"The runtime rebuilds actor {key}={value!r}; got {actor.get(key)!r}")
    if (actor.get("distribution_cfg") or {}).get("class_name") != BOUNDED_DISTRIBUTION:
        raise ValueError("The runtime rebuilds the tanh-bounded actor mean alone")
    if config.get("obs_groups", {}).get("actor") != ["policy"]:
        raise ValueError("The runtime rebuilds the single 'policy' actor group")
    if wrapper.get("gait_clock", 0) <= 0:
        raise ValueError("The runtime needs the gait clock")
    if (
        (identity.get("task_definition") or {})
        .get("reward", {})
        .get("config", {})
        .get("schedule_mirror_turns")
    ):
        raise ValueError("The runtime does not rebuild the mirrored tripod schedule")
    physics = identity.get("config") or {}
    return identity, wrapper, actor, physics


def _weights(checkpoint, hidden):
    state = checkpoint["actor_state_dict"]
    layers = []
    index = 0
    while f"mlp.{index}.weight" in state:
        layers.append(
            (
                state[f"mlp.{index}.weight"].detach().cpu().numpy().astype(np.float32),
                state[f"mlp.{index}.bias"].detach().cpu().numpy().astype(np.float32),
            )
        )
        index += 2
    widths = [w.shape[0] for w, _ in layers]
    if widths != [*hidden, 18]:
        raise ValueError(f"Actor layer widths {widths} differ from hidden dims {hidden} + 18")
    return layers


def robot_constants():
    """Neutral pose, limits, forward-kinematics frames and motor gains in JOINT_NAMES order."""
    from locomotion.robots.hexapod import JOINT_NAMES, KD, LEGS, MODEL_SHA256
    from locomotion.robots.rs05 import RATED_TORQUE_NM, STIFFNESS_NM_PER_RAD
    from locomotion.surrogate.env import MAX_JOINT_SPEED

    if sha256(MODEL) != MODEL_SHA256:
        raise ValueError("The approved model json differs from its pinned hash")
    model = json.loads(MODEL.read_text())
    stance = json.loads(STANCE.read_text())
    if list(stance["joint_names"]) != list(JOINT_NAMES):
        raise ValueError("Stance joint order differs from JOINT_NAMES")
    named = {j["name"]: j for j in model["joints"]}
    for name in JOINT_NAMES:
        if list(named[name]["axis"]) != [0.0, 0.0, 1.0]:
            raise ValueError(f"{name}: forward kinematics assumes a +Z joint axis")
    levels = ("coxa_yaw", "femur_pitch", "tibia_pitch")
    return {
        "joint_names": list(JOINT_NAMES),
        "legs": list(LEGS),
        "neutral_rad": [float(x) for x in stance["nominal_joint_position_rad"]],
        "lower_rad": [float(named[n]["lower"]) for n in JOINT_NAMES],
        "upper_rad": [float(named[n]["upper"]) for n in JOINT_NAMES],
        "joint_xyz_m": [[named[f"{leg}_{j}"]["xyz"] for leg in LEGS] for j in levels],
        "joint_quaternion_xyzw": [
            [named[f"{leg}_{j}"]["quaternion_xyzw"] for leg in LEGS] for j in levels
        ],
        "toe_local_m": stance["toe_local_points_m"],
        "kp_nm_per_rad": [STIFFNESS_NM_PER_RAD] * 18,
        "kd_nm_s_per_rad": list(KD),
        "torque_limit_nm": [RATED_TORQUE_NM] * 18,
        "max_joint_speed_rad_s": MAX_JOINT_SPEED,
        "model_sha256": MODEL_SHA256,
        "stance_sha256": sha256(STANCE),
    }


def input_constants(wrapper):
    """Input scales and the clock, ghost, contact and termination constants training used."""
    from locomotion.omniwalk.mdp import destination
    from locomotion.omniwalk.mdp.commands import GAIT_CLOCK_SCALINGS
    from locomotion.omniwalk.mdp.footfall import FOOTFALL_FEATURES, PROFILE_FEATURES
    from locomotion.omniwalk.mdp.gait import PHASE_TOLERANCE
    from locomotion.omniwalk.mdp.odometry import CONTACT_FORCE_N
    from locomotion.omniwalk.mdp.rewards import TRIPOD_A
    from locomotion.rl.wrapper import COMMAND_SCALES, FRAME_SCALES

    speeds, periods, swings = GAIT_CLOCK_SCALINGS[wrapper["gait_clock_scaling"]]
    return {
        "frame_scales": list(FRAME_SCALES),
        "frames": 5,
        "command_scales": list(COMMAND_SCALES),
        "gait_speeds_mps": list(speeds),
        "gait_periods_controls": list(periods),
        "gait_swing_share": list(swings),
        "phase_tolerance": PHASE_TOLERANCE,
        "tripod_a": list(TRIPOD_A),
        # The schedule-inferred contact fallback (gaitcontact diagnostic, 47/96 on od1_u1200).
        "schedule_contact_swing_share": 0.4,
        "control_dt_s": destination.CONTROL_DT_S,
        "distance_scale_m": destination.DISTANCE_SCALE_M,
        "velocity_scale_mps": destination.VELOCITY_SCALE_MPS,
        "longest_command_controls": destination.LONGEST_COMMAND_CONTROLS,
        "ghost_features": destination.GHOST_V11_FEATURES,
        "terrain_features": FOOTFALL_FEATURES + PROFILE_FEATURES,
        "contact_force_n": CONTACT_FORCE_N,
        # The simulator ends an episode when the body tilts past this (surrogate/env.py).
        "tilt_limit_rad": 0.85,
    }


def export(checkpoint_path, output):
    import torch

    checkpoint_path, output = Path(checkpoint_path), Path(output)
    output.mkdir(parents=True, exist_ok=False)
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    identity, wrapper, actor, physics = _checked_identity(checkpoint)
    layers = _weights(checkpoint, list(actor["hidden_dims"]))
    inputs = input_constants(wrapper)
    width = layers[0][0].shape[1]
    expected = 42 * inputs["frames"] + 3 + 18 + 2 + inputs["ghost_features"]
    expected += inputs["terrain_features"]
    if width != expected:
        raise ValueError(f"Actor input width {width} differs from the rebuilt layout {expected}")
    arrays = {}
    for index, (weight, bias) in enumerate(layers):
        arrays[f"w{index}"], arrays[f"b{index}"] = weight, bias
    weights = output / "weights.npz"
    np.savez(weights, **arrays)
    declaration = checkpoint_path.with_suffix(".json")
    bundle = {
        "schema": SCHEMA,
        "identity": {
            "checkpoint": checkpoint_path.name,
            "checkpoint_sha256": sha256(checkpoint_path),
            "declaration_sha256": sha256(declaration) if declaration.exists() else None,
            "checkpoint_update": int(checkpoint.get("iter", -1)),
            "task": identity.get("task"),
            "export_source_revision": _source_revision(),
            "weights_sha256": sha256(weights),
        },
        "actor": {
            "layers": len(layers),
            "input_width": width,
            "activation": "elu",
            "output": "tanh",
        },
        "action": {
            "scale_rad": float(physics["action_scale_rad"]),
            "slew_rad": float(physics["target_slew_rad"]),
            "smoothing": wrapper["action_smoothing"],
        },
        "inputs": inputs,
        "robot": robot_constants(),
    }
    (output / "bundle.json").write_text(json.dumps(bundle, indent=1) + "\n")
    return output / "bundle.json"


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args(argv)
    print(export(args.checkpoint, args.output))


if __name__ == "__main__":
    main()
