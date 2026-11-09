"""Load and verify a runtime bundle, and evaluate its actor with numpy."""

import hashlib
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

SCHEMA = "hexapod_runtime_bundle_v1"


@dataclass(frozen=True)
class Bundle:
    """The exported actor and every constant the runtime reads, as float64 arrays where numeric."""

    record: dict
    layers: tuple
    neutral: np.ndarray
    lower: np.ndarray
    upper: np.ndarray
    kp: np.ndarray
    kd: np.ndarray
    torque_limit: np.ndarray
    joint_xyz: np.ndarray
    joint_quaternion: np.ndarray
    toe_local: np.ndarray

    @property
    def inputs(self):
        return self.record["inputs"]

    @property
    def action(self):
        return self.record["action"]

    @property
    def robot(self):
        return self.record["robot"]

    @property
    def joint_names(self):
        return tuple(self.robot["joint_names"])


def load(path):
    """The bundle in directory ``path`` (or its ``bundle.json``), refused on any identity mismatch."""
    path = Path(path)
    directory = path.parent if path.is_file() else path
    record = json.loads((directory / "bundle.json").read_text())
    if record.get("schema") != SCHEMA:
        raise ValueError(f"Unknown runtime bundle schema {record.get('schema')!r}")
    weights = directory / "weights.npz"
    if hashlib.sha256(weights.read_bytes()).hexdigest() != record["identity"]["weights_sha256"]:
        raise ValueError("Runtime weights differ from the bundle's recorded sha256")
    with np.load(weights, allow_pickle=False) as data:
        count = record["actor"]["layers"]
        layers = tuple(
            (data[f"w{i}"].astype(np.float64), data[f"b{i}"].astype(np.float64))
            for i in range(count)
        )
    if layers[0][0].shape[1] != record["actor"]["input_width"] or layers[-1][0].shape[0] != 18:
        raise ValueError("Runtime actor shapes differ from the bundle record")
    robot = record["robot"]

    def vector(key, shape):
        value = np.asarray(robot[key], dtype=np.float64)
        if value.shape != shape or not np.all(np.isfinite(value)):
            raise ValueError(f"Invalid bundle robot.{key}")
        return value

    bundle = Bundle(
        record=record,
        layers=layers,
        neutral=vector("neutral_rad", (18,)),
        lower=vector("lower_rad", (18,)),
        upper=vector("upper_rad", (18,)),
        kp=vector("kp_nm_per_rad", (18,)),
        kd=vector("kd_nm_s_per_rad", (18,)),
        torque_limit=vector("torque_limit_nm", (18,)),
        joint_xyz=vector("joint_xyz_m", (3, 6, 3)),
        joint_quaternion=vector("joint_quaternion_xyzw", (3, 6, 4)),
        toe_local=vector("toe_local_m", (6, 3)),
    )
    if np.any(bundle.neutral < bundle.lower) or np.any(bundle.neutral > bundle.upper):
        raise ValueError("Bundle neutral pose violates its joint limits")
    return bundle


def actor(layers, inputs):
    """The deterministic action: ELU hidden layers, then tanh (the BoundedMeanGaussian mean)."""
    hidden = np.asarray(inputs, dtype=np.float64)
    for weight, bias in layers[:-1]:
        hidden = weight @ hidden + bias
        hidden = np.where(hidden > 0, hidden, np.expm1(np.minimum(hidden, 0.0)))
    weight, bias = layers[-1]
    return np.tanh(weight @ hidden + bias)
