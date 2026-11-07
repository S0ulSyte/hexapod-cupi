"""The curriculum results builder rejects foreign identities and tampered captures before publishing."""

import hashlib
import importlib.util
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
SCRIPT = ROOT / "site/assets/destination_curriculum_20261026_001/build_results.py"
CHECKPOINT = "ab" * 32
SOURCE = "cd" * 32


def load_script():
    spec = importlib.util.spec_from_file_location("build_results", SCRIPT)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def write_suite(folder, *, seed=None, checkpoint=CHECKPOINT, source=SOURCE, tamper=False):
    module = load_script()
    batch = Path(folder, "evaluation", "batch_000")
    batch.mkdir(parents=True)
    n = 10
    trace = batch / "control_trace.npz"
    np.savez(
        trace,
        root_pose_xyzw=np.tile((0.0, 0.0, 0.1, 0.0, 0.0, 0.0, 1.0), (n, 1, 1)),
        command=np.zeros((n, 1, 3)),
        terminated=np.zeros((n, 1), dtype=bool),
    )
    digest = hashlib.sha256(trace.read_bytes()).hexdigest()
    if tamper:
        trace.write_bytes(trace.read_bytes() + b"x")
    report = {
        "assigned_case_ids": ["learning:probe_stand"],
        "seed": module.SCOPE_SEED if seed is None else seed,
        "checkpoint_sha256": checkpoint,
        "source_sha256": source,
        "results": [{"pass": True, "failed_bounds": []}],
        "files": {"control_trace.npz": digest},
    }
    (batch / "report.json").write_text(json.dumps(report))
    return module


class BuildResultsTests(unittest.TestCase):
    def test_a_clean_suite_publishes_its_verified_bindings(self):
        with tempfile.TemporaryDirectory() as folder:
            module = write_suite(folder)
            suite = module.speed_suite(folder, CHECKPOINT, set())
        self.assertEqual(suite["checkpoint_sha256"], CHECKPOINT)
        self.assertEqual(suite["seed"], module.SCOPE_SEED)
        self.assertEqual(suite["cases"]["probe_stand"]["end_gap_mm"], 0.0)

    def test_a_foreign_seed_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            module = write_suite(folder, seed=999)
            with self.assertRaisesRegex(ValueError, "seed 999"):
                module.speed_suite(folder, CHECKPOINT, set())

    def test_a_foreign_checkpoint_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            module = write_suite(folder, checkpoint="ef" * 32)
            with self.assertRaisesRegex(ValueError, "not the intended"):
                module.speed_suite(folder, CHECKPOINT, set())

    def test_a_tampered_capture_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            module = write_suite(folder, tamper=True)
            with self.assertRaisesRegex(ValueError, "report-declared hash"):
                module.speed_suite(folder, CHECKPOINT, set())

    def test_a_second_evaluation_source_is_rejected(self):
        with tempfile.TemporaryDirectory() as folder:
            module = write_suite(folder, source="12" * 32)
            with self.assertRaisesRegex(ValueError, "source differs"):
                module.speed_suite(folder, CHECKPOINT, {SOURCE})
