"""Check the prepared pack entry point, modes, reward record and source tampering without Isaac."""

import json
import tempfile
import unittest
from pathlib import Path

from locomotion.spark.prepare import prepare
from locomotion.spark.sources import verify_runtime

ROOT = Path(__file__).resolve().parents[3]
REMOTE = "/srv/cupi/hexapod/runs/jcc463/pack_binding_fixture"


class PackBindingTests(unittest.TestCase):
    def test_pack_uses_one_backend_and_keeps_review_gate(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            binding = prepare(root / "pack", REMOTE, mode="diagnostic")
            self.assertNotIn("--backend", binding["command_args"])
            self.assertFalse(binding["root_review_complete"])
            self.assertEqual(binding["module"], "locomotion.scripts.diagnose")
            self.assertNotIn("--noise-probe", binding["command_args"])
            manifest = verify_runtime(root / "pack/source")
            self.assertFalse(any("/tripod/" in name for name in manifest["files"]))
            self.assertFalse((root / "pack/source/locomotion/hexapod").exists())
            shim = (root / "pack/source/triton/__init__.py").read_text()
            self.assertIn('raise ImportError("triton disabled', shim)
            nested = root / "pack/source/locomotion/robots/rs05.py"
            nested.write_text(nested.read_text() + "\n# tamper\n")
            with self.assertRaisesRegex(ValueError, "manifest differs"):
                verify_runtime(root / "pack/source")

    def test_retired_modes_fail_before_creating_pack(self):
        with tempfile.TemporaryDirectory() as directory:
            for mode in ("tripod", "throughput"):
                output = Path(directory) / mode
                with (
                    self.subTest(mode=mode),
                    self.assertRaisesRegex(ValueError, "Invalid native allocation"),
                ):
                    prepare(output, REMOTE, mode=mode)
                self.assertFalse(output.exists())

    def test_active_reward_retires_version_switches_and_requires_matching_clock(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "pack"
            for option, value in (
                ("reward_version", "2"),
                ("reward_task", "v4"),
                ("reward_options", "yaw_gate_floor=0"),
            ):
                with self.subTest(option=option), self.assertRaises(TypeError):
                    prepare(output, REMOTE, **{option: value})
                self.assertFalse(output.exists())
            with self.assertRaisesRegex(ValueError, "60-control"):
                prepare(output, REMOTE, gait_clock=40)
            self.assertFalse(output.exists())
            binding = prepare(output, REMOTE)
            self.assertFalse(
                any(value.startswith("--reward-") for value in binding["command_args"])
            )
            record = json.loads((output / "PACK.json").read_text())
            self.assertEqual(
                (record["reward_version"], record["reward_task"]), ("4", "v4_yaw_gate")
            )

    def test_pack_rejects_the_removed_hybrid_bc_option(self):
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "rejected"
            with self.assertRaises(TypeError):
                prepare(output, REMOTE, bc_warmstart=True)
            self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
