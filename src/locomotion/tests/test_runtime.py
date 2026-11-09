"""The robot runtime: numpy-only imports, the controller's safety states, and replay parity of
its inputs, actions and targets against the evaluator's policy path on the v70 checkpoint."""

import json
import math
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
CHECKPOINT = ROOT / "records/curriculum_20261109/od1/checkpoint_update001200.pt"
# Stand, then forward, turn in place, a curve, a sidestep, a diagonal and a stop: every command
# change restarts the ghost, and the turns exercise the heading path.
SCRIPT = (
    ((0.0, 0.0, 0.0), 25),
    ((0.1, 0.0, 0.0), 100),
    ((0.0, 0.0, 0.3), 75),
    ((0.1, 0.0, 0.2), 75),
    ((0.0, 0.08, 0.0), 60),
    ((0.075, -0.05, -0.15), 60),
    ((0.0, 0.0, 0.0), 40),
)


def _bundle(directory):
    from locomotion.runtime.bundle import load
    from locomotion.runtime.export import export

    export(CHECKPOINT, Path(directory) / "bundle")
    return load(Path(directory) / "bundle")


def _rollout(directory):
    """The evaluator's policy path on one surrogate robot; per control the sensor reading it
    implies, the actor inputs and action it computed and the held target the robot received."""
    import torch
    from tensordict import TensorDict

    from locomotion.omniwalk.mdp.destination import planar_pose
    from locomotion.omniwalk.mdp.odometry import observed
    from locomotion.rl.ppo import load_actor
    from locomotion.runtime.inputs import Reading
    from locomotion.surrogate.env import (
        DEFAULT_GEOMETRY,
        DEFAULT_MODEL,
        DEFAULT_STANCE,
        SurrogateConfig,
        SurrogateEnv,
    )
    from locomotion.surrogate.evaluate import DeterministicPolicy
    from locomotion.surrogate.train import wrapper_options

    checkpoint = torch.load(CHECKPOINT, map_location="cpu", weights_only=False)
    config = checkpoint["infos"]["identity"]["ppo_config"]
    cfg = SurrogateConfig(num_envs=1, seed=20261109, episode_seconds=60.0)
    env = SurrogateEnv(
        cfg,
        None,
        DEFAULT_MODEL,
        DEFAULT_GEOMETRY,
        Path(directory) / "native",
        reference_metadata=json.loads(DEFAULT_STANCE.read_text()),
        threads=0,
    )
    env.reset()
    blank = TensorDict({"policy": torch.zeros(1, 278)}, batch_size=[1])
    actor, _ = load_actor(checkpoint, config, blank, "cpu")
    policy = DeterministicPolicy(actor, env, wrapper_options(config))
    seen = {}
    build = policy.inputs

    def recorded(observation):
        seen["inputs"] = build(observation)
        return seen["inputs"]

    policy.inputs = recorded
    rows = []
    for command, controls in SCRIPT:
        for _ in range(controls):
            env.commands[:] = torch.tensor(command)
            _, _, contact = observed(env)
            state = env.current
            reading = Reading(
                gyro_body=state["angular"][0].double().numpy(),
                gravity_body=state["gravity"][0].double().numpy(),
                joint_position=state["q"][0].double().numpy(),
                joint_velocity=state["dq"][0].double().numpy(),
                contact=None if contact is None else contact[0].numpy(),
                yaw_rad=float(planar_pose(state["root"])[0, 2]),
            )
            action = policy(env.observations(state)["obs"])
            env.step(action)
            rows.append(
                (
                    reading,
                    command,
                    seen["inputs"][0].double().numpy(),
                    action[0].double().numpy(),
                    env.held[0].double().numpy(),
                )
            )
    return rows


class RuntimeImports(unittest.TestCase):
    def test_runtime_loads_without_simulator_or_torch(self):
        code = (
            "import sys, locomotion.runtime.controller, locomotion.runtime.hardware;"
            "print([m for m in ('torch', 'mujoco', 'isaaclab', 'tensordict', 'rsl_rl')"
            " if m in sys.modules])"
        )
        result = subprocess.run(
            [sys.executable, "-c", code], capture_output=True, text=True, check=True
        )
        self.assertEqual(result.stdout.strip(), "[]")


@unittest.skipUnless(CHECKPOINT.exists(), "the v70 checkpoint snapshot is not in this checkout")
class RuntimeReplay(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.directory = tempfile.TemporaryDirectory()
        cls.bundle = _bundle(cls.directory.name)
        cls.rows = _rollout(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def replay(self, *, imu_yaw=True):
        from locomotion.runtime.inputs import PolicyLoop

        loop = PolicyLoop(self.bundle)
        errors = {"inputs": 0.0, "ghost": 0.0, "action": 0.0, "target": 0.0}
        for reading, command, inputs, action, target in self.rows:
            if not imu_yaw:
                reading = type(reading)(**{**reading.__dict__, "yaw_rad": None})
            got = loop.step(reading, command)
            trace = loop.trace
            errors["inputs"] = max(errors["inputs"], float(np.abs(trace["inputs"] - inputs).max()))
            ghost = slice(233, 242)
            errors["ghost"] = max(
                errors["ghost"], float(np.abs(trace["inputs"][ghost] - inputs[ghost]).max())
            )
            errors["action"] = max(errors["action"], float(np.abs(trace["action"] - action).max()))
            errors["target"] = max(errors["target"], float(np.abs(got - target).max()))
        return errors

    def test_matches_the_evaluator_with_imu_heading(self):
        errors = self.replay()
        print("\nruntime vs evaluator (IMU yaw):", errors)
        self.assertLess(errors["inputs"], 1e-3)
        self.assertLess(errors["action"], 1e-4)
        self.assertLess(errors["target"], 1e-5)

    def test_gyro_integrated_heading_stays_close(self):
        errors = self.replay(imu_yaw=False)
        print("\nruntime vs evaluator (gyro-integrated yaw):", errors)
        self.assertLess(errors["ghost"], 0.1)
        self.assertLess(errors["action"], 0.05)


class ControllerSafety(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        if not CHECKPOINT.exists():
            raise unittest.SkipTest("the v70 checkpoint snapshot is not in this checkout")
        cls.directory = tempfile.TemporaryDirectory()
        cls.bundle = _bundle(cls.directory.name)

    @classmethod
    def tearDownClass(cls):
        cls.directory.cleanup()

    def reading(self, q=None, gravity=(0.0, 0.0, -1.0)):
        from locomotion.runtime.inputs import Reading

        return Reading(
            gyro_body=np.zeros(3),
            gravity_body=np.array(gravity),
            joint_position=self.bundle.neutral.copy() if q is None else q,
            joint_velocity=np.zeros(18),
            contact=np.ones(6, dtype=bool),
        )

    def controller(self):
        from contracts.command import CommandEnvelope
        from locomotion.runtime.controller import Controller

        envelope = CommandEnvelope((-0.15, 0.15), (-0.1, 0.1), (-0.4, 0.4))
        return Controller(self.bundle, envelope)

    def ready(self):
        from locomotion.runtime.controller import Mode

        control = self.controller()
        control.enable()
        t = 0.0
        while control.mode is not Mode.READY:
            t += 0.02
            control.step(self.reading(), t)
            self.assertLess(t, 5.0)
        return control, t

    def test_stand_up_ramps_at_the_stand_rate_then_holds_neutral(self):
        from locomotion.runtime.controller import Mode

        control = self.controller()
        control.enable()
        crouched = self.bundle.neutral + 0.2 * np.sign(self.bundle.upper)
        first = control.step(self.reading(crouched), 0.02)
        self.assertEqual(first.mode, Mode.STANDING_UP)
        start = np.clip(crouched, self.bundle.lower, self.bundle.upper)
        self.assertLessEqual(np.abs(first.position - start).max(), 0.01 + 1e-12)
        control, _ = self.ready()
        np.testing.assert_allclose(control.target, self.bundle.neutral)

    def test_disabled_and_fault_output_damping_only(self):
        from locomotion.runtime.controller import Mode

        control = self.controller()
        out = control.step(self.reading(), 0.02)
        self.assertEqual(out.mode, Mode.DISABLED)
        self.assertTrue(np.all(out.kp == 0))
        control, t = self.ready()
        control.start()
        tilted = (math.sin(1.0), 0.0, -math.cos(1.0))
        out = control.step(self.reading(gravity=tilted), t + 0.02)
        self.assertEqual(out.mode, Mode.FAULT)
        self.assertTrue(np.all(out.kp == 0))
        out = control.step(self.reading(), t + 0.04)
        self.assertEqual(out.mode, Mode.FAULT, "a fault latches until disable()")

    def test_late_or_non_finite_readings_fault(self):
        from locomotion.runtime.controller import Mode

        control, t = self.ready()
        control.start()
        self.assertEqual(control.step(self.reading(), t + 0.2).mode, Mode.FAULT)
        control, t = self.ready()
        q = self.bundle.neutral.copy()
        q[4] = math.nan
        self.assertEqual(control.step(self.reading(q), t + 0.02).mode, Mode.FAULT)

    def test_expired_command_reads_as_the_stop_command(self):
        control, t = self.ready()
        control.start()
        control.set_command(0.1, 0.0, 0.0, t)
        control.step(self.reading(), t + 0.02)
        np.testing.assert_allclose(control.loop.command, (0.1, 0.0, 0.0))
        control.step(self.reading(), t + 0.04)
        control.last_time = t + 0.5
        control.step(self.reading(), t + 0.52)
        np.testing.assert_allclose(control.loop.command, (0.0, 0.0, 0.0))

    def test_commands_are_clamped_and_non_finite_refused(self):
        control = self.controller()
        control.set_command(0.5, -0.5, 2.0, 0.0)
        np.testing.assert_allclose(control.command, (0.15, -0.1, 0.4))
        with self.assertRaises(ValueError):
            control.set_command(math.inf, 0.0, 0.0, 0.0)


class HardwareMapping(unittest.TestCase):
    def test_joint_map_round_trips_and_imu_gravity(self):
        from locomotion.runtime.hardware import ImuMount, JointMap

        names = [f"j{i}" for i in range(18)]
        record = {
            "joints": {
                n: {"motor_id": 100 + i, "sign": -1 if i % 2 else 1, "zero_rad": 0.1 * i}
                for i, n in enumerate(names)
            }
        }
        joints = JointMap.from_record(record, names)
        target = np.linspace(-0.5, 0.5, 18)
        q, _ = joints.to_joint(joints.to_motor(target), np.zeros(18))
        np.testing.assert_allclose(q, target)
        # IMU mounted yawed 90 degrees about body z, level: gravity is straight down either way.
        mount = ImuMount(np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]]))
        np.testing.assert_allclose(mount.gravity_body([0, 0, 0, 1]), (0, 0, -1), atol=1e-12)
        half = math.radians(30) / 2
        yaw = mount.yaw([0, 0, math.sin(half), math.cos(half)])
        self.assertAlmostEqual(yaw, math.radians(30) - math.pi / 2)


if __name__ == "__main__":
    unittest.main()
