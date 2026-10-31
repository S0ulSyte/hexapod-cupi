"""Give the task, the vector-environment wrapper and evaluation one walking environment over Isaac Lab."""

from functools import partial

import torch

from locomotion.omniwalk.mdp.observations import observations as assemble_observations
from locomotion.omniwalk.mdp.observations import proprio


def reset_rows(indices, num_envs, device):
    """Validate selected rows before a reset writes state."""
    rows = (
        torch.arange(num_envs, device=device)
        if indices is None
        else torch.as_tensor(indices, dtype=torch.long, device=device)
    )
    if (
        rows.ndim != 1
        or bool(torch.any(rows < 0))
        or bool(torch.any(rows >= num_envs))
        or len(rows.unique()) != len(rows)
    ):
        raise ValueError("Invalid selected reset rows")
    return rows.to(torch.long)


class OmniEnv:
    """Expose the task-facing surface of the stock Direct environment in ``isaac``.

    IsaacEnv owns ``commands``, ``history``, ``held`` and ``previous_action`` and writes them in place;
    the same names here alias those tensors, so task writes reach the observations. The task reads
    terminal observations before selected resets. A ``verify`` callable receives this environment after
    the Isaac Lab construction and before the first task reset, so its scene checks read the state of the
    SDK reset.
    """

    num_actions = 18
    observation_width = 231
    critic_width = 234
    amp_width = 61

    def __init__(
        self, cfg, asset, model_path, geometry_path, output, *, reference_metadata=None, verify=None
    ):
        from locomotion.omniwalk.env import IsaacEnv

        self.cfg, self.device, self.num_envs = cfg, cfg.device, cfg.num_envs
        self.isaac = isaac = IsaacEnv(
            cfg, asset, model_path, geometry_path, output, reference_metadata=reference_metadata
        )
        self.commands, self.history = isaac.commands, isaac.history
        self.held, self.previous_action = isaac.held, isaac.previous_action
        self.episode_steps = isaac.episode_length_buf
        self.neutral, self.lower, self.upper = isaac.neutral, isaac.lower, isaac.upper
        self.reset_height, self.origins, self.torch = isaac.reset_height, isaac.origins, isaac.torch
        self.output, self.model, self.reference_metadata = (
            isaac.output,
            isaac.model,
            isaac.reference_metadata,
        )
        self.geometry_path, self.geometry_meta = isaac.geometry_path, isaac.geometry_meta
        self.sim, self.robot, self.motor = isaac.sim, isaac.robot, isaac.motor
        self.view, self.contact, self.sensor_map, self.roots = (
            isaac.view,
            isaac.contact,
            isaac.sensor_map,
            isaac.roots,
        )
        self.joint_names, self.body_names = isaac.joint_names, isaac.body_names
        self.native_joint_names, self.native_body_names = (
            isaac.native_joint_names,
            isaac.native_body_names,
        )
        self.joint_index, self.inverse_joint_index = isaac.joint_index, isaac.inverse_joint_index
        self.body_index, self.toe_index, self.toe_local = (
            isaac.body_index,
            isaac.toe_index,
            isaac.toe_local,
        )
        self.root_com_local = isaac.root_com_local
        self.native_errors = isaac.native_errors
        self.get, self.native_array, self.save = isaac.get, isaac.native_array, isaac.save
        self.telemetry, self.total_controls = isaac.telemetry, isaac.total_controls
        self.capture = None
        if verify is not None:
            verify(self)
        self.reset()

    def reset(self, indices=None):
        rows = reset_rows(indices, self.num_envs, self.device)
        if self.native_errors:
            raise RuntimeError("Cannot reset across a native physics error")
        before = self.sim.get_physics_step_count()
        self.current = self.isaac.reset_selected(rows.to(torch.int32))
        if self.sim.get_physics_step_count() != before:
            raise RuntimeError("Reset unexpectedly advanced physics")
        if len(rows):
            expected = self.neutral.expand(len(rows), -1)
            if not torch.allclose(self.current["q"][rows], expected, atol=2e-6, rtol=0):
                raise RuntimeError("Selected joint reset readback differs")
            written = getattr(self.isaac, "last_reset_pose", None)
            if written is not None:
                # The pose the reset wrote: on a terrain it carries the local ground height and
                # a drawn heading, so the readback must match what was written, exactly as before.
                expected_root = written[rows]
            else:
                expected_root = (
                    self.torch([0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 1.0]).expand(len(rows), -1).clone()
                )
                expected_root[:, :3] = self.origins[rows]
                expected_root[:, 2] = self.reset_height
            if not torch.allclose(self.current["root"][rows], expected_root, atol=2e-6, rtol=0):
                raise RuntimeError("Selected root reset readback differs")
            self.history[rows] = proprio(self.current, self.neutral)[rows, None].expand(-1, 5, -1)
        return self.observations(self.current)

    def step(self, action):
        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.shape != (self.num_envs, 18) or not bool(torch.isfinite(action).all()):
            raise ValueError("Policy action shape/finite check failed")
        # Consumers set capture hooks here and receive this environment as the first argument.
        self.isaac.capture = None if self.capture is None else partial(self.capture, self)
        observations, _, _, _, _ = self.isaac.step(action)
        self.current, self.telemetry, self.total_controls = (
            self.isaac.current,
            self.isaac.telemetry,
            self.isaac.total_controls,
        )
        return {
            "obs": observations.pop("policy"),
            **observations,
            "terminated": self.telemetry["terminated"],
            "truncated": self.telemetry["truncated"],
        }

    def observations(self, state):
        return assemble_observations(self.history, self.commands, self.previous_action, state)
