"""Sample commands, apply the walking reward and notify an optional observer around selected episode resets."""

import hashlib
import json
import math
from dataclasses import asdict
from pathlib import Path

import torch

from locomotion.omniwalk.mdp.commands import TaskConfig, command_bank
from locomotion.omniwalk.mdp.frames import inverse_rotate, navigation
from locomotion.omniwalk.mdp.rewards import (
    CONTROL_DT_S,
    REWARD_CONFIG,
    REWARD_VERSION,
    SCHEMA,
    StrideWindow,
    displacement_velocity,
    measured_reward,
    scheduled_swing,
)


class TrainingTask:
    """Own command attribution, reward memory and explicit selected resets.

    ``observer`` is None or an object with ``reset(selected)`` and
    ``step(command, output, reward, components, previous_target, telemetry)``. The task calls ``reset``
    after it resamples the selected rows and ``step`` after each control's reward, resample and count.
    """

    def __init__(self, env, config=None, output_dir=None, reward_config=None):
        self.observer = None
        chosen = REWARD_CONFIG if reward_config is None else reward_config
        self.gate_window = StrideWindow(env.num_envs, chosen.gate_window_controls, 1, env.device)
        self.stride_window = StrideWindow(
            env.num_envs, chosen.stride_window_controls, 3, env.device
        )
        self.liftoff_world = torch.zeros(env.num_envs, 6, 3, device=env.device)
        self.lifted = torch.zeros(env.num_envs, 6, dtype=torch.bool, device=env.device)

        task_config = TaskConfig() if config is None else config
        self.reward_config = REWARD_CONFIG if reward_config is None else reward_config
        self.reward_config.validate()
        n, device = env.num_envs, env.device
        self.previous_joint_velocity = env.current["dq"].detach().clone().to(device)
        self.previous_pose = env.current["root"].detach().clone().to(device)
        self.previous_toe_world = env.current["toe_world"].detach().clone().to(device)
        self.window = StrideWindow(n, self.reward_config.tracking_window_controls, 3, device)
        self.air_time = torch.zeros(n, 6, device=device)
        self.in_contact = torch.ones(n, 6, dtype=torch.bool, device=device)
        self.command_age = torch.zeros(n, dtype=torch.long, device=device)
        self.quiet_anchor = torch.zeros(n, 18, device=device)
        self.anchor_stale = torch.ones(n, dtype=torch.bool, device=device)

        self.env = env
        self.config = task_config
        self.config.validate()
        if env.cfg.control_dt != 0.02 or env.cfg.decimation != 8:
            raise ValueError("Task requires the unchanged 50/400Hz environment")
        self.bank = torch.tensor(command_bank(self.config), dtype=torch.float32, device=env.device)
        self.rng = torch.Generator(device="cpu").manual_seed(self.config.seed)
        self.remaining_controls = torch.zeros(env.num_envs, dtype=torch.long, device=env.device)
        self.previous_target = env.held.detach().clone()
        self.command_draws = 0
        self.zero_command_draws = 0
        self.controls_completed = 0
        self.last_components = {}
        self.output_dir = None if output_dir is None else Path(output_dir)
        self.nominal_height = float(env.reference_metadata.get("root_height_m", env.reset_height))
        if self.output_dir is not None:
            self.output_dir.mkdir(parents=True, exist_ok=True)
            path = self.output_dir / "task_definition.json"
            content = json.dumps(self.declaration(), indent=2, allow_nan=False) + "\n"
            if path.exists() and path.read_text() != content:
                raise ValueError("Existing task definition differs")
            path.write_text(content)

    def reset(self, indices=None):
        selected = self._indices(indices)
        output = self.env.reset(selected)
        self._resample(selected)
        self.previous_target[selected] = self.env.held[selected]
        if self.observer is not None:
            self.observer.reset(selected)
        output = self._next_command_observation(output)
        self.previous_joint_velocity[selected] = self.env.current["dq"][selected].to(
            self.previous_joint_velocity
        )
        self.previous_pose[selected] = self.env.current["root"][selected].to(self.previous_pose)
        self.previous_toe_world[selected] = (
            self.env.current["toe_world"][selected].detach().to(self.previous_toe_world)
        )
        self.window.restart(selected, self.env.commands[selected])
        self.command_age[selected] = 0
        self.anchor_stale[selected] = True
        self.air_time[selected] = 0
        self.in_contact[selected] = True

        self.gate_window.restart(selected, self.env.commands[selected])
        self.stride_window.restart(selected, self.env.commands[selected])
        self.lifted[selected] = False
        return output

    def step(self, action):
        if bool((self.remaining_controls <= 0).any()):
            raise RuntimeError("Call task.reset before collecting a rollout")
        command = self.env.commands.detach().clone()
        output = self.env.step(action)
        reward, components, telemetry = self._reward(command, output["terminated"])
        metric_previous_target = self.previous_target
        self.previous_target = self.env.telemetry["joint_target_rad"].detach().clone()
        self.last_components = {key: value.detach().clone() for key, value in components.items()}
        self.remaining_controls -= 1
        done = output["terminated"] | output["truncated"]
        # Terminal observations remain attributed to their final command. The
        # learner supplies an explicit selected reset before collecting again.
        expired = ((self.remaining_controls == 0) & ~done).nonzero(as_tuple=False).squeeze(-1)
        self._resample(expired)
        self.controls_completed += 1
        if self.observer is not None:
            self.observer.step(
                command, output, reward, components, metric_previous_target, telemetry
            )
        result = self._next_command_observation(output)
        result["reward"] = reward
        return result

    def reward_telemetry(self, command):
        velocity = displacement_velocity(
            self.env.telemetry["root_pose_xyzw"].to(self.previous_pose), self.previous_pose
        )
        stride_yaw = self.gate_window.update(velocity[:, 2:], command)[:, 0]
        stride = self.stride_window.update(velocity, command)
        # Liftoff and touchdown by the same contact threshold as v4, read before v4 advances its contact state.
        t = self.env.telemetry
        contact = (
            torch.linalg.vector_norm(t["tibia_floor_force_world_n"], dim=-1).to(self.lifted.device)
            > self.reward_config.contact_force_n
        )
        world = t["toe_xyz_world"].detach().to(self.liftoff_world)
        liftoff = ~contact & self.in_contact.to(contact.device)
        landed = contact & ~self.in_contact.to(contact.device) & self.lifted
        pose = t["root_pose_xyzw"].to(self.liftoff_world)
        travel = navigation(
            inverse_rotate(pose[:, None, 3:].expand(-1, 6, -1), world - self.liftoff_world)
        )[..., :2]
        self.liftoff_world = torch.where(liftoff[..., None], world, self.liftoff_world)
        self.lifted = (self.lifted | liftoff) & ~(contact & ~self.in_contact.to(contact.device))

        t = self.env.telemetry
        pose = t["root_pose_xyzw"].to(self.previous_pose)
        changed = (command.to(self.window.command) != self.window.command).any(-1)
        self.command_age = (
            torch.where(
                changed.to(self.command_age.device),
                torch.zeros_like(self.command_age),
                self.command_age,
            )
            + 1
        )
        tracked = self.window.update(displacement_velocity(pose, self.previous_pose), command)
        contact = (
            torch.linalg.vector_norm(t["tibia_floor_force_world_n"], dim=-1)
            > self.reward_config.contact_force_n
        )
        touchdown = contact & ~self.in_contact
        completed = self.air_time.clone()
        self.air_time = torch.where(
            contact, torch.zeros_like(self.air_time), self.air_time + CONTROL_DT_S
        )
        self.in_contact = contact
        # Toe velocity over the floor, expressed in the body's navigation axes.
        world = t["toe_xyz_world"].detach().to(self.previous_toe_world)
        body_axes = pose[:, None, 3:].expand(-1, 6, -1)
        toe_velocity = navigation(
            inverse_rotate(body_axes, (world - self.previous_toe_world) / CONTROL_DT_S)
        )[..., :2]
        self.previous_toe_world = world.clone()
        q = t["joint_position_rad"].to(contact.device)
        # The control just completed started at episode step minus one.
        base = {
            **t,
            "tracked_velocity_nav": tracked,
            "previous_joint_velocity_rad_s": self.previous_joint_velocity,
            "previous_joint_target_rad": self.previous_target,
            "touchdown": touchdown,
            "air_time_s": completed,
            "toe_velocity_nav": toe_velocity,
            "toe_xyz_nav": navigation(t["toe_xyz_body"].detach())[..., :2],
            "scheduled_swing": scheduled_swing(self.env.episode_steps - 1, self.reward_config).to(
                contact.device
            ),
            "joint_limit_margin_rad": torch.minimum(
                q - self.env.lower.to(q), self.env.upper.to(q) - q
            ),
            "executed_action": (t["joint_target_rad"].to(q) - self.env.neutral.to(q))
            / self.env.cfg.action_scale_rad,
            "command_age_controls": self.command_age.clone(),
        }
        if self.reward_config.quiet_hold_weight:
            # The pose each quiet hold is priced against: captured on the completed control
            # closest to each command change (quiet onsets included) and after every reset.
            stale = self.anchor_stale | changed.to(self.anchor_stale.device)
            self.quiet_anchor = torch.where(stale.to(q.device)[:, None], q, self.quiet_anchor.to(q))
            self.anchor_stale = torch.zeros_like(self.anchor_stale)
            base["quiet_anchor_joint_rad"] = self.quiet_anchor.clone()
        return {
            **base,
            "gate_yaw_rate_rad_s": stride_yaw,
            "stride_velocity_nav": stride,
            "stride_touchdown": landed,
            "stride_displacement_nav": torch.where(
                landed[..., None], travel, torch.zeros_like(travel)
            ),
        }

    def _reward(self, command, terminated):
        telemetry = self.reward_telemetry(command)
        reward, components = measured_reward(
            telemetry, command, terminated, self.reward_config, self.nominal_height
        )
        self.previous_joint_velocity = (
            self.env.telemetry["joint_velocity_rad_s"]
            .detach()
            .clone()
            .to(self.previous_joint_velocity)
        )
        self.previous_pose = (
            self.env.telemetry["root_pose_xyzw"].detach().clone().to(self.previous_pose)
        )
        return reward, components, telemetry

    def _resample(self, indices):
        if not len(indices):
            return
        size = len(indices)
        bank_indices = torch.randint(len(self.bank), (size,), generator=self.rng)
        values = self.bank[bank_indices.to(self.env.device)].clone()
        # An integer quota avoids a finite pilot accidentally seeing no stops.
        before = math.ceil(self.config.zero_command_fraction * self.command_draws)
        after = math.ceil(self.config.zero_command_fraction * (self.command_draws + size))
        zero_count = after - before
        selected_zero = torch.randperm(size, generator=self.rng)[:zero_count].to(self.env.device)
        values[selected_zero] = 0
        durations = torch.randint(100, 251, (size,), generator=self.rng, dtype=torch.long).to(
            self.env.device
        )
        self.env.commands[indices] = values
        self.remaining_controls[indices] = durations
        self.command_draws += size
        self.zero_command_draws += zero_count
        share = self.reward_config.turn_draw_fraction
        if share and len(indices):
            chosen = (torch.rand(len(indices), generator=self.rng) < share).to(self.env.device)
            chosen &= (self.env.commands[indices] != 0).any(-1)
            sign = (torch.randint(0, 2, (len(indices),), generator=self.rng) * 2 - 1).to(
                self.env.commands
            )
            turns = torch.zeros(
                len(indices), 3, device=self.env.device, dtype=self.env.commands.dtype
            )
            turns[:, 2] = sign * self.config.yaw_rate_rad_s
            self.env.commands[indices[chosen]] = turns[chosen]

    def _indices(self, indices):
        result = (
            torch.arange(self.env.num_envs, device=self.env.device)
            if indices is None
            else torch.as_tensor(indices, dtype=torch.long, device=self.env.device)
        )
        if (
            result.ndim != 1
            or bool((result < 0).any())
            or bool((result >= self.env.num_envs).any())
            or len(result.unique()) != len(result)
        ):
            raise ValueError("Invalid command/reset row indices")
        return result

    def _next_command_observation(self, output):
        obs, critic = output["obs"].clone(), output["critic"].clone()
        if obs.shape != (self.env.num_envs, 231) or critic.shape != (self.env.num_envs, 234):
            raise ValueError("Task actor/critic ABI differs")
        if not torch.equal(critic[:, :231], obs):
            raise ValueError("Task received inconsistent actor/critic prefix")
        obs[:, 210:213] = self.env.commands
        critic[:, :231] = obs
        return {**output, "obs": obs, "critic": critic}

    def declaration(self):
        return {
            "schema": SCHEMA,
            "config": asdict(self.config),
            "reward_version": REWARD_VERSION,
            "reward": {"version": REWARD_VERSION, "config": asdict(self.reward_config)},
            "nonzero_command_bank": self.bank.cpu().tolist(),
            "source_sha256": hashlib.sha256(
                b"".join(
                    (Path(__file__).parent / name).read_bytes()
                    for name in ("task.py", "mdp/rewards.py", "mdp/commands.py")
                )
            ).hexdigest(),
            "physics_changed": True,
            "stage2_gates_changed": False,
            "stage2_complete": False,
        }

    def __getattr__(self, name):
        return getattr(self.env, name)
