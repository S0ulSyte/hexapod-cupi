"""Evaluate a checkpoint's no-noise policy on the learning probes, on the surrogate."""

from __future__ import annotations

import argparse
import json
import shutil
import time
from dataclasses import asdict, replace
from pathlib import Path

import numpy as np
import torch

import locomotion.rl.ppo as algorithms
import locomotion.rl.wrapper as omni_wrapper
from locomotion.amp.features import extract_features
from locomotion.spark.sources import source_files
from locomotion.surrogate.train import build_wrapper, process_threads, resolve, wrapper_options

FOCUS = ["learning:translate_0.05_0deg", "learning:quiet_20s", "learning:forward_0.05_to_stop"]
DEFAULT_TASK = "locomotion.omniwalk.task:TrainingTask"
# Profiles whose verdict rests on a static pose.
STATIC_PROFILES = ("quiet_stand", "stage2_long_quiet", "stop_to_stand")
CONSENSUS_VARIANTS = {
    "pyramidal_cone": {"cone": "pyramidal"},
    "friction_0.9": {"friction": (0.9, 0.005, 0.0001)},
}
BOUNDS_SCOPE = (
    "Unverified on the surrogate. The native check counts 400 Hz joint positions beyond a limit and joint "
    "speeds above 50.27 rad/s. The surrogate holds a loaded joint 0.5 mrad beyond its stop where native "
    "holds it within 0.1 mrad, and the touchdown model alone lets it exceed the speed limit."
)
REST_CONTROLS = 300


class DeterministicPolicy:
    """The no-noise policy path of ``locomotion/scripts/play.py``: fixed scales, gait clock, actor mean, two-control mean."""

    def __init__(
        self,
        actor,
        env,
        options,
        *,
        velocity_input_gain=1.0,
        action_mean2=False,
        adapter=None,
        schedule_mirror=False,
    ):
        self.actor, self.env, self.adapter = actor, env, adapter
        # The mirrored tripod schedule changes the footfall predictions on clockwise turns;
        # evaluation rebuilds them from the recorded task's reward flag (mdp/gait.phase_swing).
        self.schedule_mirror = bool(schedule_mirror)
        self.scaling = options.get("observation_scaling", "none")
        self.period = options.get("gait_clock", 0)
        # A speed-scaled clock follows each probe's commands, as the training task does.
        self.gait = None
        if options.get("gait_clock_scaling") is not None:
            from locomotion.omniwalk.mdp.gait import GaitPhase

            self.gait = GaitPhase(env.num_envs, options["gait_clock_scaling"], "cpu")
        self.gain = float(velocity_input_gain)
        if adapter is not None and (self.gain != 1.0 or action_mean2):
            raise ValueError(
                "The wrapper's policy_for_evaluation replaces the standard policy path; "
                "--velocity-input-gain and --action-mean2 do not reach it"
            )
        self.forced = bool(action_mean2)
        self.mean2 = adapter is None and (
            self.forced or options.get("action_smoothing", "none") == "mean2"
        )
        self.previous = None
        # The destination inputs follow each probe's command from the root pose, as play.py does.
        self.tracker = None
        if options.get("destination_input", "none") in (
            "ghost",
            "ghost_v10",
            "ghost_v11",
            "terrain_v2_critic",  # the critic alone read the terrain; the actor is ghost_v11-shaped
        ):
            from locomotion.omniwalk.mdp.destination import (
                DestinationTracker,
                GhostV10Tracker,
                GhostV11Tracker,
            )

            tracker = {
                "ghost": DestinationTracker,
                "ghost_v10": GhostV10Tracker,
                "ghost_v11": GhostV11Tracker,
                "terrain_v2_critic": GhostV11Tracker,
            }[options["destination_input"]]
            self.tracker = tracker(env.num_envs, "cpu")
        elif options.get("destination_input") in (
            "terrain_v1",
            "terrain_v2",
            "terrain_v2_ghost_critic",
        ):
            from locomotion.omniwalk.mdp.footfall import (
                TerrainV1Tracker,
                TerrainV2GhostCriticTracker,
                TerrainV2Tracker,
            )

            tracker = {
                "terrain_v1": TerrainV1Tracker,
                "terrain_v2": TerrainV2Tracker,
                "terrain_v2_ghost_critic": TerrainV2GhostCriticTracker,
            }[options["destination_input"]]
            self.tracker = tracker(
                env.num_envs,
                "cpu",
                env=env,
                scaling=options.get("gait_clock_scaling"),
                mirror=self.schedule_mirror,
            )

    def record(self, options):
        """The ``policy_path`` entry of the summary: the path that ran and the options that acted on it."""
        if self.adapter is not None:
            return {
                "source": "wrapper policy_for_evaluation",
                "velocity_input_gain": 1.0,
                "action_mean2_forced": False,
                "action_smoothing_applied": None,
                "velocity_noise_recorded": options.get("velocity_noise", 0.0),
                "velocity_noise_applied": None,
                "scope": "The wrapper builds the policy input and the action; this module applies no option to them.",
            }
        return {
            "source": "policy path of locomotion/scripts/play.py",
            "velocity_input_gain": self.gain,
            "action_mean2_forced": self.forced,
            "action_smoothing_applied": "mean2" if self.mean2 else "none",
            "velocity_noise_recorded": options.get("velocity_noise", 0.0),
            "velocity_noise_applied": 0.0,
            "scope": "Scales, gait clock, actor mean and two-control mean as the policy path of "
            "locomotion/scripts/play.py; the gain and the forced mean are stress options.",
        }

    def inputs(self, observation):
        """The actor input for raw 231-value environment observations."""
        scaled = omni_wrapper.scale_observation(observation, self.scaling)
        if self.gain != 1.0:
            scaled = scaled.clone()
            for frame in range(5):
                scaled[:, 42 * frame + 24 : 42 * frame + 42] *= self.gain
        if self.period:
            cycles = None
            if self.gait is not None:
                cycles = self.gait.follow(observation[:, 210:213], self.env.episode_steps)
            clock = omni_wrapper.clock_features(
                self.env.episode_steps, self.period, observation[:, 210:213], cycles
            ).to(scaled)
            scaled = torch.cat((scaled, clock), -1)
        if self.tracker is not None:
            extra = self.tracker.inputs(
                self.env.current["root"], observation[:, 210:213], self.env.episode_steps
            )
            scaled = torch.cat((scaled, extra.to(scaled)), -1)
        return scaled

    def __call__(self, observation):
        from tensordict import TensorDict

        with torch.inference_mode():
            if self.adapter is not None:
                return self.adapter(self.actor, observation)
            inputs = self.inputs(observation)
            groups = {"policy": inputs}
            if "proprio_history" in getattr(self.actor, "obs_groups", ()):
                from locomotion.amp.networks import actor_observation

                groups.update(actor_observation(inputs).items())
            action = self.actor(TensorDict(groups, batch_size=[self.env.num_envs]))
            if self.mean2:
                # The same two-control mean as the training wrapper, restarted with each episode.
                last = torch.zeros_like(action) if self.previous is None else self.previous
                self.previous = action.clone()
                action = omni_wrapper.smoothed_action(action, last, self.env.episode_steps)
            return action


class FootLoad:
    """Record the floor force on each foot after each control, by probe."""

    def __init__(self, env):
        self.env, self.rows, self.seen = env, {}, env.total_controls

    def record(self, case):
        if case is None or self.env.total_controls == self.seen:
            return
        self.seen = self.env.total_controls
        force = self.env.telemetry["tibia_floor_force_world_n"]
        self.rows.setdefault(case, []).append(torch.linalg.vector_norm(force[0], dim=-1).tolist())

    def rest_load(self):
        """Mean load of each foot over the last ``REST_CONTROLS`` controls of each probe."""
        return {
            case: np.asarray(rows)[-REST_CONTROLS:].mean(0).tolist()
            for case, rows in self.rows.items()
        }


class VideoWriter:
    """H.264 MP4 through OpenCV, one frame per two controls (25 fps, real time)."""

    def __init__(self, path, renderer):
        import cv2

        self.cv2, self.renderer, self.path = cv2, renderer, Path(path)
        self.writer = None
        self.frames = 0

    def add(self):
        frame = self.renderer.frame(0)
        if self.writer is None:
            height, width = frame.shape[:2]
            self.writer = self.cv2.VideoWriter(
                str(self.path), self.cv2.VideoWriter_fourcc(*"avc1"), 25, (width, height)
            )
            if not self.writer.isOpened():
                self.writer = self.cv2.VideoWriter(
                    str(self.path), self.cv2.VideoWriter_fourcc(*"mp4v"), 25, (width, height)
                )
            self.cv2.imwrite(str(self.path.with_suffix(".first_frame.png")), frame[:, :, ::-1])
        self.writer.write(frame[:, :, ::-1])
        self.frames += 1

    def close(self):
        if self.writer is not None:
            self.writer.release()


def case_summary(result, batch, limits=None, joint_names=None):
    """Headline numbers of one probe, plus gait descriptors that carry no gate."""
    metrics = result.get("metrics", {})
    row = {
        "case_id": result["case_id"],
        "pass": result["pass"],
        "passed_numeric_screen": result.get("passed_numeric_screen"),
        "failed_bounds": result.get("failed_bounds"),
        "missing_evidence": result.get("missing_evidence"),
        "batch_failure": result.get("batch_failure"),
        "metrics": {
            key: metrics.get(key)
            for key in (
                "planar_error_mps",
                "yaw_error_rad_s",
                "tilt_rms_degrees",
                "vertical_velocity_rms_mps",
                "computed_demand_over_rating_fraction",
                "nonfoot_fraction",
                "mean_velocity_mps",
                "mean_yaw_rate_rad_s",
                "max_planar_excursion_m",
                "max_heading_excursion_deg",
                "max_joint_velocity_rms_rad_s",
                "max_joint_position_range_rad",
                "max_target_step_abs_p95_rad_per_20ms",
                "max_requested_torque_saturation_fraction",
                "max_applied_torque_nm",
                "requested_saturation_fraction_400hz",
            )
            if key in metrics
        },
        "six_toe_and_nonfoot_checks": {
            key: result.get("checks", {}).get(key)
            for key in (
                "missing_six_toe_count_400hz",
                "nonfoot_contact_count_400hz",
                "minimum_non_toe_floor_m",
                "minimum_plate_height_m",
            )
            if key in result.get("checks", {})
        },
        "native_contact_screen": result.get("native_contact_screen"),
    }
    trace = batch / "control_trace.npz"
    if trace.exists():
        with np.load(trace) as data:
            start = min(result.get("window_start_control", 100), len(data["time_s"]) - 1)
            nav = data["velocity_navigation_mps"][:, 0]
            toe = data["toe_xyz_world_m"][:, 0, :, 2]
            contact = data["distal_contact"][:, 0]
            action = data["policy_action"][:, 0]
            target = data["joint_target_rad"][:, 0]
            # Toe points sit above the lowest cap point, so lift is height above each toe's stance level.
            stance = [
                float(np.median(toe[start:, k][contact[start:, k]]))
                if contact[start:, k].any()
                else None
                for k in range(6)
            ]
            row["gait"] = {
                "controls": int(len(nav)),
                "window_start_control": int(start),
                "mean_velocity_nav_mps": nav[start:].mean(0).tolist(),
                "mean_forward_speed_controls_100_399_mps": float(nav[100:400, 0].mean())
                if len(nav) >= 400
                else None,
                "root_height_mean_m": float(data["root_pose_xyzw"][start:, 0, 2].mean()),
                "final_root_height_m": float(data["root_pose_xyzw"][-1, 0, 2]),
                "toe_contact_fraction_endpoints": contact[start:].mean(0).tolist(),
                "toe_point_height_max_mm": (1000 * toe[start:].max(0)).tolist(),
                "toe_lift_above_stance_mm": [
                    None if level is None else float(1000 * (toe[start:, k].max() - level))
                    for k, level in enumerate(stance)
                ],
                "action_abs_max": float(abs(action).max()),
                "action_abs_ge_0.95_fraction": float((abs(action) >= 0.95).mean()),
                "target_step_at_limiter_fraction": float(
                    (abs(abs(np.diff(target, axis=0)) - 0.04) <= 2e-6).mean()
                ),
                "applied_torque_rms_coxa_femur_tibia_nm": [
                    float(
                        np.sqrt(
                            data["applied_torque_squared_sum_400hz"][start:, 0, k::3].mean() / 8
                        )
                    )
                    for k in range(3)
                ],
            }
            if limits is not None:
                q = data["joint_position_rad"][:, 0]
                margin = np.minimum(q - limits[:, 0], limits[:, 1] - q).min(0)
                joint = int(margin.argmin())
                row["joint_limit_margin"] = {
                    "minimum_rad": float(margin[joint]),
                    "joint": joint_names[joint],
                    "final_rad": float(
                        np.minimum(q[-1] - limits[:, 0], limits[:, 1] - q[-1]).min()
                    ),
                    "scope": "Smallest distance of a joint to a stop at the 50 Hz control ends; the termination rule fires at -2e-6 rad.",
                }
    capture = batch / "native400hz/capture.json"
    if capture.exists():
        receipt = json.loads(capture.read_text())
        row["physical_bounds"] = {
            "joint_bound_violation_steps_400hz": receipt.get("joint_bound_violation_steps"),
            "speed_bound_violation_steps_400hz": receipt.get("speed_bound_violation_steps"),
            "native_capture_or_original_physical_bounds": BOUNDS_SCOPE,
        }
    forces = batch / "force_metrics.json"
    if forces.exists():
        report = json.loads(forces.read_text())
        if report.get("status") == "available":
            window = report["cases"][0]["windows"]["full_trial"]
            row["loads_full_trial"] = {
                "per_foot_contact_fraction": {
                    leg: value["contact_fraction"] for leg, value in window["per_foot"].items()
                },
                "per_foot_peak_n": {
                    leg: value["normal_resultant_magnitude_n"]["peak"]
                    for leg, value in window["per_foot"].items()
                },
                "applied_torque_worst_joint_rms_nm": window["motor_torque"]["applied"][
                    "worst_joint_rms_nm"
                ],
                "applied_torque_worst_joint": window["motor_torque"]["applied"]["worst_joint"],
            }
    return row


def consensus(results):
    """Verdict and checks of one probe across contact models: a value where all agree, else ``unresolved``."""
    verdicts = {name: bool(result["pass"]) for name, result in results.items()}
    names = sorted(set().union(*(result.get("checks", {}) for result in results.values())))
    statuses = {
        check: {
            name: result.get("checks", {}).get(check, {}).get("status")
            for name, result in results.items()
        }
        for check in names
    }
    agreed = {
        check: next(iter(values.values())) if len(set(values.values())) == 1 else "unresolved"
        for check, values in statuses.items()
    }
    bounds = {name: sorted(result.get("failed_bounds") or []) for name, result in results.items()}
    return {
        "verdict": ("pass" if all(verdicts.values()) else "fail")
        if len(set(verdicts.values())) == 1
        else "unresolved",
        "verdicts": verdicts,
        "failed_bounds": bounds,
        "failed_bounds_in_all": sorted(
            set.intersection(*(set(value) for value in bounds.values()))
        ),
        "unresolved_checks": {
            check: statuses[check] for check, value in agreed.items() if value == "unresolved"
        },
        "checks": agreed,
        "scope": "The default, pyramidal-cone and friction-0.9 runs differ in two contact settings. Agreement shows "
        "that a surrogate verdict does not rest on those settings. It is no native verdict: "
        "locomotion/surrogate/README.md records static probes that pass in each run and fail on native.",
    }


def build_parser():
    parser = argparse.ArgumentParser(
        description=__doc__,
        allow_abbrev=False,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New directory.")
    parser.add_argument(
        "--cases", default="focus", help="focus, all, or comma-separated learning probe ids."
    )
    parser.add_argument(
        "--video",
        default="first",
        help="first, all, none, or comma-separated probe ids to render as MP4.",
    )
    parser.add_argument("--view", choices=["three_quarter", "side"], default="three_quarter")
    parser.add_argument("--seed", type=int, help="Defaults to the checkpoint record's seed.")
    parser.add_argument(
        "--task", help="module:Class used to build the actor; defaults to the checkpoint record."
    )
    parser.add_argument(
        "--wrapper", help="module:Class; defaults to the checkpoint record or OmniVecEnv."
    )
    parser.add_argument(
        "--ppo-config", type=Path, help="ppo_config.json when the checkpoint has no JSON record."
    )
    parser.add_argument(
        "--observation-scaling",
        choices=["none", "fixed"],
        help="Defaults to the checkpoint's recorded PPO configuration.",
    )
    parser.add_argument(
        "--velocity-input-gain",
        type=float,
        default=1.0,
        help="Stress option: multiply the policy's 90 joint-velocity inputs by this gain.",
    )
    parser.add_argument(
        "--action-mean2",
        action="store_true",
        help="Stress option: send the mean of each action and the previous one, as --action-smoothing mean2 trains.",
    )
    parser.add_argument(
        "--foot-load",
        action="store_true",
        help="Write foot_load.json: the floor force on each foot after each control, by probe.",
    )
    parser.add_argument("--contact", action="append", default=[], metavar="FIELD=VALUE")
    parser.add_argument(
        "--no-contact-consensus",
        action="store_true",
        help="Skip the two extra contact models on quiet and stop probes.",
    )
    parser.add_argument(
        "--keep-variant-captures",
        action="store_true",
        help="Keep the 400 Hz captures of the consensus runs (about 15 MB per probe).",
    )
    parser.add_argument(
        "--threads", type=int, default=2, help="Torch threads for this process (1 to 6)."
    )
    return parser


def task_and_wrapper(args, identity):
    """The task and wrapper to rebuild: the command line, then the record's own, then the experiment's.

    A native experiment checkpoint records ``experiment`` but no task or wrapper; its destination task and
    wrapper follow from the name, as native training builds them.
    """
    experiment = identity.get("experiment")
    task = (
        f"locomotion.omniwalk.destination_task:experiment_task({experiment})"
        if experiment is not None
        else DEFAULT_TASK
    )
    wrapper = (
        "locomotion.rl.destination_wrapper:DestinationVecEnv"
        if experiment is not None
        else "locomotion.rl.wrapper:OmniVecEnv"
    )
    return args.task or identity.get("task") or task, args.wrapper or identity.get(
        "wrapper"
    ) or wrapper


def main(argv=None):
    args = build_parser().parse_args(argv)
    # self_check calls this entry point in its own process and keeps its thread settings afterwards.
    with process_threads(args.threads):
        return evaluate_checkpoint(args)


def evaluate_checkpoint(args):
    """Run the selected probes for parsed arguments and write ``surrogate_summary.json``."""
    import mujoco

    import locomotion.evaluation.runner as evaluation
    import locomotion.omniwalk.task as task_module
    from locomotion.surrogate.env import (
        DEFAULT_GEOMETRY,
        DEFAULT_MODEL,
        DEFAULT_STANCE,
        LEGS,
        ROOT,
        Renderer,
        SurrogateConfig,
        SurrogateEnv,
        contact_overrides,
        repository_state,
        sha,
    )

    record_path = args.checkpoint.with_suffix(".json")
    record = json.loads(record_path.read_text()) if record_path.exists() else {"identity": {}}
    identity = record["identity"]
    if (
        record.get("checkpoint_sha256") is not None
        and sha(args.checkpoint) != record["checkpoint_sha256"]
    ):
        raise ValueError("Checkpoint bytes differ from the paired record")
    checkpoint = torch.load(args.checkpoint, map_location="cpu", weights_only=False)
    config = (
        json.loads(args.ppo_config.read_text()) if args.ppo_config else identity.get("ppo_config")
    )
    if config is None:
        raise ValueError("The checkpoint has no JSON record; pass --ppo-config")
    seed = args.seed if args.seed is not None else int(identity.get("seed", 20260917))
    task_spec, wrapper_spec = task_and_wrapper(args, identity)
    profiles = {case["case_id"]: case["profile"] for case in evaluation.learning_probe_cases()}
    probes = list(profiles)
    selected = (
        FOCUS if args.cases == "focus" else probes if args.cases == "all" else args.cases.split(",")
    )
    video = (
        []
        if args.video == "none"
        else selected[:1]
        if args.video == "first"
        else selected
        if args.video == "all"
        else args.video.split(",")
    )
    if any(case not in probes for case in selected) or any(case not in selected for case in video):
        raise ValueError("Unknown probe id, or a video case outside the selected probes")
    contact = contact_overrides(args.contact)
    # The checkpoint's PPO configuration records the wrapper options of its training.
    options = wrapper_options(config)
    if args.observation_scaling is not None:
        options["observation_scaling"] = args.observation_scaling
    scaling = options.get("observation_scaling", "none")
    args.output.mkdir(parents=True, exist_ok=False)
    (args.output / "videos").mkdir()
    started = time.monotonic()

    def run(model, cases, directory, videos, record_loads=False):
        """One probe suite under one contact model: the suite summary, the per-case rows, the videos and the loads."""
        torch.manual_seed(seed)
        np.random.seed(seed)
        cfg = SurrogateConfig(
            num_envs=1, seed=seed, episode_seconds=60.0, record_motion_features=True
        )
        env = SurrogateEnv(
            cfg,
            None,
            DEFAULT_MODEL,
            DEFAULT_GEOMETRY,
            directory / "native",
            reference_metadata=json.loads(DEFAULT_STANCE.read_text()),
            threads=0,
            contact=model,
        )
        task = resolve(task_spec)(env, task_module.TaskConfig(seed=seed), None)
        wrapped = build_wrapper(resolve(wrapper_spec), task, options)
        observations = wrapped.get_observations()
        if "proprio_history" in config["obs_groups"]["actor"]:
            from locomotion.amp.networks import actor_observation

            observations.update(actor_observation(wrapped.current["obs"]))
        actor, _ = algorithms.load_actor(checkpoint, config, observations, "cpu")
        # A wrapper that changes the action or observation path can supply its own deterministic policy.
        deterministic = DeterministicPolicy(
            actor,
            env,
            options,
            velocity_input_gain=args.velocity_input_gain,
            action_mean2=args.action_mean2,
            adapter=getattr(wrapped, "policy_for_evaluation", None),
            schedule_mirror=bool(
                getattr(getattr(task, "reward_config", None), "schedule_mirror_turns", None)
                or False
            ),
        )
        renderer = Renderer(env, view=args.view) if videos else None
        loads = FootLoad(env) if record_loads else None
        writers, case_index, speeds = {}, [-1], []

        def close_case():
            if case_index[0] >= 0:
                speeds.append(
                    {
                        "peak_joint_speed_rad_s": env.peak_joint_speed,
                        "speed_clamp_events": env.speed_clamp_events,
                        "touchdown_events": env.touchdown_events,
                        "touchdown_overshoot_events": env.overshoot_events,
                    }
                )
                if loads is not None:
                    loads.record(cases[case_index[0]])
            (
                env.peak_joint_speed,
                env.speed_clamp_events,
                env.touchdown_events,
                env.overshoot_events,
            ) = 0.0, 0, 0, 0

        def policy(observation):
            if int(env.episode_steps[0]) == 0:
                close_case()
                case_index[0] += 1
            elif loads is not None:
                loads.record(cases[case_index[0]])
            case = cases[case_index[0]]
            if case in videos:
                if case not in writers:
                    writers[case] = VideoWriter(
                        args.output / "videos" / (case.replace(":", "_") + ".mp4"), renderer
                    )
                if int(env.episode_steps[0]) % 2 == 0:
                    writers[case].add()
            return deterministic(observation)

        summary = evaluation.run_learning_probe_suite(
            env,
            policy,
            directory / "evaluation",
            env.geometry_extrema_path,
            args.checkpoint,
            selected_case_ids=cases,
            record_video=False,
            seed=seed,
            progress=lambda value: print(json.dumps(value), flush=True),
            motion_features=extract_features,
        )
        close_case()
        for writer in writers.values():
            writer.close()
        if renderer is not None:
            renderer.close()
        limits = torch.stack((env.lower, env.upper), dim=-1).numpy()
        rows = [
            case_summary(
                result, directory / "evaluation" / f"batch_{index:03d}", limits, env.joint_names
            )
            for index, result in enumerate(summary["results"])
        ]
        for row, speed in zip(rows, speeds):
            row.setdefault(
                "physical_bounds", {"native_capture_or_original_physical_bounds": BOUNDS_SCOPE}
            ).update(speed)
        env.close()
        return summary, rows, writers, loads, deterministic.record(options)

    summary, rows, writers, loads, policy_path = run(
        contact, selected, args.output, video, args.foot_load
    )
    foot_load = None
    if loads is not None:
        (args.output / "foot_load.json").write_text(json.dumps(loads.rows) + "\n")
        foot_load = {
            "path": "foot_load.json",
            "legs": list(LEGS),
            "rows": "Norm of the tibia floor force in N after each control, by probe; default contact model.",
            "rest_load_mean_n": loads.rest_load(),
            "rest_controls": REST_CONTROLS,
        }
    static = [case for case in selected if profiles[case] in STATIC_PROFILES]
    variants = {}
    if static and not args.no_contact_consensus:
        results = {"default": {result["case_id"]: result for result in summary["results"]}}
        for name, fields in CONSENSUS_VARIANTS.items():
            directory = args.output / "contact_consensus" / name
            directory.mkdir(parents=True)
            variant_summary, variant_rows = run(replace(contact, **fields), static, directory, [])[
                :2
            ]
            results[name] = {result["case_id"]: result for result in variant_summary["results"]}
            variants[name] = {
                "contact_model_fields": {
                    key: list(value) if isinstance(value, tuple) else value
                    for key, value in fields.items()
                },
                "cases": variant_rows,
            }
            if not args.keep_variant_captures:
                for capture in directory.glob("evaluation/batch_*/native400hz"):
                    shutil.rmtree(capture)
        for row in rows:
            if row["case_id"] in static and all(
                row["case_id"] in value for value in results.values()
            ):
                row["contact_consensus"] = consensus(
                    {name: value[row["case_id"]] for name, value in results.items()}
                )
    unresolved = [
        row["case_id"]
        for row in rows
        if row.get("contact_consensus", {}).get("verdict") == "unresolved"
    ]
    report = {
        "schema": "hexapod_surrogate_evaluation_v3",
        "surrogate": True,
        "backend": "mujoco " + mujoco.__version__,
        "scope": "CPU MuJoCo surrogate; design evidence. Native runs are the only acceptance evidence.",
        "checkpoint": str(args.checkpoint),
        "checkpoint_sha256": sha(args.checkpoint),
        "checkpoint_origin": "surrogate" if identity.get("surrogate") else "native",
        "task": task_spec,
        "wrapper": wrapper_spec,
        "wrapper_options": options,
        "observation_scaling": scaling,
        "policy_path": policy_path,
        "foot_load": foot_load,
        "seed": seed,
        "selected_case_ids": selected,
        "contact_model": asdict(contact),
        "repository": repository_state(),
        "identity": {
            "source_files": source_files(ROOT / "src/locomotion"),
            "surrogate_files": {
                p.name: sha(p) for p in sorted(Path(__file__).resolve().parent.glob("*.py"))
            },
            "model_sha256": sha(DEFAULT_MODEL),
            "stance_sha256": sha(DEFAULT_STANCE),
            "geometry_sha256": sha(DEFAULT_GEOMETRY),
            "geometry_extrema_sha256": sha(DEFAULT_GEOMETRY.parent / "geometry_extrema.npz"),
        },
        "failed_probe_cases": summary["failed_probe_cases"],
        "unresolved_probe_cases": unresolved,
        "acquisition_failure": summary["acquisition_failure"],
        "videos": {
            case: {"path": str(writer.path), "frames": writer.frames}
            for case, writer in writers.items()
        },
        "cases": rows,
        "contact_consensus_variants": variants,
        "physical_bounds_scope": BOUNDS_SCOPE,
        "wall_seconds": time.monotonic() - started,
    }
    (args.output / "surrogate_summary.json").write_text(
        json.dumps(report, indent=2, allow_nan=False) + "\n"
    )
    for row in rows:
        gait = row.get("gait", {})
        print(
            json.dumps(
                {
                    "case": row["case_id"],
                    "pass": row["pass"],
                    "failed": row["failed_bounds"],
                    "consensus": row.get("contact_consensus", {}).get("verdict"),
                    "planar_error": row["metrics"].get("planar_error_mps"),
                    "mean_velocity": gait.get("mean_velocity_nav_mps"),
                    "root_height": gait.get("root_height_mean_m"),
                    "joint_limit_margin_rad": row.get("joint_limit_margin", {}).get("minimum_rad"),
                }
            )
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
