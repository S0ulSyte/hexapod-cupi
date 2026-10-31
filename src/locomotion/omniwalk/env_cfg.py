"""Declare the environment settings and build the Isaac Lab Direct and robot configurations."""

import math
from dataclasses import asdict, dataclass
from pathlib import Path

from locomotion.robots.hexapod import JOINT_NAMES, LEGS

# Stock Isaac Lab floors: a two-triangle mesh built by the terrain utilities (the recorded floor), the grid
# ground plane, or a box spawned by CuboidCfg. A floor authors the declared collider offsets; None keeps the
# simulator defaults, which the recorded physics depends on.
FLOORS = ("isaaclab_mesh", "isaaclab_ground_plane", "isaaclab_cuboid")


@dataclass(frozen=True)
class EnvConfig:
    """Native environment settings that each record declares.

    ``record_motion_features`` has no effect on the environment, which emits no motion features. Records
    keep the field, and the AMP wrapper requires it to be True before it computes the features itself.
    """

    num_envs: int = 32
    spacing_m: float = 2.0
    physics_dt: float = 0.0025
    decimation: int = 8
    episode_seconds: float = 20.0
    action_scale_rad: float = 0.35
    target_slew_rad: float = 0.040
    reset_height_m: float = 0.08161108940839767
    command_forward_mps: float = 0.10
    command_left_mps: float = 0.0
    command_yaw_rad_s: float = 0.0
    seed: int = 20260914
    record_motion_features: bool = False
    render: bool = False
    device: str = "cuda:0"
    floor: str = "isaaclab_mesh"
    floor_contact_offset_m: float | None = None
    floor_rest_offset_m: float | None = None
    # A terrain spec (surrogate/terrain.py JSON) replaces the flat mesh floor with its height
    # field, tiled to cover the environment origins; the policy inputs read the same grid through
    # the env's ``terrain`` attribute (omniwalk/mdp/footfall.py). The mesh resolution decimates the
    # collision mesh only; the input grid keeps the spec's resolution. ``terrain_spawn`` scatters
    # each reset inside its environment's cell with a random heading, as surrogate terrain
    # training does.
    terrain_spec_path: str | None = None
    terrain_mesh_resolution_m: float = 0.08
    terrain_spawn: bool = False

    def __post_init__(self):
        if self.floor not in FLOORS:
            raise ValueError("Unknown stock floor")
        if self.terrain_spec_path is not None and self.floor != "isaaclab_mesh":
            raise ValueError("A terrain spec builds the mesh floor; declare floor isaaclab_mesh")
        if self.terrain_spawn and self.terrain_spec_path is None:
            raise ValueError("Terrain spawn randomization needs a terrain spec")
        if (
            not math.isfinite(self.terrain_mesh_resolution_m)
            or not 0.02 <= self.terrain_mesh_resolution_m <= 0.2
        ):
            raise ValueError("Terrain mesh resolution is 0.02 to 0.2 m")
        offsets = (self.floor_contact_offset_m, self.floor_rest_offset_m)
        if any(x is None for x in offsets) != all(x is None for x in offsets):
            raise ValueError("Declare both floor collider offsets or neither")
        if offsets[0] is not None and (
            not all(math.isfinite(x) for x in offsets)
            or not 0 <= self.floor_rest_offset_m < self.floor_contact_offset_m <= 0.1
        ):
            raise ValueError("Floor collider offsets need 0 <= rest < contact <= 0.1 m")
        if type(self.num_envs) is not int or not 1 <= self.num_envs <= 1024:
            raise ValueError("Pilot supports 1 through 1024 replicas")
        if self.physics_dt != 0.0025 or self.decimation != 8 or self.target_slew_rad != 0.040:
            raise ValueError("Canonical 400/50Hz servo and target slew are fixed")
        if self.spacing_m < 2 or self.device != "cuda:0":
            raise ValueError("Native CUDA pilot with at least 2m spacing required")
        if not all(
            math.isfinite(x)
            for x in (
                self.spacing_m,
                self.episode_seconds,
                self.action_scale_rad,
                self.reset_height_m,
                self.command_forward_mps,
                self.command_left_mps,
                self.command_yaw_rad_s,
            )
        ):
            raise ValueError("Configuration contains a nonfinite value")
        if not 0 < self.action_scale_rad <= 0.5 or not 0 < self.episode_seconds <= 60:
            raise ValueError("Invalid bounded pilot action scale or episode length")

    @property
    def authors_floor_offsets(self):
        return self.floor_contact_offset_m is not None

    @property
    def control_dt(self):
        return self.physics_dt * self.decimation

    def declaration(self):
        return {
            **asdict(self),
            "schema": "canonical_paper_walk_experiment_v1",
            "actor_width": 231,
            "critic_width": 234,
            "amp_width": 61,
            "joint_names": list(JOINT_NAMES),
            "legs": list(LEGS),
        }


def make_config(cfg, asset, neutral, reset_height, output):
    from isaaclab.assets import ArticulationCfg
    from isaaclab.envs import DirectRLEnvCfg
    from isaaclab.scene import InteractiveSceneCfg
    from isaaclab.sim import ArticulationRootPropertiesCfg, SimulationCfg, UsdFileCfg
    from isaaclab.utils import configclass
    from isaaclab_physx.physics import PhysxCfg

    from locomotion.robots.actuator import MotorCfg

    @configclass
    class NativeConfig(DirectRLEnvCfg):
        action_space = 18
        observation_space = 231
        state_space = 234
        decimation = 8
        episode_length_s = 20.0
        scene = InteractiveSceneCfg(
            num_envs=32, env_spacing=2.0, replicate_physics=False, filter_collisions=True
        )
        sim = SimulationCfg(
            dt=0.0025,
            render_interval=8,
            physics=PhysxCfg(enable_external_forces_every_iteration=True),
        )

    visualizers = []
    if cfg.render:
        from isaaclab_visualizers.kit import KitVisualizerCfg

        visualizers = [KitVisualizerCfg()]
    result = NativeConfig(
        seed=cfg.seed,
        episode_length_s=cfg.episode_seconds,
        scene=InteractiveSceneCfg(
            num_envs=cfg.num_envs,
            env_spacing=cfg.spacing_m,
            replicate_physics=False,
            filter_collisions=True,
        ),
        sim=SimulationCfg(
            dt=cfg.physics_dt,
            device=cfg.device,
            render_interval=cfg.decimation,
            physics=PhysxCfg(enable_external_forces_every_iteration=True),
            visualizer_cfgs=visualizers,
            log_dir=str(Path(output) / "isaac_logs"),
        ),
    )
    robot = ArticulationCfg(
        prim_path="/World/envs/env_.*/Robot",
        spawn=UsdFileCfg(
            usd_path=str((Path(asset) / "robot.usda").resolve()),
            activate_contact_sensors=True,
            articulation_props=ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=32,
                solver_velocity_iteration_count=0,
            ),
        ),
        init_state=ArticulationCfg.InitialStateCfg(
            pos=(0.0, 0.0, reset_height), joint_pos=dict(neutral), joint_vel={".*": 0.0}
        ),
        actuators={"motor": MotorCfg()},
    )
    return result, robot
