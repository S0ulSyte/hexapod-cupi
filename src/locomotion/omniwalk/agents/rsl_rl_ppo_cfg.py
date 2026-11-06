"""Build the RSL-RL learner configuration and name the admitted recipes without importing Torch."""

RATE_CLASS = "locomotion.rl.ppo:CappedRatePPO"
SEGMENT_CLASS = "locomotion.rl.ppo:CommandBootstrapPPO"


def ppo_config(
    seed,
    *,
    action_mean="unbounded",
    observation_normalization="empirical",
    observation_scaling="none",
    command_segments="continuous",
    learning_rate_max=None,
    learning_rate_min=None,
    action_std=0.15,
    action_noise_correlation=0.0,
    action_std_final=None,
    gait_clock=0,
    action_smoothing="none",
    velocity_noise=0.0,
    action_std_decay_updates=None,
    steps_per_env=24,
    destination_input="none",
    gait_clock_scaling=None,
):
    if action_mean not in ("unbounded", "tanh"):
        raise ValueError("Action mean must be unbounded or tanh")
    if observation_normalization not in ("empirical", "none"):
        raise ValueError("Observation normalization must be empirical or none")
    if observation_scaling not in ("none", "fixed") or command_segments not in (
        "continuous",
        "bootstrap",
    ):
        raise ValueError(
            "Observation scaling must be none or fixed; command segments continuous or bootstrap"
        )
    if type(gait_clock) is not int or (gait_clock and not 10 <= gait_clock <= 250):
        raise ValueError("The gait clock period is zero or 10 to 250 controls")
    if action_smoothing not in ("none", "mean2"):
        raise ValueError("Action smoothing must be none or mean2")
    if type(velocity_noise) is not float or not 0 <= velocity_noise <= 5:
        raise ValueError("The joint-velocity observation noise lies between 0 and 5 rad/s")
    if learning_rate_max is not None and not (
        type(learning_rate_max) is float and 1e-5 <= learning_rate_max <= 1e-2
    ):
        raise ValueError("The learning-rate ceiling must lie inside the stock schedule range")
    if learning_rate_min is not None and not (
        type(learning_rate_min) is float
        and learning_rate_max is not None
        and 1e-6 <= learning_rate_min <= learning_rate_max
    ):
        raise ValueError("The learning-rate floor needs the ceiling and lies from 1e-6 up to it")
    if type(action_std) is not float or not 0.01 <= action_std <= 1.0:
        raise ValueError("The initial action standard deviation must lie between 0.01 and 1")
    if (
        type(action_noise_correlation) is not float
        or not 0 <= action_noise_correlation < 1
        or (action_noise_correlation and action_mean != "tanh")
    ):
        raise ValueError("Correlated action noise needs a correlation in [0, 1) and the tanh mean")
    if action_std_final is not None and not (
        type(action_std_final) is float and 0.005 <= action_std_final <= action_std
    ):
        raise ValueError(
            "The final action deviation must lie between 0.005 and the initial deviation"
        )
    if action_std_decay_updates is not None and not (
        action_std_final is not None
        and type(action_std_decay_updates) is int
        and action_std_decay_updates >= 1
    ):
        raise ValueError(
            "The deviation decay length needs a final deviation and at least one update"
        )
    if type(steps_per_env) is not int or not 12 <= steps_per_env <= 96 or steps_per_env % 6:
        raise ValueError("The rollout length is a multiple of 6 from 12 to 96 controls per robot")
    if destination_input not in (
        "none",
        "ghost",
        "ghost_v10",
        "ghost_v11",
        "terrain_v1",
        "terrain_v2",
        "terrain_v2_critic",
    ):
        raise ValueError(
            "The destination input is none, ghost, ghost_v10, ghost_v11, terrain_v1, "
            "terrain_v2 or terrain_v2_critic"
        )
    if gait_clock_scaling not in (None, "speed_v1") or (gait_clock_scaling and not gait_clock):
        raise ValueError("The gait clock scaling is None or speed_v1 and needs the gait clock")
    distribution = (
        "GaussianDistribution"
        if action_mean == "unbounded"
        else "locomotion.rl.ppo:BoundedMeanGaussian"
    )
    config = {
        "seed": seed,
        "num_steps_per_env": 24,
        "save_interval": 50,
        "obs_groups": {"actor": ["policy"], "critic": ["critic"]},
        "actor": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "elu",
            "obs_normalization": observation_normalization == "empirical",
            "distribution_cfg": {
                "class_name": distribution,
                "init_std": action_std,
                "std_type": "log",
            },
        },
        "critic": {
            "class_name": "MLPModel",
            "hidden_dims": [256, 256, 128],
            "activation": "elu",
            "obs_normalization": observation_normalization == "empirical",
        },
        "algorithm": {
            "class_name": "PPO",
            "value_loss_coef": 1.0,
            "use_clipped_value_loss": True,
            "clip_param": 0.2,
            "entropy_coef": 0.01,
            "num_learning_epochs": 5,
            "num_mini_batches": 4,
            "learning_rate": 1e-3,
            "schedule": "adaptive",
            "gamma": 0.99,
            "lam": 0.95,
            "desired_kl": 0.01,
            "max_grad_norm": 1.0,
            "rnd_cfg": None,
            "symmetry_cfg": None,
        },
        "logger": "tensorboard",
        "check_for_nan": True,
    }
    # Options appear only when selected, so earlier checkpoints keep their recorded configuration.
    if action_noise_correlation:
        config["actor"]["distribution_cfg"].update(
            noise_correlation=action_noise_correlation,
            class_name="locomotion.rl.ppo:CorrelatedBoundedMeanGaussian",
        )
    if command_segments == "bootstrap":
        config["algorithm"]["class_name"] = SEGMENT_CLASS
    if learning_rate_max is not None:
        config["algorithm"].update(
            class_name=RATE_CLASS,
            learning_rate_max=learning_rate_max,
            learning_rate=min(1e-3, learning_rate_max),
        )
    if learning_rate_min is not None:
        config["algorithm"]["learning_rate_min"] = learning_rate_min
    if action_std_final is not None:
        config["exploration"] = {"action_std_final": action_std_final}
    if action_std_decay_updates is not None:
        config["exploration"]["action_std_decay_updates"] = action_std_decay_updates
    if steps_per_env != 24:
        # Mini-batches grow with the rollout so each keeps the 768 samples of 128 robots x 24 controls / 4.
        config["num_steps_per_env"] = steps_per_env
        config["algorithm"]["num_mini_batches"] = steps_per_env // 6
    wrapper = {
        key: value
        for key, value, default in (
            ("observation_scaling", observation_scaling, "none"),
            ("command_segments", command_segments, "continuous"),
            ("gait_clock", gait_clock, 0),
            ("action_smoothing", action_smoothing, "none"),
            ("velocity_noise", velocity_noise, 0.0),
            ("destination_input", destination_input, "none"),
            ("gait_clock_scaling", gait_clock_scaling, None),
        )
        if value != default
    }
    if wrapper:
        config["environment_wrapper"] = wrapper
    return config


# The ppo_config options of each admitted checkpoint, named after that checkpoint;
# locomotion/tests/test_recipe.py compares each entry with the checkpoint's compatibility receipt.
RECIPES = {
    "c48_long": {
        "action_mean": "tanh",
        "observation_normalization": "none",
        "observation_scaling": "fixed",
        "command_segments": "bootstrap",
        "gait_clock": 60,
        "action_smoothing": "mean2",
        "velocity_noise": 0.5,
        "learning_rate_max": 3e-4,
        "learning_rate_min": None,
        "action_std": 0.15,
        "action_noise_correlation": 0.0,
        "action_std_final": 0.05,
        "action_std_decay_updates": None,
        "steps_per_env": 48,
    },
}


# Named reviewed experiments and the destination input each adds to the PPO inputs; the reward
# and task of each live in omniwalk/destination_task.py.
EXPERIMENT_INPUTS = {
    "destination_v1": "ghost",
    "destination_v2": "ghost",
    "destination_v3": "ghost",
    "destination_v4": "ghost",
    "destination_v5": "ghost",
    "destination_v6": "ghost",
    "destination_v7": "ghost",
    "destination_v8": "ghost",
    "destination_v9": "ghost",
    "destination_v10": "ghost",
    "destination_v11": "ghost",
    "destination_v12": "ghost",
    "destination_v13": "ghost",
    "destination_v14": "ghost",
    "destination_v15": "ghost",
    "destination_v16": "ghost",
    "destination_v17": "ghost",
    "destination_v18": "ghost",
    "destination_v19": "ghost",
    "destination_v20": "ghost_v10",
    "destination_v21": "ghost_v11",
    "destination_v22": "ghost_v11",
    "destination_v23": "ghost_v11",
    "destination_v24": "ghost_v11",
    "destination_v25": "ghost_v11",
    "destination_v26": "ghost_v11",
    "destination_v27": "ghost_v11",
    "destination_v28": "ghost_v11",
    "destination_v29": "ghost_v11",
    "destination_v30": "ghost_v11",
    "destination_v32": "ghost_v11",
    "destination_v33": "terrain_v1",
    "destination_v34": "ghost_v11",
    "destination_v35": "terrain_v2",
    "destination_v36": "terrain_v2",
    "destination_v37": "terrain_v2_critic",
    "destination_v38": "terrain_v2",
    "destination_v39": "terrain_v2",
    "destination_v40": "terrain_v2",
    "destination_v41": "terrain_v2",
    "destination_v42": "terrain_v2",
    "destination_v43": "terrain_v2",
    "destination_v44": "terrain_v2",
    "destination_v45": "terrain_v2",
    "destination_v46": "terrain_v2",
    "destination_v47": "terrain_v2",
    "destination_v48": "terrain_v2",
    "destination_v49": "terrain_v2",
    "destination_v50": "terrain_v2",
    "destination_v51": "terrain_v2",
    "destination_v52": "terrain_v2",
    "destination_v53": "terrain_v2",
    "destination_v54": "terrain_v2",
    "destination_v55": "terrain_v2",
    "destination_v56": "terrain_v2",
    "destination_v57": "terrain_v2",
    "destination_v58": "terrain_v2",
    "destination_v59": "terrain_v2",
    "destination_v60": "terrain_v2",
    "destination_v61": "terrain_v2",
    "destination_v62": "terrain_v2",
    "destination_v63": "terrain_v2",
    "destination_v64": "terrain_v2",
    "destination_v65": "terrain_v2",
    "destination_v66": "terrain_v2",
    "destination_v67": "terrain_v2",
}
# The speed-scaled gait clock of each named experiment that uses one (its reward's gait_clock_scaling;
# locomotion/tests/test_destination.py checks the two agree). Its wrapper records and applies it.
EXPERIMENT_CLOCKS = {
    name: "speed_v1"
    for name in (
        f"destination_v{k}"
        for k in (
            *range(16, 31),
            32,
            33,
            34,
            35,
            36,
            37,
            38,
            39,
            40,
            41,
            42,
            43,
            44,
            45,
            46,
            47,
            48,
            49,
            50,
            51,
            52,
            53,
            54,
            55,
            56,
            57,
            58,
            59,
            60,
            61,
            62,
            63,
            64,
            65,
            66,
            67,
        )
    )
}


def recipe_options(name, **overrides):
    """Return the options of a named recipe with the overrides in place of its values; 'none' names no options."""
    return {**({} if name == "none" else RECIPES[name]), **overrides}
