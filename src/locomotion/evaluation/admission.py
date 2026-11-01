"""Check the native scene and bind a run to its model and recorded native standing checks.

Each entry point passes ``verify_scene`` to ``OmniEnv`` as ``verify``. The constructor calls it after the
Isaac Lab construction and before the first task reset, so the checks and the ``after_sdk_reset`` readback
read the scene that the SDK reset left.
"""

import argparse
import json
from pathlib import Path

from locomotion.omniwalk.env_cfg import EnvConfig
from locomotion.robots.hexapod import (
    ISAACLAB_REVISION,
    JOINT_NAMES,
    MASS_KG,
    MODEL_SHA256,
    USD_SHA256,
    sha,
)


def verify_scene(env):
    """Check the native scene against the canonical model, then save and return its readback.

    ``IsaacEnv`` guards the layout that its bindings index: decimation, articulation count, names, root
    order, contact matrix and toe points. This function checks the evidence against the canonical model.
    The environment holds the readback as ``native_readback``; captures and the evaluator read it there.
    """
    import torch

    cfg = env.cfg
    native_limits = env.get("get_dof_limits")[:, env.joint_index]
    named = {j["name"]: j for j in env.model["joints"]}
    expected_limits = env.torch([[named[n]["lower"], named[n]["upper"]] for n in JOINT_NAMES])
    if not torch.allclose(
        native_limits, expected_limits.expand(cfg.num_envs, -1, -1), atol=2e-6, rtol=0
    ):
        raise ValueError("Actual native limits differ from the canonical model")
    if torch.any(env.neutral < env.lower) or torch.any(env.neutral > env.upper):
        raise ValueError("Reference neutral violates named canonical limits")
    masses = env.get("get_masses")
    named_links = {b["name"]: b for b in env.model["links"]}
    expected_masses = env.torch([named_links[n]["mass"] for n in env.native_body_names])
    if not torch.allclose(masses, expected_masses.expand(cfg.num_envs, -1), atol=1e-6, rtol=0):
        raise ValueError("Native body mass differs from canonical mass ledger")
    for getter in ("get_dof_stiffnesses", "get_dof_dampings", "get_dof_armatures"):
        if torch.any(env.get(getter) != 0):
            raise ValueError("Unexpected implicit drive or armature: " + getter)
    sdf = env.sim.physics_sim_view.create_sdf_shape_view(
        "/World/envs/env_*/Robot/*/collisions/part_*", 1
    )
    if sdf.count != cfg.num_envs * 153 or not sdf.check():
        raise ValueError("Detailed collision shapes absent or changed")
    readback = {
        "native_joint_names": env.native_joint_names,
        "native_body_names": env.native_body_names,
        "canonical_joint_names": env.joint_names,
        "root_paths": env.roots,
        "mass_per_replica_kg": masses.sum(-1).cpu().tolist(),
        "masses": masses.cpu().tolist(),
        "limits": native_limits.cpu().tolist(),
        "native_max_velocity": env.get("get_dof_max_velocities")[:, env.joint_index].cpu().tolist(),
        "sdf_shapes": sdf.count,
        "solver_iterations_requested": [32, 0],
        "implicit_drive_and_armature_zero": True,
        "asset_mass_target_kg": MASS_KG,
        "config": cfg.declaration(),
    }
    env.native_readback = readback
    env.save("native_readback.json", readback)
    verify_native_recipe(env, "after_sdk_reset")
    readback["backend"] = "direct"
    readback["isaaclab_revision_expected"] = ISAACLAB_REVISION
    readback["episode_owner"] = "external RewardTask; stock done flags are zero"
    readback["force_readback_phase"] = (
        "native force after stock write before physics; state and contact after stock update"
    )
    env.save("native_readback.json", readback)
    return readback


def verify_native_recipe(env, phase):
    """Read actual scene/articulation/material values, not only authored intent.

    The MuJoCo surrogate has no PhysX scene; it checks its own fixed recipe with its method of this name.
    """
    if hasattr(env, "verify_native_recipe"):
        return env.verify_native_recipe(phase)
    import torch
    from pxr import PhysxSchema, UsdPhysics, UsdShade

    solver = {}
    for root in env.roots:
        prim = env.sim.stage.GetPrimAtPath(root + "/body")
        api = PhysxSchema.PhysxArticulationAPI(prim)
        values = [
            api.GetSolverPositionIterationCountAttr().Get(),
            api.GetSolverVelocityIterationCountAttr().Get(),
        ]
        if values != [32, 0]:
            raise ValueError("Actual solver attributes differ: " + root)
        if prim.GetAttribute("physxArticulation:enabledSelfCollisions").Get() is not True:
            raise ValueError("Canonical self-collision setting differs")
        solver[root] = values
    scene = env.sim.stage.GetPrimAtPath(env.isaac.scene_prim_path)
    scene_values = {
        name: scene.GetAttribute(name).Get()
        for name in (
            "physxScene:timeStepsPerSecond",
            "physxScene:enableExternalForcesEveryIteration",
        )
    }
    if (
        scene_values["physxScene:timeStepsPerSecond"] != 400
        or scene_values["physxScene:enableExternalForcesEveryIteration"] is not True
    ):
        raise ValueError("Actual simulation timing/external-force configuration differs")
    material = env.get("get_material_properties")
    offsets = env.get("get_contact_offsets")
    rests = env.get("get_rest_offsets")
    if material.shape != (env.num_envs, 153, 3) or not torch.allclose(
        material, env.torch([1.0, 1.0, 0.0]).expand_as(material), atol=1e-7, rtol=0
    ):
        raise ValueError("Actual native materials differ")
    if (
        offsets.shape != (env.num_envs, 153)
        or not torch.allclose(offsets, torch.full_like(offsets, 0.001), atol=1e-9, rtol=0)
        or torch.any(rests != 0)
    ):
        raise ValueError("Actual native contact/rest offsets differ")
    terrain = getattr(env.isaac, "terrain", None) is not None
    if terrain:
        # The terrain floor is chunked colliders sharing the recipe that locomotion.terrain
        # declares; every chunk is read back against it, including the exact-triangle
        # approximation. The stock floor keeps the config-declared contract, where None
        # means no authored offsets.
        from locomotion.terrain import TERRAIN_CONTACT_OFFSET_M, TERRAIN_REST_OFFSET_M

        floor_paths = list(env.isaac.ground_collision_paths)
        declared = (TERRAIN_CONTACT_OFFSET_M, TERRAIN_REST_OFFSET_M)
    else:
        floor_paths = [env.isaac.ground_collision_path]
        declared = (env.cfg.floor_contact_offset_m, env.cfg.floor_rest_offset_m)
    floor = None
    for path in floor_paths:
        collider = env.sim.stage.GetPrimAtPath(path)
        collision = PhysxSchema.PhysxCollisionAPI(collider)
        entry = {
            "path": path,
            "type": collider.GetTypeName(),
            "kind": env.cfg.floor,
            "contact_offset_m": collision.GetContactOffsetAttr().Get(),
            "rest_offset_m": collision.GetRestOffsetAttr().Get(),
            "approximation": (
                UsdPhysics.MeshCollisionAPI(collider).GetApproximationAttr().Get()
                if collider.HasAPI(UsdPhysics.MeshCollisionAPI)
                else None
            ),
        }
        actual = (entry["contact_offset_m"], entry["rest_offset_m"])
        if (
            not collider.HasAPI(UsdPhysics.CollisionAPI)
            or (declared[0] is None and actual != (None, None))
            or (
                declared[0] is not None
                and (None in actual or any(abs(a - d) > 1e-9 for a, d in zip(actual, declared)))
            )
            or (terrain and entry["approximation"] != "none")
        ):
            raise ValueError("Actual floor collider differs from the declared floor")
        bound, _ = UsdShade.MaterialBindingAPI(collider).ComputeBoundMaterial(
            materialPurpose="physics"
        )
        material_prim = bound.GetPrim() if bound else None
        if material_prim is None or not material_prim.IsValid():
            raise ValueError("Actual floor has no bound physics material")
        physics_api, physx_api = (
            UsdPhysics.MaterialAPI(material_prim),
            PhysxSchema.PhysxMaterialAPI(material_prim),
        )
        entry["material"] = {
            "path": str(material_prim.GetPath()),
            "static_friction": physics_api.GetStaticFrictionAttr().Get(),
            "dynamic_friction": physics_api.GetDynamicFrictionAttr().Get(),
            "restitution": physics_api.GetRestitutionAttr().Get(),
            "friction_combine_mode": physx_api.GetFrictionCombineModeAttr().Get(),
            "restitution_combine_mode": physx_api.GetRestitutionCombineModeAttr().Get(),
        }
        expected_material = {
            "static_friction": 1.0,
            "dynamic_friction": 1.0,
            "restitution": 0.0,
            "friction_combine_mode": "multiply",
            "restitution_combine_mode": "multiply",
        }
        if any(
            entry["material"][key] is None or entry["material"][key] != value
            for key, value in expected_material.items()
        ):
            raise ValueError("Actual floor material differs from the canonical material")
        if floor is None:
            floor = entry
    if terrain:
        floor["chunk_count"] = len(floor_paths)
    if env.native_errors:
        raise RuntimeError("Native physics errors prevent acceptance")
    env.native_readback.setdefault("recipe_readbacks", {})[phase] = {
        "solver_attributes": solver,
        "scene_attributes": scene_values,
        "floor_collider": floor,
        "material_properties": material.cpu().tolist(),
        "contact_offsets": offsets.cpu().tolist(),
        "rest_offsets": rests.cpu().tolist(),
        "backend_iteration_introspection": False,
        "native_error_events": list(env.native_errors),
    }
    env.save("native_readback.json", env.native_readback)


def require_admission(path, identity, cfg):
    """New training can consume only authentic matching one/batch standing passes."""
    if path is None:
        raise ValueError("Training/recording requires fresh model/source-bound standing admission")
    value = json.loads(Path(path).read_text())
    return validate_admission(value, identity, cfg)


def validate_admission(value, identity, cfg, mounts=None):
    def resolve(name):
        path = Path(name)
        for target, source in (mounts or {}).items():
            if path.is_relative_to(target):
                return Path(source) / path.relative_to(target)
        return path

    if value.get("schema") != "hexapod_locomotion_standing_admission_v1":
        raise ValueError("Wrong standing admission schema")
    if value.get("model_sha256") != MODEL_SHA256 or value.get("usd_sha256") != USD_SHA256:
        raise ValueError("Standing admission model differs")
    if (
        value.get("physics_source_files") != identity["physics_source_files"]
        or value.get("stance_sha256") != identity["stance_sha256"]
    ):
        raise ValueError("Standing source or neutral stance differs")
    if value.get("physics_config") != identity["physics_config"]:
        raise ValueError("Standing physics configuration differs")
    for key in ("geometry_sha256", "geometry_extrema_sha256"):
        if value.get(key) != identity[key]:
            raise ValueError("Standing geometry identity differs")
    if type(value.get("num_envs")) is not int or value["num_envs"] < 32:
        raise ValueError("At least 32 same-source standing replicas are required")
    for name, count in (("one", 1), ("batch", value.get("num_envs"))):
        row = value.get(name, {})
        report_path = resolve(row.get("report_path", ""))
        if not report_path.is_file() or sha(report_path) != row.get("report_sha256"):
            raise ValueError("Missing/changed actual standing report: " + name)
        report = json.loads(report_path.read_text())
        if (
            report.get("all_pass") is not True
            or report.get("num_envs") != count
            or report.get("controls") != 1000
            or report.get("substeps") != 8000
            or len(report.get("replicas", [])) != count
            or [r.get("env") for r in report["replicas"]] != list(range(count))
            or not all(
                type(r.get("env")) is int
                and r.get("pass") is True
                and r.get("failed_physical_bounds") == []
                and r.get("quiet", {}).get("pass") is True
                and r.get("quiet", {}).get("failed_bounds") == []
                for r in report["replicas"]
            )
        ):
            raise ValueError("Standing quality rejected: " + name)
        state_path = resolve(row.get("state_path", ""))
        if not state_path.is_file() or sha(state_path) != row.get("state_sha256"):
            raise ValueError("Missing/changed native standing acquisition: " + name)
        state = json.loads(state_path.read_text())
        if (
            state.get("status") != "completed"
            or state.get("standing_gate_pass") is not True
            or state.get("errors")
        ):
            raise ValueError("Native standing acquisition failed: " + name)
        observed = state.get("identity", {})
        for key in (
            "physics_source_files",
            "physics_config",
            "stance_sha256",
            "model_sha256",
            "usd_sha256",
            "geometry_sha256",
            "geometry_extrema_sha256",
        ):
            if observed.get(key) != identity[key]:
                raise ValueError("Native standing identity differs: " + name + ":" + key)
        if observed.get("config", {}).get("num_envs") != count:
            raise ValueError("Native standing replica count differs: " + name)
    if cfg.num_envs != 1 and value.get("num_envs") != cfg.num_envs:
        raise ValueError("Training replica layout lacks matching standing admission")
    return value


def create(one, batch, output, inputs):
    """Recompute both native captures before writing admission and launch inputs."""
    # The capture module imports torch; keep it off the host until a capture is scored.
    from locomotion.evaluation.capture import score_diagnostic

    one, batch, output, inputs = map(Path, (one, batch, output, inputs))
    roots = {"one": one.resolve(), "batch": batch.resolve()}
    states = {name: json.loads((root / "state.json").read_text()) for name, root in roots.items()}
    identity = states["batch"]["identity"]
    count = identity["config"]["num_envs"]
    keys = (
        "model_sha256",
        "usd_sha256",
        "physics_source_files",
        "physics_config",
        "stance_sha256",
        "geometry_sha256",
        "geometry_extrema_sha256",
    )
    value = {
        "schema": "hexapod_locomotion_standing_admission_v1",
        "num_envs": count,
        **{key: identity[key] for key in keys},
    }
    declared = json.loads(inputs.read_text())
    mounts = {}
    for name, root in roots.items():
        report_path = root / "standing/standing_report.json"
        report = json.loads(report_path.read_text())
        if score_diagnostic(root / "standing") != report:
            raise ValueError("Standing report differs from raw recomputation: " + name)
        target = "/standing_" + name
        mounts[target] = root
        value[name] = {
            "report_path": target + "/standing/standing_report.json",
            "report_sha256": sha(report_path),
            "state_path": target + "/state.json",
            "state_sha256": sha(root / "state.json"),
        }
        declared["extra_mounts"].append([str(root), target])
        for path in (report_path, root / "state.json"):
            declared["input_files"][str(path)] = sha(path)
    validate_admission(value, identity, EnvConfig(num_envs=count), mounts)
    output.mkdir(parents=True, exist_ok=False)
    path = output / "admission.json"
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    declared["extra_mounts"].append([str(output.resolve()), "/admission"])
    declared["input_files"][str(path.resolve())] = sha(path)
    (output / "inputs.json").write_text(json.dumps(declared, indent=2, allow_nan=False) + "\n")
    return value


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ("one", "batch", "output", "inputs"):
        parser.add_argument("--" + name, type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(create(**vars(args)), indent=2))


if __name__ == "__main__":
    main()
