"""Check the native scene checks and their readback on a CPU stand-in, without claiming native evidence."""

import copy
import json
import re
import sys
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest import mock

import torch

from locomotion.evaluation.admission import verify_scene
from locomotion.omniwalk.adapter import OmniEnv
from locomotion.omniwalk.env_cfg import EnvConfig
from locomotion.robots.hexapod import (
    BODY_NAMES,
    ISAACLAB_REVISION,
    JOINT_NAMES,
    MASS_KG,
    MODEL_SHA256,
    sha,
)
from locomotion.terrain import TERRAIN_CONTACT_OFFSET_M, TERRAIN_REST_OFFSET_M

ROOT = Path(__file__).resolve().parents[3]
MODEL_PATH = ROOT / "robot/hexapod_mkii_updated_v1/model_rs05_mass_corrected.json"
SCENE_PRIM = "/physicsScene"
GROUND = "/Ground/mesh"
GROUND_MATERIAL = "/Ground/physicsMaterial"
N = 2


def value(item):
    return SimpleNamespace(Get=lambda: item)


class Prim:
    """A USD prim stand-in that holds its attributes, applied schemas and, for an articulation root, its solver counts."""

    def __init__(self, solver=None, apis=(), type_name="Xform", **attributes):
        self.solver, self.attributes, self.apis, self.type_name = (
            solver,
            attributes,
            apis,
            type_name,
        )

    def GetAttribute(self, name):
        return value(self.attributes[name])

    def HasAPI(self, api):
        return getattr(api, "schema", api) in self.apis

    def GetTypeName(self):
        return self.type_name

    def GetPath(self):
        return self.attributes.get("path")

    def IsValid(self):
        return True


def articulation_api(prim):
    return SimpleNamespace(
        GetSolverPositionIterationCountAttr=lambda: value(prim.solver[0]),
        GetSolverVelocityIterationCountAttr=lambda: value(prim.solver[1]),
    )


def collision_api(prim):
    return SimpleNamespace(
        GetContactOffsetAttr=lambda: value(prim.attributes.get("physxCollision:contactOffset")),
        GetRestOffsetAttr=lambda: value(prim.attributes.get("physxCollision:restOffset")),
    )


def attribute_api(prefix, schema=None):
    """A schema stand-in whose Get<Name>Attr readers map to ``prefix:camelName`` attributes."""

    def api(prim):
        class Reader:
            def __getattr__(self, name):
                field = name[3:-4]
                key = prefix + ":" + field[0].lower() + field[1:]
                return lambda: value(prim.attributes.get(key))

        return Reader()

    api.schema = schema
    return api


def material_binding_api(prim):
    def bound(materialPurpose):
        material = prim.attributes.get("material:binding:physics")
        return (SimpleNamespace(GetPrim=lambda: material) if material else None), None

    return SimpleNamespace(ComputeBoundMaterial=bound)


PXR = SimpleNamespace(
    PhysxSchema=SimpleNamespace(
        PhysxArticulationAPI=articulation_api,
        PhysxCollisionAPI=collision_api,
        PhysxMaterialAPI=attribute_api("physxMaterial"),
    ),
    UsdPhysics=SimpleNamespace(
        CollisionAPI="CollisionAPI",
        MeshCollisionAPI=attribute_api("physics", "MeshCollisionAPI"),
        MaterialAPI=attribute_api("physics"),
    ),
    UsdShade=SimpleNamespace(MaterialBindingAPI=material_binding_api),
)


class Scene:
    """Stand in for IsaacEnv with the canonical values that the scene checks read.

    The native joint and body orders differ from the canonical orders, so the checks must index them.
    A reset clamps the joints to the native limits, as PhysX does.
    """

    def __init__(self, output):
        model = json.loads(MODEL_PATH.read_text())
        joints = {joint["name"]: joint for joint in model["joints"]}
        links = {link["name"]: link for link in model["links"]}
        config = EnvConfig(num_envs=N)
        self.cfg = SimpleNamespace(
            num_envs=N,
            device="cpu",
            declaration=config.declaration,
            floor=config.floor,
            floor_contact_offset_m=config.floor_contact_offset_m,
            floor_rest_offset_m=config.floor_rest_offset_m,
        )
        self.output, self.model, self.events = Path(output), model, []
        self.torch = lambda x: torch.as_tensor(x, dtype=torch.float32)
        self.joint_names, self.body_names = list(JOINT_NAMES), list(BODY_NAMES)
        self.native_joint_names, self.native_body_names = (
            self.joint_names[::-1],
            self.body_names[::-1],
        )
        self.joint_index = torch.tensor(
            [self.native_joint_names.index(name) for name in JOINT_NAMES]
        )
        self.lower = self.torch([joints[name]["lower"] for name in JOINT_NAMES])
        self.upper = self.torch([joints[name]["upper"] for name in JOINT_NAMES])
        self.neutral = self.torch([0.0, -0.3, 0.4] * 6)
        self.reset_height, self.origins = 0.12, self.torch([[0.0, 0.0, 0.0], [2.0, 0.0, 0.0]])
        self.roots = [f"/World/envs/env_{index}/Robot" for index in range(N)]
        self.scene_prim_path, self.ground_collision_path, self.native_errors = (
            SCENE_PRIM,
            GROUND,
            [],
        )
        limits = self.torch(
            [[joints[name]["lower"], joints[name]["upper"]] for name in self.native_joint_names]
        )
        self.native = {
            "get_dof_limits": limits.expand(N, -1, -1).clone(),
            "get_masses": self.torch([links[name]["mass"] for name in self.native_body_names])
            .expand(N, -1)
            .clone(),
            "get_dof_max_velocities": torch.full((N, 18), 33.0),
            "get_material_properties": self.torch([1.0, 1.0, 0.0]).expand(N, 153, -1).clone(),
            "get_contact_offsets": torch.full((N, 153), 0.001),
            "get_rest_offsets": torch.zeros(N, 153),
        }
        for getter in ("get_dof_stiffnesses", "get_dof_dampings", "get_dof_armatures"):
            self.native[getter] = torch.zeros(N, 18)
        self.prims = {
            root + "/body": Prim([32, 0], **{"physxArticulation:enabledSelfCollisions": True})
            for root in self.roots
        }
        self.prims[SCENE_PRIM] = Prim(
            **{
                "physxScene:timeStepsPerSecond": 400,
                "physxScene:enableExternalForcesEveryIteration": True,
            }
        )
        self.prims[GROUND_MATERIAL] = Prim(
            path=GROUND_MATERIAL,
            **{
                "physics:staticFriction": 1.0,
                "physics:dynamicFriction": 1.0,
                "physics:restitution": 0.0,
                "physxMaterial:frictionCombineMode": "multiply",
                "physxMaterial:restitutionCombineMode": "multiply",
            },
        )
        self.prims[GROUND] = Prim(
            apis=("CollisionAPI", "MeshCollisionAPI"),
            type_name="Mesh",
            **{
                "physics:approximation": "none",
                "material:binding:physics": self.prims[GROUND_MATERIAL],
            },
        )
        self.sdf_count, self.sdf_valid = N * 153, True
        self.sim = SimpleNamespace(
            get_physics_step_count=lambda: 0,
            stage=SimpleNamespace(GetPrimAtPath=lambda path: self.prims[path]),
            physics_sim_view=SimpleNamespace(
                create_sdf_shape_view=lambda pattern, count: SimpleNamespace(
                    count=self.sdf_count, check=lambda: self.sdf_valid
                )
            ),
        )
        self.commands, self.history = torch.zeros(N, 3), torch.zeros(N, 5, 42)
        self.held, self.previous_action = self.neutral.expand(N, -1).clone(), torch.zeros(N, 18)
        self.episode_length_buf = torch.zeros(N, dtype=torch.long)
        self.saved = []
        # OmniEnv aliases these; neither the scene checks nor the reset reads them.
        for name in (
            "reference_metadata",
            "geometry_path",
            "geometry_meta",
            "robot",
            "motor",
            "view",
            "contact",
            "sensor_map",
            "inverse_joint_index",
            "body_index",
            "toe_index",
            "toe_local",
            "root_com_local",
            "native_array",
            "telemetry",
            "total_controls",
        ):
            setattr(self, name, None)

    def get(self, getter):
        return self.native[getter].clone()

    def save(self, name, record):
        self.events.append("save")
        self.saved.append(copy.deepcopy(record))
        (self.output / name).write_text(json.dumps(record, indent=2, allow_nan=False) + "\n")

    def reset_selected(self, rows):
        self.events.append("reset")
        root = torch.cat((self.origins, self.torch([0.0, 0.0, 0.0, 0.0]).expand(N, -1)), -1)
        root[:, 2], root[:, 6] = self.reset_height, 1.0
        return {
            "q": torch.clamp(self.neutral, self.lower, self.upper).expand(N, -1).clone(),
            "dq": torch.zeros(N, 18),
            "root": root,
            "linear": torch.zeros(N, 3),
            "angular": torch.zeros(N, 3),
            "gravity": self.torch([0.0, 0.0, -1.0]).expand(N, -1).clone(),
            "toe_world": torch.zeros(N, 6, 3),
            "toe_body": torch.zeros(N, 6, 3),
        }


def construct(scene, verify=verify_scene):
    """Build OmniEnv over the stand-in, as cli.create_env and replay_native build it over IsaacEnv."""
    with mock.patch.dict(
        sys.modules,
        {
            "locomotion.omniwalk.env": SimpleNamespace(IsaacEnv=lambda *args, **kwargs: scene),
            "pxr": PXR,
        },
    ):
        return OmniEnv(scene.cfg, None, None, None, None, verify=verify)


CHUNKS = ["/Ground/chunk_0", "/Ground/chunk_1"]


def terrain_scene(output):
    """The stand-in with the chunked terrain floor in place of the single flat mesh."""
    scene = Scene(output)
    scene.terrain = SimpleNamespace(kind="height map stand-in")
    scene.ground_collision_paths = list(CHUNKS)
    for path in CHUNKS:
        scene.prims[path] = Prim(
            apis=("CollisionAPI", "MeshCollisionAPI"),
            type_name="Mesh",
            **{
                "physics:approximation": "none",
                "physxCollision:contactOffset": TERRAIN_CONTACT_OFFSET_M,
                "physxCollision:restOffset": TERRAIN_REST_OFFSET_M,
                "material:binding:physics": scene.prims[GROUND_MATERIAL],
            },
        )
    return scene


class SceneCheckTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.output = Path(directory.name)
        self.assertEqual(sha(MODEL_PATH), MODEL_SHA256)

    def test_canonical_scene_writes_the_readback_and_the_sdk_reset_recipe_before_the_first_reset(
        self,
    ):
        scene = Scene(self.output)
        env = construct(scene)
        # The three readback writes precede the first task reset.
        self.assertEqual(scene.events, ["save", "save", "save", "reset"])
        recipe = {"recipe_readbacks"}
        final = {"backend", "isaaclab_revision_expected", "episode_owner", "force_readback_phase"}
        checks = {
            "native_joint_names",
            "native_body_names",
            "canonical_joint_names",
            "root_paths",
            "mass_per_replica_kg",
            "masses",
            "limits",
            "native_max_velocity",
            "sdf_shapes",
            "solver_iterations_requested",
            "implicit_drive_and_armature_zero",
            "asset_mass_target_kg",
            "config",
        }
        self.assertEqual(
            [set(record) for record in scene.saved],
            [checks, checks | recipe, checks | recipe | final],
        )
        record = json.loads((self.output / "native_readback.json").read_text())
        self.assertEqual(record, env.native_readback)
        self.assertEqual(record["backend"], "direct")
        self.assertEqual(record["isaaclab_revision_expected"], ISAACLAB_REVISION)
        self.assertEqual(record["sdf_shapes"], N * 153)
        self.assertEqual(record["root_paths"], scene.roots)
        self.assertEqual(record["config"], EnvConfig(num_envs=N).declaration())
        self.assertEqual(
            (record["asset_mass_target_kg"], record["solver_iterations_requested"]),
            (MASS_KG, [32, 0]),
        )
        for total in record["mass_per_replica_kg"]:
            self.assertAlmostEqual(total, MASS_KG, places=4)
        torch.testing.assert_close(
            torch.tensor(record["limits"]),
            torch.stack((scene.lower, scene.upper), -1).expand(N, -1, -1),
        )
        self.assertEqual(set(record["recipe_readbacks"]), {"after_sdk_reset"})
        readback = record["recipe_readbacks"]["after_sdk_reset"]
        self.assertEqual(readback["solver_attributes"], {root: [32, 0] for root in scene.roots})
        self.assertEqual(
            readback["scene_attributes"],
            {
                "physxScene:timeStepsPerSecond": 400,
                "physxScene:enableExternalForcesEveryIteration": True,
            },
        )
        self.assertEqual(readback["native_error_events"], [])
        self.assertEqual(
            readback["floor_collider"],
            {
                "path": GROUND,
                "type": "Mesh",
                "kind": "isaaclab_mesh",
                "contact_offset_m": None,
                "rest_offset_m": None,
                "approximation": "none",
                "material": {
                    "path": GROUND_MATERIAL,
                    "static_friction": 1.0,
                    "dynamic_friction": 1.0,
                    "restitution": 0.0,
                    "friction_combine_mode": "multiply",
                    "restitution_combine_mode": "multiply",
                },
            },
        )
        self.assertFalse(readback["backend_iteration_introspection"])
        self.assertEqual(torch.tensor(readback["material_properties"]).shape, (N, 153, 3))
        self.assertEqual(torch.tensor(readback["contact_offsets"]).shape, (N, 153))

    def test_neutral_outside_the_limits_fails_the_scene_check_before_the_reset(self):
        # PhysX clamps an out-of-limit neutral; without the scene check, the reset readback reports the clamp.
        unchecked = Scene(self.output)
        unchecked.neutral[2] = unchecked.lower[2] - 0.05
        with self.assertRaisesRegex(RuntimeError, "Selected joint reset readback differs"):
            construct(unchecked, verify=None)
        scene = Scene(self.output)
        scene.neutral[2] = scene.lower[2] - 0.05
        with self.assertRaisesRegex(
            ValueError, "Reference neutral violates named canonical limits"
        ):
            construct(scene)
        self.assertEqual(scene.events, [])

    def test_each_changed_scene_value_raises_its_named_error(self):
        def native(getter, index, item):
            return lambda scene: scene.native[getter].__setitem__(index, item)

        def prim(path, name, item):
            return lambda scene: scene.prims[path].attributes.__setitem__(name, item)

        root = "/World/envs/env_1/Robot"
        cases = [
            (
                "limits",
                native("get_dof_limits", (1, 4, 1), 1.0),
                ValueError,
                "Actual native limits differ from the canonical model",
            ),
            (
                "neutral",
                lambda scene: scene.neutral.__setitem__(0, float(scene.upper[0]) + 0.01),
                ValueError,
                "Reference neutral violates named canonical limits",
            ),
            (
                "masses",
                native("get_masses", (0, 3), 0.5),
                ValueError,
                "Native body mass differs from canonical mass ledger",
            ),
            *[
                (
                    getter,
                    native(getter, (1, 7), 0.2),
                    ValueError,
                    "Unexpected implicit drive or armature: " + getter,
                )
                for getter in ("get_dof_stiffnesses", "get_dof_dampings", "get_dof_armatures")
            ],
            (
                "sdf count",
                lambda scene: setattr(scene, "sdf_count", N * 153 - 1),
                ValueError,
                "Detailed collision shapes absent or changed",
            ),
            (
                "sdf check",
                lambda scene: setattr(scene, "sdf_valid", False),
                ValueError,
                "Detailed collision shapes absent or changed",
            ),
            (
                "solver",
                lambda scene: setattr(scene.prims[root + "/body"], "solver", [16, 0]),
                ValueError,
                "Actual solver attributes differ: " + root,
            ),
            (
                "self collision",
                prim(root + "/body", "physxArticulation:enabledSelfCollisions", False),
                ValueError,
                "Canonical self-collision setting differs",
            ),
            (
                "timing",
                prim(SCENE_PRIM, "physxScene:timeStepsPerSecond", 200),
                ValueError,
                "Actual simulation timing/external-force configuration differs",
            ),
            (
                "materials",
                native("get_material_properties", (0, 7, 0), 0.8),
                ValueError,
                "Actual native materials differ",
            ),
            (
                "contact offsets",
                native("get_contact_offsets", (1, 9), 0.002),
                ValueError,
                "Actual native contact/rest offsets differ",
            ),
            (
                "rest offsets",
                native("get_rest_offsets", (0, 2), 0.001),
                ValueError,
                "Actual native contact/rest offsets differ",
            ),
            (
                "authored floor contact offset",
                prim(GROUND, "physxCollision:contactOffset", 0.02),
                ValueError,
                "Actual floor collider differs from the declared floor",
            ),
            (
                "authored floor rest offset",
                prim(GROUND, "physxCollision:restOffset", 0.0),
                ValueError,
                "Actual floor collider differs from the declared floor",
            ),
            (
                "floor collider",
                lambda scene: setattr(scene.prims[GROUND], "apis", ()),
                ValueError,
                "Actual floor collider differs from the declared floor",
            ),
            (
                "floor material unbound",
                prim(GROUND, "material:binding:physics", None),
                ValueError,
                "Actual floor has no bound physics material",
            ),
            (
                "floor friction",
                prim(GROUND_MATERIAL, "physics:dynamicFriction", 0.5),
                ValueError,
                "Actual floor material differs from the canonical material",
            ),
            (
                "floor combine mode",
                prim(GROUND_MATERIAL, "physxMaterial:frictionCombineMode", "average"),
                ValueError,
                "Actual floor material differs from the canonical material",
            ),
            (
                "native error",
                lambda scene: scene.native_errors.append("contact buffer overflow"),
                RuntimeError,
                "Native physics errors prevent acceptance",
            ),
        ]
        for name, change, error, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                scene = Scene(directory)
                change(scene)
                with self.assertRaisesRegex(error, "^" + re.escape(message) + "$"):
                    construct(scene)
                self.assertNotIn("reset", scene.events)

    def test_terrain_chunks_verify_against_the_terrain_recipe(self):
        env = construct(terrain_scene(self.output))
        self.assertEqual(
            env.native_readback["recipe_readbacks"]["after_sdk_reset"]["floor_collider"],
            {
                "path": CHUNKS[0],
                "type": "Mesh",
                "kind": "isaaclab_mesh",
                "contact_offset_m": TERRAIN_CONTACT_OFFSET_M,
                "rest_offset_m": TERRAIN_REST_OFFSET_M,
                "approximation": "none",
                "material": {
                    "path": GROUND_MATERIAL,
                    "static_friction": 1.0,
                    "dynamic_friction": 1.0,
                    "restitution": 0.0,
                    "friction_combine_mode": "multiply",
                    "restitution_combine_mode": "multiply",
                },
                "chunk_count": len(CHUNKS),
            },
        )

    def test_each_changed_terrain_chunk_raises_its_named_error(self):
        # Every fault sits on the second chunk: the readback must check past the first.
        target = CHUNKS[1]

        def attribute(name, item):
            return lambda scene: scene.prims[target].attributes.__setitem__(name, item)

        floor_error = "Actual floor collider differs from the declared floor"
        cases = [
            (
                "contact offset",
                attribute("physxCollision:contactOffset", TERRAIN_CONTACT_OFFSET_M + 0.01),
                floor_error,
            ),
            (
                "unauthored offsets",
                lambda scene: [
                    scene.prims[target].attributes.pop("physxCollision:contactOffset"),
                    scene.prims[target].attributes.pop("physxCollision:restOffset"),
                ],
                floor_error,
            ),
            (
                "collision api",
                lambda scene: setattr(scene.prims[target], "apis", ()),
                floor_error,
            ),
            (
                "approximation",
                attribute("physics:approximation", "convexHull"),
                floor_error,
            ),
            (
                "material unbound",
                attribute("material:binding:physics", None),
                "Actual floor has no bound physics material",
            ),
        ]
        for name, change, message in cases:
            with self.subTest(name=name), tempfile.TemporaryDirectory() as directory:
                scene = terrain_scene(directory)
                change(scene)
                with self.assertRaisesRegex(ValueError, "^" + re.escape(message) + "$"):
                    construct(scene)
                self.assertNotIn("reset", scene.events)


if __name__ == "__main__":
    unittest.main()
