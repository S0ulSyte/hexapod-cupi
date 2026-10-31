"""Hash the runtime closure, including nested implementation modules."""

import ast
import hashlib
import importlib.util
import json
from pathlib import Path

SCHEMA = "hexapod_runtime_sources_v2"
MANIFEST = "RUNTIME_SOURCES.json"


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def runtime_paths(package):
    """Select the files a native allocation runs.

    The tripod task runs through stock Isaac Lab entry points. Evaluation and AMP
    read the motion features in amp/features.py, and prepare copies the other
    prior modules it needs for AMP or replay.
    """
    package = Path(package)
    return sorted(
        path
        for path in package.rglob("*.py")
        if not set(path.relative_to(package).parts)
        & {"tests", "__pycache__", "hexapod", "surrogate"}
        and (
            "amp" not in path.relative_to(package).parts
            or path.name in {"__init__.py", "features.py"}
        )
    )


def source_files(package):
    package = Path(package)
    return {path.relative_to(package).as_posix(): digest(path) for path in runtime_paths(package)}


def physics_files(files, package=None):
    """Follow project imports from the native environment, its scene checks, the standing capture and the scorer."""
    package = Path(package) if package is not None else Path(__file__).resolve().parents[1]
    pending, seen = (
        [
            "omniwalk/adapter.py",
            "omniwalk/env_cfg.py",
            "omniwalk/env.py",
            "evaluation/admission.py",
            "evaluation/capture.py",
            "evaluation/scoring.py",
        ],
        set(),
    )
    while pending:
        name = pending.pop()
        if name in seen or name not in files:
            continue
        seen.add(name)
        path = Path(name)
        parent = "locomotion." + ".".join(path.parent.parts) if path.parent.parts else "locomotion"
        for directory in (path.parent, *path.parent.parents):
            pending.append((directory / "__init__.py").as_posix())
        for node in ast.walk(ast.parse((package / name).read_text())):
            modules = []
            if isinstance(node, ast.Import):
                modules = [alias.name for alias in node.names]
            elif isinstance(node, ast.ImportFrom):
                base = importlib.util.resolve_name("." * node.level + (node.module or ""), parent)
                modules = [base, *(base + "." + alias.name for alias in node.names)]
            for module in modules:
                if module == "locomotion":
                    pending.append("__init__.py")
                elif module.startswith("locomotion."):
                    relative = module.removeprefix("locomotion.").replace(".", "/")
                    pending.extend((relative + ".py", relative + "/__init__.py"))
    return {name: files[name] for name in sorted(seen)}


def runtime_manifest(source):
    source = Path(source)
    paths = sorted(source.rglob("*.py"))
    if source.is_symlink() or any(path.is_symlink() for path in source.rglob("*")):
        raise ValueError("Runtime dependency uses a symbolic path")
    return {
        "schema": SCHEMA,
        "files": {
            path.relative_to(source).as_posix(): digest(path)
            for path in paths
            if "__pycache__" not in path.parts
        },
    }


def write_runtime(source):
    value = runtime_manifest(source)
    (Path(source) / MANIFEST).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
    return value


def verify_runtime(source):
    path = Path(source) / MANIFEST
    recorded = json.loads(path.read_text())
    if recorded != runtime_manifest(source):
        raise ValueError("Runtime dependency manifest differs")
    return recorded
