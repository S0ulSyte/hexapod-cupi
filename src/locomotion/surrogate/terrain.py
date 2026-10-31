"""Height maps for surrogate scenes; the implementation lives in ``locomotion/terrain.py``,
the neutral home both the surrogate and the native terrain floor import (native packs hold no
surrogate file)."""

from locomotion.terrain import FEATURES, Terrain, feature_height, validate

__all__ = ["FEATURES", "Terrain", "feature_height", "validate"]
