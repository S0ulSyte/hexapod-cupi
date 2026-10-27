"""Height maps for surrogate scenes, built from a list of terrain features.

A scene places everything in the viewer's navigation frame: ``forward`` and ``left`` in metres from the world
origin, with a heading of 0 facing forward and positive headings turning left. MuJoCo's world has x = left and
y = -forward, so a robot with the identity orientation faces forward. Heights add up over the features:

- ``box``: ``center``, ``size`` [along, across], ``height`` (negative for a pit), ``heading_deg``;
- ``ramp``: ``center``, ``size``, ``rise``, ``heading_deg``; 0 at the back edge, ``rise`` at the front edge;
- ``steps``: ``center``, ``size``, ``rise``, ``count``, ``heading_deg``; ``count`` equal steps up to ``rise``;
- ``bump``: ``center``, ``radius``, ``height``; a raised cosine;
- ``noise``: ``amplitude``, ``wavelength``, ``seed``; smooth value noise in [-amplitude, amplitude] over the
  whole terrain.

``viewer/src/scene/terrain.js`` computes the same heights for the Scene page's editor; the viewer's node test
compares the two on ``viewer/test/fixtures/sample_terrain.json``.
"""

import math

import numpy as np

FEATURES = {
    "box": ("center", "size", "height"),
    "ramp": ("center", "size", "rise"),
    "steps": ("center", "size", "rise", "count"),
    "bump": ("center", "radius", "height"),
    "noise": ("amplitude", "wavelength", "seed"),
}


def _local(feature, forward, left):
    """Coordinates along and across a feature's heading, from its centre."""
    f0, l0 = feature["center"]
    angle = math.radians(feature.get("heading_deg", 0.0))
    df, dl = forward - f0, left - l0
    return df * math.cos(angle) + dl * math.sin(angle), -df * math.sin(angle) + dl * math.cos(angle)


def _hash(ix, iy, seed):
    """A value in [0, 1) for each integer lattice point; the same 32-bit arithmetic as terrain.js."""
    mask = 0xFFFFFFFF
    h = (ix * 374761393 + iy * 668265263 + seed * 2147483647) & mask
    h = ((h ^ (h >> 13)) * 1274126177) & mask
    h = h ^ (h >> 16)
    return h / 4294967296.0


def _noise(feature, forward, left):
    wavelength = feature["wavelength"]
    u, v = forward / wavelength, left / wavelength
    i, j = np.floor(u).astype(np.int64), np.floor(v).astype(np.int64)
    a, b = u - i, v - j
    a, b = a * a * (3 - 2 * a), b * b * (3 - 2 * b)
    seed = int(feature["seed"]) % 1000003
    c00, c10 = _hash(i, j, seed), _hash(i + 1, j, seed)
    c01, c11 = _hash(i, j + 1, seed), _hash(i + 1, j + 1, seed)
    value = (c00 * (1 - a) + c10 * a) * (1 - b) + (c01 * (1 - a) + c11 * a) * b
    return feature["amplitude"] * (2 * value - 1)


def feature_height(feature, forward, left):
    """One feature's height at arrays of forward and left positions, in metres."""
    kind = feature["type"]
    if kind == "noise":
        return _noise(feature, np.asarray(forward, float), np.asarray(left, float))
    forward, left = np.asarray(forward, float), np.asarray(left, float)
    if kind == "bump":
        r = np.hypot(forward - feature["center"][0], left - feature["center"][1])
        return np.where(
            r < feature["radius"],
            feature["height"] * 0.5 * (1 + np.cos(np.pi * r / feature["radius"])),
            0.0,
        )
    u, v = _local(feature, forward, left)
    length, width = feature["size"]
    inside = (np.abs(u) <= length / 2) & (np.abs(v) <= width / 2)
    if kind == "box":
        return np.where(inside, feature["height"], 0.0)
    share = np.clip(u / length + 0.5, 0.0, 1.0)
    if kind == "ramp":
        return np.where(inside, feature["rise"] * share, 0.0)
    count = int(feature["count"])
    step = np.minimum(np.floor(share * count), count - 1) + 1
    return np.where(inside, feature["rise"] * step / count, 0.0)


def validate(spec):
    """Raise ValueError unless ``spec`` is a terrain: extent, resolution and known, complete features."""
    size = spec.get("size_m")
    resolution = spec.get("resolution_m", 0.02)
    if (
        not isinstance(size, (list, tuple))
        or len(size) != 2
        or not all(isinstance(x, (int, float)) and 0.5 <= x <= 50 for x in size)
    ):
        raise ValueError("Terrain size_m is [forward, left] extents from 0.5 to 50 m")
    if not isinstance(resolution, (int, float)) or not 0.005 <= resolution <= 0.2:
        raise ValueError("Terrain resolution_m is from 0.005 to 0.2 m")
    if max(size) / resolution > 2000:
        raise ValueError("A terrain holds at most 2000 cells along each side")
    for feature in spec.get("features", []):
        kind = feature.get("type")
        if kind not in FEATURES:
            raise ValueError(f"Unknown terrain feature {kind!r}; known: {', '.join(FEATURES)}")
        missing = [k for k in FEATURES[kind] if k not in feature]
        if missing:
            raise ValueError(f"A {kind} needs {', '.join(missing)}")
        if kind == "steps" and (int(feature["count"]) != feature["count"] or feature["count"] < 1):
            raise ValueError("Steps need a whole count of at least 1")
        if kind in ("bump", "noise") and feature.get("radius", feature.get("wavelength")) <= 0:
            raise ValueError("A bump radius or noise wavelength is positive")
        if kind == "noise" and (int(feature["seed"]) != feature["seed"] or feature["seed"] < 0):
            raise ValueError("A noise seed is a whole number of at least 0")


class Terrain:
    """The height map of a terrain spec on MuJoCo's world grid: rows along y, columns along x."""

    def __init__(self, spec):
        validate(spec)
        self.spec = spec
        self.resolution = float(spec.get("resolution_m", 0.02))
        center = spec.get("center_m", [0.0, 0.0])
        length, width = spec["size_m"]
        # World x is left and y is -forward.
        self.center_xy = (float(center[1]), -float(center[0]))
        self.half = (width / 2, length / 2)
        self.nx = int(round(width / self.resolution)) + 1
        self.ny = int(round(length / self.resolution)) + 1
        self.xs = self.center_xy[0] + np.linspace(-self.half[0], self.half[0], self.nx)
        self.ys = self.center_xy[1] + np.linspace(-self.half[1], self.half[1], self.ny)
        x, y = np.meshgrid(self.xs, self.ys)
        self.heights = sum(
            (feature_height(f, -y, x) for f in spec.get("features", [])), np.zeros_like(x)
        )

    @property
    def flat(self):
        return float(np.ptp(self.heights)) < 1e-9 and abs(float(self.heights.max())) < 1e-9

    def height_at(self, forward, left):
        """Bilinear height of the grid at a navigation-frame point (the floor's level outside the grid)."""
        x, y = left, -forward
        u = np.clip((x - self.xs[0]) / self.resolution, 0, self.nx - 1)
        v = np.clip((y - self.ys[0]) / self.resolution, 0, self.ny - 1)
        i, j = min(int(u), self.nx - 2), min(int(v), self.ny - 2)
        a, b = u - i, v - j
        h = self.heights
        return float(
            (h[j, i] * (1 - a) + h[j, i + 1] * a) * (1 - b)
            + (h[j + 1, i] * (1 - a) + h[j + 1, i + 1] * a) * b
        )

    def highest_under(self, forward, left, radius):
        """The highest grid point within ``radius`` of a navigation-frame point."""
        x, y = left, -forward
        i0, i1 = np.searchsorted(self.xs, [x - radius, x + radius + 1e-12])
        j0, j1 = np.searchsorted(self.ys, [y - radius, y + radius + 1e-12])
        gx, gy = np.meshgrid(self.xs[i0:i1], self.ys[j0:j1])
        near = (gx - x) ** 2 + (gy - y) ** 2 <= radius**2
        if not near.any():
            return self.height_at(forward, left)
        return float(self.heights[j0:j1, i0:i1][near].max())

    def spawner(self, margin, seed):
        """Random starts at least ``margin`` metres inside the terrain, facing any way: rows -> [n, 4] of world
        x, y, the highest ground within 0.25 m, and yaw (for ``SurrogateEnv(spawn=...)``)."""
        rng = np.random.default_rng(seed)
        (x0, x1), (y0, y1) = (
            (self.xs[0] + margin, self.xs[-1] - margin),
            (self.ys[0] + margin, self.ys[-1] - margin),
        )
        if x0 >= x1 or y0 >= y1:
            raise ValueError("The spawn margin leaves no room on the terrain")

        def spawn(rows):
            out = np.zeros((len(rows), 4))
            for k in range(len(rows)):
                x, y = rng.uniform(x0, x1), rng.uniform(y0, y1)
                out[k] = (x, y, self.highest_under(-y, x, 0.25), rng.uniform(-np.pi, np.pi))
            return out

        return spawn
