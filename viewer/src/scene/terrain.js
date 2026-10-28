// Terrain heights for the Scene page, the same as src/locomotion/surrogate/terrain.py computes them.
// Positions are the navigation frame's forward and left in metres; heights add up over the features.
// test/terrain.test.mjs compares both on test/fixtures/sample_terrain.json.

export const FEATURES = {
  box: { center: [0, 0], size: [0.6, 0.6], height: 0.03, heading_deg: 0 },
  ramp: { center: [0, 0], size: [1, 0.8], rise: 0.04, heading_deg: 0 },
  steps: { center: [0, 0], size: [1.2, 0.8], rise: 0.03, count: 3, heading_deg: 0 },
  bump: { center: [0, 0], radius: 0.3, height: 0.02 },
  hill: { center: [0, 0], radius: 1.5, top_radius: 0.5, height: 0.14 },
  noise: { amplitude: 0.005, wavelength: 0.3, seed: 1 },
};

function local(feature, forward, left) {
  const [f0, l0] = feature.center;
  const a = ((feature.heading_deg ?? 0) * Math.PI) / 180;
  const df = forward - f0, dl = left - l0;
  return [df * Math.cos(a) + dl * Math.sin(a), -df * Math.sin(a) + dl * Math.cos(a)];
}

// A value in [0, 1) per integer lattice point, with the 32-bit arithmetic of terrain.py's _hash.
function hash(ix, iy, seed) {
  let h = (Math.imul(ix, 374761393) + Math.imul(iy, 668265263) + Math.imul(seed, 2147483647)) >>> 0;
  h = Math.imul((h ^ (h >>> 13)) >>> 0, 1274126177) >>> 0;
  h = (h ^ (h >>> 16)) >>> 0;
  return h / 4294967296;
}

function noise(feature, forward, left) {
  const u = forward / feature.wavelength, v = left / feature.wavelength;
  const i = Math.floor(u), j = Math.floor(v);
  let a = u - i, b = v - j;
  a = a * a * (3 - 2 * a);
  b = b * b * (3 - 2 * b);
  const seed = Math.trunc(feature.seed) % 1000003;
  const value =
    (hash(i, j, seed) * (1 - a) + hash(i + 1, j, seed) * a) * (1 - b) +
    (hash(i, j + 1, seed) * (1 - a) + hash(i + 1, j + 1, seed) * a) * b;
  return feature.amplitude * (2 * value - 1);
}

export function featureHeight(feature, forward, left) {
  switch (feature.type) {
    case 'noise':
      return noise(feature, forward, left);
    case 'bump': {
      const r = Math.hypot(forward - feature.center[0], left - feature.center[1]);
      return r < feature.radius ? feature.height * 0.5 * (1 + Math.cos((Math.PI * r) / feature.radius)) : 0;
    }
    case 'hill': {
      const r = Math.hypot(forward - feature.center[0], left - feature.center[1]);
      const share = (feature.radius - r) / (feature.radius - feature.top_radius);
      return feature.height * Math.min(1, Math.max(0, share));
    }
    default: {
      const [u, v] = local(feature, forward, left);
      const [length, width] = feature.size;
      if (Math.abs(u) > length / 2 || Math.abs(v) > width / 2) return 0;
      if (feature.type === 'box') return feature.height;
      const share = Math.min(1, Math.max(0, u / length + 0.5));
      if (feature.type === 'ramp') return feature.rise * share;
      const count = Math.trunc(feature.count);
      return (feature.rise * (Math.min(Math.floor(share * count), count - 1) + 1)) / count;
    }
  }
}

export function heightAt(terrain, forward, left) {
  return (terrain.features ?? []).reduce((s, f) => s + featureHeight(f, forward, left), 0);
}

// The terrain's bounds in the navigation frame: [forward min, forward max, left min, left max].
export function bounds(terrain) {
  const [cf, cl] = terrain.center_m ?? [0, 0];
  const [length, width] = terrain.size_m;
  return [cf - length / 2, cf + length / 2, cl - width / 2, cl + width / 2];
}

// A height grid for drawing: computed from the features (the editor) or read from a scene file's simulated
// grid. `at(forward, left)` interpolates; `step` is the grid spacing in metres.
export function featureGrid(terrain, step) {
  const [f0, f1, l0, l1] = bounds(terrain);
  const nf = Math.max(2, Math.round((f1 - f0) / step) + 1), nl = Math.max(2, Math.round((l1 - l0) / step) + 1);
  const heights = new Float32Array(nf * nl);
  for (let a = 0; a < nf; a++)
    for (let b = 0; b < nl; b++)
      heights[a * nl + b] = heightAt(terrain, f0 + ((f1 - f0) * a) / (nf - 1), l0 + ((l1 - l0) * b) / (nl - 1));
  return gridOf({ f0, f1, l0, l1, nf, nl, heights });
}

// The simulated grid of a scene file: MuJoCo's rows along y (= -forward) and columns along x (= left).
export function simulatedGrid(t, heightsMm) {
  const nf = t.ny, nl = t.nx;
  const heights = new Float32Array(nf * nl);
  // Row r is y = y0 + r * resolution, so forward = -y runs from -y0 down; store rows from the forward minimum.
  for (let a = 0; a < nf; a++) for (let b = 0; b < nl; b++) heights[a * nl + b] = heightsMm[(nf - 1 - a) * nl + b] / 1000;
  const fMax = -t.y0_m, fMin = fMax - (nf - 1) * t.resolution_m;
  return gridOf({ f0: fMin, f1: fMax, l0: t.x0_m, l1: t.x0_m + (nl - 1) * t.resolution_m, nf, nl, heights });
}

function gridOf(g) {
  const df = (g.f1 - g.f0) / (g.nf - 1), dl = (g.l1 - g.l0) / (g.nl - 1);
  let low = Infinity, high = -Infinity;
  for (const h of g.heights) {
    low = Math.min(low, h);
    high = Math.max(high, h);
  }
  return {
    ...g,
    low,
    high,
    at(forward, left) {
      const u = Math.min(g.nf - 1, Math.max(0, (forward - g.f0) / df));
      const v = Math.min(g.nl - 1, Math.max(0, (left - g.l0) / dl));
      const i = Math.min(Math.floor(u), g.nf - 2), j = Math.min(Math.floor(v), g.nl - 2);
      const a = u - i, b = v - j, h = g.heights, n = g.nl;
      return (h[i * n + j] * (1 - b) + h[i * n + j + 1] * b) * (1 - a) + (h[(i + 1) * n + j] * (1 - b) + h[(i + 1) * n + j + 1] * b) * a;
    },
  };
}
