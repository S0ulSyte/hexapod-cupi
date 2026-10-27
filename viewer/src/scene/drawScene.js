// Draw a scene: the terrain as shaded height cells, then every robot as the Runs page draws one (draw.js), with
// its body path, ghost and command on the ground. Before a scene is simulated it shows each robot's start and the
// path its commands lead along.
import { center } from '../runs/runFile.js';
import { drawCommand, robotItems } from '../runs/draw.js';

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();
const ROBOT_COLOURS = ['--run-0', '--run-1', '--run-2', '--run-3'];
const MAX_CELLS = 110;

function rgb(hex) {
  const v = parseInt(hex.replace('#', ''), 16);
  return [(v >> 16) & 255, (v >> 8) & 255, v & 255];
}

function path(g, pts) {
  g.beginPath();
  pts.forEach((q, k) => (k ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1])));
}

// The canvas at its CSS size (the page) or at a fixed pixel size (a video frame).
function prepare(canvas, size) {
  const g = canvas.getContext('2d');
  if (size) {
    if (canvas.width !== size[0] || canvas.height !== size[1]) [canvas.width, canvas.height] = size;
    g.setTransform(1, 0, 0, 1, 0, 0);
    return [g, size[0], size[1]];
  }
  const r = canvas.getBoundingClientRect(), d = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(r.width * d) || canvas.height !== Math.round(r.height * d)) {
    canvas.width = Math.round(r.width * d);
    canvas.height = Math.round(r.height * d);
  }
  g.setTransform(d, 0, 0, d, 0, 0);
  return [g, r.width, r.height];
}

// The camera looks at `target` [forward, left, height] from azimuth `az` and elevation `el`; `span` metres fit the
// shorter side of the canvas.
export function projector(camera, W, H) {
  const { az, el, target, span } = camera;
  const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
  const S = Math.min(W, H) / span;
  return (fw, lf, z) => {
    const E = -(lf - target[1]), N = fw - target[0], e = E * ca - N * sa, n = E * sa + N * ca;
    const dz = z - target[2];
    return [W / 2 + e * S, H / 2 - (dz * ce + n * se) * S, n * ce - dz * se];
  };
}

function drawTerrain(g, grid, proj, W, H) {
  const stride = Math.max(1, Math.ceil(Math.max(grid.nf, grid.nl) / MAX_CELLS));
  const low = rgb(css('--terrain-low') || '#22252b'), high = rgb(css('--terrain-high') || '#565b66');
  const range = Math.max(1e-6, grid.high - grid.low);
  const df = (grid.f1 - grid.f0) / (grid.nf - 1), dl = (grid.l1 - grid.l0) / (grid.nl - 1);
  const h = (a, b) => grid.heights[Math.min(a, grid.nf - 1) * grid.nl + Math.min(b, grid.nl - 1)];
  const cells = [];
  for (let a = 0; a < grid.nf - 1; a += stride) {
    for (let b = 0; b < grid.nl - 1; b += stride) {
      const a1 = Math.min(a + stride, grid.nf - 1), b1 = Math.min(b + stride, grid.nl - 1);
      const f0 = grid.f0 + a * df, f1 = grid.f0 + a1 * df, l0 = grid.l0 + b * dl, l1 = grid.l0 + b1 * dl;
      const z = [h(a, b), h(a, b1), h(a1, b1), h(a1, b)];
      const q = [proj(f0, l0, z[0]), proj(f0, l1, z[1]), proj(f1, l1, z[2]), proj(f1, l0, z[3])];
      if (q.every((p) => p[0] < -20) || q.every((p) => p[0] > W + 20) || q.every((p) => p[1] < -20) || q.every((p) => p[1] > H + 20)) continue;
      // Light from the forward-left, above: the slope darkens cells that face away from it.
      const sf = (z[3] + z[2] - z[0] - z[1]) / 2 / (f1 - f0), sl = (z[1] + z[2] - z[0] - z[3]) / 2 / (l1 - l0);
      const shade = Math.max(0.55, Math.min(1.25, 1 - (sf * 0.6 + sl * 0.4) * 2.5));
      const mix = ((z[0] + z[1] + z[2] + z[3]) / 4 - grid.low) / range;
      const colour = low.map((c, k) => Math.round(Math.min(255, (c + (high[k] - c) * mix) * shade)));
      cells.push({ q, depth: (q[0][2] + q[1][2] + q[2][2] + q[3][2]) / 4, colour });
    }
  }
  cells.sort((x, y) => y.depth - x.depth);
  g.lineWidth = 0.6;
  g.strokeStyle = css('--grid');
  for (const c of cells) {
    path(g, c.q);
    g.closePath();
    g.fillStyle = `rgb(${c.colour.join(',')})`;
    g.fill();
    g.stroke();
  }
}

function label(g, text, p, colour) {
  g.font = '600 12px system-ui';
  const w = g.measureText(text).width;
  g.fillStyle = 'rgba(20,22,26,0.75)';
  g.fillRect(p[0] - w / 2 - 5, p[1] - 30, w + 10, 18);
  g.fillStyle = colour;
  g.fillText(text, p[0] - w / 2, p[1] - 17);
}

// One frame of the scene. `robots` are simulated cases (with `frames`, points and ghost) or, before simulation,
// planned starts and paths; `t` is seconds; `size` draws at a fixed pixel size for video.
export function drawScene(canvas, { grid, robots, planned, t, frameSeconds, camera, title, size }) {
  const [g, W, H] = prepare(canvas, size);
  g.fillStyle = css('--bg');
  g.fillRect(0, 0, W, H);
  const proj = projector(camera, W, H);
  const ground = (f, l) => (grid ? grid.at(f, l) : 0) + 0.002;
  if (grid) drawTerrain(g, grid, proj, W, H);
  const items = [];
  (robots ?? []).forEach((c, r) => {
    const colour = css(ROBOT_COLOURS[r % 4]);
    const f = Math.min(c.frames - 1, t / frameSeconds), i = Math.round(f);
    const [cf, cl] = center(c, f);
    g.strokeStyle = colour;
    g.lineWidth = 2;
    path(g, Array.from({ length: i + 1 }, (_, j) => {
      const [a, b] = center(c, j);
      return proj(a, b, ground(a, b));
    }));
    g.stroke();
    g.strokeStyle = g.fillStyle = css('--cmd');
    g.setLineDash([5, 4]);
    g.lineWidth = 1.5;
    path(g, c.ghost_mm.slice(0, i + 1).map(([a, b]) => proj(a / 1000, b / 1000, ground(a / 1000, b / 1000))));
    g.stroke();
    g.setLineDash([]);
    const [ga, gb] = c.ghost_mm[i];
    const dot = proj(ga / 1000, gb / 1000, ground(ga / 1000, gb / 1000));
    g.beginPath();
    g.arc(dot[0], dot[1], 5, 0, 7);
    g.fill();
    drawCommand(g, c, f, i, proj, cf, cl, ground(cf, cl));
    items.push(...robotItems(g, c, f, proj));
    const top = proj(cf, cl, ground(cf, cl) + 0.2);
    items.push({ depth: -Infinity, draw: () => label(g, c.name + (c.fell_at_s != null && t >= c.fell_at_s ? ' · fell' : ''), top, colour) });
  });
  (planned ?? []).forEach((p, r) => {
    const colour = css(ROBOT_COLOURS[r % 4]);
    const [f0, l0, h0] = p.path[0];
    g.strokeStyle = css('--cmd');
    g.setLineDash([5, 4]);
    g.lineWidth = 1.5;
    path(g, p.path.map(([a, b]) => proj(a, b, ground(a, b))));
    g.stroke();
    g.setLineDash([]);
    // The start: the body's footprint and its facing.
    const ring = Array.from({ length: 24 }, (_, k) => {
      const a = (k / 24) * 2 * Math.PI;
      return proj(f0 + 0.17 * Math.cos(a), l0 + 0.12 * Math.sin(a), ground(f0, l0));
    });
    g.strokeStyle = colour;
    g.lineWidth = 2;
    path(g, ring);
    g.closePath();
    g.stroke();
    const tip = proj(f0 + 0.28 * Math.cos(h0), l0 + 0.28 * Math.sin(h0), ground(f0, l0));
    path(g, [proj(f0, l0, ground(f0, l0)), tip]);
    g.stroke();
    label(g, p.name, proj(f0, l0, ground(f0, l0) + 0.15), colour);
  });
  items.sort((a, b) => b.depth - a.depth).forEach((x) => x.draw());
  g.font = '13px system-ui';
  g.fillStyle = css('--ink');
  g.fillText(`${title ?? ''}${title ? ' · ' : ''}${t.toFixed(1)} s`, 10, 20);
}
