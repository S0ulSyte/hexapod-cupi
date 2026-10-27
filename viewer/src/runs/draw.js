import { center, point, touching } from './runFile.js';

// Legs lf, lm, lr, rf, rm, rr; tripod A is lf, lr and rm.
const TRIPOD_A = [true, false, true, false, true, false];
export const VIEWS = {
  top: { az: 0, el: Math.PI / 2 - 0.001 },
  '3d': { az: -0.6, el: 0.5 },
  side: { az: Math.PI / 2, el: 0.02 },
};

const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

export function fit(canvas) {
  const r = canvas.getBoundingClientRect();
  const d = window.devicePixelRatio || 1;
  if (canvas.width !== Math.round(r.width * d) || canvas.height !== Math.round(r.height * d)) {
    canvas.width = Math.round(r.width * d);
    canvas.height = Math.round(r.height * d);
  }
  const g = canvas.getContext('2d');
  g.setTransform(d, 0, 0, d, 0, 0);
  return [g, r.width, r.height];
}

// The robot at time `t` seen from `view`, following the body: floor grid, body path, ghost path and dot,
// commanded direction or turn, then the body plate and legs drawn back to front.
export function drawRobot(canvas, c, t, view, frameSeconds) {
  const [g, W, H] = fit(canvas);
  g.clearRect(0, 0, W, H);
  const f = Math.min(c.frames - 1, t / frameSeconds);
  const [cf, cl] = center(c, f);
  const { az, el } = view;
  const ca = Math.cos(az), sa = Math.sin(az), ce = Math.cos(el), se = Math.sin(el);
  const S = Math.min(W, H) / (0.62 + 0.2 * se);
  const cy = H * (0.5 + 0.12 * ce);
  const proj = (fw, lf, z) => {
    const E = -(lf - cl), N = fw - cf, e = E * ca - N * sa, n = E * sa + N * ca;
    return [W / 2 + e * S, cy - (z * ce + n * se) * S, n * ce - z * se];
  };
  const line = (pts) => {
    g.beginPath();
    pts.forEach((q, k) => (k ? g.lineTo(q[0], q[1]) : g.moveTo(q[0], q[1])));
  };

  g.lineWidth = 1;
  g.strokeStyle = css('--grid');
  const G = 0.1, R = 0.5, f0 = Math.round(cf / G) * G, l0 = Math.round(cl / G) * G;
  for (let k = -R; k <= R + 1e-9; k += G) {
    line([proj(f0 + k, l0 - R, 0), proj(f0 + k, l0 + R, 0)]);
    g.stroke();
    line([proj(f0 - R, l0 + k, 0), proj(f0 + R, l0 + k, 0)]);
    g.stroke();
  }
  const i = Math.round(f);
  g.strokeStyle = css('--trail');
  g.lineWidth = 2;
  line(Array.from({ length: i + 1 }, (_, j) => proj(...center(c, j), 0)));
  g.stroke();
  g.strokeStyle = g.fillStyle = css('--cmd');
  g.setLineDash([5, 4]);
  g.lineWidth = 1.5;
  line(c.ghost_mm.slice(0, i + 1).map(([a, b]) => proj(a / 1000, b / 1000, 0)));
  g.stroke();
  g.setLineDash([]);
  const [ga, gb] = c.ghost_mm[i];
  const dot = proj(ga / 1000, gb / 1000, 0);
  g.beginPath();
  g.arc(dot[0], dot[1], 5, 0, 7);
  g.fill();

  // The command arrow uses the body's current axes from the front and rear coxa joints.
  const front = [0, 3].map((leg) => point(c, f, leg, 0));
  const rear = [2, 5].map((leg) => point(c, f, leg, 0));
  let hf = (front[0][0] + front[1][0] - rear[0][0] - rear[1][0]) / 2;
  let hl = (front[0][1] + front[1][1] - rear[0][1] - rear[1][1]) / 2;
  const hn = Math.hypot(hf, hl) || 1;
  hf /= hn;
  hl /= hn;
  const cmd = c.command[i];
  g.lineWidth = 3;
  if (Math.hypot(cmd[0], cmd[1]) > 1e-6) {
    const vf = cmd[0] * hf - cmd[1] * hl, vl = cmd[0] * hl + cmd[1] * hf;
    const a = proj(cf, cl, 0), b = proj(cf + vf * 3.2, cl + vl * 3.2, 0);
    line([a, b]);
    g.stroke();
    const ang = Math.atan2(b[1] - a[1], b[0] - a[0]);
    line([b, [b[0] - 10 * Math.cos(ang - 0.4), b[1] - 10 * Math.sin(ang - 0.4)], [b[0] - 10 * Math.cos(ang + 0.4), b[1] - 10 * Math.sin(ang + 0.4)]]);
    g.fill();
  }
  if (Math.abs(cmd[2]) > 1e-6) {
    const a = proj(cf, cl, 0), r = 48, end = -Math.PI / 2 + (cmd[2] > 0 ? -0.9 : 0.9);
    g.beginPath();
    g.arc(a[0], a[1], r, -Math.PI / 2 - 0.9, -Math.PI / 2 + 0.9);
    g.stroke();
    g.beginPath();
    g.arc(a[0] + r * Math.cos(end), a[1] + r * Math.sin(end), 5, 0, 7);
    g.fill();
  }

  const items = [];
  const plate = [0, 1, 2, 5, 4, 3].map((leg) => proj(...point(c, f, leg, 0)));
  items.push({
    depth: plate.reduce((s, q) => s + q[2], 0) / 6,
    draw() {
      g.fillStyle = css('--body') + '55';
      g.strokeStyle = css('--body');
      g.lineWidth = 2;
      line(plate);
      g.closePath();
      g.fill();
      g.stroke();
      const a = point(c, f, 0, 0), b = point(c, f, 3, 0);
      const m = proj((a[0] + b[0]) / 2, (a[1] + b[1]) / 2, (a[2] + b[2]) / 2);
      g.fillStyle = css('--ink');
      g.beginPath();
      g.arc(m[0], m[1], 4, 0, 7);
      g.fill();
      g.font = '12px system-ui';
      g.fillText('front', m[0] + 6, m[1] - 6);
    },
  });
  for (let leg = 0; leg < 6; leg++) {
    const s = [0, 1, 2, 3].map((k) => proj(...point(c, f, leg, k)));
    const colour = css(TRIPOD_A[leg] ? '--legA' : '--legB');
    const on = touching(c, f, leg);
    items.push({
      depth: (s[1][2] + s[2][2] + s[3][2]) / 3,
      draw() {
        g.strokeStyle = g.fillStyle = colour;
        g.lineWidth = 4;
        g.lineCap = g.lineJoin = 'round';
        line(s);
        g.stroke();
        for (let k = 0; k < 3; k++) {
          g.beginPath();
          g.arc(s[k][0], s[k][1], 3.5, 0, 7);
          g.fill();
        }
        g.beginPath();
        g.arc(s[3][0], s[3][1], 6, 0, 7);
        g.lineWidth = 2.5;
        if (on) g.fill();
        else {
          g.fillStyle = css('--card');
          g.fill();
          g.stroke();
        }
      },
    });
  }
  items.sort((a, b) => b.depth - a.depth).forEach((x) => x.draw());
  g.font = '13px system-ui';
  g.fillStyle = css('--ink');
  const deg = c.heading_deg[i], want = c.commanded_heading_deg[i];
  g.fillText(`heading ${deg > 0 ? '+' : ''}${Math.round(deg)}°  (commanded ${Math.round(want)}°)`, 8, 18);
}

// One or more series against time with a cursor at `t`; each series is {values, colour, dashed}.
export function drawSeries(canvas, { series, seconds, frameSeconds, t, unit, title }) {
  const [g, W, H] = fit(canvas);
  g.clearRect(0, 0, W, H);
  const L = 48, Rt = 10, T = 14, B = 22, w = W - L - Rt, h = H - T - B;
  const all = series.flatMap((s) => s.values);
  let lo = Math.min(0, ...all), hi = Math.max(1e-6, ...all);
  if (hi - lo < 1e-6) hi = lo + 1;
  const pad = (hi - lo) * 0.08;
  lo -= pad;
  hi += pad;
  const X = (s) => L + (s / seconds) * w, Y = (v) => T + ((hi - v) / (hi - lo)) * h;
  // Degrees step by 45 and 90 past 20; other units by 1, 2 and 5.
  const steps = unit === '°'
    ? [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 45, 90, 180, 360]
    : [0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000];
  const step = steps.find((s) => (hi - lo) / s <= 6) ?? 10000;
  g.font = '11px system-ui';
  g.lineWidth = 1;
  g.strokeStyle = css('--grid');
  g.fillStyle = css('--muted');
  for (let v = Math.ceil(lo / step) * step; v <= hi; v += step) {
    g.beginPath();
    g.moveTo(L, Y(v));
    g.lineTo(W - Rt, Y(v));
    g.stroke();
    g.fillText(`${Number(v.toFixed(2))} ${unit}`, 2, Y(v) + 4);
  }
  for (let s = 0; s <= seconds; s += seconds > 30 ? 10 : 5) g.fillText(`${s} s`, X(s) - 8, H - 6);
  for (const s of series) {
    g.strokeStyle = s.colour;
    g.lineWidth = s.dashed ? 2 : 2.5;
    g.setLineDash(s.dashed ? [6, 5] : []);
    g.beginPath();
    s.values.forEach((v, j) => (j ? g.lineTo(X(j * frameSeconds), Y(v)) : g.moveTo(X(0), Y(v))));
    g.stroke();
  }
  g.setLineDash([]);
  g.strokeStyle = css('--ink');
  g.beginPath();
  g.moveTo(X(Math.min(seconds, t)), T);
  g.lineTo(X(Math.min(seconds, t)), T + h);
  g.stroke();
  g.fillStyle = css('--muted');
  g.fillText(title, L + 4, T + 10);
}
