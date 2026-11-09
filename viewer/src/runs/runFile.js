// Read a run file that viewer/run_view.py writes from a native Spark or surrogate evaluation folder.
// The file is JSON with one gzip block: int16 leg points (millimetres, forward/left/up, 6 legs x 4 points per
// frame) for every case, then one contact byte per frame (bit k set when leg k's toe touches the floor).

export const SCHEMA = 'hexapod_run_view_v1';

function bytesOf(base64) {
  const text = atob(base64);
  const out = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i);
  return out;
}

export async function readRun(value) {
  const run = typeof value === 'string' ? JSON.parse(value) : value;
  if (run.schema !== SCHEMA) throw new Error(`Not a run file (schema ${run.schema ?? 'missing'})`);
  const stream = new Blob([bytesOf(run.data)]).stream().pipeThrough(new DecompressionStream('gzip'));
  const buffer = await new Response(stream).arrayBuffer();
  for (const c of run.cases) {
    c.points = new Int16Array(buffer, c.points_offset, c.frames * 72);
    // Optional overlays: more robots on the same frames, drawn translucent under the main one.
    for (const o of c.overlays ?? []) o.points = new Int16Array(buffer, o.points_offset, c.frames * 72);
    c.contacts = new Uint8Array(buffer, run.contacts_base + c.contacts_offset, c.frames);
    c.seconds = c.frames * run.frame_seconds;
  }
  return run;
}

// Point k of leg `leg` (0 coxa, 1 femur, 2 tibia joint, 3 toe) at fractional frame `f`, in metres.
export function point(c, f, leg, k, points = c.points) {
  const i0 = Math.max(0, Math.min(c.frames - 1, Math.floor(f)));
  const i1 = Math.min(c.frames - 1, i0 + 1);
  const a = Math.max(0, Math.min(1, f - i0));
  const o0 = ((i0 * 6 + leg) * 4 + k) * 3;
  const o1 = ((i1 * 6 + leg) * 4 + k) * 3;
  const p = points;
  return [0, 1, 2].map((d) => (p[o0 + d] + (p[o1 + d] - p[o0 + d]) * a) / 1000);
}

// The body centre as the mean of the six coxa joints, [forward, left] in metres.
export function center(c, f) {
  let forward = 0;
  let left = 0;
  for (let leg = 0; leg < 6; leg++) {
    const q = point(c, f, leg, 0);
    forward += q[0] / 6;
    left += q[1] / 6;
  }
  return [forward, left];
}

export function touching(c, f, leg) {
  return ((c.contacts[Math.min(c.frames - 1, Math.round(f))] >> leg) & 1) === 1;
}

// Plain words for the evaluator's failed checks.
const WORDS = {
  complete_requested_window: (v) => `missed ${Math.round(v)} controls of the scored window`,
  terminated: () => 'fell',
  planar_error_mps: 'planar speed error, m/s',
  yaw_error_rad_s: 'turn-rate error, rad/s',
  computed_demand_over_rating_fraction: 'share of time motors were asked above their rating',
  tilt_rms_degrees: 'body tilt RMS, deg',
  native_capture_or_original_physical_bounds: () => 'native 400 Hz joint-speed or contact screen',
  max_joint_velocity_rms_rad_s: 'joint motion while still, rad/s RMS',
  max_joint_position_range_rad: 'joint travel while still, rad',
  max_target_step_abs_p95_rad_per_20ms: 'target chatter, rad per control (p95)',
  missing_six_toe_count_400hz: 'physics steps without all six toes down',
};

export function describeFailure({ check, value, bound }) {
  const word = WORDS[check];
  if (typeof word === 'function') return word(value);
  const label = word ?? check.replaceAll('_', ' ');
  return typeof value === 'number' ? `${label} ${Number(value.toPrecision(3))} (limit ${bound})` : label;
}
