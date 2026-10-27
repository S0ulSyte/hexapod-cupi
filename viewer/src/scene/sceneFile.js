// Scene files: the scene a user edits (hexapod_scene_v1, what src/locomotion/surrogate/scene.py runs) and the
// simulated result that viewer/scene_view.py writes (hexapod_scene_view_v1). A result holds one case per robot
// in the run file's layout (leg points, contacts, command, ghost), all in the world's navigation axes, then the
// terrain's simulated height grid as int16 millimetres, then each robot's body pose and joint angles.
import { simulatedGrid } from './terrain.js';

export const SCENE = 'hexapod_scene_v1';
export const SCENE_VIEW = 'hexapod_scene_view_v1';
export const CONTROL_SECONDS = 0.02;

export function exampleScene() {
  return {
    schema: SCENE,
    name: 'Ramp, steps and bumps',
    seconds: null,
    terrain: {
      size_m: [5, 4],
      center_m: [1.5, 0],
      resolution_m: 0.02,
      features: [
        { type: 'ramp', center: [1, -0.8], size: [1, 0.8], rise: 0.04, heading_deg: 0 },
        { type: 'box', center: [1.9, -0.8], size: [0.8, 0.8], height: 0.04, heading_deg: 0 },
        { type: 'steps', center: [1, 0.8], size: [1.2, 0.8], rise: 0.03, count: 3, heading_deg: 0 },
        { type: 'bump', center: [2.6, 0], radius: 0.35, height: 0.02 },
        { type: 'noise', amplitude: 0.003, wavelength: 0.3, seed: 7 },
      ],
    },
    robots: [
      {
        name: 'ramp walker',
        checkpoint: 'checkpoint_update000600.pt',
        start: { forward_m: 0, left_m: -0.8, heading_deg: 0 },
        commands: [
          { seconds: 2, command: [0, 0, 0] },
          { seconds: 16, command: [0.15, 0, 0] },
          { seconds: 6, command: [0, 0, 0.2] },
        ],
      },
      {
        name: 'steps walker',
        checkpoint: 'checkpoint_update000600.pt',
        start: { forward_m: 0, left_m: 0.8, heading_deg: 0 },
        commands: [
          { seconds: 2, command: [0, 0, 0] },
          { seconds: 20, command: [0.1, 0, 0] },
        ],
      },
    ],
  };
}

export function sceneSeconds(scene) {
  const planned = scene.robots.map((r) => (r.commands ?? []).reduce((s, c) => s + Number(c.seconds), 0));
  return Number(scene.seconds) || Math.max(1, ...planned);
}

// The path each command list leads a robot along from its start: [forward, left, heading] every frame.
export function plannedPath(robot, seconds, frameSeconds = 0.04) {
  const steps = Math.round(frameSeconds / CONTROL_SECONDS);
  const s = robot.start ?? {};
  let f = Number(s.forward_m ?? 0), l = Number(s.left_m ?? 0), h = (Number(s.heading_deg ?? 0) * Math.PI) / 180;
  const out = [[f, l, h]];
  const rows = [];
  for (const c of robot.commands ?? []) for (let k = 0; k < Math.round(c.seconds / CONTROL_SECONDS); k++) rows.push(c.command);
  const total = Math.round(seconds / CONTROL_SECONDS);
  for (let k = 0; k < total; k++) {
    const [vf, vl, w] = rows[k] ?? [0, 0, 0];
    const m = h + (w * CONTROL_SECONDS) / 2;
    f += (vf * Math.cos(m) - vl * Math.sin(m)) * CONTROL_SECONDS;
    l += (vf * Math.sin(m) + vl * Math.cos(m)) * CONTROL_SECONDS;
    h += w * CONTROL_SECONDS;
    if ((k + 1) % steps === 0) out.push([f, l, h]);
  }
  return out;
}

function bytesOf(base64) {
  const text = atob(base64);
  const out = new Uint8Array(text.length);
  for (let i = 0; i < text.length; i++) out[i] = text.charCodeAt(i);
  return out;
}

// A scene (to edit) or a simulated scene (to play); other files are refused.
export async function readSceneFile(value) {
  const file = typeof value === 'string' ? JSON.parse(value) : value;
  if (file.schema === SCENE) return { scene: file, result: null };
  if (file.schema !== SCENE_VIEW) throw new Error(`Not a scene file (schema ${file.schema ?? 'missing'})`);
  const stream = new Blob([bytesOf(file.data)]).stream().pipeThrough(new DecompressionStream('gzip'));
  const buffer = await new Response(stream).arrayBuffer();
  for (const c of file.cases) {
    c.points = new Int16Array(buffer, c.points_offset, c.frames * 72);
    c.contacts = new Uint8Array(buffer, file.contacts_base + c.contacts_offset, c.frames);
    c.seconds = c.frames * file.frame_seconds;
    // Body pose in MuJoCo's world (x, y, z, quaternion xyzw) and the 18 joint angles, every frame.
    c.pose = new Float32Array(buffer, file.pose_base + c.pose_offset, c.frames * 7);
    c.joints = new Float32Array(buffer, file.joints_base + c.joints_offset, c.frames * 18);
  }
  const t = file.terrain;
  file.grid = simulatedGrid(t, new Int16Array(buffer, t.base, t.nx * t.ny));
  return { scene: file.scene ?? null, result: file };
}
