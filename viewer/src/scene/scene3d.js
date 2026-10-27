// The Scene page's 3D view in the look of the native Spark (Isaac) videos: the robot's CAD parts in white on a pale
// floor with a thin grid, soft light and shadows. Robots are posed from the simulated body pose and joint angles,
// on the terrain's height field. World axes are MuJoCo's: x = left, y = -forward, z up.
import * as THREE from 'three';
import { OrbitControls } from 'three/examples/jsm/controls/OrbitControls.js';
import { STLLoader } from 'three/examples/jsm/loaders/STLLoader.js';
import { toCreasedNormals } from 'three/examples/jsm/utils/BufferGeometryUtils.js';
import activeModel from '../../../robot/active_model.json';

// Vite serves the robot directory at the site root.
const MODEL_URL = '/' + activeModel.model.path.replace(/^robot\//, '');
const ROBOT_URL = MODEL_URL.replace(/[^/]+$/, '');
const TILE_M = 0.5;
const COLOURS = { background: '#d3dae0', floor: '#a9b4be', line: '#dfe5ea', robot: '#f3f4f5' };
// Views as the camera's direction from its target (x left, y back, z up).
export const VIEWS = { '3d': [0.75, -1.1, 0.75], top: [0, -0.001, 1], side: [1, 0, 0.1] };

// Fewer triangles for real-time play: vertices merge on a grid a fraction of the part's size (0.25 to 2 mm).
function simplify(geometry) {
  const p = geometry.getAttribute('position').array;
  geometry.computeBoundingBox();
  const size = geometry.boundingBox.getSize(new THREE.Vector3()).length();
  const cell = Math.min(0.002, Math.max(0.00025, size / 70));
  const index = new Map(), sums = [], vertex = new Uint32Array(p.length / 3);
  for (let v = 0; v < p.length / 3; v++) {
    const key = `${Math.round(p[3 * v] / cell)},${Math.round(p[3 * v + 1] / cell)},${Math.round(p[3 * v + 2] / cell)}`;
    let k = index.get(key);
    if (k === undefined) {
      k = sums.length / 4;
      index.set(key, k);
      sums.push(0, 0, 0, 0);
    }
    sums[4 * k] += p[3 * v];
    sums[4 * k + 1] += p[3 * v + 1];
    sums[4 * k + 2] += p[3 * v + 2];
    sums[4 * k + 3] += 1;
    vertex[v] = k;
  }
  const positions = new Float32Array(sums.length * 3 / 4);
  for (let k = 0; k < sums.length / 4; k++)
    for (let d = 0; d < 3; d++) positions[3 * k + d] = sums[4 * k + d] / sums[4 * k + 3];
  const triangles = [];
  for (let t = 0; t < vertex.length; t += 3) {
    const a = vertex[t], b = vertex[t + 1], c = vertex[t + 2];
    if (a !== b && b !== c && a !== c) triangles.push(a, b, c);
  }
  const out = new THREE.BufferGeometry();
  out.setAttribute('position', new THREE.BufferAttribute(positions, 3));
  out.setIndex(triangles);
  return toCreasedNormals(out, Math.PI / 5);
}

// The CAD parts of every link, their simplified meshes and the standing pose; `progress(done, total)` reports loading.
export async function loadRobot(progress) {
  const model = await (await fetch(MODEL_URL)).json();
  const stance = await (await fetch(ROBOT_URL + 'stance.json')).json();
  const loader = new STLLoader(), names = [...new Set(model.parts.map((p) => p.mesh))], geometries = new Map();
  let done = 0, triangles = 0;
  const queue = [...names];
  const worker = async () => {
    while (queue.length) {
      const name = queue.shift();
      const g = simplify(await loader.loadAsync(ROBOT_URL + 'meshes/' + encodeURIComponent(name)));
      geometries.set(name, g);
      progress?.(++done, names.length);
    }
  };
  await Promise.all(Array.from({ length: 6 }, worker));
  for (const p of model.parts) triangles += geometries.get(p.mesh).getAttribute('position').count / 3;
  return { model, stance, geometries, triangles };
}

// One robot: a group per link in the joint tree, each link's parts as instanced meshes.
function buildRobot(robot, material) {
  const { model, geometries } = robot;
  const root = new THREE.Group(), links = new Map();
  for (const l of model.links) links.set(l.name, new THREE.Group());
  root.add(links.get('body'));
  for (const j of model.joints) {
    const g = links.get(j.child);
    g.position.fromArray(j.xyz);
    links.get(j.parent).add(g);
  }
  const batches = new Map();
  for (const p of model.parts) {
    const key = p.link + '|' + p.mesh;
    if (!batches.has(key)) batches.set(key, { link: p.link, mesh: p.mesh, parts: [] });
    batches.get(key).parts.push(p);
  }
  const m = new THREE.Object3D();
  for (const b of batches.values()) {
    const mesh = new THREE.InstancedMesh(geometries.get(b.mesh), material, b.parts.length);
    b.parts.forEach((p, i) => {
      m.position.fromArray(p.xyz);
      m.quaternion.fromArray(p.quaternion_xyzw);
      m.updateMatrix();
      mesh.setMatrixAt(i, m.matrix);
    });
    mesh.castShadow = mesh.receiveShadow = true;
    links.get(b.link).add(mesh);
  }
  const joints = model.joints.map((j) => ({
    group: links.get(j.child),
    base: new THREE.Quaternion().fromArray(j.quaternion_xyzw),
    axis: new THREE.Vector3(...j.axis).normalize(),
  }));
  return { root, joints };
}

function gridTexture() {
  const c = document.createElement('canvas');
  c.width = c.height = 512;
  const g = c.getContext('2d');
  g.fillStyle = COLOURS.floor;
  g.fillRect(0, 0, 512, 512);
  g.strokeStyle = COLOURS.line;
  g.lineWidth = 3;
  g.strokeRect(0, 0, 512, 512);
  const t = new THREE.CanvasTexture(c);
  t.wrapS = t.wrapT = THREE.RepeatWrapping;
  t.repeat.set(1 / TILE_M, 1 / TILE_M);
  t.anisotropy = 8;
  t.colorSpace = THREE.SRGBColorSpace;
  return t;
}

// Planar UVs in metres, so the grid texture tiles the same on the floor and the terrain.
function planarUV(geometry) {
  const p = geometry.getAttribute('position');
  const uv = new Float32Array(p.count * 2);
  for (let i = 0; i < p.count; i++) {
    uv[2 * i] = p.getX(i);
    uv[2 * i + 1] = p.getY(i);
  }
  geometry.setAttribute('uv', new THREE.BufferAttribute(uv, 2));
}

export class SceneRenderer {
  constructor(canvas) {
    this.renderer = new THREE.WebGLRenderer({ canvas, antialias: true, preserveDrawingBuffer: true });
    this.renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
    this.renderer.shadowMap.enabled = true;
    this.renderer.shadowMap.type = THREE.PCFSoftShadowMap;
    this.renderer.outputColorSpace = THREE.SRGBColorSpace;
    this.renderer.toneMapping = THREE.ACESFilmicToneMapping;
    this.renderer.toneMappingExposure = 0.95;
    this.scene = new THREE.Scene();
    this.scene.background = new THREE.Color(COLOURS.background);
    this.scene.fog = new THREE.Fog(COLOURS.background, 8, 35);
    this.camera = new THREE.PerspectiveCamera(35, 1, 0.02, 200);
    this.camera.up.set(0, 0, 1);
    this.controls = new OrbitControls(this.camera, canvas);
    this.controls.enableDamping = false;
    this.scene.add(new THREE.HemisphereLight('#ffffff', '#7f8a96', 1.25));
    this.sun = new THREE.DirectionalLight('#ffffff', 2.8);
    this.sun.castShadow = true;
    this.sun.shadow.mapSize.set(4096, 4096);
    this.sun.shadow.bias = -0.0002;
    this.sun.shadow.normalBias = 0.002;
    this.scene.add(this.sun, this.sun.target);
    this.fill = new THREE.DirectionalLight('#ffffff', 0.6);
    this.fill.position.set(-3, 4, 2);
    this.scene.add(this.fill);
    this.ground = new THREE.MeshStandardMaterial({ map: gridTexture(), roughness: 0.85, metalness: 0 });
    this.material = new THREE.MeshStandardMaterial({ color: COLOURS.robot, roughness: 0.45, metalness: 0.05 });
    this.overlay = new THREE.Group();
    this.scene.add(this.overlay);
    this.robots = [];
    this.terrain = null;
  }

  // The terrain mesh over [x0, x1] x [y0, y1] with heights `at(x, y)`, and the floor plane around it.
  setTerrain(at, [x0, x1, y0, y1]) {
    if (this.terrain) {
      this.scene.remove(this.terrain);
      this.terrain.traverse((o) => o.geometry?.dispose());
    }
    const step = Math.max(0.02, Math.max(x1 - x0, y1 - y0) / 320);
    const nx = Math.round((x1 - x0) / step), ny = Math.round((y1 - y0) / step);
    const field = new THREE.PlaneGeometry(x1 - x0, y1 - y0, nx, ny);
    field.translate((x0 + x1) / 2, (y0 + y1) / 2, 0);
    const p = field.getAttribute('position');
    for (let i = 0; i < p.count; i++) p.setZ(i, at(p.getX(i), p.getY(i)));
    field.computeVertexNormals();
    planarUV(field);
    const shape = new THREE.Shape([[-60, -60], [60, -60], [60, 60], [-60, 60]].map(([a, b]) => new THREE.Vector2(a, b)));
    shape.holes.push(new THREE.Path([[x0, y0], [x0, y1], [x1, y1], [x1, y0]].map(([a, b]) => new THREE.Vector2(a, b))));
    const floor = new THREE.ShapeGeometry(shape);
    planarUV(floor);
    const group = new THREE.Group();
    for (const g of [field, floor]) {
      const mesh = new THREE.Mesh(g, this.ground);
      mesh.receiveShadow = true;
      group.add(mesh);
    }
    this.terrain = group;
    this.bounds = [x0, x1, y0, y1];
    this.scene.add(group);
  }

  setRobots(robot, count) {
    for (const r of this.robots) this.scene.remove(r.root);
    this.robots = Array.from({ length: count }, () => buildRobot(robot, this.material));
    for (const r of this.robots) this.scene.add(r.root);
  }

  // Robot k at world position, quaternion xyzw and 18 joint angles.
  pose(k, position, quaternion, angles) {
    const r = this.robots[k];
    r.root.position.set(position[0], position[1], position[2]);
    r.root.quaternion.set(quaternion[0], quaternion[1], quaternion[2], quaternion[3]);
    const turn = new THREE.Quaternion();
    r.joints.forEach((j, i) => j.group.quaternion.copy(j.base).multiply(turn.setFromAxisAngle(j.axis, angles[i])));
  }

  // Lines drawn on the ground: each is {points: [[x, y, z]], colour, dashed}.
  setLines(lines) {
    this.overlay.traverse((o) => o.geometry?.dispose());
    this.overlay.clear();
    for (const l of lines) {
      if (l.points.length < 2) continue;
      const g = new THREE.BufferGeometry().setFromPoints(l.points.map((q) => new THREE.Vector3(...q)));
      const m = l.dashed
        ? new THREE.LineDashedMaterial({ color: l.colour, dashSize: 0.03, gapSize: 0.02 })
        : new THREE.LineBasicMaterial({ color: l.colour });
      const line = new THREE.Line(g, m);
      if (l.dashed) line.computeLineDistances();
      this.overlay.add(line);
    }
  }

  // Aim the camera at `target` from direction `view` at `distance`, keeping a user's orbit when `keep` is set.
  aim(target, view, distance, keep) {
    const t = new THREE.Vector3(...target);
    const offset = keep ? this.camera.position.clone().sub(this.controls.target) : new THREE.Vector3(...view).normalize().multiplyScalar(distance);
    this.controls.target.copy(t);
    this.camera.position.copy(t).add(offset);
    // The sun follows the target, so the shadow map covers what the camera sees.
    const reach = Math.max(1.5, Math.min(12, offset.length() * 0.9));
    this.sun.position.copy(t).add(new THREE.Vector3(1.2, 2.0, 4.0));
    this.sun.target.position.copy(t);
    Object.assign(this.sun.shadow.camera, { left: -reach, right: reach, top: reach, bottom: -reach, near: 0.1, far: 20 });
    this.sun.shadow.camera.updateProjectionMatrix();
  }

  render(width, height) {
    const size = this.renderer.getSize(new THREE.Vector2());
    if (size.x !== width || size.y !== height) this.renderer.setSize(width, height, false);
    this.camera.aspect = width / Math.max(1, height);
    this.camera.updateProjectionMatrix();
    this.controls.update();
    this.renderer.render(this.scene, this.camera);
  }

  dispose() {
    this.controls.dispose();
    this.renderer.dispose();
  }
}
