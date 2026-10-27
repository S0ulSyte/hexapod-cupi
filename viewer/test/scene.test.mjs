// Read the fixtures that tests/test_scene_view.py writes from src/locomotion/surrogate/terrain.py and scene.py.
import assert from 'node:assert/strict';
import { readFile } from 'node:fs/promises';
import test from 'node:test';
import { center, touching } from '../src/runs/runFile.js';
import { featureGrid, featureHeight, heightAt } from '../src/scene/terrain.js';
import { exampleScene, plannedPath, readSceneFile, sceneSeconds } from '../src/scene/sceneFile.js';

const read = async (name) => readFile(new URL(`./fixtures/${name}`, import.meta.url), 'utf8');

test('the editor computes the same terrain heights as the surrogate', async () => {
  const fixture = JSON.parse(await read('sample_terrain.json'));
  fixture.terrain.features.forEach((feature, k) => {
    fixture.points.forEach(([f, l], j) => {
      const want = fixture.features[k][j];
      assert.ok(Math.abs(featureHeight(feature, f, l) - want) < 1e-9, `${feature.type} at ${f}, ${l}: ${featureHeight(feature, f, l)} vs ${want}`);
    });
  });
  const [f, l] = fixture.points[0];
  const sum = fixture.features.reduce((s, heights) => s + heights[0], 0);
  assert.ok(Math.abs(heightAt(fixture.terrain, f, l) - sum) < 1e-9);
  const grid = featureGrid(fixture.terrain, 0.05);
  assert.ok(Math.abs(grid.at(1, -0.4) - heightAt(fixture.terrain, 1, -0.4)) < 0.004);
});

test('a simulated scene decodes into robots on the simulated terrain', async () => {
  const { scene, result } = await readSceneFile(await read('sample_scene_view.json'));
  assert.equal(scene.name, 'fixture');
  assert.deepEqual(result.cases.map((c) => [c.name, c.frames]), [['a', 5], ['b', 5]]);
  const [a, b] = result.cases;
  // Robot a stands over the 20 mm box at 0.3 m forward and 0.2 m left; b on the floor.
  const [af, al] = center(a, 4);
  assert.ok(Math.abs(af - 0.3) < 0.02 && Math.abs(al - 0.2) < 0.02);
  assert.ok(Math.abs(result.grid.at(af, al) - 0.02) < 1e-9);
  assert.ok(Math.abs(result.grid.at(...center(b, 4))) < 1e-9);
  assert.equal(touching(a, 4, 0), true);
  await assert.rejects(readSceneFile({ schema: 'other' }), /Not a scene file/);
});

test('a scene plans where its commands lead', () => {
  const scene = exampleScene();
  assert.equal(sceneSeconds(scene), 24);
  const robot = { start: { forward_m: 1, left_m: 0, heading_deg: 90 }, commands: [{ seconds: 2, command: [0.1, 0, 0] }] };
  const path = plannedPath(robot, 3);
  // Facing left, 2 s at 0.1 m/s moves 0.2 m left, then the robot stands.
  const [f, l, h] = path.at(-1);
  assert.ok(Math.abs(f - 1) < 1e-9 && Math.abs(l - 0.2) < 1e-9 && Math.abs(h - Math.PI / 2) < 1e-9);
  assert.equal(path.length, 3 / 0.04 + 1);
});
