# Viewer

The viewer has three pages. **Robot** is the joint inspector for the active robot model. **Runs** replays
evaluation runs from native Spark attempts and from the surrogate side by side on one clock. **Scene** builds
terrain scenes with several robots and long command lists, plays their surrogate simulation and records it as
a video.

```sh
npm ci
npm run dev
```

Open the address Vite prints. Add `#runs` for the Runs page and `#scene` for the Scene page.

## Runs

Both evaluators write `evaluation/batch_*/report.json` and `control_trace.npz`. `run_view.py` turns one
evaluation into a run file:

```sh
uv run python viewer/run_view.py /path/to/eval_attempt --output attempt.json
uv run python viewer/run_view.py /path/to/surrogate_output --output surrogate.json
```

The folder can be a native attempt (`<attempt>/run/standing/evaluation`), a surrogate output folder
(`<output>/evaluation`) or the evaluation folder itself. Copy native attempts from Spark first; only batches
with a control trace are drawn, and the file lists the others.

On the Runs page, open one or more run files or drop them on the page. Pick the cases to show from the
table; each case shows:

- the robot from the top, in 3D or from the side, with its legs drawn from the robot model's joint frames;
- the body path in grey, and toes filled while on the floor;
- the ghost, as a dashed orange line and dot. It starts at the reset pose (the origin, heading zero), follows
  the command exactly once per control and restarts at the body when the command changes, as the destination
  experiments train it;
- the commanded direction or turn, the heading against the commanded heading, and the distance to the ghost;
- the evaluator's own pass or fail and failed checks.

A server that serves the built viewer and the run files can open them directly: `?run=attempt.json#runs`,
with `run` repeated for more files.

`npm test` reads `test/fixtures/sample_run.json`. `tests/test_run_view.py` checks that `run_view.py` writes
the same file from its synthetic attempt.

## Scene

A scene is a terrain, one or more robots and each robot's start and command list. On the Scene page:

1. Shape the terrain from features (box, ramp, steps, bump, noise) and set each robot's checkpoint, start
   (forward and left in metres, heading in degrees, turning left) and commands ([forward m/s, left m/s,
   turn rad/s] for a number of seconds, as many as you like). The view shows the terrain, each start and the
   path its commands lead along. **Save scene** downloads the scene file.
2. Simulate it on the surrogate, with checkpoint paths relative to the scene file:

   ```sh
   uv run python -m locomotion.surrogate.scene scene.json --output scene_out
   uv run python viewer/scene_view.py scene_out --output scene_view.json
   ```

3. Open `scene_view.json` on the page (or `?scene=scene_view.json#scene` from a server that serves it). It plays
   every robot on the simulated terrain in the look of the native Spark videos: the robot's 1,753 CAD parts in
   white on a pale floor with a 0.5 m grid, soft light and shadows, posed from each robot's simulated body pose and
   joint angles. **paths** adds each body path and its ghost. The style shows this CAD view, the Runs page's
   line drawing (legs, toes on the ground, body path, ghost and command), or both side by side. Pick the camera
   (whole terrain or follow a robot; 3D, top or side; drag to orbit, scroll to zoom), then **Record video** plays
   the whole scene once in real time at the chosen speed and downloads a WebM at the chosen size, with both
   views side by side when both are shown. Keep the page visible while it records.

The page loads the CAD meshes once (about 15 s) and merges their vertices on a grid a fraction of each part's size
(0.25 to 2 mm), so several robots play in real time; the full meshes hold about 22 million triangles per robot.

A robot's **slope inputs** option corrects what its policy feels for a slope: whenever a front leg and a rear leg
touch the ground, their height difference over their distance gives the ground's pitch along the robot, and the
policy's gravity inputs become gravity relative to that ground (`SlopeInputs` in `surrogate/scene.py`).

Each robot runs alone on its own copy of the terrain, so robots pass through each other. The terrain is the
whole floor: robots start at least 0.35 m inside it, and one that walks off its edge falls. The policies were
trained on flat ground and see no terrain, and the surrogate's touchdown model is off on terrain because it
assumes a flat floor; a scene is a design tool, not evidence.

`src/locomotion/surrogate/terrain.py` and `src/scene/terrain.js` compute the same heights;
`tests/test_scene_view.py` writes the fixtures `npm test` compares them on.
