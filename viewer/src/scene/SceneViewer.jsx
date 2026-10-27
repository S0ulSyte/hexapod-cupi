import React, { useCallback, useEffect, useMemo, useRef, useState } from 'react';
import { VIEWS } from '../runs/draw.js';
import { center } from '../runs/runFile.js';
import { drawScene } from './drawScene.js';
import { FEATURES, bounds, featureGrid } from './terrain.js';
import { exampleScene, plannedPath, readSceneFile, sceneSeconds } from './sceneFile.js';
import '../runs/runs.css';
import './scene.css';

const PREVIEW_STEP_M = 0.04;
// locomotion.surrogate.scene refuses starts closer to the terrain's edge: the terrain is the whole floor.
const EDGE_MARGIN_M = 0.35;
const SIZES = { '1280×720': [1280, 720], '1920×1080': [1920, 1080], '1080×1080': [1080, 1080] };
const FIELD_LABELS = {
  center: ['centre forward', 'centre left'],
  size: ['length', 'width'],
  height: 'height',
  rise: 'rise',
  count: 'steps',
  heading_deg: 'heading °',
  radius: 'radius',
  amplitude: 'amplitude',
  wavelength: 'wavelength',
  seed: 'seed',
};

function Num({ value, onChange, step = 0.01, label }) {
  const [text, setText] = useState(String(value ?? ''));
  useEffect(() => setText(String(value ?? '')), [value]);
  return (
    <label className="num">
      <span>{label}</span>
      <input
        type="number"
        step={step}
        value={text}
        onChange={(e) => {
          setText(e.target.value);
          if (e.target.value !== '' && Number.isFinite(Number(e.target.value))) onChange(Number(e.target.value));
        }}
      />
    </label>
  );
}

function download(name, blob) {
  const a = document.createElement('a');
  a.href = URL.createObjectURL(blob);
  a.download = name;
  a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 1000);
}

const slug = (text) => (text || 'scene').toLowerCase().replace(/[^a-z0-9]+/g, '_').replace(/^_|_$/g, '');

// Build a scene (terrain features, robots, their starts and commands), save it for the surrogate to run, then
// open the simulated result to play it and record it as a video.
export default function SceneViewer() {
  const [scene, setScene] = useState(exampleScene);
  const [result, setResult] = useState(null);
  const [showResult, setShowResult] = useState(false);
  const [error, setError] = useState(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(1);
  const [viewName, setViewName] = useState('3d');
  const [follow, setFollow] = useState(-1);
  const [recording, setRecording] = useState(null);
  const [videoSize, setVideoSize] = useState('1280×720');
  const [fps, setFps] = useState(25);
  const canvas = useRef(null);
  const state = useRef({ t: 0, view: { ...VIEWS['3d'] }, span: null, dirty: true });
  const slider = useRef(null);
  const clock = useRef(null);

  const update = (change) => setScene((s) => change(structuredClone(s)));
  const grid = useMemo(() => {
    try {
      return featureGrid(scene.terrain, Math.max(PREVIEW_STEP_M, Number(scene.terrain.resolution_m) || PREVIEW_STEP_M));
    } catch {
      return null;
    }
  }, [scene.terrain]);
  const live = showResult && result;
  const seconds = live ? result.seconds : sceneSeconds(scene);
  const planned = useMemo(
    () => scene.robots.map((r) => ({ name: r.name, path: plannedPath(r, sceneSeconds(scene)) })),
    [scene],
  );

  const open = useCallback(async (files) => {
    setError(null);
    try {
      for (const file of files) {
        const { scene: loaded, result: simulated } = await readSceneFile(await file.text());
        if (loaded) setScene(loaded);
        if (simulated) {
          setResult(simulated);
          setShowResult(true);
          state.current.t = 0;
        }
      }
    } catch (e) {
      setError(String(e.message ?? e));
    }
  }, []);

  // ?scene=<url> opens a scene or a simulated scene that the same server serves.
  useEffect(() => {
    const urls = new URLSearchParams(window.location.search).getAll('scene');
    if (!urls.length) return;
    Promise.all(
      urls.map(async (url) => {
        // A regenerated result keeps its name, so skip the cache.
        const response = await fetch(url, { cache: 'no-store' });
        if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
        return { text: () => response.text() };
      }),
    )
      .then(open)
      .catch((e) => setError(String(e.message ?? e)));
  }, [open]);

  // The camera: the whole terrain, or one robot followed at a closer span.
  const camera = useCallback(
    (t) => {
      const s = state.current;
      const [f0, f1, l0, l1] = bounds(scene.terrain);
      let target = [(f0 + f1) / 2, (l0 + l1) / 2, 0];
      let span = s.span ?? Math.max(f1 - f0, l1 - l0) * (s.view.el > 1.2 ? 1.05 : 0.95);
      if (follow >= 0) {
        if (live && result.cases[follow]) {
          const c = result.cases[follow];
          const [cf, cl] = center(c, Math.min(c.frames - 1, t / result.frame_seconds));
          target = [cf, cl, live ? result.grid.at(cf, cl) : 0];
        } else if (planned[follow]) {
          const p = planned[follow].path[Math.min(planned[follow].path.length - 1, Math.round(t / 0.04))];
          target = [p[0], p[1], grid ? grid.at(p[0], p[1]) : 0];
        }
        span = s.span ?? 1.8;
      }
      return { ...s.view, target, span };
    },
    [scene.terrain, follow, live, result, planned, grid],
  );

  const frame = useCallback(
    (target, t, size) =>
      drawScene(target, {
        grid: live ? result.grid : grid,
        robots: live ? result.cases : null,
        planned: live ? null : planned,
        t,
        frameSeconds: live ? result.frame_seconds : 0.04,
        camera: camera(t),
        title: live ? result.name : `${scene.name} (plan — not simulated)`,
        size,
      }),
    [live, result, grid, planned, camera, scene.name],
  );

  useEffect(() => {
    let last = null, id = 0;
    const tick = (now) => {
      const s = state.current;
      if (last != null && playing && !recording) {
        s.t += ((now - last) / 1000) * speed;
        if (s.t > seconds) s.t = 0;
        s.dirty = true;
      }
      last = now;
      if (s.dirty && canvas.current && !recording) {
        s.dirty = false;
        if (slider.current) slider.current.value = s.t;
        if (clock.current) clock.current.textContent = `${s.t.toFixed(1)} s`;
        frame(canvas.current, s.t);
      }
      id = requestAnimationFrame(tick);
    };
    id = requestAnimationFrame(tick);
    const redraw = () => (state.current.dirty = true);
    window.addEventListener('resize', redraw);
    state.current.dirty = true;
    return () => {
      cancelAnimationFrame(id);
      window.removeEventListener('resize', redraw);
    };
  }, [playing, speed, seconds, frame, recording]);

  const rotate = (e) => {
    const s = state.current;
    const start = [e.clientX, e.clientY, s.view.az, s.view.el];
    const move = (m) => {
      s.view = {
        az: start[2] - (m.clientX - start[0]) * 0.01,
        el: Math.max(0.02, Math.min(Math.PI / 2 - 0.001, start[3] + (m.clientY - start[1]) * 0.01)),
      };
      s.dirty = true;
      setViewName(null);
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };
  const zoom = (e) => {
    const s = state.current;
    s.span = Math.max(0.3, Math.min(60, (s.span ?? camera(s.t).span) * Math.exp(e.deltaY * 0.001)));
    s.dirty = true;
  };

  // Play the scene once at real time onto a fixed-size canvas and record it with the browser's MediaRecorder.
  const record = async () => {
    const size = SIZES[videoSize];
    const target = document.createElement('canvas');
    [target.width, target.height] = size;
    const type = ['video/webm;codecs=vp9', 'video/webm;codecs=vp8', 'video/webm', 'video/mp4'].find(
      (t) => window.MediaRecorder?.isTypeSupported?.(t),
    );
    if (!type) {
      setError('This browser cannot record canvas video (no MediaRecorder).');
      return;
    }
    const stream = target.captureStream(fps);
    const recorder = new MediaRecorder(stream, { mimeType: type, videoBitsPerSecond: size[0] * size[1] * 6 });
    const chunks = [];
    recorder.ondataavailable = (e) => e.data.size && chunks.push(e.data);
    const done = new Promise((resolve) => (recorder.onstop = resolve));
    const frames = Math.ceil((seconds / speed) * fps);
    setRecording({ frame: 0, frames });
    frame(target, 0, size);
    recorder.start(1000);
    const started = performance.now();
    for (let k = 0; k <= frames; k++) {
      const wait = started + (k * 1000) / fps - performance.now();
      if (wait > 0) await new Promise((r) => setTimeout(r, wait));
      frame(target, Math.min(seconds, (k / fps) * speed), size);
      if (k % fps === 0) setRecording({ frame: k, frames });
    }
    recorder.stop();
    await done;
    stream.getTracks().forEach((track) => track.stop());
    download(`${slug(live ? result.name : scene.name)}.${type.includes('mp4') ? 'mp4' : 'webm'}`, new Blob(chunks, { type }));
    setRecording(null);
    state.current.dirty = true;
  };

  const setView = (name) => {
    state.current.view = { ...VIEWS[name] };
    state.current.span = null;
    state.current.dirty = true;
    setViewName(name);
  };

  const featureFields = (f, k) =>
    Object.keys(FEATURES[f.type]).map((key) => {
      const labels = FIELD_LABELS[key];
      if (Array.isArray(f[key]))
        return f[key].map((v, j) => (
          <Num key={key + j} label={labels[j]} value={v} onChange={(x) => update((s) => ((s.terrain.features[k][key][j] = x), s))} />
        ));
      return (
        <Num
          key={key}
          label={labels}
          step={key === 'count' || key === 'seed' ? 1 : key === 'heading_deg' ? 5 : 0.005}
          value={f[key]}
          onChange={(x) => update((s) => ((s.terrain.features[k][key] = x), s))}
        />
      );
    });

  const t = scene.terrain;
  const [tf0, tf1, tl0, tl1] = bounds(t);
  const warnings = scene.robots.flatMap((r) => {
    const f = r.start?.forward_m ?? 0, l = r.start?.left_m ?? 0, out = [];
    if (f < tf0 + EDGE_MARGIN_M || f > tf1 - EDGE_MARGIN_M || l < tl0 + EDGE_MARGIN_M || l > tl1 - EDGE_MARGIN_M)
      out.push(`${r.name} starts within ${EDGE_MARGIN_M} m of the terrain's edge or outside it`);
    if (r.policy !== 'stand' && !r.checkpoint) out.push(`${r.name} needs a checkpoint`);
    return out;
  });
  if (new Set(scene.robots.map((r) => r.name)).size !== scene.robots.length) warnings.push('every robot needs its own name');
  const scenePath = `${slug(scene.name)}.json`;
  return (
    <div
      className="runs-page scene-page"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        open(e.dataTransfer.files);
      }}
    >
      <aside className="editor">
        <h1>Scene</h1>
        <p className="sub">
          Shape a terrain, place robots and give each a command list, then save the scene and simulate it on the
          surrogate. Open the simulated file to play it and record a video. Positions are forward and left in metres;
          headings turn left.
        </p>
        <div className="row">
          <label className="open">
            Open scene or result…
            <input type="file" accept=".json,application/json" multiple onChange={(e) => open(e.target.files)} />
          </label>
          <button onClick={() => download(scenePath, new Blob([JSON.stringify(scene, null, 1)], { type: 'application/json' }))}>
            Save scene
          </button>
          <button onClick={() => (setScene(exampleScene()), setShowResult(false))}>Example</button>
        </div>
        {error && <p className="error">{error}</p>}
        <details className="howto">
          <summary>Simulate this scene</summary>
          <p>Save the scene next to the checkpoints it names (paths are relative to the scene file), then run:</p>
          <code>uv run python -m locomotion.surrogate.scene {scenePath} --output {slug(scene.name)}_out</code>
          <code>uv run python viewer/scene_view.py {slug(scene.name)}_out --output {slug(scene.name)}_view.json</code>
          <p>and open <code>{slug(scene.name)}_view.json</code> here.</p>
        </details>

        <h2>Scene</h2>
        <label className="text">
          <span>name</span>
          <input value={scene.name} onChange={(e) => update((s) => ((s.name = e.target.value), s))} />
        </label>
        <Num label="seconds (blank: the longest command list)" step={1} value={scene.seconds ?? ''} onChange={(x) => update((s) => ((s.seconds = x || null), s))} />

        <h2>Terrain</h2>
        <div className="grid2">
          <Num label="length (forward)" step={0.5} value={t.size_m[0]} onChange={(x) => update((s) => ((s.terrain.size_m[0] = x), s))} />
          <Num label="width (left)" step={0.5} value={t.size_m[1]} onChange={(x) => update((s) => ((s.terrain.size_m[1] = x), s))} />
          <Num label="centre forward" step={0.1} value={(t.center_m ?? [0, 0])[0]} onChange={(x) => update((s) => ((s.terrain.center_m = [x, (s.terrain.center_m ?? [0, 0])[1]]), s))} />
          <Num label="centre left" step={0.1} value={(t.center_m ?? [0, 0])[1]} onChange={(x) => update((s) => ((s.terrain.center_m = [(s.terrain.center_m ?? [0, 0])[0], x]), s))} />
          <Num label="resolution" step={0.005} value={t.resolution_m ?? 0.02} onChange={(x) => update((s) => ((s.terrain.resolution_m = x), s))} />
        </div>
        {t.features.map((f, k) => (
          <div key={k} className="card">
            <div className="head">
              <b>{f.type}</b>
              <button onClick={() => update((s) => (s.terrain.features.splice(k, 1), s))}>Remove</button>
            </div>
            <div className="grid2">{featureFields(f, k)}</div>
          </div>
        ))}
        <div className="row">
          {Object.keys(FEATURES).map((type) => (
            <button key={type} onClick={() => update((s) => (s.terrain.features.push({ type, ...structuredClone(FEATURES[type]) }), s))}>
              + {type}
            </button>
          ))}
        </div>

        <h2>Robots</h2>
        {warnings.map((w) => <p key={w} className="warn">{w}</p>)}
        {scene.robots.map((r, k) => (
          <div key={k} className="card" style={{ borderLeftColor: `var(--run-${k % 4})` }}>
            <div className="head">
              <input className="name" value={r.name} onChange={(e) => update((s) => ((s.robots[k].name = e.target.value), s))} />
              <button onClick={() => update((s) => (s.robots.splice(k, 1), s))}>Remove</button>
            </div>
            <label className="check">
              <input
                type="checkbox"
                checked={r.policy === 'stand'}
                onChange={(e) =>
                  update((s) => {
                    const robot = s.robots[k];
                    if (e.target.checked) {
                      delete robot.checkpoint;
                      robot.policy = 'stand';
                    } else {
                      delete robot.policy;
                      robot.checkpoint = 'checkpoint_update000600.pt';
                    }
                    return s;
                  })
                }
              />
              stand still (no checkpoint)
            </label>
            {r.policy !== 'stand' && (
              <label className="text">
                <span>checkpoint (relative to the scene file)</span>
                <input value={r.checkpoint ?? ''} onChange={(e) => update((s) => ((s.robots[k].checkpoint = e.target.value), s))} />
              </label>
            )}
            <div className="grid3">
              <Num label="start forward" step={0.1} value={r.start?.forward_m ?? 0} onChange={(x) => update((s) => ((s.robots[k].start = { ...s.robots[k].start, forward_m: x }), s))} />
              <Num label="start left" step={0.1} value={r.start?.left_m ?? 0} onChange={(x) => update((s) => ((s.robots[k].start = { ...s.robots[k].start, left_m: x }), s))} />
              <Num label="heading °" step={15} value={r.start?.heading_deg ?? 0} onChange={(x) => update((s) => ((s.robots[k].start = { ...s.robots[k].start, heading_deg: x }), s))} />
            </div>
            <table className="commands">
              <thead>
                <tr><th>seconds</th><th>forward m/s</th><th>left m/s</th><th>turn rad/s</th><th></th></tr>
              </thead>
              <tbody>
                {(r.commands ?? []).map((c, j) => (
                  <tr key={j}>
                    <td><Num step={1} value={c.seconds} onChange={(x) => update((s) => ((s.robots[k].commands[j].seconds = x), s))} /></td>
                    {[0, 1, 2].map((d) => (
                      <td key={d}><Num step={0.01} value={c.command[d]} onChange={(x) => update((s) => ((s.robots[k].commands[j].command[d] = x), s))} /></td>
                    ))}
                    <td><button onClick={() => update((s) => (s.robots[k].commands.splice(j, 1), s))}>×</button></td>
                  </tr>
                ))}
              </tbody>
            </table>
            <button onClick={() => update((s) => ((s.robots[k].commands ??= []).push({ seconds: 4, command: [0.1, 0, 0] }), s))}>+ command</button>
          </div>
        ))}
        <button
          onClick={() =>
            update((s) => {
              s.robots.push({
                name: `robot ${s.robots.length + 1}`,
                checkpoint: 'checkpoint_update000600.pt',
                // Inside the terrain's back edge, spread across it.
                start: {
                  forward_m: Number((bounds(s.terrain)[0] + EDGE_MARGIN_M + 0.1).toFixed(2)),
                  left_m: Number(((s.terrain.center_m ?? [0, 0])[1] + 0.5 * ((s.robots.length % 3) - 1)).toFixed(2)),
                  heading_deg: 0,
                },
                commands: [{ seconds: 10, command: [0.1, 0, 0] }],
              });
              return s;
            })
          }
        >
          + robot
        </button>
      </aside>

      <main className="stage">
        <div className="bar">
          {result && (
            <span className="toggle">
              <button className={live ? 'on' : ''} onClick={() => setShowResult(true)}>Simulated</button>
              <button className={live ? '' : 'on'} onClick={() => setShowResult(false)}>Plan</button>
            </span>
          )}
          <button onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play'}</button>
          <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Speed">
            {[0.5, 1, 2, 4].map((s) => <option key={s} value={s}>{s}×</option>)}
          </select>
          {Object.keys(VIEWS).map((name) => (
            <button key={name} className={viewName === name ? 'on' : ''} onClick={() => setView(name)}>
              {name === '3d' ? '3D' : name[0].toUpperCase() + name.slice(1)}
            </button>
          ))}
          <select
            value={follow}
            onChange={(e) => {
              setFollow(Number(e.target.value));
              state.current.span = null;
              state.current.dirty = true;
            }}
            aria-label="Camera"
          >
            <option value={-1}>Whole terrain</option>
            {(live ? result.cases.map((c) => c.name) : scene.robots.map((r) => r.name)).map((name, k) => (
              <option key={k} value={k}>Follow {name}</option>
            ))}
          </select>
          <input
            ref={slider}
            type="range"
            min="0"
            max={seconds}
            step="0.04"
            aria-label="Time"
            onInput={(e) => {
              state.current.t = Number(e.target.value);
              state.current.dirty = true;
            }}
          />
          <span ref={clock} className="clock">0.0 s</span>
        </div>
        <canvas ref={canvas} className="scene-view" onPointerDown={rotate} onWheel={zoom} />
        <div className="bar record">
          <select value={videoSize} onChange={(e) => setVideoSize(e.target.value)} aria-label="Video size">
            {Object.keys(SIZES).map((k) => <option key={k}>{k}</option>)}
          </select>
          <select value={fps} onChange={(e) => setFps(Number(e.target.value))} aria-label="Frames per second">
            {[25, 30, 50].map((k) => <option key={k} value={k}>{k} fps</option>)}
          </select>
          <button disabled={!!recording} onClick={record}>
            {recording ? `Recording… ${Math.round((100 * recording.frame) / recording.frames)}%` : 'Record video'}
          </button>
          <span className="note">
            Records the whole {seconds.toFixed(0)} s at {speed}× ({(seconds / speed).toFixed(0)} s of video) with the current camera.
            {live
              ? result.cases.some((c) => c.fell_at_s != null) && ` Fell: ${result.cases.filter((c) => c.fell_at_s != null).map((c) => `${c.name} at ${c.fell_at_s} s`).join(', ')}.`
              : ' Showing the plan: starts and where the commands lead.'}
          </span>
        </div>
      </main>
    </div>
  );
}
