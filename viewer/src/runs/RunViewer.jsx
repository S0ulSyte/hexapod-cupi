import React, { useCallback, useEffect, useRef, useState } from 'react';
import { describeFailure, readRun } from './runFile.js';
import { VIEWS, drawRobot, drawSeries } from './draw.js';
import './runs.css';

const RUN_COLOURS = ['--run-0', '--run-1', '--run-2', '--run-3'];
const MAX_PANELS = 6;
const css = (name) => getComputedStyle(document.documentElement).getPropertyValue(name).trim();

// Replays the cases of one or more run files side by side on one clock.
export default function RunViewer() {
  const [runs, setRuns] = useState([]);
  const [selected, setSelected] = useState([]);
  const [error, setError] = useState(null);
  const [playing, setPlaying] = useState(true);
  const [speed, setSpeed] = useState(2);
  const [viewName, setViewName] = useState('top');
  const clock = useRef({ t: 0, view: { ...VIEWS.top }, dirty: true });
  const panels = useRef(new Map());
  const slider = useRef(null);
  const label = useRef(null);
  const loaded = useRef(0);

  const open = useCallback(async (files) => {
    setError(null);
    try {
      const added = await Promise.all([...files].map(async (file) => readRun(await file.text())));
      const first = loaded.current;
      loaded.current += added.length;
      setRuns((current) => [...current, ...added]);
      // The first file opened shows its first four cases; later files add to the table only.
      setSelected((chosen) => (chosen.length ? chosen : added[0].cases.slice(0, 4).map((_, j) => `${first}:${j}`)));
    } catch (e) {
      setError(String(e.message ?? e));
    }
  }, []);

  // ?run=<url> (repeatable, before the #runs hash) opens run files the same server serves.
  useEffect(() => {
    const urls = new URLSearchParams(window.location.search).getAll('run');
    if (!urls.length) return;
    Promise.all(
      urls.map(async (url) => {
        const response = await fetch(url);
        if (!response.ok) throw new Error(`${url}: HTTP ${response.status}`);
        return { text: () => response.text() };
      }),
    )
      .then(open)
      .catch((e) => setError(String(e.message ?? e)));
  }, [open]);

  const shown = selected
    .map((key) => key.split(':').map(Number))
    .filter(([r, c]) => runs[r]?.cases[c])
    .map(([r, c]) => ({ key: `${r}:${c}`, run: runs[r], runIndex: r, item: runs[r].cases[c] }));
  const longest = Math.max(1, ...shown.map((s) => s.item.seconds));

  useEffect(() => {
    let last = null;
    let frame = 0;
    const tick = (now) => {
      const state = clock.current;
      if (last != null && playing) {
        state.t += ((now - last) / 1000) * speed;
        if (state.t > longest) state.t = 0;
        state.dirty = true;
      }
      last = now;
      if (state.dirty) {
        state.dirty = false;
        if (slider.current) slider.current.value = state.t;
        if (label.current) label.current.textContent = `${state.t.toFixed(1)} s`;
        for (const s of shown) {
          const canvases = panels.current.get(s.key);
          if (!canvases?.view) continue;
          const seconds = s.item.seconds, fs = s.run.frame_seconds;
          drawRobot(canvases.view, s.item, state.t, state.view, fs);
          if (s.item.charts) {
            // A run file may bring its own charts (the leg stand: joint angles, real against simulated).
            s.item.charts.forEach((chart, i) =>
              drawSeries(canvases[`chart${i}`], {
                series: chart.series.map((line) => ({
                  values: line.values,
                  colour: line.colour?.startsWith('#') ? line.colour : css(line.colour ?? RUN_COLOURS[s.runIndex % 4]),
                  dashed: !!line.dashed,
                })),
                seconds, frameSeconds: fs, t: state.t, unit: chart.unit, title: chart.title,
              }),
            );
          } else {
            drawSeries(canvases.heading, {
              series: [
                { values: s.item.commanded_heading_deg, colour: css('--cmd'), dashed: true },
                { values: s.item.heading_deg, colour: css(RUN_COLOURS[s.runIndex % 4]) },
              ],
              seconds, frameSeconds: fs, t: state.t, unit: '°', title: 'heading — dashed: commanded',
            });
            drawSeries(canvases.ghost, {
              series: [{ values: s.item.ghost_distance_mm, colour: css(RUN_COLOURS[s.runIndex % 4]) }],
              seconds, frameSeconds: fs, t: state.t, unit: 'mm', title: 'distance to the ghost, where the command leads',
            });
          }
        }
      }
      frame = requestAnimationFrame(tick);
    };
    frame = requestAnimationFrame(tick);
    const redraw = () => (clock.current.dirty = true);
    window.addEventListener('resize', redraw);
    clock.current.dirty = true;
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('resize', redraw);
    };
  }, [playing, speed, longest, shown.map((s) => s.key).join(','), runs]);

  const toggle = (key) =>
    setSelected((chosen) =>
      chosen.includes(key) ? chosen.filter((k) => k !== key) : [...chosen, key].slice(-MAX_PANELS),
    );
  const register = (key, name) => (node) => {
    const entry = panels.current.get(key) ?? {};
    entry[name] = node;
    panels.current.set(key, entry);
    clock.current.dirty = true;
  };
  const rotate = (e) => {
    const start = [e.clientX, e.clientY, clock.current.view.az, clock.current.view.el];
    const move = (m) => {
      clock.current.view = {
        az: start[2] - (m.clientX - start[0]) * 0.01,
        el: Math.max(0.02, Math.min(Math.PI / 2 - 0.001, start[3] + (m.clientY - start[1]) * 0.01)),
      };
      clock.current.dirty = true;
      setViewName(null);
    };
    const up = () => {
      window.removeEventListener('pointermove', move);
      window.removeEventListener('pointerup', up);
    };
    window.addEventListener('pointermove', move);
    window.addEventListener('pointerup', up);
  };

  return (
    <div
      className="runs-page"
      onDragOver={(e) => e.preventDefault()}
      onDrop={(e) => {
        e.preventDefault();
        open(e.dataTransfer.files);
      }}
    >
      <main>
        <h1>Runs</h1>
        <p className="sub">
          Open one or more run files made by <code>uv run python viewer/run_view.py &lt;evaluation folder&gt; --output run.json</code>{' '}
          from a native Spark attempt or a surrogate evaluation, drop them here, or add <code>?run=&lt;url&gt;</code> to this page's address. Grey is the body path; the dashed orange line
          and dot are the ghost, which starts at the reset pose, follows the command exactly and restarts at the body when the
          command changes. Filled toes are on the floor. Drag a view to rotate it.
        </p>
        <label className="open">
          Open run files…
          <input type="file" accept=".json,application/json" multiple onChange={(e) => open(e.target.files)} />
        </label>
        {error && <p className="error">{error}</p>}
        {runs.length > 0 && (
          <div className="run-list">
            {runs.map((run, r) => (
              <div key={r} className="run-chip" style={{ borderColor: `var(${RUN_COLOURS[r % 4]})` }}>
                <b>{run.name}</b> · {run.source} · {run.cases.filter((c) => c.pass).length} of {run.cases.length} cases passed
                {run.cases_without_trace.length > 0 && ` · ${run.cases_without_trace.length} more without a trace`}
              </div>
            ))}
          </div>
        )}
        {runs.length > 0 && (
          <table className="cases">
            <thead>
              <tr>
                <th></th><th>Run</th><th>Case</th><th>Result</th><th>Seconds</th><th>Off the ghost at the end</th><th>Failed checks</th>
              </tr>
            </thead>
            <tbody>
              {runs.flatMap((run, r) =>
                run.cases.map((c, j) => {
                  const key = `${r}:${j}`;
                  return (
                    <tr key={key} className={selected.includes(key) ? 'on' : ''} onClick={() => toggle(key)}>
                      <td><input type="checkbox" readOnly checked={selected.includes(key)} /></td>
                      <td>{run.name}</td>
                      <td>{c.case_id}</td>
                      <td className={c.pass ? 'pass' : 'fail'}>{c.pass ? 'passed' : c.fell ? 'failed, fell' : 'failed'}</td>
                      <td>{c.seconds.toFixed(1)}</td>
                      <td>{c.ghost_distance_end_mm.toFixed(1)} mm</td>
                      <td>{c.failed.map(describeFailure).join('; ')}</td>
                    </tr>
                  );
                }),
              )}
            </tbody>
          </table>
        )}
        {shown.length > 0 && (
          <div className="bar">
            <button onClick={() => setPlaying(!playing)}>{playing ? 'Pause' : 'Play'}</button>
            <select value={speed} onChange={(e) => setSpeed(Number(e.target.value))} aria-label="Speed">
              {[1, 2, 4].map((s) => <option key={s} value={s}>{s}×</option>)}
            </select>
            {Object.keys(VIEWS).map((name) => (
              <button
                key={name}
                className={viewName === name ? 'on' : ''}
                onClick={() => {
                  clock.current.view = { ...VIEWS[name] };
                  clock.current.dirty = true;
                  setViewName(name);
                }}
              >
                {name === '3d' ? '3D' : name[0].toUpperCase() + name.slice(1)}
              </button>
            ))}
            <input
              ref={slider}
              type="range"
              min="0"
              max={longest}
              step="0.04"
              aria-label="Time"
              onInput={(e) => {
                clock.current.t = Number(e.target.value);
                clock.current.dirty = true;
              }}
            />
            <span ref={label} className="clock">0.0 s</span>
          </div>
        )}
        <div className="panels">
          {shown.map((s) => (
            <section key={s.key} className="panel" style={{ borderTopColor: `var(${RUN_COLOURS[s.runIndex % 4]})` }}>
              <h2>{s.item.case_id}</h2>
              <p className="meta">
                {s.run.name} · {s.run.source} ·{' '}
                <span className={s.item.pass ? 'pass' : 'fail'}>{s.item.pass ? '✓ passed' : '✗ failed'}</span>
                {s.item.failed.length > 0 && ` — ${s.item.failed.map(describeFailure).join(', ')}`}
              </p>
              <canvas className="view" ref={register(s.key, 'view')} onPointerDown={rotate} />
              {s.item.charts ? (
                s.item.charts.map((_, i) => (
                  <canvas key={i} className="chart" ref={register(s.key, `chart${i}`)} />
                ))
              ) : (
                <>
                  <canvas className="chart" ref={register(s.key, 'heading')} />
                  <canvas className="chart" ref={register(s.key, 'ghost')} />
                </>
              )}
            </section>
          ))}
        </div>
      </main>
    </div>
  );
}
