import React, { useEffect, useState } from 'react';
import activeModel from '../../robot/active_model.json';
import RunViewer from './runs/RunViewer.jsx';
import SceneViewer from './scene/SceneViewer.jsx';

const tabStyle = (on) => ({
  font: '14px system-ui', padding: '6px 12px', borderRadius: 8, textDecoration: 'none',
  border: `1px solid ${on ? '#ecebe6' : '#343841'}`, background: '#1d2026', color: '#ecebe6', fontWeight: on ? 600 : 400,
});
const PAGES = { '#runs': 'runs', '#scene': 'scene' };
const pageOf = () => PAGES[window.location.hash] ?? 'robot';

// The robot inspector by default; #runs replays evaluation runs that viewer/run_view.py writes, and #scene
// builds terrain scenes for locomotion.surrogate.scene and plays their results (viewer/scene_view.py).
export default function App() {
  const [page, setPage] = useState(pageOf);
  useEffect(() => {
    const follow = () => setPage(pageOf());
    window.addEventListener('hashchange', follow);
    return () => window.removeEventListener('hashchange', follow);
  }, []);
  // The same detailed CAD inspector is used by the repository quickstart.
  // Historical checkpoints keep their own explicit asset paths and task IDs.
  // Vite's public directory serves the explicit file; a directory URL falls
  // through to the application shell and would recursively embed this page.
  const preview = '/' + activeModel.preview.replace(/^robot\//, '') + 'index.html';
  return (
    <div style={{ display: 'flex', flexDirection: 'column', height: '100%' }}>
      <nav style={{ display: 'flex', gap: 8, padding: '8px 12px', background: '#14161a', borderBottom: '1px solid #343841' }}>
        <a href="#robot" style={tabStyle(page === 'robot')}>Robot</a>
        <a href="#runs" style={tabStyle(page === 'runs')}>Runs</a>
        <a href="#scene" style={tabStyle(page === 'scene')}>Scene</a>
      </nav>
      <div style={{ flex: 1, minHeight: 0 }}>
        {page === 'runs' ? (
          <RunViewer />
        ) : page === 'scene' ? (
          <SceneViewer />
        ) : (
          <iframe title="Hexapod main robot: parts and joint travel"
            src={preview} style={{ width: '100%', height: '100%', border: 0, display: 'block' }} />
        )}
      </div>
    </div>
  );
}
