import React from 'react';

export default function Waveform({ peaks, progress = 0 }) {
  const maximum = Math.max(...peaks, .001);
  return <svg viewBox="0 0 768 120" preserveAspectRatio="none" aria-hidden="true">
    {peaks.map((peak, i) => <line key={i} x1={i * 4 + 2} x2={i * 4 + 2} y1={60 - 52 * peak / maximum} y2={60 + 52 * peak / maximum} />)}
    {progress > 0 && <path className="ov-playhead" d={`M ${Math.min(1, progress) * 768} 0 v120`} />}
  </svg>;
}
