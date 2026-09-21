import React from 'react';
import './audio-capture.css';

const clock = seconds => `0:${String(seconds).padStart(2, '0')}`;

export default function AudioCaptureButton({recorder, target, label, minimum = 0.5, maximum = 30, disabled}) {
  const own = recorder.target === target;
  const phase = own ? recorder.phase : 'idle';
  const recording = phase === 'recording';
  const requesting = phase === 'requesting';
  const encoding = phase === 'encoding';
  const text = recording ? 'Stop recording' : requesting ? 'Cancel microphone request' : encoding ? 'Preparing audio…' : 'Record audio';
  return <div className="audio-capture" data-phase={phase}>
    <button className="audio-capture-button" type="button" disabled={disabled || encoding || (recorder.busy && !own)}
      aria-label={`${text} — ${label}`} onClick={() => recording ? recorder.stop() : requesting ? recorder.cancel() : recorder.start(target, {minimum, maximum})}>
      {recording ? <span className="capture-stop" aria-hidden="true"/> : <svg aria-hidden="true" viewBox="0 0 24 24"><rect x="9" y="2" width="6" height="12" rx="3"/><path d="M5 10v2a7 7 0 0 0 14 0v-2M12 19v3M8 22h8"/></svg>}
      <span>{text}</span>{recording && <time aria-hidden="true">{clock(recorder.elapsed)}</time>}
    </button>
    {phase !== 'idle' && <div className="audio-capture-status">
      <span role="status">{recording ? `Stops at ${maximum} seconds` : requesting ? 'Allow microphone access to begin.' : 'Preparing your recording.'}</span>
      {recording && <button type="button" className="audio-capture-discard" onClick={recorder.cancel} aria-label={`Discard recording — ${label}`}>Discard</button>}
    </div>}
  </div>;
}
