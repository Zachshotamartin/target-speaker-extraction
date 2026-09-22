import React, {useEffect, useMemo, useState} from 'react';
import {suggestions, waveformPeaks} from './naturalEdits.js';

export default function NaturalEditorTools({words, edit, bounds, audio, duration, change, preview}) {
  const [cut, setCut] = useState({start: 0, end: 0}), [text, setText] = useState(''), [pause, setPause] = useState(.45);
  const start = bounds?.[0], end = bounds?.[1];
  useEffect(() => {if (start !== undefined) setCut({start: words[start].start, end: words[end].end});}, [start, end]);
  const selectedText = words[start]?.text;
  useEffect(() => {if (selectedText !== undefined) setText(selectedText);}, [start, selectedText]);
  const ids = start === undefined ? [] : Array.from({length: end - start + 1}, (_, i) => start + i);
  const ideas = useMemo(() => suggestions(words, 1.2, pause), [words, pause]);
  const remaining = ideas.filter(idea => idea.cut ? !edit.customCuts.some(c => c.start === idea.start && c.end === idea.end) : !idea.ids.every(i => edit.removed.includes(i)));
  const left = start === undefined ? 0 : Math.max(0, words[start].start - 1), right = end === undefined ? 0 : Math.min(duration, words[end].end + 1);
  const peaks = useMemo(() => audio && right > left ? waveformPeaks(audio.channels[0], left, right, audio.sampleRate) : [], [audio, left, right]);
  function mark(state) {
    const decisions = {...edit.decisions}; ids.forEach(i => {if (state) decisions[i] = state; else delete decisions[i];});
    change({decisions}, state ? 'Speaker label updated. Audio is unchanged.' : 'Model labels restored.');
  }
  return <>
    {bounds && <div className="editor-refinements">
      <h4>Correct text or speaker</h4><p>Corrections update the transcript and labels. They do not cut audio.</p>
      {ids.length === 1 && <form onSubmit={e => {e.preventDefault(); if (text.trim()) change({corrections: {...edit.corrections, [start]: text.trim()}}, 'Text corrected. Audio is unchanged.');}}><label>Correct selected word<input value={text} maxLength={120} onChange={e => setText(e.target.value)}/></label><button type="submit">Save text</button></form>}
      <div className="editor-history"><button onClick={() => mark('accepted')}>This is the selected voice</button><button onClick={() => mark('excluded')}>Different voice</button><button onClick={() => mark(null)}>Restore model labels</button></div>
      {audio && <div className="waveform-cut"><h4>Fine-tune a cut</h4><p>Adjust the highlighted region, preview it, then apply the cut.</p>
        <svg viewBox="0 0 600 90" preserveAspectRatio="none" role="img" aria-label="Selected voice waveform">
          <rect x={Math.max(0, (cut.start - left) / (right - left) * 600)} width={Math.max(0, (Math.min(right, cut.end) - Math.max(left, cut.start)) / (right - left) * 600)} height="90" fill="currentColor" opacity=".1"/>
          {peaks.map((peak, i) => <line key={i} x1={i * 2.5} x2={i * 2.5} y1={45 - peak * 43} y2={45 + peak * 43} stroke="currentColor" strokeWidth="1.5"/>)}
        </svg>
        <div className="cut-boundaries"><label>Cut start (seconds)<input type="number" min="0" max={duration} step="0.01" value={cut.start} onChange={e => setCut({...cut, start: Number(e.target.value)})}/><input aria-label="Adjust cut start" type="range" min={left} max={right} step="0.01" value={cut.start} onChange={e => setCut({...cut, start: Number(e.target.value)})}/></label>
          <label>Cut end (seconds)<input type="number" min="0" max={duration} step="0.01" value={cut.end} onChange={e => setCut({...cut, end: Number(e.target.value)})}/><input aria-label="Adjust cut end" type="range" min={left} max={right} step="0.01" value={cut.end} onChange={e => setCut({...cut, end: Number(e.target.value)})}/></label>
          <button onClick={() => preview(Math.max(0, cut.start - .3), Math.min(duration, cut.end + .3))}>Preview boundary</button>
          <button disabled={!Number.isFinite(cut.start + cut.end) || cut.start < 0 || cut.end > duration || cut.end - cut.start < .01} onClick={() => change({customCuts: [...edit.customCuts, cut]}, 'Precise cut applied. Undo is available.')}>Apply precise cut</button>
        </div>
      </div>}
    </div>}
    <details className="editor-suggestions"><summary>Natural cuts &amp; suggestions</summary>
      <div className="cut-boundaries"><label>Breathing room around kept words<input type="range" min="0" max="0.3" step="0.01" value={edit.padding} onChange={e => change({padding: Number(e.target.value)}, 'Breathing room updated.')}/><span>{Math.round(edit.padding * 1000)} ms</span></label>
      <label>Keep this much of long pauses<select value={pause} onChange={e => setPause(Number(e.target.value))}><option value="0.3">0.3 seconds</option><option value="0.45">0.45 seconds</option><option value="0.75">0.75 seconds</option></select></label></div>
      <p>Suggestions are optional. Repetition may be intentional; listen before removing a take.</p>
      <ul>{remaining.map((idea, i) => <li key={`${idea.start}-${i}`}><span>{idea.label}</span><button disabled={!audio} onClick={() => preview(Math.max(0, idea.start - .4), Math.min(duration, idea.end + .4))}>Preview</button><button onClick={() => change(idea.cut ? {customCuts: [...edit.customCuts, {start: idea.start, end: idea.end}]} : {removed: [...new Set([...edit.removed, ...idea.ids])]}, 'Suggestion applied. Undo is available.')}>Apply</button></li>)}</ul>
      {!remaining.length && <p>No unapplied filler, pause, or repeated-phrase suggestions.</p>}
    </details>
  </>;
}
