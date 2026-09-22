import React, {useEffect, useRef} from 'react';
import {animateChange, useMotionState} from './motion.js';
import {useDismissibleDetails} from './useDismissibleDetails.js';
import {comparisonRows} from './comparisonRows.js';
import TranscriptEditor from './TranscriptEditor.jsx';
import './transcription-results.css';

import {TRANSCRIPTION_API as API} from './transcriptionApi.js';
const clock = time => `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, '0')}`;
const views = [['compare', 'Compare'], ['selected', 'Selected voice'], ['edit', 'Edit audio']];
const variants = [['original', 'Without One Voice'], ['extracted', 'With One Voice']];

function TranscriptLines({segments, seek, empty}) {
  return segments.length ? <ol className="transcript-lines">{segments.map((segment, index) =>
    <li key={index}><button type="button" onClick={() => seek(segment.start)} aria-label={`Play from ${clock(segment.start)}`}>
      <time>{clock(segment.start)}</time><span>{segment.text.trim()}</span>
    </button></li>)}</ol> : <p className="transcript-empty">{empty}</p>;
}

export default function TranscriptionResults({result, jobId, referenceName, inputsChanged, active, onDelete, onNotice, onError}) {
  const [view, setView] = useMotionState(result.comparison ? 'compare' : 'selected');
  const players = useRef({});
  const playing = useRef(null);
  const position = useRef(0);
  const exportMenu = useRef(null);
  useDismissibleDetails(exportMenu, '.selected-toolbar');
  const accepted = result.segments.filter(segment => segment.attribution === 'accepted');
  const uncertain = result.segments.filter(segment => segment.attribution === 'uncertain');
  const rows = comparisonRows(result.comparison, result.duration);

  useEffect(() => {
    Object.values(players.current).forEach(player => player?.pause());
  }, [view, active]);

  async function copy(text, label) {
    try { await navigator.clipboard.writeText(text); onNotice(`${label} copied.`); }
    catch { onError('Clipboard unavailable. Try selecting the text directly.'); }
  }

  function playFrom(key, seconds) {
    const player = players.current[key];
    if (!player) return;
    position.current = seconds;
    player.currentTime = Math.min(seconds, player.duration || seconds);
    player.play().catch(() => onNotice('Press Play to hear this section.'));
  }

  function audioProps(key) {
    return {
      ref: element => { players.current[key] = element; },
      onPlay: event => {
        const player = event.currentTarget;
        if (playing.current !== player) {
          const next = position.current >= player.duration ? 0 : position.current;
          player.currentTime = next;
        }
        playing.current = player;
        Object.values(players.current).forEach(other => { if (other !== player) other?.pause(); });
      },
      onTimeUpdate: event => { if (!event.currentTarget.paused) position.current = event.currentTarget.currentTime; },
      onSeeking: event => { position.current = event.currentTarget.currentTime; },
    };
  }

  return <>
    <div className="result-context">
      <strong>{referenceName || 'Your selected voice'}</strong>
      <span className="result-metrics">{clock(result.duration)} recording</span>
    </div>
    {inputsChanged && <p className="result-changed" role="status">Inputs changed. Transcribe again to update this result.</p>}
    <div className="result-tabs" role="tablist" aria-label="Result views">
      {views.map(([name, label], index) => <button key={name} id={`result-tab-${name}`} type="button" role="tab"
        aria-selected={view === name} aria-controls={`result-${name}`} tabIndex={view === name ? 0 : -1}
        onClick={() => setView(name)} onKeyDown={event => {
          const next = event.key === 'ArrowRight' ? (index + 1) % views.length : event.key === 'ArrowLeft' ? (index + views.length - 1) % views.length : event.key === 'Home' ? 0 : event.key === 'End' ? views.length - 1 : null;
          if (next !== null) { event.preventDefault(); setView(views[next][0]); document.getElementById(`result-tab-${views[next][0]}`).focus(); }
        }}>{label}{name === 'selected' && <span>{result.coverage?.accepted_words || 0} {(result.coverage?.accepted_words || 0) === 1 ? 'word' : 'words'}</span>}</button>)}
    </div>

    <div id="result-compare" role="tabpanel" aria-labelledby="result-tab-compare" hidden={view !== 'compare'} tabIndex={0} className="result-content comparison-view">
      {result.comparison ? <>
        <p className="result-explanation">Same recording and Whisper settings. Compare the original audio with the isolated voice.</p>
        <div className="comparison-players">
          {variants.map(([key, label]) => <section key={key} aria-label={label}>
            <h3>{label}</h3>
            <p>{key === 'original' ? 'Original recording → text' : 'Isolated voice → text'}</p>
            <audio {...audioProps(key)} controls preload="metadata" src={`${API}/transcriptions/${jobId}/audio/${key}`} aria-label={`${label} audio`}/>
          </section>)}
        </div>
        <div className="comparison-reading-guide"><span>Click text to play that section</span><span>Audio position stays linked</span></div>
        <div className="comparison-rows">
          {rows.map(row => <section className="comparison-row" key={row.start} aria-label={`${clock(row.start)} to ${clock(row.end)}`}>
            <div className="comparison-time"><time>{clock(row.start)}–{clock(row.end)}</time></div>
            {variants.map(([key, label]) => <div key={key} className="comparison-cell">
              <span className="comparison-mobile-label">{label}</span>
              {row[key] ? <button type="button" className="comparison-text" aria-label={`Play ${label.toLowerCase()} from ${clock(row.start)}`} onClick={() => playFrom(key, row.start)}>{row[key]}</button>
                : <p className="comparison-silence">No words transcribed in this section.</p>}
            </div>)}
          </section>)}
        </div>
        <div className="comparison-copy-actions">
          <button className="poc-text-button" type="button" disabled={!result.comparison.raw.text} onClick={() => copy(result.comparison.raw.text, 'Transcript without One Voice')}>Copy without One Voice</button>
          <button className="poc-text-button" type="button" disabled={!result.comparison.one_voice.text} onClick={() => copy(result.comparison.one_voice.text, 'Transcript with One Voice')}>Copy with One Voice</button>
        </div>
        <p className="comparison-method">Both transcripts are shown before speaker filtering. “Selected voice” contains the filtered text used for exports.</p>
      </> : <p className="transcript-empty">{result.outcome === 'no_speech' ? 'No speech was detected, so neither input was sent to Whisper.' : 'Comparison was off for this run. Enable it in Advanced and transcribe again to compare both versions.'}</p>}
    </div>

    <div id="result-selected" role="tabpanel" aria-labelledby="result-tab-selected" hidden={view !== 'selected'} tabIndex={0} className="result-content selected-view">
      <div className="selected-toolbar">
        <div><h3>Selected voice transcript</h3><p>Words matched to your voice reference.</p></div>
        <div className="result-actions">
          <button className="poc-text-button" type="button" disabled={!accepted.length} onClick={() => copy(accepted.map(segment => segment.text.trim()).join('\n'), 'Selected voice transcript')}>Copy text</button>
          <details className="poc-export-menu" ref={exportMenu} onClick={event => {
            if (event.target.closest('a')) { const menu = event.currentTarget; animateChange(() => { menu.open = false; }, '.selected-toolbar'); }
          }}>
            <summary>Export <span aria-hidden="true">↓</span></summary>
            <div className="export-options">
              <a href={`${API}/transcriptions/${jobId}/export/txt`} download><b>Text</b><span>Selected voice · .txt</span></a>
              <a href={`${API}/transcriptions/${jobId}/export/srt`} download><b>Subtitles</b><span>Selected voice with timestamps · .srt</span></a>
              <a href={`${API}/transcriptions/${jobId}/audio/extracted`} download><b>Isolated audio</b><span>One Voice output · .wav</span></a>
              <a href={`${API}/transcriptions/${jobId}/export/json`} download><b>Full report</b><span>All words and evidence · .json</span></a>
              <button type="button" onClick={onDelete}>Delete this result</button>
            </div>
          </details>
        </div>
      </div>
      <audio {...audioProps('selected')} className="selected-audio" controls preload="metadata" src={`${API}/transcriptions/${jobId}/audio/extracted`} aria-label="Selected voice audio"/>
      <p className="transcript-caption">Click a line to listen</p>
      <TranscriptLines segments={accepted} seek={seconds => playFrom('selected', seconds)} empty={result.outcome === 'no_speech' ? 'No speech was detected in this recording.' : 'No words confidently matched this reference. Review the comparison or try another reference sample.'}/>
      {uncertain.length > 0 && <details className="poc-review"><summary><span className="review-indicator" aria-hidden="true"/> {uncertain.length} uncertain {uncertain.length === 1 ? 'region' : 'regions'} <span aria-hidden="true">+</span></summary>
        <p>These words are not included in the selected transcript or its exports.</p>
        <TranscriptLines segments={uncertain} seek={seconds => playFrom('selected', seconds)}/>
      </details>}
    </div>

    <div id="result-edit" role="tabpanel" aria-labelledby="result-tab-edit" hidden={view !== 'edit'} tabIndex={0} className="result-content">
      <TranscriptEditor result={result} jobId={jobId} active={active && view === 'edit'}/>
    </div>

    <details className="run-details">
      <summary><span>Run details</span><span>{result.processing_seconds.toFixed(1)}s processing <span aria-hidden="true">+</span></span></summary>
      <div className="run-details-content">
        <dl className="run-metadata">
          <div><dt>Voice isolation</dt><dd>One Voice · epoch {result.model_manifest.one_voice.epoch}</dd></div>
          <div><dt>Transcriber</dt><dd>Whisper small.en · CPU int8</dd></div>
          <div><dt>Checkpoint</dt><dd><code>{result.model_manifest.one_voice.sha256.slice(0, 12)}</code></dd></div>
          <div><dt>Peak worker memory</dt><dd>{((result.peak_worker_rss_bytes || 0) / 1024 ** 3).toFixed(2)} GiB</dd></div>
        </dl>
        <h3>How this result was made</h3>
        <dl className="model-roles">
          <div><dt>One Voice</dt><dd>Isolates the speaker identified by your reference.</dd></div>
          <div><dt>Whisper</dt><dd>Transcribes audio. It never receives the voice reference.</dd></div>
          <div><dt>ECAPA</dt><dd>Checks voice similarity to filter the selected transcript.</dd></div>
          <div><dt>Silero</dt><dd>Finds speech and silence.</dd></div>
        </dl>
        <details className="poc-evidence"><summary>Voice-match scores <span aria-hidden="true">+</span></summary>
          <p>Similarity scores are not probabilities. Acceptance also requires evidence in the original audio.</p>
          <div className="poc-table-scroll" tabIndex={0} role="region" aria-label="Voice-match scores"><table><thead><tr><th>Time</th><th>Original</th><th>Isolated</th><th>Decision</th></tr></thead><tbody>{result.windows.map((window, index) => <tr key={index}><td>{clock(window.start)}–{clock(window.end)}</td><td>{window.original_similarity.toFixed(3)}</td><td>{window.extracted_similarity.toFixed(3)}</td><td>{window.attribution}</td></tr>)}</tbody></table></div>
        </details>
        <p className="accuracy-note">Isolation and voice matching can introduce errors. Timestamps are approximate; listen to the audio when reviewing the text.</p>
      </div>
    </details>
    <div className="result-footer">Temporary result · 15-minute storage</div>
  </>;
}
