import React from 'react';
import Dialog from '../ui/Dialog.jsx';
const clock = time => `${Math.floor(time / 60)}:${(time % 60).toFixed(1).padStart(4, '0')}`;
export default function ResultDetails({result, notice}) {
  return <Dialog label="Run details" className="result-details">{notice && <p className="accuracy-note run-details-notice">{notice}</p>}
      <div className="run-details-content">
        <dl className="run-metadata">
          <div><dt>Processing time</dt><dd>{result.processing_seconds.toFixed(1)} seconds</dd></div>
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
  </Dialog>;
}
