import AudioPlayer from './ui/AudioPlayer.jsx';
import SplitLayout from './ui/SplitLayout.jsx';
import ResultNavigation from './workspace/ResultNavigation.jsx';
import ResultDetails from './workspace/ResultDetails.jsx';
import ComparisonView from './workspace/ComparisonView.jsx';
import Heading from './ui/Heading.jsx';
import {transcriptPhrases} from './transcriptPhrases.js';
import TranscriptLines from './TranscriptLines.jsx';
import React, {useEffect, useMemo, useRef} from 'react';
import {animateChange, useMotionState} from './motion.js';
import {useDismissibleDetails} from './useDismissibleDetails.js';
import {comparisonRows} from './comparisonRows.js';
import TranscriptEditor from './TranscriptEditor.jsx';
import {useAudioUrl} from './useAudioUrl.js';
import {captionsSrt, transcriptWords} from './audioEdit.js';
import {emptyEdit, correctedWords} from './naturalEdits.js';
import './transcription-results.css';

import {TRANSCRIPTION_API as API} from './transcriptionApi.js';
const clock = time => `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, '0')}`;



export default function TranscriptionResults({result, jobId, referenceName, inputsChanged, active, onDelete, onNotice, onError, audioFiles, savedEdit, onEdit, onRenderVideo, videoBusy, notice}) {
  const [view, setView] = useMotionState('selected');
  const players = useRef({});
  const playing = useRef(null);
  const position = useRef(0);
  const exportMenu = useRef(null);
  useDismissibleDetails(exportMenu, '.selected-toolbar');
  const originalUrl = useAudioUrl(audioFiles?.original), extractedUrl = useAudioUrl(audioFiles?.extracted);
  const reviewed = useMemo(() => correctedWords(transcriptWords(result), {...emptyEdit, ...savedEdit}), [result, savedEdit]);
  const selectedWords = reviewed.filter(w => w.attribution === 'accepted');
  const textBlob = useMemo(() => new Blob([selectedWords.map(w => w.text).join(' ')], {type: 'text/plain'}), [reviewed]);
  const srtBlob = useMemo(() => new Blob([captionsSrt(selectedWords)], {type: 'application/x-subrip'}), [reviewed]);
  const reportBlob = useMemo(() => new Blob([JSON.stringify({result, review: savedEdit}, null, 2)], {type: 'application/json'}), [result, savedEdit]);
  const textUrl = useAudioUrl(textBlob), srtUrl = useAudioUrl(srtBlob), reportUrl = useAudioUrl(reportBlob);
  const accepted = audioFiles ? selectedWords : result.segments.filter(segment => segment.attribution === 'accepted');
  const uncertain = audioFiles ? reviewed.filter(w => w.attribution === 'uncertain') : result.segments.filter(segment => segment.attribution === 'uncertain');
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

  return <SplitLayout className="result-workbench" sidebar={<ResultNavigation value={view} onChange={setView} name={referenceName} duration={clock(result.duration)} words={selectedWords.length}><ResultDetails result={result} notice={notice}/><span className="result-save-note">{audioFiles ? 'Saved on this device' : 'Temporary result · 15 minutes'}</span></ResultNavigation>}>
    {inputsChanged && <p className="result-changed" role="status">Inputs changed. Transcribe again to update this result.</p>}
    <div id="result-compare" role="tabpanel" aria-labelledby="result-tab-compare" hidden={view !== 'compare'} tabIndex={0} className="result-content comparison-view">
      <ComparisonView result={result} rows={rows} audioProps={audioProps} playFrom={playFrom} copy={copy}
        originalUrl={audioFiles ? originalUrl : `${API}/transcriptions/${jobId}/audio/original`}
        extractedUrl={audioFiles ? extractedUrl : `${API}/transcriptions/${jobId}/audio/extracted`}/>
    </div>

    <div id="result-selected" role="tabpanel" aria-labelledby="result-tab-selected" hidden={view !== 'selected'} tabIndex={0} className="result-content selected-view">
      <div className="selected-toolbar">
        <Heading>Transcript</Heading>
        <div className="result-actions">
          <button className="poc-text-button" type="button" disabled={!accepted.length} onClick={() => copy(accepted.map(segment => segment.text.trim()).join('\n'), 'Selected voice transcript')}>Copy text</button>
          <details className="poc-export-menu" ref={exportMenu} onClick={event => {
            if (event.target.closest('a')) { const menu = event.currentTarget; animateChange(() => { menu.open = false; }, '.selected-toolbar'); }
          }}>
            <summary>Export <span aria-hidden="true">↓</span></summary>
            <div className="export-options">
              <a href={audioFiles ? textUrl : `${API}/transcriptions/${jobId}/export/txt`} download="onevoice.txt"><b>Text</b><span>Selected voice · .txt</span></a>
              <a href={audioFiles ? srtUrl : `${API}/transcriptions/${jobId}/export/srt`} download="onevoice.srt"><b>Subtitles</b><span>Selected voice with timestamps · .srt</span></a>
              <a href={audioFiles ? extractedUrl : `${API}/transcriptions/${jobId}/audio/extracted`} download="onevoice-isolated.wav"><b>Isolated audio</b><span>One Voice output · .wav</span></a>
              <a href={audioFiles ? reportUrl : `${API}/transcriptions/${jobId}/export/json`} download="onevoice.json"><b>Full report</b><span>All words and evidence · .json</span></a>
              <button type="button" onClick={onDelete}>Delete this result</button>
            </div>
          </details>
        </div>
      </div>
      <AudioPlayer {...audioProps('selected')} className="selected-audio" src={audioFiles ? extractedUrl : `${API}/transcriptions/${jobId}/audio/extracted`} label="Selected voice audio"/>
      <p className="transcript-caption">Words matched to your reference. Click a passage to listen.</p>
      <TranscriptLines segments={transcriptPhrases(accepted)} seek={seconds => playFrom('selected', seconds)} empty={result.outcome === 'no_speech' ? 'No speech was detected in this recording.' : 'No words confidently matched this reference. Review the comparison or try another reference sample.'}/>
      {uncertain.length > 0 && <details className="poc-review"><summary><span className="review-indicator" aria-hidden="true"/> {uncertain.length} unconfirmed {audioFiles ? (uncertain.length === 1 ? 'word' : 'words') : (uncertain.length === 1 ? 'passage' : 'passages')} <span aria-hidden="true">+</span></summary>
        <p>These words are not included in the selected transcript or its exports.</p>
        <TranscriptLines segments={transcriptPhrases(uncertain)} seek={seconds => playFrom('selected', seconds)}/>
      </details>}
    </div>

    <div id="result-edit" role="tabpanel" aria-labelledby="result-tab-edit" hidden={view !== 'edit'} tabIndex={0} className="result-content">
      <TranscriptEditor result={result} jobId={jobId} active={active && view === 'edit'} audioFiles={audioFiles} savedEdit={savedEdit} onEdit={onEdit} onRenderVideo={onRenderVideo} videoBusy={videoBusy}/>
    </div>

  </SplitLayout>;
}
