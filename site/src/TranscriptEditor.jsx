import React, {useEffect, useMemo, useReducer, useRef, useState} from 'react';
import {TRANSCRIPTION_API as API} from './transcriptionApi.js';
import {animateChange} from './motion.js';
import {captionsSrt, editedDuration, editedToSource, editedWords,
  renderEdit, sourceToEdited, transcriptWords, wavBytes} from './audioEdit.js';
import './transcript-editor.css';
import {emptyEdit, historyStep, naturalPlan, correctedWords} from './naturalEdits.js';
import NaturalEditorTools from './NaturalEditorTools.jsx';

const clock = time => `${Math.floor(time / 60)}:${(time % 60).toFixed(1).padStart(4, '0')}`;
const tracks = [['edited', 'Edited voice'], ['isolated', 'Unedited voice'], ['original', 'Original recording']];
function useBlobUrl(blob) {
  const [url, setUrl] = useState('');
  useEffect(() => {
    if (!blob) {setUrl(''); return;}
    const next = URL.createObjectURL(blob); setUrl(next);
    return () => URL.revokeObjectURL(next);
  }, [blob]);
  return url;
}

export default function TranscriptEditor({result, jobId, active, audioFiles, savedEdit, onEdit, onRenderVideo, videoBusy}) {
  const rawWords = useMemo(() => transcriptWords(result), [result]);
  const [history, dispatch] = useReducer(historyStep, {past: [], present: {...emptyEdit, ...savedEdit}, future: []});
  const edit = history.present;
  const words = useMemo(() => correctedWords(rawWords, edit), [rawWords, edit]);
  useEffect(() => {onEdit?.(edit);}, [edit]);
  const [audio, setAudio] = useState(null), [error, setError] = useState(''), [attempt, retry] = useState(0);
  const [track, setTrack] = useState('edited');
  const [selection, setSelection] = useState(null), [awaitingEnd, setAwaitingEnd] = useState(false);
  const [time, setTime] = useState(0), [message, setMessage] = useState('');
  const [editPending, setEditPending] = useState(false);
  const player = useRef(null), sourceTime = useRef(0), stopAt = useRef(null), editing = useRef(false);
  const pendingSeek = useRef(null), pendingPreview = useRef(null), frame = useRef(0);
  const clips = useMemo(() => naturalPlan(rawWords, edit, result.duration), [rawWords, edit.removed, edit.customCuts, edit.padding, result.duration]);
  const kept = useMemo(() => editedWords(words, edit.removed, clips), [words, edit.removed, clips]);
  const removed = useMemo(() => new Set(words.flatMap((word, index) =>
    edit.removed.includes(index) || !clips.some(clip => clip.end > word.start && clip.start < word.end) ? [index] : [])), [words, edit.removed, clips]);
  const output = useMemo(() => {
    if (!audio || !clips.length) return null;
    return new Blob([wavBytes(renderEdit(audio.channels, audio.sampleRate, clips), audio.sampleRate)], {type: 'audio/wav'});
  }, [audio, clips]);
  const subtitles = useMemo(() => new Blob([captionsSrt(kept)], {type: 'application/x-subrip'}), [kept]);
  const text = useMemo(() => new Blob([kept.map(word => word.text).join(' ') + '\n'], {type: 'text/plain'}), [kept]);
  const originalUrl = useBlobUrl(audio?.original), isolatedUrl = useBlobUrl(audio?.isolated);
  const editedUrl = useBlobUrl(output), srtUrl = useBlobUrl(subtitles), textUrl = useBlobUrl(text);
  const url = {original: originalUrl, isolated: isolatedUrl, edited: editedUrl}[track];
  const bounds = selection ? [Math.min(...selection), Math.max(...selection)] : null;
  const selectedIds = bounds ? words.map((_, index) => index).slice(bounds[0], bounds[1] + 1) : [];
  const currentWord = player.current && !player.current.paused ? words.findIndex((word, index) => time >= word.start && time < word.end && (track !== 'edited' || !removed.has(index))) : -1;

  useEffect(() => {
    if (!active || audio) return;
    const controller = new AbortController();
    setError('');
    async function load() {
      try {
        const files = audioFiles ? [audioFiles.original, audioFiles.extracted] : await Promise.all(['original', 'extracted'].map(async name => {
          const response = await fetch(`${API}/transcriptions/${jobId}/audio/${name}`, {signal: controller.signal});
          if (!response.ok) throw new Error(response.status === 404 ? 'This audio result has expired. Transcribe again to edit it.' : 'Audio could not be loaded. Try again.');
          if (!response.headers.get('content-type')?.startsWith('audio/')) throw new Error('The service did not return an audio file.');
          return response.blob();
        }));
        const context = new OfflineAudioContext(1, 1, result.sample_rate || 16000);
        const buffer = await context.decodeAudioData(await files[1].arrayBuffer());
        if (Math.abs(buffer.duration - result.duration) > 0.05) throw new Error('Audio and transcript durations do not match. Transcribe again before editing.');
        if (!controller.signal.aborted) setAudio({original: files[0], isolated: files[1], sampleRate: buffer.sampleRate,
          channels: Array.from({length: buffer.numberOfChannels}, (_, index) => buffer.getChannelData(index))});
      } catch (failure) { if (!controller.signal.aborted) setError(failure.message || 'This browser could not decode the audio.'); }
    }
    load();
    return () => controller.abort();
  }, [active, audio, attempt, jobId, result.duration, result.sample_rate, audioFiles?.original, audioFiles?.extracted]);

  useEffect(() => { if (!active) {player.current?.pause(); stopAt.current = null; pendingPreview.current = null; cancelAnimationFrame(frame.current);} }, [active]);
  useEffect(() => () => cancelAnimationFrame(frame.current), []);
  useEffect(() => {editing.current = false;}, [history]);

  function readPosition() {
    if (pendingSeek.current !== null) return;
    const seconds = player.current?.currentTime || 0;
    const source = track === 'edited' ? editedToSource(seconds, clips) : seconds;
    sourceTime.current = source; setTime(source);
    if (stopAt.current !== null && source >= stopAt.current) {player.current?.pause(); stopAt.current = null;}
  }
  function followPlayback() {
    cancelAnimationFrame(frame.current);
    const tick = () => {
      readPosition();
      if (player.current && !player.current.paused) frame.current = requestAnimationFrame(tick);
    };
    frame.current = requestAnimationFrame(tick);
  }
  function seek(source, play = false) {
    sourceTime.current = source; setTime(source);
    if (!player.current) return;
    const seconds = track === 'edited' ? sourceToEdited(source, clips) : source;
    player.current.currentTime = Math.min(seconds, Number.isFinite(player.current.duration) ? player.current.duration : seconds);
    if (play) player.current.play().catch(() => setMessage('Press Play to listen.'));
  }
  function changeTrack(next) {
    if (next === track) return;
    player.current?.pause(); stopAt.current = null; pendingPreview.current = null;
    pendingSeek.current = sourceTime.current; setTrack(next);
  }
  function selectWord(index, extend = false) {
    player.current?.pause(); stopAt.current = null; pendingPreview.current = null;
    if (extend || awaitingEnd) {
      setSelection([selection?.[0] ?? index, index]); setAwaitingEnd(false);
    } else {setSelection([index, index]); setAwaitingEnd(true);}
    setMessage('');
  }
  function playSelection() {
    if (!bounds || !audio) return;
    if (!player.current.paused && stopAt.current !== null) {player.current.pause(); stopAt.current = null; return;}
    const start = words[bounds[0]].start, end = words[bounds[1]].end;
    // Always audition the uncut passage, including words the user may restore.
    if (track !== 'isolated') {
      player.current.pause(); pendingSeek.current = start; pendingPreview.current = end; setTrack('isolated');
    } else {stopAt.current = end; seek(start, true);}
  }
  function changeEdits(action, notice) {
    if (editing.current || historyStep(history, action.type === 'set' ? {value: {removed: action.removed}} : action) === history) return;
    editing.current = true;
    setEditPending(true);
    player.current?.pause(); stopAt.current = null; pendingPreview.current = null; sourceTime.current = 0; setTime(0);
    pendingSeek.current = 0;
    // Guard immediately; commit history and its derived export artifacts together.
    animateChange(() => {dispatch(action.type === 'set' ? {value: {removed: action.removed}} : action); setTrack('edited'); setAwaitingEnd(false); editing.current = false; setEditPending(false); setMessage(notice);}, '.transcript-editor');
  }
  function apply(operation) {
    if (!bounds) return;
    const selected = new Set(selectedIds);
    if (operation === 'restore') {
      const start = words[bounds[0]].start, end = words[bounds[1]].end;
      const customCuts = edit.customCuts.flatMap(cut => cut.end <= start || cut.start >= end ? [cut] : [
        {start: cut.start, end: Math.min(start, cut.end)}, {start: Math.max(end, cut.start), end: cut.end},
      ].filter(cut => cut.end - cut.start >= .01));
      changeEdits({value: {removed: edit.removed.filter(index => !selected.has(index)), customCuts}}, 'Selected words restored.');
      return;
    }
    const next = operation === 'keep' ? words.map((_, index) => index).filter(index => !selected.has(index))
      : operation === 'restore' ? edit.removed.filter(index => !selected.has(index))
      : [...edit.removed, ...selectedIds];
    changeEdits({type: 'set', removed: next}, operation === 'keep' ? 'Only this passage remains. Listen to your edit below.' : operation === 'restore' ? 'Words restored. Listen to your edit below.' : 'Passage removed. Listen below, or select another passage.');
  }
  function keyboard(event) {
    if (/INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') {
      event.preventDefault(); changeEdits({type: event.shiftKey ? 'redo' : 'undo'}, event.shiftKey ? 'Edit redone.' : 'Edit undone.');
    } else if (event.key === 'Escape') {setSelection(null); setAwaitingEnd(false);}
  }

  return <section className="transcript-editor" aria-label="Transcript audio editor" onKeyDown={keyboard}>
    <p className="editor-intro">Edit the isolated voice in three steps. Your original recording stays intact.</p>
    {!audio && !error && <p role="status" className="editor-status">Preparing audio for local editing…</p>}
    {error && <div className="editor-error" role="alert"><p>{error}</p><button type="button" onClick={() => retry(value => value + 1)}>Retry audio</button></div>}
    <ol className="editor-steps" aria-label="Audio editing steps">
      <li className="editor-step">
        <div className="editor-step-heading"><span className="editor-step-number" aria-hidden="true">01</span><h3>Select words</h3></div>
        <p id="editor-word-help" className="editor-step-help">Click the first and last word of a passage. For a single word, click it once.</p>
        <div className="editor-transcript" role="group" aria-label="Editable transcript" aria-describedby="editor-word-help">
          {words.length ? words.map((word, index) => <React.Fragment key={index}>
            {index > 0 && (word.start - words[index - 1].end > 1 || index % 24 === 0) && <span className="editor-break" aria-hidden="true"/>}
            <button type="button" className={`editor-word${removed.has(index) ? ' is-removed' : ''}${bounds && index >= bounds[0] && index <= bounds[1] ? ' is-selected' : ''}${currentWord === index ? ' is-current' : ''}${word.attribution && word.attribution !== 'accepted' ? ' is-uncertain' : ''}`}
              aria-label={`${word.text}, ${clock(word.start)}${removed.has(index) ? ', removed' : ''}${word.attribution && word.attribution !== 'accepted' ? ', voice match unconfirmed' : ''}`}
              aria-pressed={Boolean(bounds && index >= bounds[0] && index <= bounds[1])} aria-current={currentWord === index ? 'true' : undefined}
              title={`${clock(word.start)}–${clock(word.end)}`} onClick={event => selectWord(index, event.shiftKey)}
              onKeyDown={event => {
                if (event.key !== 'ArrowRight' && event.key !== 'ArrowLeft') return;
                event.preventDefault();
                const next = Math.max(0, Math.min(words.length - 1, index + (event.key === 'ArrowRight' ? 1 : -1)));
                if (event.shiftKey) {setSelection([selection?.[0] ?? index, next]); setAwaitingEnd(false);}
                event.currentTarget.parentElement.querySelectorAll('.editor-word')[next]?.focus();
              }}>{word.text}</button>{' '}
          </React.Fragment>) : <p>No timed words were found. Skip to step 3 to listen or download the full isolated audio.</p>}
        </div>
        <p className="editor-legend"><span>Crossed out: removed</span><span>Dotted underline: speaker unconfirmed</span></p>
        <div className="editor-selection" aria-label="Passage selection">
          <p aria-live="polite">{bounds ? <><strong>{selectedIds.length} {selectedIds.length === 1 ? 'word' : 'words'} selected</strong><span> · {clock(words[bounds[0]].start)}–{clock(words[bounds[1]].end)}</span>{awaitingEnd && <span className="editor-selection-hint">Click another word to extend, or edit this word below.</span>}</> : 'Choose a passage above to get started.'}</p>
          <div className="editor-selection-actions">
            <button type="button" disabled={editPending || !bounds || !audio} onClick={playSelection}>{player.current && !player.current.paused && stopAt.current !== null ? 'Stop preview' : 'Play selection'}</button>
            <button type="button" className="editor-text-button" disabled={!bounds} onClick={() => {setSelection(null); setAwaitingEnd(false);}}>Clear selection</button>
          </div>
        </div>
      </li>
      <li className="editor-step">
        <div className="editor-step-heading"><span className="editor-step-number" aria-hidden="true">02</span><h3>Make your edit</h3></div>
        <p className="editor-step-help">Choose what happens to your selection. Every edit can be undone.</p>
        <div className="editor-edit-actions">
          <div><button type="button" disabled={editPending || !bounds || selectedIds.every(index => removed.has(index))} onClick={() => apply('remove')}>Remove selected</button><p>Cut this passage out.</p></div>
          <div><button type="button" disabled={editPending || !bounds} onClick={() => apply('keep')}>Keep only selected</button><p>Remove everything else.</p></div>
        </div>
        <div className="editor-history" aria-label="Edit history">
          <button type="button" disabled={editPending || !history.past.length} onClick={() => changeEdits({type: 'undo'}, 'Edit undone.')}>Undo</button>
          <button type="button" disabled={editPending || !history.future.length} onClick={() => changeEdits({type: 'redo'}, 'Edit redone.')}>Redo</button>
          <button type="button" className="editor-text-button" disabled={editPending || (!edit.removed.length && !edit.customCuts.length && !Object.keys(edit.corrections).length && !Object.keys(edit.decisions).length)} onClick={() => changeEdits({type: 'reset'}, 'All edits reset. You can undo this reset.')}>Reset edits</button>
          {bounds && selectedIds.some(index => removed.has(index)) && <button type="button" disabled={editPending} onClick={() => apply('restore')}>Restore selected</button>}
        </div>
        <NaturalEditorTools words={words} edit={edit} bounds={bounds} audio={audio} duration={result.duration}
          change={(value, notice) => changeEdits({value}, notice)} preview={(start, end) => {
            player.current?.pause();
            if (track !== 'isolated') {pendingSeek.current = start; pendingPreview.current = end; setTrack('isolated');}
            else {stopAt.current = end; seek(start, true);}
          }}/>
        <p className="editor-announcement" role="status">{message}</p>
      </li>
      <li className="editor-step">
        <div className="editor-step-heading"><span className="editor-step-number" aria-hidden="true">03</span><h3>Listen &amp; download</h3></div>
        <p className="editor-step-help">Check your edit, then save it. Switch tracks to compare with the original.</p>
        <p className="editor-duration"><strong>{clock(editedDuration(clips))}</strong> edited <span>· {clock(result.duration)} original</span></p>
        <div className="editor-listening">
          <label htmlFor="editor-track">Preview track<select id="editor-track" value={track} onChange={event => changeTrack(event.target.value)} disabled={!audio || editPending}>
            {tracks.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
          </select></label>
          <audio ref={player} controls preload="metadata" src={url || undefined} aria-label="Editor audio"
            onLoadedMetadata={() => {
              const next = pendingSeek.current ?? sourceTime.current, end = pendingPreview.current;
              pendingSeek.current = null; pendingPreview.current = null;
              if (end !== null) stopAt.current = end;
              seek(next, end !== null && active);
            }}
            onPlay={followPlayback} onPause={() => {cancelAnimationFrame(frame.current); setTime(sourceTime.current);}} onTimeUpdate={readPosition} onSeeked={readPosition}
            onError={() => {if (url) setMessage('Audio playback failed. Try switching tracks or reloading the audio.');}}/>
        </div>
        {!clips.length && <p className="editor-status">All audio is removed. Undo an edit or restore words in step 2 to download.</p>}
        <div className="editor-export-links">
          {editedUrl && clips.length ? <a href={editedUrl} download="onevoice-edited.wav">Download audio (.wav)</a> : <button disabled>Download audio (.wav)</button>}
          {audio && kept.length ? <><a href={srtUrl} download="onevoice-edited.srt">Subtitles (.srt)</a><a href={textUrl} download="onevoice-edited.txt">Transcript (.txt)</a></> : <><button disabled>Subtitles (.srt)</button><button disabled>Transcript (.txt)</button></>}
        </div>
        {onRenderVideo && <button className="poc-primary" type="button" disabled={!output || videoBusy} onClick={() => onRenderVideo({audio: output, clips, words: kept})}>{videoBusy ? 'Processing…' : 'Export captioned video (.mp4)'}</button>}
        <p className="editor-export-note">Downloads always contain your edit. Subtitle times follow the edited audio.</p>
      </li>
    </ol>
    <p className="editor-footnote">Word timings are approximate—listen before saving. {audioFiles ? 'Audio and edits are saved in this browser with your project.' : 'Download to keep your work before this temporary result expires.'}</p>
  </section>;
}
