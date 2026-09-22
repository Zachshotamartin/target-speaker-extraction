import React, {useEffect, useMemo, useReducer, useRef, useState} from 'react';
import {TRANSCRIPTION_API as API} from './transcriptionApi.js';
import {animateChange} from './motion.js';
import {captionsSrt, editedDuration, editedToSource, editedWords, editHistory, editPlan,
  initialEdits, renderEdit, sourceToEdited, transcriptWords, wavBytes} from './audioEdit.js';
import './transcript-editor.css';

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

export default function TranscriptEditor({result, jobId, active}) {
  const words = useMemo(() => transcriptWords(result), [result]);
  const [history, dispatch] = useReducer(editHistory, initialEdits);
  const [audio, setAudio] = useState(null), [error, setError] = useState(''), [attempt, retry] = useState(0);
  const [track, setTrack] = useState('edited'), [mode, setMode] = useState('listen');
  const [selection, setSelection] = useState(null), [awaitingEnd, setAwaitingEnd] = useState(false);
  const [time, setTime] = useState(0), [message, setMessage] = useState('');
  const [editPending, setEditPending] = useState(false);
  const player = useRef(null), sourceTime = useRef(0), stopAt = useRef(null), editing = useRef(false);
  const pendingSeek = useRef(null), frame = useRef(0);
  const clips = useMemo(() => editPlan(words, history.removed, result.duration), [words, history.removed, result.duration]);
  const kept = useMemo(() => editedWords(words, history.removed, clips), [words, history.removed, clips]);
  const removed = useMemo(() => new Set(history.removed), [history.removed]);
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
  const currentWord = words.findIndex((word, index) => time >= word.start && time < word.end && (track !== 'edited' || !removed.has(index)));

  useEffect(() => {
    if (!active || audio) return;
    const controller = new AbortController();
    setError('');
    async function load() {
      try {
        const files = await Promise.all(['original', 'extracted'].map(async name => {
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
  }, [active, audio, attempt, jobId, result.duration, result.sample_rate]);

  useEffect(() => { if (!active) {player.current?.pause(); stopAt.current = null; cancelAnimationFrame(frame.current);} }, [active]);
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
    player.current?.pause(); stopAt.current = null; pendingSeek.current = sourceTime.current; setTrack(next);
  }
  function selectWord(index, extend = false) {
    if (mode === 'listen' && !extend) {stopAt.current = null; seek(words[index].start, true); return;}
    if (extend || awaitingEnd) {
      setSelection([selection?.[0] ?? index, index]); setAwaitingEnd(false);
    } else {setSelection([index, index]); setAwaitingEnd(true);}
    setMessage('');
  }
  function changeEdits(action, notice) {
    if (editing.current || editHistory(history, action) === history) return;
    editing.current = true;
    setEditPending(true);
    player.current?.pause(); stopAt.current = null; sourceTime.current = 0; setTime(0);
    if (track === 'edited') pendingSeek.current = 0;
    else if (player.current) player.current.currentTime = 0;
    // Guard immediately; commit history and its derived export artifacts together.
    animateChange(() => {dispatch(action); editing.current = false; setEditPending(false); setMessage(notice);}, '.transcript-editor');
  }
  function apply(operation) {
    if (!bounds) return;
    const selected = new Set(selectedIds);
    const next = operation === 'keep' ? words.map((_, index) => index).filter(index => !selected.has(index))
      : operation === 'restore' ? history.removed.filter(index => !selected.has(index))
      : [...history.removed, ...selectedIds];
    changeEdits({type: 'set', removed: next}, operation === 'keep' ? 'Only the selected passage is kept.' : operation === 'restore' ? 'Selected words restored.' : 'Selected words removed. Undo is available.');
  }
  function keyboard(event) {
    if (/INPUT|TEXTAREA|SELECT/.test(event.target.tagName)) return;
    if ((event.metaKey || event.ctrlKey) && event.key.toLowerCase() === 'z') {
      event.preventDefault(); changeEdits({type: event.shiftKey ? 'redo' : 'undo'}, event.shiftKey ? 'Edit redone.' : 'Edit undone.');
    } else if (event.key === 'Escape') {setSelection(null); setAwaitingEnd(false);}
  }

  return <section className="transcript-editor" aria-label="Transcript audio editor" onKeyDown={keyboard}>
    <div className="editor-heading"><div><h3>Edit by selecting words</h3><p>Keep a passage or remove words from the isolated voice. Your original stays intact.</p></div>
      <span className="editor-duration">{clock(editedDuration(clips))} <span>of {clock(result.duration)}</span></span></div>
    {!audio && !error && <p role="status" className="editor-status">Preparing audio for local editing…</p>}
    {error && <div className="editor-error" role="alert"><p>{error}</p><button type="button" onClick={() => retry(value => value + 1)}>Retry audio</button></div>}
    <div className="editor-listening">
      <label htmlFor="editor-track">Listen to<select id="editor-track" value={track} onChange={event => changeTrack(event.target.value)} disabled={!audio || editPending}>
        {tracks.map(([value, label]) => <option key={value} value={value}>{label}</option>)}
      </select></label>
      <audio ref={player} controls preload="metadata" src={url || undefined} aria-label="Editor audio"
        onLoadedMetadata={() => {const next = pendingSeek.current ?? sourceTime.current; pendingSeek.current = null; seek(next);}}
        onPlay={followPlayback} onPause={() => cancelAnimationFrame(frame.current)} onTimeUpdate={readPosition} onSeeked={readPosition}
        onError={() => {if (url) setMessage('Audio playback failed. Try switching tracks or reloading the audio.');}}/>
    </div>
    <div className="editor-tools">
      <div className="editor-mode" role="group" aria-label="Word interaction">
        <button type="button" aria-pressed={mode === 'listen'} onClick={() => {setMode('listen'); setAwaitingEnd(false);}}>Listen</button>
        <button type="button" aria-pressed={mode === 'select'} onClick={() => {setMode('select'); setAwaitingEnd(false);}}>Select words</button>
      </div>
      <div className="editor-history"><button type="button" disabled={editPending || !history.past.length} onClick={() => changeEdits({type: 'undo'}, 'Edit undone.')}>Undo</button>
        <button type="button" disabled={editPending || !history.future.length} onClick={() => changeEdits({type: 'redo'}, 'Edit redone.')}>Redo</button>
        <button type="button" disabled={editPending || !history.removed.length} onClick={() => changeEdits({type: 'set', removed: []}, 'All words restored. You can undo this reset.')}>Reset edits</button></div>
    </div>
    <p id="editor-word-help" className="editor-help">{mode === 'listen' ? 'Click a word to listen. Switch to Select words to make a cut.' : awaitingEnd ? 'Now choose the last word of your passage, or use the selection below.' : 'Choose the first and last word of a passage. Shift-click also extends a selection.'}</p>
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
      </React.Fragment>) : <p>No timed words were found. You can still listen to and export the full isolated audio.</p>}
    </div>
    <p className="editor-legend">Struck through = removed. Dotted underline = voice match unconfirmed. All transcribed words are available here; listen before keeping them.</p>
    <div className="editor-selection" aria-label="Passage selection">
      <p>{bounds ? `${selectedIds.length} ${selectedIds.length === 1 ? 'word' : 'words'} selected · ${clock(words[bounds[0]].start)}–${clock(words[bounds[1]].end)}` : 'Select a passage to edit.'}</p>
      <div className="editor-selection-actions">
        <button type="button" disabled={editPending || !bounds || !audio || (track === 'edited' && selectedIds.every(index => removed.has(index)))} onClick={() => {stopAt.current = words[bounds[1]].end; seek(words[bounds[0]].start, true);}}>Play selection</button>
        <button type="button" disabled={editPending || !bounds || selectedIds.every(index => removed.has(index))} onClick={() => apply('remove')}>Remove</button>
        <button type="button" disabled={editPending || !bounds} onClick={() => apply('keep')}>Keep only</button>
        <button type="button" disabled={editPending || !bounds || !selectedIds.some(index => removed.has(index))} onClick={() => apply('restore')}>Restore</button>
        <button type="button" disabled={!bounds} onClick={() => {setSelection(null); setAwaitingEnd(false);}}>Clear selection</button>
      </div>
    </div>
    {!clips.length && <p className="editor-status">All audio is removed. Restore words or undo an edit to export.</p>}
    <p className="editor-announcement" role="status">{message}</p>
    <div className="editor-export"><div><h4>Export your edit</h4><p>Subtitles use the edited audio’s timeline.</p></div>
      <div className="editor-export-links">
        {editedUrl && clips.length ? <a href={editedUrl} download="onevoice-edited.wav">Download WAV</a> : <button disabled>Download WAV</button>}
        {audio && kept.length ? <><a href={srtUrl} download="onevoice-edited.srt">Subtitles (.srt)</a><a href={textUrl} download="onevoice-edited.txt">Transcript (.txt)</a></> : <><button disabled>Subtitles (.srt)</button><button disabled>Transcript (.txt)</button></>}
      </div></div>
    <p className="editor-footnote">Cuts and exports run in this browser. Word timings are approximate; audition cuts before exporting. Edits last until you reload, replace this result, or its 15-minute expiry.</p>
  </section>;
}
