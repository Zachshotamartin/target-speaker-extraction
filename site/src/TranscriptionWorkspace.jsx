import React, {useEffect, useRef, useState} from 'react';
import {animateChange, useMotionState} from './motion.js';
import {profiles} from './voiceProfiles.js';
import './transcription.css';

const API = '/api/poc';
const AUDIO_TYPES = 'audio/*,.wav,.flac,.mp3,.m4a,.webm,.ogg';
const terminal = new Set(['ready', 'failed', 'cancelled', 'expired']);
const clock = time => `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, '0')}`;

async function request(path, options) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    let reason;
    try { reason = (await response.json()).detail; } catch { /* Proxy errors are not JSON. */ }
    throw new Error(typeof reason === 'string' ? reason : `Local service returned ${response.status}. Check that the transcription service is running.`);
  }
  return response;
}

function useAudioUrl(blob) {
  const [url, setUrl] = useMotionState(null);
  useEffect(() => {
    if (!blob) { setUrl(null); return; }
    const value = URL.createObjectURL(blob);
    setUrl(value);
    return () => URL.revokeObjectURL(value);
  }, [blob]);
  return url;
}

function Transcript({segments, seek, empty}) {
  return segments.length ? <ol className="transcript-lines">{segments.map((segment, index) =>
    <li key={index}><button type="button" onClick={() => seek(segment.start)} aria-label={`Play from ${clock(segment.start)}`}>
      <time>{clock(segment.start)}</time><span>{segment.text.trim()}</span>
    </button></li>)}</ol> : <p className="transcript-empty">{empty}</p>;
}

export default function TranscriptionWorkspace({active = true}) {
  const [health, setHealth] = useMotionState(null);
  const [resultView, setResultView] = useMotionState('transcript');
  const [demos, setDemos] = useMotionState([]);
  const [demo, setDemo] = useMotionState(0);
  const [condition, setCondition] = useMotionState('mixture');
  const [source, setSource] = useMotionState('files');
  const [reference, setReference] = useMotionState(null);
  const [recording, setRecording] = useMotionState(null);
  const [referenceName, setReferenceName] = useMotionState('');
  const [recordingName, setRecordingName] = useMotionState('');
  const [saved, setSaved] = useMotionState([]);
  const [profileId, setProfileId] = useMotionState('');
  const [profileName, setProfileName] = useState('My voice');
  const [compare, setCompare] = useMotionState(true);
  const [job, setJob] = useMotionState(null);
  const [submittedInput, setSubmittedInput] = useMotionState(null);
  const [sending, setSending] = useMotionState(false);
  const [error, setError] = useMotionState('');
  const [notice, setNotice] = useMotionState('');
  const [capturing, setCapturing] = useMotionState(false);
  const [requestingMic, setRequestingMic] = useMotionState(false);
  const [elapsed, setElapsed] = useState(0);
  const [track, setTrack] = useMotionState('extracted');
  const [loadingExample, setLoadingExample] = useMotionState(false);
  const mounted = useRef(true);
  const workspace = useRef(null);
  const fileDraft = useRef(null);
  const sourceChoice = useRef('files');
  const pendingExample = useRef(false);
  const activeRef = useRef(active);
  activeRef.current = active;
  const recorder = useRef(null);
  const stream = useRef(null);
  const recordTimer = useRef(null);
  const audio = useRef(null);
  const position = useRef(0);
  const latestJob = useRef(null);
  const submitting = useRef(false);
  const referenceUrl = useAudioUrl(reference);
  const recordingUrl = useAudioUrl(recording);
  const busy = sending || (job && !terminal.has(job.status));
  latestJob.current = job;

  useEffect(() => {
    mounted.current = true;
    Promise.all([request('/health').then(r => r.json()), request('/demos').then(r => r.json())])
      .then(([info, list]) => { if (mounted.current) { setHealth(info); setDemos(list); } })
      .catch(() => { if (mounted.current) setError('The local transcription service is unavailable. Start it with the command in poc/README.md.'); });
    profiles('list').then(list => { if (mounted.current) setSaved(list); }).catch(e => setError(e.message));
    return () => {
      mounted.current = false;
      clearInterval(recordTimer.current);
      if (recorder.current?.state === 'recording') recorder.current.stop();
      stream.current?.getTracks().forEach(t => t.stop());
      const active = latestJob.current;
      if (active && !terminal.has(active.status)) fetch(`${API}/transcriptions/${active.id}`, {method: 'DELETE', keepalive: true}).catch(() => {});
    };
  }, []);

  useEffect(() => {
    if (source !== 'public' || !demos.length) return;
    const controller = new AbortController();
    setLoadingExample(true); setReference(null); setRecording(null); setError(''); setProfileId('');
    Promise.all(['reference', condition].map(part => request(`/demos/${demo}/${part}`, {signal: controller.signal}).then(r => r.blob())))
      .then(([ref, mix]) => {
        if (controller.signal.aborted || sourceChoice.current !== 'public') return;
        const selected = demos.find(item => item.id === Number(demo));
        setReference(ref); setRecording(mix);
        setReferenceName(`Conversation ${selected.conversation}, voice ${selected.voice}`);
        setRecordingName({mixture: 'Two voices overlapping', target: 'Selected voice alone', absent: 'Other voice alone', silence: 'Silence'}[condition]);
      }).catch(e => { if (!controller.signal.aborted) setError(e.message); })
      .finally(() => { if (!controller.signal.aborted) {
        setLoadingExample(false);
        animateChange(() => {pendingExample.current = false;}, '#transcribe');
      } });
    return () => controller.abort();
  }, [demo, condition, source, demos]);

  useEffect(() => {
    if (!job?.id || terminal.has(job.status)) return;
    let cancelled = false, timer;
    const poll = async () => {
      try {
        const next = await request(`/transcriptions/${job.id}`).then(r => r.json());
        if (!cancelled) {
          if (next.status !== latestJob.current?.status || next.stage !== latestJob.current?.stage) setJob(next);
          setError('');
        }
      } catch (e) { if (!cancelled) setError(`${e.message} Your job may still be running; reconnecting…`); }
      if (!cancelled) timer = setTimeout(poll, 1500);
    };
    timer = setTimeout(poll, 500);
    return () => { cancelled = true; clearTimeout(timer); };
  }, [job?.id, job?.status]);

  useEffect(() => {
    if (job?.status !== 'ready') return;
    const timer = setTimeout(() => setJob(previous => previous?.id === job.id ? {...previous, status: 'expired', result: null} : previous),
      Math.max(0, job.expires_at * 1000 - Date.now()));
    return () => clearTimeout(timer);
  }, [job?.id, job?.status, job?.expires_at]);

  useEffect(() => {
    if (!active) {
      workspace.current?.querySelectorAll('audio').forEach(player => player.pause());
      if (recorder.current?.state === 'recording') recorder.current.stop();
    }
  }, [active]);

  function chooseSource(next) {
    if (next === sourceChoice.current) return;
    if (sourceChoice.current === 'files' && source === 'files') fileDraft.current = {reference, recording, referenceName, recordingName, profileId};
    sourceChoice.current = next;
    pendingExample.current = next === 'public';
    const draft = next === 'files' ? fileDraft.current : null;
    setReference(draft?.reference || null); setRecording(draft?.recording || null);
    setReferenceName(draft?.referenceName || ''); setRecordingName(draft?.recordingName || '');
    setProfileId(draft?.profileId || ''); setLoadingExample(false); setSource(next); setError(''); setNotice('');
  }

  async function start() {
    if (submitting.current || busy || pendingExample.current) return;
    submitting.current = true;
    setError(''); setNotice(''); setSending(true); setResultView('transcript'); position.current = 0; setTrack('extracted');
    try {
      if (!reference || !recording) throw new Error('Choose a reference voice and a recording first.');
      if (reference.size + recording.size > 4 * 1024 * 1024 - 2048) throw new Error('The two audio files must total less than 4 MiB.');
      if (job?.id) await request(`/transcriptions/${job.id}`, {method: 'DELETE'}).catch(() => {});
      setJob(null);
      const data = new FormData();
      data.append('reference', reference, 'reference.audio'); data.append('mixture', recording, 'recording.audio');
      data.append('compare', String(compare));
      const next = await request('/transcriptions', {method: 'POST', body: data}).then(r => r.json());
      setSubmittedInput({reference, recording, referenceName, recordingName}); setJob(next);
      if (activeRef.current && window.matchMedia('(max-width: 680px)').matches) {
        requestAnimationFrame(() => document.getElementById('result-title')?.scrollIntoView({
          behavior: window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'instant' : 'smooth', block: 'start',
        }));
      }
    } catch (e) { setError(e.message); } finally { setSending(false); animateChange(() => {submitting.current = false;}, '#transcribe'); }
  }

  async function removeJob() {
    try { if (job?.id) await request(`/transcriptions/${job.id}`, {method: 'DELETE'}); setJob(null); setError(''); }
    catch (e) { setError(e.message); }
  }

  async function saveProfile() {
    if (!reference) return;
    try {
      const id = crypto.randomUUID();
      await profiles('save', {id, name: profileName.trim().slice(0, 60) || 'Voice reference', audio: reference});
      setSaved(await profiles('list')); setProfileId(id); setNotice('Reference saved in this browser on this device.');
    } catch (e) { setError(e.message); }
  }

  async function deleteProfile() {
    try {
      await profiles('delete', profileId); setSaved(await profiles('list'));
      setProfileId(''); setReference(null); setReferenceName(''); setNotice('Saved reference deleted from this browser.');
    } catch (e) { setError(e.message); }
  }

  async function capture() {
    setError(''); setNotice('');
    if (recorder.current?.state === 'recording') { recorder.current.stop(); return; }
    if (requestingMic) return;
    setRequestingMic(true);
    try {
      if (!navigator.mediaDevices?.getUserMedia || !window.MediaRecorder) throw new Error('Microphone capture is unavailable here. Upload a recording instead.');
      stream.current = await navigator.mediaDevices.getUserMedia({audio: {echoCancellation: false, noiseSuppression: false, autoGainControl: false}});
      if (!mounted.current || !activeRef.current) { stream.current.getTracks().forEach(t => t.stop()); return; }
      const preferred = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/webm'].find(type => MediaRecorder.isTypeSupported(type));
      const chunks = [];
      const captureRecorder = new MediaRecorder(stream.current, preferred ? {mimeType: preferred, audioBitsPerSecond: 64000} : undefined);
      recorder.current = captureRecorder;
      captureRecorder.ondataavailable = event => { if (event.data.size) chunks.push(event.data); };
      captureRecorder.onstop = () => {
        clearInterval(recordTimer.current); stream.current?.getTracks().forEach(t => t.stop());
        if (mounted.current) {
          setCapturing(false); setRecording(new Blob(chunks, {type: captureRecorder.mimeType})); setRecordingName('Microphone recording');
        }
      };
      captureRecorder.onerror = () => { setError('Recording failed. Try uploading a file.'); if (captureRecorder.state !== 'inactive') captureRecorder.stop(); };
      captureRecorder.start(250); setCapturing(true); setElapsed(0);
      const began = Date.now();
      recordTimer.current = setInterval(() => {
        const seconds = (Date.now() - began) / 1000; setElapsed(seconds);
        // Stop slightly early to allow the final encoded frame within the 30s server limit.
        if (seconds >= 29 && captureRecorder.state === 'recording') captureRecorder.stop();
      }, 100);
    } catch (e) { stream.current?.getTracks().forEach(t => t.stop()); setError(e.message || 'Microphone permission was denied.'); }
    finally { if (mounted.current) setRequestingMic(false); }
  }

  function seek(seconds) {
    if (!audio.current) return;
    audio.current.currentTime = seconds;
    audio.current.play().catch(() => setNotice('Press Play to hear this segment.'));
  }

  const result = !sending && job?.status === 'ready' ? job.result : null;
  const accepted = result?.segments.filter(s => s.attribution === 'accepted') || [];
  const uncertain = result?.segments.filter(s => s.attribution === 'uncertain') || [];

  const processingStep = !job || job.status === 'queued' ? 0 : /Transcribing/.test(job.stage) ? 3 : /Checking the selected/.test(job.stage) ? 2 : /Extracting/.test(job.stage) ? 1 : 0;

  return <section ref={workspace} id="transcribe" className="transcription" aria-labelledby="transcription-title" onPlay={event => {
    workspace.current?.querySelectorAll('audio').forEach(player => { if (player !== event.target) player.pause(); });
  }}>
    <a className="workspace-breadcrumb" href="#"><span aria-hidden="true">←</span> Overview</a>
    <header className="transcription-heading">
      <div><h1 id="transcription-title">Speech to text</h1></div>
      <div className="workspace-actions"><div className={`poc-connection ${health?.ready ? 'is-ready' : ''}`}>
        <span className="connection-dot" aria-hidden="true"/>
        <span>{health?.ready ? 'Local processing' : health ? 'Models need setup' : 'Connecting…'}</span>
        {!health?.ready && <button type="button" onClick={async () => {try {
          const [info, list] = await Promise.all([request('/health').then(r => r.json()), request('/demos').then(r => r.json())]);
          setHealth(info); setDemos(list); setError('');
        } catch (e) {setError(e.message);}}}>Retry</button>}
      </div><button className="poc-primary" type="button" onClick={start} disabled={!health?.ready || busy || capturing || requestingMic || !reference || !recording || loadingExample}>
            {busy ? 'Processing…' : 'Transcribe'}<span aria-hidden="true">↗</span>
          </button></div>
    </header>

    <div className="workspace-grid">
      <aside className="setup-panel" aria-labelledby="setup-title">
        <div className="setup-panel-heading"><h2 id="setup-title">Audio setup</h2><span>English</span></div>
        <div className="poc-source-switch" role="group" aria-label="Recording source" data-source={source}>
          {[['files', 'Upload audio'], ['public', 'Use example']].map(([value, label]) =>
            <button key={value} type="button" aria-pressed={source === value} disabled={busy || capturing || requestingMic} onClick={() => chooseSource(value)}>{label}</button>)}
        </div>

        <fieldset className="setup-fields" disabled={busy || capturing || requestingMic}>
          <legend className="sr-only">Choose reference voice and recording</legend>
          <section className="setup-group" aria-labelledby="reference-heading">
            <h3 id="reference-heading">Voice reference</h3>
            {source === 'public' ? <>
              <label className="sr-only" htmlFor="demo-voice">Voice to select</label>
              <select id="demo-voice" value={demo} onChange={event => {pendingExample.current = true; setDemo(Number(event.target.value));}}>
                {demos.map(item => <option key={item.id} value={item.id}>Conversation {item.conversation} · Voice {item.voice}</option>)}
              </select>
            </> : <>
              <p className="field-help">3–10 seconds · one speaker</p>
              {saved.length > 0 && <div className="poc-profile-row">
                <label htmlFor="saved-voice">Saved on this device</label>
                <div><select id="saved-voice" value={profileId} onChange={e => {
                  setProfileId(e.target.value); const selected = saved.find(p => p.id === e.target.value);
                  if (selected) {setReference(selected.audio); setReferenceName(selected.name);} else {setReference(null); setReferenceName('');}
                }}><option value="">Choose a saved reference</option>{saved.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}</select>
                  <button className="poc-text-button" type="button" disabled={!profileId} onClick={deleteProfile}>Delete</button>
                </div>
              </div>}
              <label className="upload-field"><span className="sr-only">Reference audio</span><span className="file-picker"><span aria-hidden="true">↑</span>{reference ? 'Replace reference' : 'Upload reference'}<input aria-label="Reference audio" type="file" accept={AUDIO_TYPES} onChange={e => {setReference(e.target.files?.[0] || null); setReferenceName(e.target.files?.[0]?.name || ''); setProfileId('');}}/></span></label>
              {referenceName && <p className="poc-filename" title={referenceName}>{referenceName}</p>}
            </>}
            {loadingExample ? <div className="audio-loading" role="status">Loading reference…</div> : referenceUrl && <audio controls preload="metadata" src={referenceUrl} aria-label="Reference voice sample"/>}
            {source === 'files' && reference && <details className="poc-profile-save"><summary>Save voice</summary>
              <label>Profile name<input maxLength={60} value={profileName} onChange={e => setProfileName(e.target.value)}/></label>
              <button type="button" onClick={saveProfile}>Save reference</button>
              <p>This stores the audio in this browser. It does not train a model or upload it to a cloud service.</p>
            </details>}
          </section>

          <section className="setup-group" aria-labelledby="recording-heading">
            <h3 id="recording-heading">Recording</h3>
            {source === 'public' ? <>
              <label className="sr-only" htmlFor="demo-condition">Recording condition</label>
              <select id="demo-condition" value={condition} onChange={event => {pendingExample.current = true; setCondition(event.target.value);}}>
                <option value="mixture">Two voices overlapping</option><option value="target">Selected voice alone</option>
                <option value="absent">Other voice only (target absent)</option><option value="silence">Silence</option>
              </select>
            </> : <>
              <label className="upload-field"><span className="sr-only">Recording file</span><span className="file-picker"><span aria-hidden="true">↑</span>{recording ? 'Replace recording' : 'Upload recording'}<input aria-label="Recording file" type="file" accept={AUDIO_TYPES} onChange={e => {setRecording(e.target.files?.[0] || null); setRecordingName(e.target.files?.[0]?.name || '');}}/></span></label>
              {recordingName && <p className="poc-filename" title={recordingName}>{recordingName}</p>}
            </>}
            {loadingExample ? <div className="audio-loading" role="status">Loading recording…</div> : recordingUrl && <audio controls preload="metadata" src={recordingUrl} aria-label="Input recording"/>}
          </section>
        </fieldset>
        {source === 'files' && <div className="poc-microphone">
          <button type="button" onClick={capture} disabled={busy || requestingMic} className={capturing ? 'is-recording' : ''}>
            {requestingMic ? 'Opening microphone…' : capturing ? `Stop recording · ${clock(elapsed)}` : 'Record audio'}
          </button>
          {capturing && <p role="status">Recording. Stops automatically at 29 seconds.</p>}
        </div>}

        <details className="poc-options"><summary>Advanced <span>Comparison {compare ? 'on' : 'off'}</span></summary>
          <label className="poc-checkbox"><input type="checkbox" checked={compare} disabled={busy} onChange={e => setCompare(e.target.checked)}/> Compare with original audio</label>
          <p>The same Whisper model transcribes both inputs, so you can see what One Voice changes.</p>
        </details>
        <p className="setup-limit">Up to 30s · 4 MiB total</p>
        {error && <p className="poc-error" role="alert">{error}</p>}
        {notice && <p className="poc-notice" role="status">{notice}</p>}
      </aside>

      <section className={`result-panel ${result ? 'has-result' : ''}`} aria-labelledby="result-title">
        <header className="result-toolbar">
          <h2 id="result-title">Transcript</h2>
          {result ? <div className="result-actions">
            <button type="button" className="poc-text-button" onClick={async () => {try {await navigator.clipboard.writeText(accepted.map(s => s.text.trim()).join('\n')); setNotice('Attributed transcript copied.');} catch {setError('Clipboard unavailable. Download the TXT instead.');}}} disabled={!accepted.length}>Copy text</button>
            <details className="poc-export-menu" onClick={event => {if (event.target.closest('a')) {const menu = event.currentTarget; animateChange(() => {menu.open = false;});}}}>
              <summary>Export <span aria-hidden="true">↓</span></summary>
              <div className="export-options">
                <a href={`${API}/transcriptions/${job.id}/export/txt`} download><b>Text</b><span>Accepted words · .txt</span></a>
                <a href={`${API}/transcriptions/${job.id}/export/srt`} download><b>Subtitles</b><span>With timestamps · .srt</span></a>
                <a href={`${API}/transcriptions/${job.id}/audio/extracted`} download><b>Extracted audio</b><span>One Voice output · .wav</span></a>
                <a href={`${API}/transcriptions/${job.id}/export/json`} download><b>Full report</b><span>All words & evidence · .json</span></a>
                <button type="button" onClick={removeJob}>Delete this result</button>
              </div>
            </details>
          </div> : <span className="result-state">{busy ? 'In progress' : job?.status === 'failed' ? 'Needs attention' : job?.status === 'expired' ? 'Expired' : 'New transcript'}</span>}
        </header>

        {result ? <>
          <div className="result-context">
            <div><strong>{submittedInput?.referenceName || 'Your reference'}</strong></div>
            <div className="result-metrics"><span>{clock(result.duration)} audio</span></div>
          </div>
          {(reference !== submittedInput?.reference || recording !== submittedInput?.recording) && <p className="result-changed" role="status">Inputs changed. Transcribe again to update this result.</p>}
          <div className="result-tabs" role="tablist" aria-label="Result views">
            {[['transcript', 'Transcript'], ['compare', 'Compare'], ['details', 'Details']].map(([name, label], index) =>
              <button key={name} id={`result-tab-${name}`} type="button" role="tab" aria-selected={resultView === name} aria-controls={`result-${name}`} tabIndex={resultView === name ? 0 : -1} onClick={() => setResultView(name)} onKeyDown={event => {
                const names = ['transcript', 'compare', 'details'];
                const next = event.key === 'ArrowRight' ? (index + 1) % 3 : event.key === 'ArrowLeft' ? (index + 2) % 3 : event.key === 'Home' ? 0 : event.key === 'End' ? 2 : null;
                if (next !== null) {event.preventDefault(); setResultView(names[next]); document.getElementById(`result-tab-${names[next]}`).focus();}
              }}>{label}{name === 'transcript' && <span>{result.coverage?.accepted_words || 0}</span>}</button>)}
          </div>
          <div className="result-audio">
            <div className="audio-source-switch" role="group" aria-label="Playback source">
              {[['extracted', 'One Voice output'], ['original', 'Original']].map(([value, label]) =>
                <button key={value} type="button" aria-pressed={track === value} onClick={() => {position.current = audio.current?.currentTime || 0; setTrack(value);}}>{label}</button>)}
            </div>
            <audio ref={audio} controls preload="metadata" src={`${API}/transcriptions/${job.id}/audio/${track}`} aria-label={track === 'extracted' ? 'One Voice extracted audio' : 'Original recording for comparison'} onLoadedMetadata={() => {if (audio.current) audio.current.currentTime = Math.min(position.current, audio.current.duration || 0);}}/>
          </div>

          <div id="result-transcript" role="tabpanel" aria-labelledby="result-tab-transcript" hidden={resultView !== 'transcript'} tabIndex={0} className="result-content">
            <div className="transcript-caption"><span>Click a line to listen</span></div>
            <Transcript segments={accepted} seek={seek} empty={result.outcome === 'no_speech' ? 'No speech was detected in this recording.' : 'No words confidently matched this reference. Try another sample or review the comparison.'}/>
            {uncertain.length > 0 && <details className="poc-review"><summary><span className="review-indicator" aria-hidden="true"/> {uncertain.length} uncertain {uncertain.length === 1 ? 'region' : 'regions'} to review <span aria-hidden="true">+</span></summary>
              <p>Excluded from the transcript and exports.</p><Transcript segments={uncertain} seek={seek}/>
            </details>}

          </div>

          <div id="result-compare" role="tabpanel" aria-labelledby="result-tab-compare" hidden={resultView !== 'compare'} tabIndex={0} className="result-content">

            {result.comparison ? <div className="comparison-columns">
              <section><h4>Original audio</h4><p className="comparison-text">{result.comparison.raw.text || 'No text returned.'}</p></section>
              <section><h4>One Voice output</h4><p className="comparison-text">{result.comparison.one_voice.text || 'No text returned.'}</p></section>
            </div> : <p className="transcript-empty">{result.outcome === 'no_speech' ? 'No speech was detected, so neither input was sent to Whisper.' : 'Comparison was switched off for this run. Enable it in Advanced and transcribe again.'}</p>}
            <p className="comparison-method">Same Whisper model and settings.</p>
          </div>

          <div id="result-details" role="tabpanel" aria-labelledby="result-tab-details" hidden={resultView !== 'details'} tabIndex={0} className="result-content">
            <div className="result-section-intro"><h3>Processing details</h3></div>
            <dl className="model-roles"><div><dt>One Voice</dt><dd>Extracts the voice using your reference.</dd></div><div><dt>ECAPA</dt><dd>Compares voice similarity. Does not change audio.</dd></div><div><dt>Whisper</dt><dd>Transcribes audio. Never receives your voice reference.</dd></div><div><dt>Silero</dt><dd>Finds speech and silence. Does not select a speaker.</dd></div></dl>
            <dl className="run-metadata"><div><dt>Processing time</dt><dd>{result.processing_seconds.toFixed(1)}s</dd></div><div><dt>One Voice checkpoint</dt><dd>Epoch {result.model_manifest.one_voice.epoch} · <code>{result.model_manifest.one_voice.sha256.slice(0, 12)}</code></dd></div><div><dt>Transcriber</dt><dd>small.en · CPU int8</dd></div><div><dt>Peak worker memory</dt><dd>{((result.peak_worker_rss_bytes || 0) / 1024 ** 3).toFixed(2)} GiB</dd></div></dl>
            <details className="poc-evidence accuracy-notes"><summary>Accuracy notes <span aria-hidden="true">+</span></summary><p className="poc-attribution-note">Voice matching can miss speech or accept another speaker. Review the audio before relying on the text. Timestamps are approximate.</p><p className="poc-attribution-note">The Transcript tab adds voice attribution after recognition. One Voice may help with overlap, but can introduce errors. Direct Whisper may be more accurate on already-clean speech.</p></details>
            <details className="poc-evidence"><summary>Voice-match evidence <span aria-hidden="true">+</span></summary>
              <p>Cosine similarity scores, not probabilities. Extracted audio is influenced by the reference, so its score is not independent proof of identity. Acceptance also requires evidence in the original audio.</p>
              <div className="poc-table-scroll" tabIndex={0} role="region" aria-label="Voice-match scores"><table><thead><tr><th>Time</th><th>Original</th><th>Extracted</th><th>Decision</th></tr></thead><tbody>{result.windows.map((w, i) => <tr key={i}><td>{clock(w.start)}–{clock(w.end)}</td><td>{w.original_similarity.toFixed(3)}</td><td>{w.extracted_similarity.toFixed(3)}</td><td>{w.attribution}</td></tr>)}</tbody></table></div>
            </details>
          </div>
          <div className="result-footer"><span>Temporary result · 15-minute storage</span></div>
        </> : busy ? <div className="processing-state">
          <div className="processing-symbol" aria-hidden="true"><span/><span/><span/><span/><span/></div>
          <h3>{sending ? 'Preparing your recording' : job?.status === 'queued' ? 'Waiting for the worker' : ['Check audio', 'Separate voice', 'Match speaker', 'Transcribe'][processingStep]}</h3>
          <p className="sr-only" role="status" aria-live="polite">{sending ? 'Sending audio to the local service…' : job.stage}</p>
          <ol className="processing-stages" aria-label="Processing stages">{['Check audio', 'Separate voice', 'Match speaker', 'Transcribe'].map((label, index) => <li key={label} aria-current={processingStep === index ? 'step' : undefined} className={processingStep > index ? 'is-complete' : ''}><span>{processingStep > index ? '✓' : index + 1}</span>{label}</li>)}</ol>
          <button type="button" onClick={removeJob} disabled={sending}>Cancel job</button>

        </div> : <div className={`result-empty ${job?.status === 'failed' ? 'has-error' : ''}`}>
          <div className="empty-audio" aria-hidden="true"><span/><span/><span/><span/><span/></div>
          <h3>{job?.status === 'failed' ? 'This recording needs another try' : job?.status === 'expired' ? 'This result has expired' : 'No transcript yet'}</h3>
          <p>{job?.status === 'failed' ? job.stage : job?.status === 'expired' ? 'Transcribe your recording again to create a fresh result.' : 'Add audio and a voice reference to begin.'}</p>
          {job && <button className="poc-text-button" type="button" onClick={removeJob}>Dismiss</button>}

        </div>}
      </section>
    </div>
    <div className="workspace-footnote"><details><summary>About this tool</summary><p>One Voice separates your selected speaker; Whisper transcribes the audio. Processing stays on this Mac. Saved voice references stay in this browser. Temporary results expire after 15 minutes. No training happens here.</p></details></div>
  </section>;
}
