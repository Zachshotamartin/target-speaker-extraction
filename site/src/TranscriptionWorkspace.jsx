import React, {useEffect, useRef, useState} from 'react';
import {animateChange, useMotionState} from './motion.js';
import {useDismissibleDetails} from './useDismissibleDetails.js';
import {profiles} from './voiceProfiles.js';
import TranscriptionResults from './TranscriptionResults.jsx';
import AudioCaptureButton from './AudioCaptureButton.jsx';
import {useAudioRecorder} from './useAudioRecorder.js';
import {useAudioUrl} from './useAudioUrl.js';
import './transcription.css';

const API = '/api/poc';
const AUDIO_TYPES = 'audio/*,.wav,.flac,.mp3,.m4a,.webm,.ogg';
const terminal = new Set(['ready', 'failed', 'cancelled', 'expired']);

async function request(path, options) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    let reason;
    try { reason = (await response.json()).detail; } catch { /* Proxy errors are not JSON. */ }
    throw new Error(typeof reason === 'string' ? reason : `Local service returned ${response.status}. Check that the transcription service is running.`);
  }
  return response;
}

export default function TranscriptionWorkspace({active = true}) {
  const [health, setHealth] = useMotionState(null);
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
  const [loadingExample, setLoadingExample] = useMotionState(false);
  const mounted = useRef(true);
  const workspace = useRef(null);
  const toolHelp = useRef(null);
  useDismissibleDetails(toolHelp, '.workspace-footnote');
  const fileDraft = useRef(null);
  const sourceChoice = useRef('files');
  const pendingExample = useRef(false);
  const activeRef = useRef(active);
  activeRef.current = active;
  const latestJob = useRef(null);
  const submitting = useRef(false);
  const referenceUrl = useAudioUrl(reference);
  const recordingUrl = useAudioUrl(recording);
  const busy = sending || (job && !terminal.has(job.status));
  const microphone = useAudioRecorder({active, scope: '#transcribe',
    onStart: () => {setError(''); setNotice('');}, onError: setError,
    onComplete: (target, blob) => {
      if (target === 'reference') {setReference(blob); setReferenceName('Recorded voice reference'); setProfileId('');}
      else {setRecording(blob); setRecordingName('Microphone recording');}
    },
  });
  const selectedDemo = demos.find(item => item.id === Number(demo));
  latestJob.current = job;

  useEffect(() => {
    mounted.current = true;
    Promise.all([request('/health').then(r => r.json()), request('/demos').then(r => r.json())])
      .then(([info, list]) => { if (mounted.current) { setHealth(info); setDemos(list); } })
      .catch(() => { if (mounted.current) setError('The local transcription service is unavailable. Start it with the command in poc/README.md.'); });
    profiles('list').then(list => { if (mounted.current) setSaved(list); }).catch(e => setError(e.message));
    return () => {
      mounted.current = false;
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
    }
  }, [active]);

  function chooseSource(next) {
    if (microphone.isBusy()) return;
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
    if (submitting.current || busy || pendingExample.current || microphone.isBusy()) return;
    submitting.current = true;
    setError(''); setNotice(''); setSending(true);
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


  const result = !sending && job?.status === 'ready' ? job.result : null;

  const processingStep = !job || job.status === 'queued' ? 0 : /Transcribing/.test(job.stage) ? 3 : /Checking the selected/.test(job.stage) ? 2 : /Extracting/.test(job.stage) ? 1 : 0;

  return <section ref={workspace} id="transcribe" className="transcription" aria-labelledby="transcription-title" onPlay={event => {
    if (microphone.isBusy()) {event.target.pause(); return;}
    workspace.current?.querySelectorAll('audio').forEach(player => { if (player !== event.target) player.pause(); });
  }}>
    <header className="transcription-heading">
      <div><h1 id="transcription-title">Speech to text</h1></div>
      <div className="workspace-actions"><div className={`poc-connection ${health?.ready ? 'is-ready' : ''}`}>
        <span className="connection-dot" aria-hidden="true"/>
        <span>{health?.ready ? 'Local processing' : health ? 'Models need setup' : 'Connecting…'}</span>
        {!health?.ready && <button type="button" onClick={async () => {try {
          const [info, list] = await Promise.all([request('/health').then(r => r.json()), request('/demos').then(r => r.json())]);
          setHealth(info); setDemos(list); setError('');
        } catch (e) {setError(e.message);}}}>Retry</button>}
      </div><button className="poc-primary" type="button" onClick={start} disabled={!health?.ready || busy || microphone.busy || !reference || !recording || loadingExample}>
            {busy ? 'Processing…' : 'Transcribe'}<span aria-hidden="true">↗</span>
          </button></div>
    </header>

    <div className="workspace-grid">
      <aside className="setup-panel" aria-labelledby="setup-title">
        <div className="setup-panel-heading"><h2 id="setup-title">Audio setup</h2><span>English</span></div>
        <div className="poc-source-switch" role="group" aria-label="Recording source" data-source={source}>
          {[['files', 'Upload audio'], ['public', 'Use example']].map(([value, label]) =>
            <button key={value} type="button" aria-pressed={source === value} disabled={busy || microphone.busy} onClick={() => chooseSource(value)}>{label}</button>)}
        </div>

        <fieldset className="setup-fields" disabled={busy}>
          <legend className="sr-only">Choose reference voice and recording</legend>
          {source === 'public' && <div className="example-choice">
            <label htmlFor="demo-voice">Example</label>
            <select id="demo-voice" value={demo} onChange={event => {pendingExample.current = true; setDemo(Number(event.target.value));}}>
              {demos.map(item => <option key={item.id} value={item.id}>Conversation {item.conversation} · Voice {item.voice}</option>)}
            </select>
          </div>}
          <section className="setup-group" aria-labelledby="reference-heading">
            <h3 id="reference-heading">Voice reference</h3>
            {source === 'public' ? <p className="field-help">Voice {selectedDemo?.voice} · separate recording</p> : <>
              <p className="field-help">3–10 seconds · one speaker</p>
              {saved.length > 0 && <div className="poc-profile-row">
                <label htmlFor="saved-voice">Saved on this device</label>
                <div><select id="saved-voice" value={profileId} disabled={microphone.busy} onChange={e => {
                  setProfileId(e.target.value); const selected = saved.find(p => p.id === e.target.value);
                  if (selected) {setReference(selected.audio); setReferenceName(selected.name);} else {setReference(null); setReferenceName('');}
                }}><option value="">Choose a saved reference</option>{saved.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}</select>
                  <button className="poc-text-button" type="button" disabled={!profileId || microphone.busy} onClick={deleteProfile}>Delete</button>
                </div>
              </div>}
              <label className="upload-field"><span className="sr-only">Reference audio</span><span className="file-picker"><span aria-hidden="true">↑</span>{reference ? 'Replace reference' : 'Upload reference'}<input aria-label="Reference audio" type="file" accept={AUDIO_TYPES} disabled={microphone.busy} onChange={e => {if (!e.target.files?.[0]) return; setReference(e.target.files[0]); setReferenceName(e.target.files[0].name); setProfileId(''); e.target.value = '';}}/></span></label>
              <AudioCaptureButton recorder={microphone} target="reference" label="voice reference" minimum={3} maximum={10} disabled={busy}/>
              {referenceName && <p className="poc-filename" title={referenceName}>{referenceName}</p>}
            </>}
            {loadingExample ? <div className="audio-loading" role="status">Loading reference…</div> : referenceUrl && <audio controls preload="metadata" src={referenceUrl} aria-label="Reference voice sample"/>}
            {source === 'files' && reference && <details className="poc-profile-save"><summary>Save voice</summary>
              <label>Profile name<input maxLength={60} value={profileName} onChange={e => setProfileName(e.target.value)}/></label>
              <button type="button" disabled={microphone.busy} onClick={saveProfile}>Save reference</button>
              <p>This stores the audio in this browser. It does not train a model or upload it to a cloud service.</p>
            </details>}
          </section>

          <section className="setup-group" aria-labelledby="recording-heading">
            <h3 id="recording-heading">Recording</h3>
            {source === 'public' ? <>
              <p className="field-help">Conversation {selectedDemo?.conversation}</p>
              <label className="sr-only" htmlFor="demo-condition">Recording condition</label>
              <select id="demo-condition" value={condition} onChange={event => {pendingExample.current = true; setCondition(event.target.value);}}>
                <option value="mixture">Two voices overlapping</option><option value="target">Selected voice alone</option>
                <option value="absent">Other voice only (target absent)</option><option value="silence">Silence</option>
              </select>
            </> : <>
              <p className="field-help">Up to 30 seconds</p>
              <label className="upload-field"><span className="sr-only">Recording file</span><span className="file-picker"><span aria-hidden="true">↑</span>{recording ? 'Replace recording' : 'Upload recording'}<input aria-label="Recording file" type="file" accept={AUDIO_TYPES} disabled={microphone.busy} onChange={e => {if (!e.target.files?.[0]) return; setRecording(e.target.files[0]); setRecordingName(e.target.files[0].name); e.target.value = '';}}/></span></label>
              <AudioCaptureButton recorder={microphone} target="recording" label="recording" disabled={busy}/>
              {recordingName && <p className="poc-filename" title={recordingName}>{recordingName}</p>}
            </>}
            {loadingExample ? <div className="audio-loading" role="status">Loading recording…</div> : recordingUrl && <audio controls preload="metadata" src={recordingUrl} aria-label="Input recording"/>}
          </section>
        </fieldset>


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
          <h2 id="result-title">Results</h2>
          <span className="result-state">{result ? 'Ready' : busy ? 'In progress' : job?.status === 'failed' ? 'Needs attention' : job?.status === 'expired' ? 'Expired' : 'New transcript'}</span>
        </header>
        {result ? <TranscriptionResults key={job.id} result={result} jobId={job.id}
          referenceName={submittedInput?.referenceName}
          inputsChanged={reference !== submittedInput?.reference || recording !== submittedInput?.recording}
          active={active} onDelete={removeJob} onNotice={setNotice} onError={setError}/> : busy ? <div className="processing-state">
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
    <div className="workspace-footnote"><details ref={toolHelp}><summary>About this tool</summary><p>One Voice separates your selected speaker; Whisper transcribes the audio. Processing stays on this Mac. Saved voice references stay in this browser. Temporary results expire after 15 minutes. No training happens here.</p></details></div>
  </section>;
}
