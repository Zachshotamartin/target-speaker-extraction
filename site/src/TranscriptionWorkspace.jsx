import React, {useEffect, useRef, useState} from 'react';
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
  const [url, setUrl] = useState(null);
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

export default function TranscriptionWorkspace() {
  const [health, setHealth] = useState(null);
  const [demos, setDemos] = useState([]);
  const [demo, setDemo] = useState(0);
  const [condition, setCondition] = useState('mixture');
  const [source, setSource] = useState('public');
  const [reference, setReference] = useState(null);
  const [recording, setRecording] = useState(null);
  const [referenceName, setReferenceName] = useState('');
  const [recordingName, setRecordingName] = useState('');
  const [saved, setSaved] = useState([]);
  const [profileId, setProfileId] = useState('');
  const [profileName, setProfileName] = useState('My voice');
  const [compare, setCompare] = useState(true);
  const [job, setJob] = useState(null);
  const [submittedInput, setSubmittedInput] = useState(null);
  const [sending, setSending] = useState(false);
  const [error, setError] = useState('');
  const [notice, setNotice] = useState('');
  const [capturing, setCapturing] = useState(false);
  const [requestingMic, setRequestingMic] = useState(false);
  const [elapsed, setElapsed] = useState(0);
  const [track, setTrack] = useState('extracted');
  const [loadingExample, setLoadingExample] = useState(false);
  const mounted = useRef(true);
  const recorder = useRef(null);
  const stream = useRef(null);
  const recordTimer = useRef(null);
  const audio = useRef(null);
  const position = useRef(0);
  const latestJob = useRef(null);
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
        if (controller.signal.aborted) return;
        const selected = demos.find(item => item.id === Number(demo));
        setReference(ref); setRecording(mix);
        setReferenceName(`Conversation ${selected.conversation}, voice ${selected.voice}`);
        setRecordingName({mixture: 'Two voices overlapping', target: 'Selected voice alone', absent: 'Other voice alone', silence: 'Silence'}[condition]);
      }).catch(e => { if (!controller.signal.aborted) setError(e.message); })
      .finally(() => { if (!controller.signal.aborted) setLoadingExample(false); });
    return () => controller.abort();
  }, [demo, condition, source, demos]);

  useEffect(() => {
    if (!job?.id || terminal.has(job.status)) return;
    let cancelled = false, timer;
    const poll = async () => {
      try {
        const next = await request(`/transcriptions/${job.id}`).then(r => r.json());
        if (!cancelled) { setJob(next); setError(''); }
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

  async function start() {
    setError(''); setNotice(''); setSending(true); position.current = 0; setTrack('extracted');
    try {
      if (!reference || !recording) throw new Error('Choose a reference voice and a recording first.');
      if (reference.size + recording.size > 4 * 1024 * 1024 - 2048) throw new Error('The two audio files must total less than 4 MiB.');
      if (job?.id) await request(`/transcriptions/${job.id}`, {method: 'DELETE'}).catch(() => {});
      const data = new FormData();
      data.append('reference', reference, 'reference.audio'); data.append('mixture', recording, 'recording.audio');
      data.append('compare', String(compare));
      const next = await request('/transcriptions', {method: 'POST', body: data}).then(r => r.json());
      setSubmittedInput({reference, recording, referenceName, recordingName}); setJob(next);
    } catch (e) { setError(e.message); } finally { setSending(false); }
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
      if (!mounted.current) { stream.current.getTracks().forEach(t => t.stop()); return; }
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

  const result = job?.status === 'ready' ? job.result : null;
  const accepted = result?.segments.filter(s => s.attribution === 'accepted') || [];
  const uncertain = result?.segments.filter(s => s.attribution === 'uncertain') || [];

  return <section id="transcribe" className="transcription" aria-labelledby="transcription-title">
    <div className="transcription-heading"><div><p className="eyebrow">LOCAL TRANSCRIPTION · PROOF OF CONCEPT</p><h2 id="transcription-title">One voice. In words.</h2></div><p>Choose the voice to keep. One Voice separates the audio, then Whisper writes it down.</p></div>
    <div className="pipeline-strip" aria-label="Pipeline"><span>01 <b>Choose a reference</b></span><span>02 <b>One Voice extracts</b></span><span>03 <b>Whisper transcribes</b></span></div>
    <div className="poc-source-controls">
      <div className="poc-toggle" aria-label="Recording source">{[['public', 'Public examples'], ['files', 'My recordings']].map(([value, label]) =>
        <button key={value} type="button" aria-pressed={source === value} disabled={busy || capturing} onClick={() => {setSource(value); setLoadingExample(false);}}>{label}</button>)}</div>
      <span className="poc-local">{health?.ready ? 'Models ready · processing on this Mac' : health ? 'Local models need setup' : 'Connecting to local service…'}</span>
      {!health?.ready && <button type="button" onClick={async () => {try {const [info, list] = await Promise.all([request('/health').then(r => r.json()), request('/demos').then(r => r.json())]); setHealth(info); setDemos(list); setError('');} catch (e) {setError(e.message);}}}>Retry connection</button>}
    </div>
    <fieldset className="poc-inputs" disabled={busy || capturing}>
      <legend className="sr-only">Choose reference voice and recording</legend>
      {source === 'public' && <div className="poc-demo-controls">
        <label>Voice to select<select value={demo} onChange={event => setDemo(Number(event.target.value))}>{demos.map(item => <option key={item.id} value={item.id}>Conversation {item.conversation} · Voice {item.voice}</option>)}</select></label>
        <label>Try a condition<select value={condition} onChange={event => setCondition(event.target.value)}><option value="mixture">Two voices overlapping</option><option value="target">Selected voice alone</option><option value="absent">Other voice alone — target absent</option><option value="silence">Silence — neither voice</option></select></label>
        <p>Public LibriSpeech examples. No personal recording needed.</p>
      </div>}
      <div className="poc-input-columns">
        <div><h3><span>01</span> Voice to keep</h3><p>A clean 3–10 second sample identifies the speaker. It is used for inference, never training.</p>
          {source === 'files' && <><label>Reference audio<input type="file" accept={AUDIO_TYPES} onChange={e => {setReference(e.target.files?.[0] || null); setReferenceName(e.target.files?.[0]?.name || ''); setProfileId('');}}/></label>
            {saved.length > 0 && <div className="poc-profile-row"><label>Saved on this device<select value={profileId} onChange={e => {
              setProfileId(e.target.value); const selected = saved.find(p => p.id === e.target.value);
              if (selected) {setReference(selected.audio); setReferenceName(selected.name);} else {setReference(null); setReferenceName('');}
            }}><option value="">Choose a saved reference</option>{saved.map(p => <option key={p.id} value={p.id}>{p.name}</option>)}</select></label>
              <button type="button" disabled={!profileId} onClick={deleteProfile}>Delete profile</button></div>}
          </>}
          <p className="poc-filename">{loadingExample ? 'Loading reference…' : referenceName || 'No reference selected'}</p>
          {referenceUrl && <audio controls preload="metadata" src={referenceUrl} aria-label="Reference voice sample"/>}
          {source === 'files' && reference && <details className="poc-profile-save"><summary>Remember this voice on this device</summary><div><label>Profile name<input maxLength={60} value={profileName} onChange={e => setProfileName(e.target.value)}/></label><button type="button" onClick={saveProfile}>Save reference</button></div><p>This stores the audio in this browser. It does not train a model or upload it to a cloud service.</p></details>}
        </div>
        <div><h3><span>02</span> Recording to transcribe</h3><p>English audio, up to 30 seconds. The reference and recording must total less than 4 MiB.</p>
          {source === 'files' && <label>Recording file<input type="file" accept={AUDIO_TYPES} onChange={e => {setRecording(e.target.files?.[0] || null); setRecordingName(e.target.files?.[0]?.name || '');}}/></label>}
          <p className="poc-filename">{loadingExample ? 'Loading recording…' : recordingName || 'No recording selected'}</p>
          {recordingUrl && <audio controls preload="metadata" src={recordingUrl} aria-label="Input recording"/>}
        </div>
      </div>
    </fieldset>
    {source === 'files' && <div className="poc-microphone"><button type="button" onClick={capture} disabled={busy || requestingMic}>{requestingMic ? 'Opening microphone…' : capturing ? `Stop recording · ${clock(elapsed)}` : 'Record from microphone'}</button><span>{capturing ? 'Recording… stops automatically after 29 seconds.' : 'Optional. Recording starts only when you press the button.'}</span></div>}
    <div className="poc-actions"><div><label className="poc-checkbox"><input type="checkbox" checked={compare} disabled={busy} onChange={e => setCompare(e.target.checked)}/> Compare against Whisper on the original audio</label><p>Same recognizer, same settings. Neither the reference nor a speaker prompt is sent to Whisper.</p></div>
      <button className="poc-primary" type="button" onClick={start} disabled={!health?.ready || busy || capturing || !reference || !recording || loadingExample}>{sending ? 'Submitting…' : 'Transcribe selected voice ↗'}</button>
    </div>
    {error && <p className="poc-error" role="alert">{error}</p>}
    {notice && <p className="poc-notice" role="status">{notice}</p>}
    {job && <div className={`poc-job poc-job--${job.status}`}><div role="status" aria-live="polite"><span className="poc-status-dot"/><b>{job.status === 'expired' ? 'Result expired. Run the recording again.' : job.stage}</b>{busy && <span> One CPU worker · training continues separately</span>}</div><button type="button" onClick={removeJob}>{busy ? 'Cancel job' : 'Delete result'}</button></div>}
    {result && <div className="poc-result">
      <div className="poc-result-heading"><h3>Selected voice transcript</h3><p>{result.processing_seconds.toFixed(1)}s processing · {clock(result.duration)} audio · One Voice epoch {result.model_manifest.one_voice.epoch}</p></div>
      <p className="poc-result-reference">Reference used: {submittedInput?.referenceName || 'Selected reference'} · {submittedInput?.recordingName || 'Uploaded recording'}</p>
      {(reference !== submittedInput?.reference || recording !== submittedInput?.recording) && <p className="poc-notice">Inputs have changed. This result belongs to the reference and recording named above; transcribe again to use your new selection.</p>}
      <div className="poc-result-audio"><div className="poc-toggle" aria-label="Playback source">{[['extracted', 'One Voice output'], ['original', 'Original recording']].map(([value, label]) =>
        <button key={value} type="button" aria-pressed={track === value} onClick={() => { position.current = audio.current?.currentTime || 0; setTrack(value); }}>{label}</button>)}</div>
        <audio ref={audio} controls preload="metadata" src={`${API}/transcriptions/${job.id}/audio/${track}`} aria-label={track === 'extracted' ? 'One Voice extracted audio' : 'Original recording for comparison'} onLoadedMetadata={() => { if (audio.current) audio.current.currentTime = Math.min(position.current, audio.current.duration || 0); }}/>
      </div>
      <Transcript segments={accepted} seek={seek} empty={result.outcome === 'no_speech' ? 'No speech was detected in this recording.' : 'No words confidently matched this reference. Review the audio and uncertain regions, or try a cleaner reference.'}/>
      <p className="poc-attribution-note">Voice attribution is an experimental similarity check. It can miss speech or accept another speaker; review the result before relying on it. Word timestamps are approximate.</p>
      {uncertain.length > 0 && <details className="poc-review"><summary>Review {uncertain.length} uncertain {uncertain.length === 1 ? 'region' : 'regions'} · excluded from TXT/SRT</summary><Transcript segments={uncertain} seek={seek}/></details>}
      <div className="poc-exports"><button type="button" onClick={async () => {try {await navigator.clipboard.writeText(accepted.map(s => s.text.trim()).join('\n')); setNotice('Attributed transcript copied.');} catch {setError('Clipboard unavailable. Download the TXT instead.');}}} disabled={!accepted.length}>Copy text</button>
        {['txt', 'srt', 'json'].map(kind => <a key={kind} href={`${API}/transcriptions/${job.id}/export/${kind}`} download>{kind === 'json' ? 'JSON + evidence' : kind.toUpperCase()} ↓</a>)}
        <a href={`${API}/transcriptions/${job.id}/audio/extracted`} download>Extracted WAV ↓</a><span>Results expire 15 minutes after completion.</span>
      </div>
      {result.comparison && <div className="poc-comparison"><h3>What did One Voice add?</h3><p>These transcripts use identical Whisper settings. The second input is the audio separated by One Voice. The voice filter above is a separate step.</p>
        <div className="poc-comparison-columns"><section><h4>Original → Whisper</h4><p>{result.comparison.raw.text || 'No text returned.'}</p></section><section><h4>One Voice → Whisper</h4><p>{result.comparison.one_voice.text || 'No text returned.'}</p></section></div>
        <p className="poc-attribution-note">Whisper can handle some overlapping audio on its own. One Voice may help, make no difference, or introduce errors. Listen to both and compare the words you wanted. For already-clean speech, direct Whisper may be more accurate.</p>
      </div>}
      <details className="poc-evidence"><summary>Model roles and attribution evidence</summary><p>Silero marks speech. One Voice alone separates waveforms. ECAPA compares voices; it never edits audio. faster-whisper transcribes the supplied audio without a voice reference, denoising or diarization stage.</p>
        <p>Extracted audio is influenced by the reference, so its match score is not independent proof of identity. Acceptance also requires evidence from the original audio. Scores are cosine similarities, not probabilities.</p>
        <p>Checkpoint: <code>{result.model_manifest.one_voice.sha256.slice(0, 16)}</code> · ASR: small.en, CPU int8 · {((result.peak_worker_rss_bytes || 0) / 1024 ** 3).toFixed(2)} GiB peak worker memory.</p>
        <div className="poc-table-scroll"><table><thead><tr><th>Time</th><th>Original match</th><th>Extracted match</th><th>Decision</th></tr></thead><tbody>{result.windows.map((w, i) => <tr key={i}><td>{clock(w.start)}–{clock(w.end)}</td><td>{w.original_similarity.toFixed(3)}</td><td>{w.extracted_similarity.toFixed(3)}</td><td>{w.attribution}</td></tr>)}</tbody></table></div>
      </details>
    </div>}
    <p className="poc-footer-note">Local files stay on this Mac. Only explicitly saved references persist in this browser; temporary jobs are deleted after 15 minutes or when you delete the result. No training happens in this workspace.</p>
  </section>;
}
