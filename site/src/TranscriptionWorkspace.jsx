import React, {useEffect, useRef, useState} from 'react';
import TranscriptionResults from './TranscriptionResults.jsx';
import AudioCaptureButton from './AudioCaptureButton.jsx';
import {useAudioRecorder} from './useAudioRecorder.js';
import {useAudioUrl} from './useAudioUrl.js';
import {profiles} from './voiceProfiles.js';
import {projects, newProject} from './recordingProjects.js';
import {json, request, post, upload, asset} from './workspaceApi.js';
import {wavBytes, transcriptWords, captionsSrt, editedWords, renderEdit} from './audioEdit.js';
import {emptyEdit, correctedWords, naturalPlan} from './naturalEdits.js';
import './transcription.css';
import './recording-workspace.css';

const terminal = new Set(['ready', 'failed', 'cancelled', 'expired']);
const fileTypes = 'audio/*,video/*,.wav,.flac,.mp3,.m4a,.mp4,.mov,.webm,.ogg';
const clock = time => `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, '0')}`;
function Audio({blob, label}) {const url = useAudioUrl(blob); return <audio src={url || undefined} controls preload="metadata" aria-label={label}/>;}
function download(blob, name) {const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 60000);}

export default function TranscriptionWorkspace({active = true}) {
  const [project, setProject] = useState(newProject), [library, setLibrary] = useState([]), [savedProfiles, setSavedProfiles] = useState([]);
  const [health, setHealth] = useState(null), [demos, setDemos] = useState([]), [demo, setDemo] = useState('0');
  const [busy, setBusy] = useState(false), [message, setMessage] = useState(''), [error, setError] = useState(''), [saveStatus, setSaveStatus] = useState('');
  const [voice, setVoice] = useState(0), [excerpt, setExcerpt] = useState({start: 0, end: 4}), [exportBusy, setExportBusy] = useState(false);
  const current = useRef(project), queue = useRef(Promise.resolve()), operation = useRef(null), polling = useRef(null), workspace = useRef(null);
  const mounted = useRef(true), importing = useRef(false);
  current.current = project;
  const job = project.job;
  const processing = busy || Boolean(job && !terminal.has(job.status));
  const recordingUrl = useAudioUrl(project.recording);
  const microphone = useAudioRecorder({active, scope: '#transcribe', onStart: () => setError(''), onError: setError,
    onComplete: (target, blob) => target === 'reference' ? addReference(blob, 'Recorded reference') : loadRecording(blob, 'Microphone recording')});

  function update(patch) {setProject(p => ({...p, ...(typeof patch === 'function' ? patch(p) : patch), updated: Date.now()}));}
  function persist(snapshot) {
    queue.current = queue.current.catch(() => {}).then(() => projects('save', snapshot));
    return queue.current;
  }
  function refreshLibrary() {projects('list').then(items => setLibrary(items.sort((a, b) => b.updated - a.updated))).catch(e => setError(e.message));}
  useEffect(() => {
    mounted.current = true;
    refreshLibrary(); profiles('list').then(setSavedProfiles).catch(e => setError(e.message));
    Promise.all([json('/health'), json('/demos')]).then(([h, d]) => {if (mounted.current) {setHealth(h); setDemos(d);}}).catch(e => {if (mounted.current) setError(e.message);});
    return () => {mounted.current = false; operation.current?.abort(); if (current.current.recording) persist(current.current).catch(() => {});};
  }, []);
  useEffect(() => {
    if (!project.recording) return;
    setSaveStatus('Saving on this device…');
    const timer = setTimeout(() => persist(project).then(() => {if (current.current.id === project.id) setSaveStatus('Saved on this device'); refreshLibrary();}).catch(e => {setSaveStatus('Not saved'); setError(e.message);}), 450);
    return () => clearTimeout(timer);
  }, [project]);
  useEffect(() => {if (!active) workspace.current?.querySelectorAll('audio,video').forEach(p => p.pause());}, [active]);

  async function collect(next, signal) {
    const result = next.result;
    if (result.kind === 'discover') {
      const candidates = [];
      for (const item of result.candidates) candidates.push({...item, blob: await asset(next.id, item.asset, signal)});
      const original = await asset(next.id, 'original.wav', signal);
      update({candidates, original, duration: result.duration, job: null});
      setMessage(candidates.length ? 'Listen to the suggested samples and add the voices you want.' : 'No clear voice sample found. Select a solo passage below or upload a reference.');
    } else if (result.kind === 'extract') {
      const original = await asset(next.id, 'original.wav', signal), tracks = [];
      for (const track of result.tracks) tracks.push({...track, result: track.result || JSON.parse(await (await asset(next.id, track.report_asset, signal)).text()), audio: await asset(next.id, track.asset, signal)});
      update({original, tracks, duration: result.duration, edits: {}, job: null, notice: result.notice}); setVoice(0);
      setMessage('Tracks are ready. Your audio and edits are saved on this device.');
    } else {
      const video = await asset(next.id, result.asset, signal);
      update({videoExport: video, job: null}); download(video, 'onevoice-captioned.mp4'); setMessage('Captioned video is ready.');
    }
  }
  useEffect(() => {
    if (!job?.id || ['failed', 'cancelled', 'expired'].includes(job.status)) return;
    const controller = new AbortController(); polling.current = controller; let timer;
    async function poll() {
      try {
        const next = await json(`/workspace/jobs/${job.id}`, {signal: controller.signal});
        if (controller.signal.aborted) return;
        if (next.status === 'ready') {setMessage('Saving completed files to your project…'); await collect(next, controller.signal); return;}
        if (next.status === 'failed' || next.status === 'cancelled') {update({job: next}); setError(next.error || next.stage || 'Processing stopped. Your recording is saved; you can try again.'); return;}
        setMessage(next.stage || 'Waiting for the model service');
      } catch (e) {
        if (controller.signal.aborted) return;
        if (e.status === 404) {update({job: null}); setError('The temporary server result expired. Your saved input and previous tracks are still available. Run the recording again.'); return;}
        setError(`${e.message} Reconnecting to the saved job…`);
      }
      timer = setTimeout(poll, 2000);
    }
    poll(); return () => {controller.abort(); clearTimeout(timer);};
  }, [job?.id, project.id]);

  async function loadRecording(blob, name) {
    if (!blob || processing || importing.current) return;
    if (blob.size > 128 * 1024 * 1024) {setError('Choose a recording up to 128 MiB and 10 minutes.'); return;}
    importing.current = true; setBusy(true); setError('');
    try {
      if (current.current.recording) await persist(current.current);
      const fresh = {...newProject(), recording: blob, name, video: blob.type.startsWith('video/') || /\.(mp4|mov|mkv|webm)$/i.test(name)};
      await persist(fresh); setProject(fresh); setVoice(0); setMessage('Recording saved. Find voices or add a clean reference.');
    } catch (e) {setError(e.message);} finally {importing.current = false; setBusy(false);}
  }
  function addReference(blob, label) {
    if (!blob || !current.current.recording) return;
    if (current.current.references.length >= 4) {setError('Choose at most four voices per project.'); return;}
    if (blob.size > 4 * 1024 * 1024) {setError('A reference must be smaller than 4 MiB and 3–10 seconds.'); return;}
    update(p => ({references: [...p.references, {id: crypto.randomUUID(), blob, label}]})); setError('');
  }
  async function example() {
    if (processing) return;
    setBusy(true); setError('');
    try {
      const [recording, reference] = await Promise.all(['mixture', 'reference'].map(part => request(`/demos/${demo}/${part}`).then(r => r.blob())));
      if (current.current.recording) await persist(current.current);
      const item = demos.find(d => d.id === Number(demo));
      setProject({...newProject(), name: `Conversation ${item?.conversation || Number(demo) + 1}`, recording,
        references: [{id: crypto.randomUUID(), blob: reference, label: `Voice ${item?.voice || 'A'}`}], video: false}); setVoice(0); setMessage('Example loaded. Extract the selected voice, or find more voices.');
    } catch (e) {setError(e.message);} finally {setBusy(false);}
  }
  async function send(kind, videoOptions) {
    if (processing || operation.current || microphone.isBusy()) return;
    setBusy(true); setError('');
    const controller = new AbortController(); operation.current = controller;
    const snapshot = current.current;
    async function sendFile(key, blob) {
      return upload(blob, snapshot.uploads?.[key], state => {
        update(p => ({uploads: {...p.uploads, [key]: state.id}})); setMessage(`Uploading ${key} · ${Math.round(state.received / state.size * 100)}%`);
      }, controller.signal);
    }
    try {
      const signature = JSON.stringify({kind, refs: snapshot.references.map(r => [r.id, r.label]), clips: videoOptions?.clips, words: videoOptions?.words});
      const pending = snapshot.pending?.signature === signature ? snapshot.pending : {id: crypto.randomUUID(), signature};
      const resumed = snapshot.pending?.id === pending.id ? await json(`/workspace/requests/${pending.id}`, {signal: controller.signal}).catch(e => {if (e.status !== 404) throw e;}) : null;
      if (resumed) {update({job: resumed, pending: null, uploads: {}}); return;}
      update({pending}); await persist({...current.current, pending});
      const body = {kind, request_id: pending.id, recording: await sendFile('recording', snapshot.recording)};
      if (kind === 'extract') {
        body.references = [];
        for (const ref of snapshot.references) body.references.push({upload: await sendFile(ref.id, ref.blob), label: ref.label});
        body.compare = true;
      }
      if (kind === 'video') Object.assign(body, {audio: await sendFile(`edited-${pending.id}`, videoOptions.audio), clips: videoOptions.clips, words: videoOptions.words, captions: true});
      const next = await post('/workspace/jobs', body, controller.signal);
      const saved = {...current.current, uploads: {}, pending: null, job: next, updated: Date.now()};
      setProject(saved); await persist(saved); setMessage(next.stage || 'Queued');
    } catch (e) {if (!controller.signal.aborted) setError(e.message); else setMessage('Upload paused. Run again to resume the saved chunks.');}
    finally {setBusy(false); operation.current = null;}
  }
  async function cancel() {
    operation.current?.abort();
    if (job?.id) {
      try {await json(`/workspace/jobs/${job.id}`, {method: 'DELETE'}); polling.current?.abort(); update({job: null}); setMessage('Processing cancelled. Saved audio and edits are unchanged.');}
      catch (e) {setError(e.message);}
    }
  }
  async function excerptReference() {
    if (!Number.isFinite(excerpt.start + excerpt.end) || excerpt.start < 0 || excerpt.end - excerpt.start < 3 || excerpt.end - excerpt.start > 10) {setError('Select a passage lasting 3–10 seconds.'); return;}
    setBusy(true); setError('');
    try {
      const context = new OfflineAudioContext(1, 1, 16000);
      const buffer = await context.decodeAudioData(await (project.original || project.recording).arrayBuffer());
      if (excerpt.end > buffer.duration) throw new Error('The selected passage ends after the recording.');
      const samples = new Float32Array(Math.round((excerpt.end - excerpt.start) * buffer.sampleRate));
      for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
        const input = buffer.getChannelData(ch); for (let i = 0; i < samples.length; i++) samples[i] += input[Math.floor(excerpt.start * buffer.sampleRate) + i] / buffer.numberOfChannels;
      }
      addReference(new Blob([wavBytes([samples], buffer.sampleRate)], {type: 'audio/wav'}), `Voice at ${clock(excerpt.start)}`);
    } catch (e) {setError(e.message || 'This browser cannot decode the video audio. Use Find voices first, then select a passage.');} finally {setBusy(false);}
  }
  async function openProject(id) {
    if (processing) return;
    try {if (current.current.recording) await persist(current.current); const saved = await projects('get', id); if (!saved) throw new Error('Project is no longer saved here.'); setProject(saved); setVoice(0); setError(''); setMessage('Saved project opened.');}
    catch (e) {setError(e.message);}
  }
  async function exportAll() {
    setExportBusy(true); setError('');
    try {
      const {zipSync, strToU8} = await import('fflate'); const files = {};
      for (const track of project.tracks) {
        const edit = {...emptyEdit, ...project.edits[track.id]}, words = correctedWords(transcriptWords(track.result), edit);
        const clips = naturalPlan(words, edit, track.result.duration), kept = editedWords(words, edit.removed, clips);
        if (!clips.length) continue;
        const context = new OfflineAudioContext(1, 1, 16000), buffer = await context.decodeAudioData(await track.audio.arrayBuffer());
        const name = `${track.id}-${track.label.replace(/[^a-z0-9-]/gi, '-').slice(0, 60)}`;
        files[`${name}.wav`] = new Uint8Array(wavBytes(renderEdit([buffer.getChannelData(0)], buffer.sampleRate, clips), buffer.sampleRate));
        files[`${name}.srt`] = strToU8(captionsSrt(kept)); files[`${name}.txt`] = strToU8(kept.map(w => w.text).join(' '));
        files[`${name}-review.json`] = strToU8(JSON.stringify({edits: edit, original_report: track.result}, null, 2));
      }
      if (!Object.keys(files).length) throw new Error('All audio has been removed. Restore a passage before exporting.');
      download(new Blob([zipSync(files, {level: 0})], {type: 'application/zip'}), 'onevoice-speaker-tracks.zip');
    } catch (e) {setError(e.message);} finally {setExportBusy(false);}
  }
  const selected = project.tracks[voice] || project.tracks[0];
  return <section id="transcribe" ref={workspace} className="recording-workspace" onPlayCapture={e => workspace.current?.querySelectorAll('audio,video').forEach(p => {if (p !== e.target) p.pause();})}>
    <header className="recording-heading"><p className="eyebrow">Recording workspace</p><h1>Choose a voice.<br/>Make it yours.</h1><p>Isolate speakers, review the transcript, and edit audio or video. Start with a recording or try an example.</p></header>
    <div className="project-bar"><label>Project name<input aria-label="Project name" value={project.name} maxLength={100} onChange={e => update({name: e.target.value})}/></label><span role="status">{saveStatus || 'Projects are saved on this device'}</span>
      <details><summary>Saved projects ({library.length})</summary><ul>{library.map(item => <li key={item.id}><button disabled={processing} onClick={() => openProject(item.id)}>{item.name} <small>{item.tracks} tracks</small></button><button disabled={processing} aria-label={`Delete saved project ${item.name}`} onClick={async () => {try {await queue.current; await projects('delete', item.id); if (project.id === item.id) {setProject(newProject()); setSaveStatus('');} refreshLibrary();} catch (e) {setError(e.message);}}}>Delete</button></li>)}</ul>{!library.length && <p>Your saved projects will appear here.</p>}</details>
    </div>
    {error && <p className="workspace-alert" role="alert">{error}</p>}
    {(message || processing) && <div className="workspace-progress" role="status"><span>{message || 'Working…'}</span>{processing && <button onClick={cancel}>{busy ? 'Pause upload / cancel' : 'Cancel processing'}</button>}</div>}
    <div className="recording-setup">
      <section className="recording-step"><h2><span>01</span> Add a recording</h2><p>Audio or video · up to 10 minutes / 128 MiB.</p>
        <label className="workspace-file">Choose audio or video<input type="file" accept={fileTypes} disabled={processing || microphone.isBusy()} onChange={e => {loadRecording(e.target.files?.[0], e.target.files?.[0]?.name); e.target.value = '';}}/></label>
        <AudioCaptureButton target="recording" recorder={microphone} disabled={processing} label="Record with microphone"/>
        {project.recording && <><p className="recording-filename">{project.name}</p>{project.video ? <video src={recordingUrl || undefined} controls preload="metadata" aria-label="Imported video"/> : <Audio blob={project.recording} label="Imported recording"/>}</>}
        <details><summary>Try a public example</summary><div className="example-choice"><label>Example<select value={demo} onChange={e => setDemo(e.target.value)}>{demos.map(d => <option key={d.id} value={d.id}>Conversation {d.conversation} · voice {d.voice}</option>)}</select></label><button disabled={processing || !demos.length} onClick={example}>Load example</button></div></details>
      </section>
      <section className="recording-step"><h2><span>02</span> Choose your voices</h2><p>Find samples in the recording, or provide 3–10 seconds of a speaker on their own.</p>
        <button className="poc-primary" disabled={processing || !project.recording || !health?.workspace} onClick={() => send('discover')}>Find voices in recording</button>
        {!health?.workspace && <p>The recording service is connecting. <button onClick={() => json('/health').then(setHealth).catch(e => setError(e.message))}>Retry connection</button></p>}
        {!!project.candidates.length && <div className="voice-candidates"><p>Suggested samples, not verified identities. If voices overlap, choose a clean solo passage instead.</p>{project.candidates.map(c => <div key={c.id}><strong>{c.label} · {clock(c.start)}</strong><Audio blob={c.blob} label={`${c.label} suggested sample`}/><button disabled={processing || project.references.length >= 4} onClick={() => addReference(c.blob, c.label)}>Use this voice</button></div>)}</div>}
        <details><summary>Select a passage or upload a reference</summary><div className="reference-tools">
          <p>Listen above and choose a clean passage containing only the person you want.</p><div className="cut-boundaries"><label>Start (seconds)<input type="number" step="0.1" min="0" value={excerpt.start} onChange={e => setExcerpt({...excerpt, start: Number(e.target.value)})}/></label><label>End (seconds)<input type="number" step="0.1" min="3" value={excerpt.end} onChange={e => setExcerpt({...excerpt, end: Number(e.target.value)})}/></label><button disabled={processing || !project.recording} onClick={excerptReference}>Use this passage</button></div>
          <label className="workspace-file">Upload voice reference<input type="file" accept="audio/*" disabled={processing || !project.recording} onChange={e => {addReference(e.target.files?.[0], e.target.files?.[0]?.name || 'Voice'); e.target.value = '';}}/></label>
          <AudioCaptureButton target="reference" minimum={3} maximum={10} recorder={microphone} disabled={processing || !project.recording} label="Record voice reference"/>
          {!!savedProfiles.length && <label>Saved voice<select defaultValue="" onChange={e => {const profile = savedProfiles.find(p => p.id === e.target.value); if (profile) addReference(profile.audio, profile.name); e.target.value = '';}}><option value="">Choose a saved voice</option>{savedProfiles.map(p => <option value={p.id} key={p.id}>{p.name}</option>)}</select></label>}
        </div></details>
        <div className="chosen-voices">{project.references.map((ref, index) => <div key={ref.id}><label>Voice {index + 1}<input aria-label={`Voice ${index + 1} name`} value={ref.label} maxLength={60} disabled={processing} onChange={e => update(p => ({references: p.references.map(r => r.id === ref.id ? {...r, label: e.target.value} : r)}))}/></label><Audio blob={ref.blob} label={`Reference ${index + 1}`}/><div className="voice-actions"><button disabled={processing} onClick={() => update(p => ({references: p.references.filter(r => r.id !== ref.id)}))}>Remove</button><button onClick={() => profiles('save', {id: ref.id, name: ref.label, audio: ref.blob}).then(() => profiles('list')).then(setSavedProfiles).then(() => setMessage('Reference saved on this device.')).catch(e => setError(e.message))}>Save voice</button></div></div>)}</div>
        <button className="poc-primary" disabled={processing || !project.recording || !project.references.length || project.references.some(r => !r.label.trim()) || !health?.workspace} onClick={() => send('extract')}>Extract {project.references.length > 1 ? `${project.references.length} voices` : 'selected voice'} &amp; transcribe</button>
        <p className="workspace-note">Three or four simultaneous speakers are experimental. Longer recordings take longer to process; you can return to a saved project while the server works.</p>
      </section>
    </div>
    <section className="recording-results"><div className="recording-result-heading"><h2><span>03</span> Review, edit &amp; export</h2>{!!project.tracks.length && <button disabled={exportBusy} onClick={exportAll}>{exportBusy ? 'Preparing tracks…' : 'Download all tracks (.zip)'}</button>}</div>
      {!selected ? <p>Your isolated tracks and transcript editor will appear here.</p> : <><div className="voice-tabs" role="group" aria-label="Speaker tracks">{project.tracks.map((track, index) => <button key={track.id} aria-pressed={voice === index} onClick={() => setVoice(index)}>{track.label}</button>)}</div><p className="workspace-note">{project.notice}</p>
        <TranscriptionResults key={`${project.id}-${selected.id}-${selected.result.processing_seconds}`} result={selected.result} referenceName={selected.label} jobId={null} active={active}
          audioFiles={{original: project.original, extracted: selected.audio}} savedEdit={project.edits[selected.id]} onEdit={edit => update(p => ({edits: {...p.edits, [selected.id]: edit}}))}
          onRenderVideo={project.video ? options => send('video', options) : undefined} videoBusy={processing}
          onNotice={setMessage} onError={setError} onDelete={() => update(p => ({tracks: p.tracks.filter(t => t.id !== selected.id)}))}/></>}
      {project.videoExport && <button onClick={() => download(project.videoExport, 'onevoice-captioned.mp4')}>Download last captioned video</button>}
    </section>
    <p className="workspace-footnote">Recordings and edits are saved in this browser, not a cloud account. Browser storage can be cleared—download important work. Processing sends the selected files to the service; incomplete uploads expire after an hour and completed server results after 15 minutes. Audio is not used for training. <a href="#privacy">Privacy details</a></p>
  </section>;
}
