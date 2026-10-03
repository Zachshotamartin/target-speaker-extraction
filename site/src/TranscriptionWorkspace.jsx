import ProcessingStatus from './workspace/ProcessingStatus.jsx';
import React, {useEffect, useRef, useState} from 'react';
import {useAudioRecorder} from './useAudioRecorder.js';
import {animateChange, animateStepChange, useMotionState} from './motion.js';
import Section from './ui/Section.jsx';
import Heading from './ui/Heading.jsx';
import Text from './ui/Text.jsx';
import Stack from './ui/Stack.jsx';
import Inline from './ui/Inline.jsx';
import Button from './ui/Button.jsx';
import Notice from './ui/Notice.jsx';
import WorkspacePrivacy from './workspace/WorkspacePrivacy.jsx';
import WorkspaceFeedback from './workspace/WorkspaceFeedback.jsx';
import StepNavigation from './ui/StepNavigation.jsx';
import ProjectLibrary from './workspace/ProjectLibrary.jsx';
import AudioStep from './workspace/AudioStep.jsx';
import VoiceStep from './workspace/VoiceStep.jsx';
import ResultsStep from './workspace/ResultsStep.jsx';
import {profiles} from './voiceProfiles.js';
import {projects, newProject, projectWriter} from './recordingProjects.js';
import {json, request, post, upload, asset} from './workspaceApi.js';
import {wavBytes, transcriptWords, captionsSrt, editedWords, renderEdit} from './audioEdit.js';
import {emptyEdit, correctedWords, naturalPlan} from './naturalEdits.js';
import './transcription.css';
import './recording-workspace.css';

const terminal = new Set(['ready', 'failed', 'cancelled', 'expired']);
const clock = time => `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, '0')}`;
function download(blob, name) {const url = URL.createObjectURL(blob), link = document.createElement('a'); link.href = url; link.download = name; link.click(); setTimeout(() => URL.revokeObjectURL(url), 60000);}

export default function TranscriptionWorkspace({active = true}) {
  const [project, setProject] = useState(newProject), [library, setLibrary] = useState([]), [savedProfiles, setSavedProfiles] = useState([]);
  const [health, setHealth] = useState(null), [demos, setDemos] = useState([]), [demo, setDemo] = useState('0');
  const [loadingDemos, setLoadingDemos] = useState(true);
  const [step, setStep] = useState('audio'), [source, setSource] = useMotionState('example');
  const [busy, setBusy] = useState(false), [message, setMessage] = useState(''), [error, setError] = useState(''), [saveStatus, setSaveStatus] = useState('');
  const [voice, setVoice] = useState(0), [excerpt, setExcerpt] = useState({start: 0, end: 4}), [exportBusy, setExportBusy] = useState(false);
  const current = useRef(project), storage = useRef(null), operation = useRef(null), polling = useRef(null), workspace = useRef(null);
  if (!storage.current) storage.current = projectWriter();
  const mounted = useRef(true), importing = useRef(false);
  current.current = project;
  const job = project.job;
  const processing = busy || Boolean(job && !terminal.has(job.status));
  const microphone = useAudioRecorder({active: active && step !== 'results', scope: '#transcribe', onStart: () => setError(''), onError: setError,
    onComplete: (target, blob) => target === 'reference' ? addReference(blob, 'Recorded reference') : loadRecording(blob, 'Microphone recording')});

  function goToStep(next) {
    if (microphone.isBusy() || step === next) return;
    workspace.current?.querySelectorAll('audio,video').forEach(player => player.pause());
    animateStepChange(() => setStep(next), () => {
      workspace.current?.querySelector(`[data-step="${next}"]`)?.focus({preventScroll: true});
      // Keep the step navigator visible when returning from a long transcript.
      const nav = workspace.current?.querySelector('.ui-steps');
      if (active && nav?.getBoundingClientRect().top < 0) nav.scrollIntoView({block: 'start', behavior: 'auto'});
    });
  }
  function replaceProject(next, nextStep) {
    animateStepChange(() => {setProject(next); setVoice(0); setStep(nextStep);}, () => {
      workspace.current?.querySelector(`[data-step="${nextStep}"]`)?.focus({preventScroll: true});
    });
  }
  async function loadDemos() {
    setLoadingDemos(true);
    try {const items = await json('/demos'); if (mounted.current) setDemos(items);}
    catch (e) {if (mounted.current) setError(e.message);}
    finally {if (mounted.current) setLoadingDemos(false);}
  }
  function refreshHealth() {json('/health').then(setHealth).catch(e => setError(e.message));}
  function update(patch) {setProject(p => ({...p, ...(typeof patch === 'function' ? patch(p) : patch), updated: Date.now()}));}
  function persist(snapshot) {
    return storage.current.save(snapshot);
  }
  function refreshLibrary() {projects('list').then(items => setLibrary(items.sort((a, b) => b.updated - a.updated))).catch(e => setError(e.message));}
  useEffect(() => {
    mounted.current = true;
    refreshLibrary(); profiles('list').then(setSavedProfiles).catch(e => setError(e.message));
    loadDemos(); refreshHealth();
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
      animateStepChange(() => {update({candidates, original, duration: result.duration, job: null}); setStep('voice');});
      setMessage(candidates.length ? 'Listen to the suggested samples and add the voices you want.' : 'No clear voice sample found. Select a solo passage below or upload a reference.');
    } else if (result.kind === 'extract') {
      const original = await asset(next.id, 'original.wav', signal), tracks = [];
      for (const track of result.tracks) tracks.push({...track, result: track.result || JSON.parse(await (await asset(next.id, track.report_asset, signal)).text()), audio: await asset(next.id, track.asset, signal)});
      animateStepChange(() => {update({original, tracks, duration: result.duration, edits: {}, job: null, notice: result.notice}); setVoice(0); setStep('results');});
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
    importing.current = true; setBusy(true); setError(''); setMessage('Saving your recording…');
    try {
      if (current.current.recording) await persist(current.current);
      const fresh = {...newProject(), recording: blob, name, video: blob.type.startsWith('video/') || (!blob.type.startsWith('audio/') && /\.(mp4|mov|mkv|webm)$/i.test(name))};
      await persist(fresh); replaceProject(fresh, 'voice'); setMessage('Recording saved. Choose the person to keep.');
    } catch (e) {setError(e.message);} finally {importing.current = false; setBusy(false);}
  }
  function addReference(blob, label) {
    if (!blob || !current.current.recording) return;
    if (current.current.references.length >= 4) {setError('Choose at most four voices per project.'); return;}
    if (blob.size > 4 * 1024 * 1024) {setError('A reference must be smaller than 4 MiB and 3–10 seconds.'); return;}
    animateChange(() => update(p => ({references: [...p.references, {id: crypto.randomUUID(), blob, label: label.slice(0, 60)}]})), '#transcribe'); setError('');
  }
  async function example() {
    if (processing || microphone.isBusy()) return;
    setBusy(true); setError(''); setMessage('Loading example audio…');
    try {
      const [recording, reference] = await Promise.all(['mixture', 'reference'].map(part => request(`/demos/${demo}/${part}`).then(r => r.blob())));
      if (current.current.recording) await persist(current.current);
      const item = demos.find(d => d.id === Number(demo));
      replaceProject({...newProject(), name: `Conversation ${item?.conversation || Number(demo) + 1}`, recording,
        references: [{id: crypto.randomUUID(), blob: reference, label: `Voice ${item?.voice || 'A'}`}], video: false}, 'voice'); setMessage('Example loaded with a matching voice reference.');
    } catch (e) {setError(e.message);} finally {setBusy(false);}
  }
  async function send(kind, videoOptions) {
    if (processing || operation.current || microphone.isBusy()) return;
    setBusy(true); setError(''); setMessage('Preparing your recording…');
    if (kind === 'extract' || kind === 'video') goToStep('results');
    const controller = new AbortController(); operation.current = controller;
    const snapshot = current.current;
    async function sendFile(key, blob) {
      return upload(blob, snapshot.uploads?.[key], state => {
        update(p => ({uploads: {...p.uploads, [key]: state.id}})); setMessage(`Uploading ${key} · ${Math.round(state.received / state.size * 100)}%`);
      }, controller.signal);
    }
    try {
      const signature = JSON.stringify({kind, refs: snapshot.references.map(r => [r.id, r.label]), track: videoOptions?.trackId, clips: videoOptions?.clips, words: videoOptions?.words});
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
    setBusy(true); setError(''); setMessage('Preparing the voice sample…');
    try {
      const context = new OfflineAudioContext(1, 1, 16000);
      const buffer = await context.decodeAudioData(await (project.original || project.recording).arrayBuffer());
      if (excerpt.end > buffer.duration) throw new Error('The selected passage ends after the recording.');
      const samples = new Float32Array(Math.round((excerpt.end - excerpt.start) * buffer.sampleRate));
      for (let ch = 0; ch < buffer.numberOfChannels; ch++) {
        const input = buffer.getChannelData(ch); for (let i = 0; i < samples.length; i++) samples[i] += input[Math.floor(excerpt.start * buffer.sampleRate) + i] / buffer.numberOfChannels;
      }
      addReference(new Blob([wavBytes([samples], buffer.sampleRate)], {type: 'audio/wav'}), `Voice at ${clock(excerpt.start)}`); setMessage('Voice sample added.');
    } catch (e) {setError(e.message || 'This browser cannot decode the video audio. Use Find voices first, then select a passage.');} finally {setBusy(false);}
  }
  async function openProject(id) {
    if (processing || importing.current || microphone.isBusy()) return;
    importing.current = true; setBusy(true); setMessage('Opening saved project…');
    try {if (current.current.recording) await persist(current.current); const saved = await projects('get', id); if (!saved) throw new Error('Project is no longer saved here.'); replaceProject(saved, saved.tracks.length || (saved.job && saved.job.kind !== 'discover' && !terminal.has(saved.job.status)) ? 'results' : 'voice'); setError(''); setMessage('Saved project opened.');}
    catch (e) {setError(e.message);} finally {importing.current = false; setBusy(false);}
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
  async function deleteProject(id) {
    try {
      await storage.current.delete(id);
      if (project.id === id) {replaceProject(newProject(), 'audio'); setSaveStatus(''); setMessage('');}
      refreshLibrary();
    } catch (e) {setError(e.message);}
  }
  const blocked = processing || microphone.isBusy();
  const steps = [
    {id: 'audio', label: 'Audio', disabled: microphone.isBusy()},
    {id: 'voice', label: 'Voice', disabled: !project.recording || microphone.isBusy()},
    {id: 'results', label: 'Results', disabled: (!project.tracks.length && step !== 'results') || microphone.isBusy()},
  ];
  return <Section id="transcribe" ref={workspace} spacing="workspace" className="recording-workspace" onPlayCapture={e => workspace.current?.querySelectorAll('audio,video').forEach(p => {if (p !== e.target) p.pause();})}>
    <Stack gap="large">
      <header className="recording-heading">
        <Heading as="h1" size="title">Speech to text</Heading>
        <StepNavigation steps={steps} current={step} onChange={goToStep}/>
        <ProjectLibrary project={project} library={library} saveStatus={saveStatus} disabled={blocked}
          onRename={name => update({name})} onOpen={openProject} onDelete={deleteProject}/>
      </header>
      {error && <Notice error><Inline className="workspace-status"><span>{error}</span><Button variant="quiet" onClick={() => setError('')}>Dismiss</Button></Inline></Notice>}
      {processing && <ProcessingStatus message={message} active={active} uploading={Boolean(operation.current)} canCancel={Boolean(operation.current || job)} onCancel={cancel}/>}
      {!processing && <WorkspaceFeedback message={message}/>}
      <div className="workspace-step-content" data-step={step} tabIndex={-1}>
        {step === 'audio' && <AudioStep project={project} mode={source} setMode={setSource} demos={demos} demo={demo} setDemo={setDemo}
          loadingDemos={loadingDemos} onRetry={loadDemos} disabled={processing} microphone={microphone} onExample={example}
          onUpload={loadRecording} onContinue={() => goToStep('voice')}/>}
        {step === 'voice' && <VoiceStep project={project} disabled={processing} serviceReady={health?.workspace} microphone={microphone}
          savedProfiles={savedProfiles} excerpt={excerpt} setExcerpt={setExcerpt} onDiscover={() => send('discover')} onReference={addReference}
          onExcerpt={excerptReference} onRetry={refreshHealth} onBack={() => goToStep('audio')} onExtract={() => send('extract')}
          onRename={(id, label) => update(p => ({references: p.references.map(r => r.id === id ? {...r, label} : r)}))}
          onRemove={id => animateChange(() => update(p => ({references: p.references.filter(r => r.id !== id)})), '#transcribe')}
          onSave={ref => profiles('save', {id: ref.id, name: ref.label, audio: ref.blob}).then(() => profiles('list')).then(setSavedProfiles).then(() => setMessage('Reference saved on this device.')).catch(e => setError(e.message))}/>}
        {step === 'results' && <ResultsStep project={project} voice={voice} setVoice={setVoice} processing={processing} exportBusy={exportBusy} onExport={exportAll}
          active={active} onBack={() => goToStep('voice')} onVideo={options => send('video', options)}
          onVideoDownload={() => download(project.videoExport, 'onevoice-captioned.mp4')}
          onEdit={(id, edit) => update(p => ({edits: {...p.edits, [id]: edit}}))}
          onDelete={id => update(p => ({tracks: p.tracks.filter(t => t.id !== id)}))} onNotice={setMessage} onError={setError}/>}
      </div>
      <div className="workspace-utilities"><Text size="small">Audio is never used for training.</Text><WorkspacePrivacy/></div>
    </Stack>
  </Section>;
}
