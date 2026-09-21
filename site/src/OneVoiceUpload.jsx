import { useEffect, useRef } from 'react';
import {useMotionState} from './motion.js';
import {useAudioRecorder} from './useAudioRecorder.js';
import {useAudioUrl} from './useAudioUrl.js';
import AudioCaptureButton from './AudioCaptureButton.jsx';
const useUploadState = initial => useMotionState(initial, '.ov-upload');

export default function OneVoiceUpload({active = true}) {
  const [model, setModel] = useUploadState(null), [notice, setNotice] = useUploadState('Checking upload availability…');
  const [files, setFiles] = useUploadState({}), [busy, setBusy] = useUploadState(false), [output, setOutput] = useUploadState('');
  const request = useRef(null), resultUrl = useRef('');
  const mixtureUrl = useAudioUrl(files.mixture, '.ov-upload');
  const referenceUrl = useAudioUrl(files.reference, '.ov-upload');
  const microphone = useAudioRecorder({active, scope: '.ov-upload',
    onStart: () => setNotice(''), onError: setNotice,
    onComplete: (target, blob) => change(target, new File([blob], `${target}-recording.wav`, {type: 'audio/wav'})),
  });
  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/one-voice/model', {signal: controller.signal}).then(async response => {
      const data = await response.json(); if (!response.ok) throw new Error(data.detail);
      setModel(data); setNotice('');
    }).catch(error => {if (error.name !== 'AbortError') setNotice(error.message || 'Uploads are temporarily unavailable.');});
    return () => {controller.abort(); request.current?.abort(); URL.revokeObjectURL(resultUrl.current);};
  }, []);
  function change(name, file) {if (!file) return; setFiles(previous => ({...previous, [name]: file})); URL.revokeObjectURL(resultUrl.current); resultUrl.current=''; setOutput('');}
  async function extract(event) {
    event.preventDefault(); if (!files.mixture || !files.reference || busy || microphone.isBusy()) return;
    if (files.mixture.size + files.reference.size > 4 * 1024 * 1024 - 4096) {setNotice('Choose recordings with a combined size under 4 MiB.'); return;}
    setBusy(true); setNotice('Extracting the selected voice. This may take a minute.');
    URL.revokeObjectURL(resultUrl.current); setOutput('');
    const controller = new AbortController(); request.current = controller;
    try {
      const form = new FormData(); form.append('mixture', files.mixture); form.append('reference', files.reference);
      const response = await fetch('/api/one-voice/extract', {method: 'POST', body: form, signal: controller.signal});
      if (!response.ok) {const data = await response.json(); throw new Error(data.detail || 'Extraction failed.');}
      if (!response.headers.get('content-type')?.startsWith('audio/wav')) throw new Error('The service did not return audio.');
      const url = URL.createObjectURL(await response.blob()); resultUrl.current=url; setOutput(url); setNotice('Ready. Compare the result with your original recording.');
    } catch (error) {setNotice(error.name === 'AbortError' ? 'Stopped waiting for the result.' : error.message);}
    finally {setBusy(false); request.current=null;}
  }
  return <section className="ov-upload" aria-labelledby="ov-upload-title" onPlay={event => {
    if (microphone.isBusy()) {event.target.pause(); return;}
    event.currentTarget.querySelectorAll('audio').forEach(player => {if (player !== event.target) player.pause();});
  }}>
    <h2 id="ov-upload-title">Try your own recordings</h2>
    <p>Upload or record overlapping speech and a separate, clean sample of the person you want to keep. The sample should contain only that person speaking.</p>
    <form onSubmit={extract}>
      <div className="ov-upload-fields">
        {[
          {name: 'mixture', title: 'Overlapping speech', help: 'WAV or FLAC · up to 30 seconds', url: mixtureUrl, minimum: 0.5, maximum: 30},
          {name: 'reference', title: 'Voice to keep', help: 'A separate sample · 3–10 seconds · WAV or FLAC', url: referenceUrl, minimum: 3, maximum: 10},
        ].map(field => <section className="ov-capture-field" key={field.name} aria-labelledby={`ov-input-${field.name}`}>
          <h3 id={`ov-input-${field.name}`}>{field.title}</h3><p>{field.help}</p>
          <label className="upload-field"><span className="file-picker"><span aria-hidden="true">↑</span>{files[field.name] ? 'Replace audio' : 'Upload audio'}<input type="file" aria-label={field.title} accept=".wav,.flac,audio/wav,audio/flac" disabled={busy || microphone.busy} onChange={event => {change(field.name, event.target.files?.[0]); event.target.value = '';}}/></span></label>
          <AudioCaptureButton recorder={microphone} target={field.name} label={field.title.toLowerCase()} minimum={field.minimum} maximum={field.maximum} disabled={busy}/>
          {files[field.name] && <p className="poc-filename" title={files[field.name].name}>{files[field.name].name}</p>}
          {field.url && <audio className="ov-capture-preview" controls src={field.url} aria-label={`${field.title} preview`}/>}
        </section>)}
      </div>
      <p className="ov-listening-note">Up to 4 MiB combined. Audio is sent to the extraction service for processing and is not saved or used for training. Only upload recordings you have permission to use. <a href="#privacy">Privacy policy</a>.</p>
      <div className="ov-upload-actions">
        <button disabled={!model || !files.mixture || !files.reference || busy || microphone.busy} type="submit">{busy ? 'Extracting…' : 'Extract voice'}</button>
        {busy && <button type="button" onClick={() => request.current?.abort()}>Cancel</button>}
      </div>
      <p role="status">{notice}</p>
    </form>
    {output && <div className="ov-upload-result"><h3>Extracted voice</h3><audio controls src={output} /><a href={output} download="extracted-voice.wav">Download extracted voice</a></div>}
    <p className="ov-listening-note">Results vary with speakers, noise, and recording conditions.</p>
  </section>;
}
