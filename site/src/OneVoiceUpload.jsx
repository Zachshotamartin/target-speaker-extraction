import { useEffect, useRef, useState } from 'react';

export default function OneVoiceUpload() {
  const [model, setModel] = useState(null), [notice, setNotice] = useState('Checking upload availability…');
  const [files, setFiles] = useState({}), [busy, setBusy] = useState(false), [output, setOutput] = useState('');
  const request = useRef(null), resultUrl = useRef('');
  useEffect(() => {
    const controller = new AbortController();
    fetch('/api/one-voice/model', {signal: controller.signal}).then(async response => {
      const data = await response.json(); if (!response.ok) throw new Error(data.detail);
      setModel(data); setNotice('');
    }).catch(error => {if (error.name !== 'AbortError') setNotice(error.message || 'Uploads are temporarily unavailable.');});
    return () => {controller.abort(); request.current?.abort(); URL.revokeObjectURL(resultUrl.current);};
  }, []);
  function change(name, file) {setFiles(previous => ({...previous, [name]: file})); URL.revokeObjectURL(resultUrl.current); resultUrl.current=''; setOutput('');}
  async function extract(event) {
    event.preventDefault(); if (!files.mixture || !files.reference || busy) return;
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
  return <section className="ov-upload" aria-labelledby="ov-upload-title">
    <h1 id="ov-upload-title">Isolate a voice</h1>
    <p>Get the extracted audio from your recording. Upload overlapping speech and a separate, clean sample containing only the person you want to keep.</p>
    <form onSubmit={extract}>
      <div className="ov-upload-fields">
        <label>Overlapping speech <small>WAV or FLAC · up to 30 seconds</small><input type="file" accept=".wav,.flac,audio/wav,audio/flac" disabled={busy || !model} onChange={e => change('mixture',e.target.files[0])} required /></label>
        <label>Voice to keep <small>A separate WAV or FLAC · 3–10 seconds</small><input type="file" accept=".wav,.flac,audio/wav,audio/flac" disabled={busy || !model} onChange={e => change('reference',e.target.files[0])} required /></label>
      </div>
      <p className="ov-listening-note">Up to 4 MiB combined. Audio is sent to the extraction service for processing and is not saved or used for training. Only upload recordings you have permission to use.</p>
      <div className="ov-upload-actions">
        <button disabled={!model || !files.mixture || !files.reference || busy} type="submit">{busy ? 'Extracting…' : 'Extract voice'}</button>
        {busy && <button type="button" onClick={() => request.current?.abort()}>Cancel</button>}
      </div>
      <p role="status">{notice}</p>
    </form>
    {output && <div className="ov-upload-result"><h3>Extracted voice</h3><audio controls src={output} /><a href={output} download="extracted-voice.wav">Download extracted voice</a></div>}
    <p className="ov-listening-note">Results vary with speakers, noise, and recording conditions.</p>
  </section>;
}
