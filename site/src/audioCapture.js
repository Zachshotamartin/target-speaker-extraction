const SAMPLE_RATE = 16000;

export function encodeWav(samples, sampleRate = SAMPLE_RATE) {
  const buffer = new ArrayBuffer(44 + samples.length * 2);
  const view = new DataView(buffer);
  const text = (offset, value) => [...value].forEach((character, index) => view.setUint8(offset + index, character.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, buffer.byteLength - 8, true); text(8, 'WAVE');
  text(12, 'fmt '); view.setUint32(16, 16, true); view.setUint16(20, 1, true);
  view.setUint16(22, 1, true); view.setUint32(24, sampleRate, true);
  view.setUint32(28, sampleRate * 2, true); view.setUint16(32, 2, true); view.setUint16(34, 16, true);
  text(36, 'data'); view.setUint32(40, samples.length * 2, true);
  samples.forEach((sample, index) => {
    const value = Math.max(-1, Math.min(1, Number.isFinite(sample) ? sample : 0));
    view.setInt16(44 + index * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  });
  return new Blob([buffer], {type: 'audio/wav'});
}

export async function recordingToWav(blob, {minimum, maximum}) {
  if (!blob.size) throw new Error('No audio was captured. Try recording again.');
  const context = new AudioContext();
  let decoded;
  try { decoded = await context.decodeAudioData(await blob.arrayBuffer()); }
  finally { await context.close(); }
  const duration = Math.min(decoded.duration, maximum);
  if (duration < minimum) throw new Error(`Record at least ${minimum} seconds${minimum === 3 ? ' for the voice reference' : ''}, then try again.`);
  const output = new OfflineAudioContext(1, Math.floor(duration * SAMPLE_RATE), SAMPLE_RATE);
  const source = output.createBufferSource();
  source.buffer = decoded; source.connect(output.destination); source.start();
  const rendered = await output.startRendering();
  return encodeWav(rendered.getChannelData(0));
}

const idle = {phase: 'idle', target: null, elapsed: 0};

// One session owns the microphone. Cancellation also invalidates pending permission
// and conversion promises, so late results cannot replace a newer input.
export class AudioCapture {
  constructor({onState, onComplete, onError, mediaDevices = globalThis.navigator?.mediaDevices,
    Recorder = globalThis.MediaRecorder, encode = recordingToWav, now = () => performance.now(),
    every = setInterval, clear = clearInterval}) {
    Object.assign(this, {onState, onComplete, onError, mediaDevices, Recorder, encode, now, every, clear});
    this.session = null;
  }

  async start(target, limits) {
    if (this.session) return;
    const session = {target, ...limits, chunks: [], phase: 'requesting'};
    this.session = session;
    this.onState({phase: 'requesting', target, elapsed: 0});
    try {
      if (!this.mediaDevices?.getUserMedia || !this.Recorder) throw new Error('Microphone recording is unavailable in this browser. You can upload audio instead.');
      const stream = await this.mediaDevices.getUserMedia({audio: {echoCancellation: false, noiseSuppression: false, autoGainControl: false}});
      if (this.session !== session) { stream.getTracks().forEach(track => track.stop()); return; }
      session.stream = stream;
      const mimeType = ['audio/webm;codecs=opus', 'audio/mp4', 'audio/webm'].find(type => this.Recorder.isTypeSupported(type));
      const recorder = new this.Recorder(stream, mimeType ? {mimeType} : undefined);
      session.recorder = recorder;
      recorder.ondataavailable = event => { if (this.session === session && event.data.size) session.chunks.push(event.data); };
      recorder.onerror = () => this.fail(session, new Error('Recording failed. Check your microphone and try again.'));
      recorder.onstop = () => this.finish(session);
      recorder.start(250);
      session.started = this.now(); session.phase = 'recording';
      this.onState({phase: 'recording', target, elapsed: 0});
      session.timer = this.every(() => {
        if (this.session !== session || session.phase !== 'recording') return;
        const elapsed = (this.now() - session.started) / 1000;
        this.onState({phase: 'recording', target, elapsed: Math.floor(elapsed)});
        if (elapsed >= session.maximum - 0.2) this.stop();
      }, 100);
    } catch (error) { this.fail(session, error); }
  }

  release(session) {
    this.clear(session.timer);
    session.stream?.getTracks().forEach(track => track.stop());
  }

  stop() {
    const session = this.session;
    if (!session || session.phase !== 'recording') return;
    session.phase = 'encoding';
    this.onState({phase: 'encoding', target: session.target, elapsed: 0});
    try { session.recorder.stop(); }
    catch (error) { this.fail(session, error); }
    finally { this.release(session); }
  }

  async finish(session) {
    if (this.session !== session) return;
    this.release(session);
    session.phase = 'encoding';
    this.onState({phase: 'encoding', target: session.target, elapsed: 0});
    try {
      const file = await this.encode(new Blob(session.chunks, {type: session.recorder.mimeType}), session);
      if (this.session !== session) return;
      this.onComplete(session.target, file);
      this.session = null;
      this.onState(idle);
    } catch (error) { this.fail(session, error); }
  }

  cancel() {
    const session = this.session;
    if (!session) return;
    this.session = null;
    this.release(session);
    if (session.recorder?.state !== 'inactive') {
      try { session.recorder?.stop(); } catch { /* Device may already have stopped. */ }
    }
    this.onState(idle);
  }

  fail(session, error) {
    if (this.session !== session) return;
    this.cancel();
    this.onError(error.name === 'NotAllowedError' ? 'Microphone access was denied. Allow it in your browser settings or upload audio.'
      : error.name === 'NotFoundError' ? 'No microphone was found. Connect one or upload audio.'
        : error.message || 'Could not record audio. Please try again.');
  }
}
