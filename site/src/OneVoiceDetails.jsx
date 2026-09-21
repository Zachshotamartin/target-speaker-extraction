import { useEffect, useRef, useState } from 'react';
import snapshot from './snapshot.json';
import './one-voice.css';
import OneVoiceUpload from './OneVoiceUpload.jsx';

const labels = { mixture: 'Both voices', estimate: 'Extracted voice', target: 'Clean target' };
const clock = value => `${Math.floor(value / 60)}:${String(Math.floor(value % 60)).padStart(2, '0')}`;

function Waveform({ peaks, progress = 0 }) {
  const maximum = Math.max(...peaks, .001);
  return <svg viewBox="0 0 768 120" preserveAspectRatio="none" aria-hidden="true">
    {peaks.map((peak, i) => <line key={i} x1={i * 4 + 2} x2={i * 4 + 2} y1={60 - 52 * peak / maximum} y2={60 + 52 * peak / maximum} />)}
    {progress > 0 && <path className="ov-playhead" d={`M ${Math.min(1, progress) * 768} 0 v120`} />}
  </svg>;
}

export default function OneVoiceDetails() {
  const [index, setIndex] = useState(0);
  const [track, setTrack] = useState('mixture');
  const [playing, setPlaying] = useState(false);
  const [referencePlaying, setReferencePlaying] = useState(false);
  const [time, setTime] = useState(0);
  const [error, setError] = useState('');
  const player = useRef(null), reference = useRef(null);
  const pending = useRef({ time: 0, play: false });
  const item = snapshot.items[index], active = item.tracks[track];
  const duration = active.duration;
  useEffect(() => {
    const a = player.current, b = reference.current;
    return () => { a?.pause(); b?.pause(); };
  }, []);
  function play(audio) {
    setError('');
    audio.play().catch(() => setError('Audio could not play. Press Play to try again.'));
  }
  function chooseTrack(next) {
    if (next === track) return;
    pending.current = { time: player.current.currentTime || 0, play: !player.current.paused };
    player.current.pause();
    reference.current.pause();
    setTrack(next); setError('');
  }
  function chooseExample(next) {
    if (next === index) return;
    player.current.pause(); reference.current.pause();
    pending.current = { time: 0, play: false };
    setTime(0); setIndex(next); setError('');
  }
  function loaded() {
    const action = pending.current;
    player.current.currentTime = Math.min(action.time, Math.max(0, duration - .01));
    setTime(player.current.currentTime);
    pending.current = { time: 0, play: false };
    if (action.play) play(player.current);
  }
  return <>
    <section className="study-live one-voice" aria-labelledby="one-voice-listen" data-checkpoint={snapshot.checkpointSHA256}>
      <div className="study-section-heading"><h2 id="one-voice-listen">One conversation. Either voice.</h2><p>Six conversations · two voices each</p></div>
      <p>Listen to the overlapping voices, choose who to keep, then switch to the extraction.</p>
      <div className="ov-controls">
        <fieldset><legend>Conversation</legend><div className="ov-switch">
          {[...new Set(snapshot.items.map(example => example.conversation))].map(n => <button key={n} aria-pressed={item.conversation === n} onClick={() => chooseExample((n - 1) * 2 + index % 2)}>Conversation {n}</button>)}
        </div></fieldset>
        <fieldset><legend>Voice to keep</legend><div className="ov-switch">
          {['A', 'B'].map((voice, n) => <button key={voice} aria-pressed={item.voice === voice} onClick={() => chooseExample(Math.floor(index / 2) * 2 + n)}>Voice {voice}</button>)}
        </div></fieldset>
      </div>
      <div className="ov-reference">
        <button aria-pressed={referencePlaying} onClick={() => { player.current.pause(); if (reference.current.paused) play(reference.current); else reference.current.pause(); }}>
          {referencePlaying ? 'Pause' : 'Hear'} voice {item.voice}’s sample
        </button><p>A separate recording identifies the speaker. It is not the clean answer.</p>
      </div>
      <div className="ov-waveforms">
        {['mixture', 'estimate'].map(name => <button key={name} className={`ov-wave ov-wave--${name}`} aria-pressed={track === name} onClick={() => chooseTrack(name)}>
          <span><strong>{labels[name]}</strong><small>{name === 'mixture' ? 'Original overlapping speech' : `Model output · voice ${item.voice}`}</small></span>
          <Waveform peaks={item.tracks[name].peaks} progress={track === name ? time / duration : 0} />
          <span className="ov-wave-choice">{track === name ? 'Selected' : 'Listen to this track'} <span aria-hidden="true">↗</span></span>
        </button>)}
      </div>
      <div className="ov-player">
        <button className="ov-play" onClick={() => { reference.current.pause(); if (player.current.paused) play(player.current); else player.current.pause(); }}>{playing ? 'Pause' : 'Play'} {labels[track].toLowerCase()}</button>
        <label className="ov-timeline"><span className="sr-only">Playback position</span><input aria-label="Playback position" type="range" min="0" max={duration} step="0.01" value={Math.min(time, duration)} onChange={event => { const value = Number(event.target.value); player.current.currentTime = value; setTime(value); }} /></label>
        <span className="ov-time">{clock(time)} / {clock(duration)}</span>
      </div>
      <audio ref={player} src={active.src} preload="auto" onLoadedMetadata={loaded} onPlay={() => setPlaying(true)} onPause={() => setPlaying(false)} onEnded={() => setPlaying(false)} onTimeUpdate={() => setTime(player.current.currentTime)} onError={() => setError('This audio could not load. Try another conversation or reload the page.')} />
      <audio ref={reference} src={item.tracks.reference.src} preload="metadata" onPlay={() => setReferencePlaying(true)} onPause={() => setReferencePlaying(false)} onEnded={() => setReferencePlaying(false)} onError={() => setError('The voice sample could not load. Please reload the page.')} />
      {error && <p role="alert">{error}</p>}
      <p className="ov-listening-note">Switch tracks while playing to compare the same moment. Playback levels are matched; no extra denoising is applied.</p>
      <details className="ov-details"><summary>Compare with the clean target</summary><div>
        <p>This is the original speaker’s recording before mixing. The model does not receive it during extraction.</p>
        <button aria-pressed={track === 'target'} onClick={() => chooseTrack('target')}>Listen to clean target</button>
        <a href={item.tracks.estimate.raw} download>Download raw model output ↗</a>
        <p>This example: {item.improvement.toFixed(2)} dB SI-SDR improvement over the mixture. Higher means better separation against the clean target; it is not a listening-quality rating.</p>
      </div></details>
    </section>
    <OneVoiceUpload />
    <section className="ov-about"><h2>A sample tells One Voice who to keep.</h2><p>Use a separate recording of the person speaking alone. One Voice uses that voice sample to extract their speech from an overlapping conversation. It does not clone voices or generate new speech.</p><p>Results can contain distortion or other speakers, especially with noise or unfamiliar recording conditions. Listen to the output before relying on it.</p><p className="ov-listening-note">Examples: LibriSpeech / Libri2Mix, <a href="https://www.openslr.org/12/">OpenSLR</a>, <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Audio has been mixed and model-processed.</p><section className="ov-about ov-research" aria-labelledby="ov-research-title"><h2 id="ov-research-title">Research behind One Voice</h2><p>One Voice is an independent implementation informed by published target speaker extraction research. Its reference-conditioned model draws on Junjie Li and colleagues’ <a href="https://arxiv.org/abs/2409.09589">On the effectiveness of enrollment speech augmentation for Target Speaker Extraction</a> (2024), with a smaller configuration for local training. It does not reproduce the paper’s full experiments or reported results.</p><p>The separator follows ideas from Yi Luo and Jianwei Yu’s <a href="https://arxiv.org/abs/2209.15174">Music Source Separation with Band-split RNN</a> (2022). <a href="https://github.com/BUTSpeechFIT/speakerbeam">SpeakerBeam</a> and <a href="https://arxiv.org/abs/2004.08326">SpEx</a> informed the target-speaker formulation and speaker supervision. No pretrained weights from these systems are used.</p><p>Thanks to the researchers and the <a href="https://www.openslr.org/12/">LibriSpeech</a> and <a href="https://github.com/JorisCos/LibriMix">LibriMix</a> dataset contributors. <a href="https://github.com/Zachshotamartin/target-speaker-extraction/blob/main/docs/SOURCES.md">Full sources and attribution ↗</a></p></section>
</section>
  </>;
}
