import Waveform from './ui/Waveform.jsx';
import Disclosure from './ui/Disclosure.jsx';
import Section from './ui/Section.jsx';
import Text from './ui/Text.jsx';
import Heading from './ui/Heading.jsx';
import AudioPlayer from './ui/AudioPlayer.jsx';
import { useEffect, useRef, useState } from 'react';
import {useMotionState} from './motion.js';
import bestSnapshot from './snapshot.json';
import TrainingProgress from './TrainingProgress.jsx';
import releaseAssets from './release-assets.json';
import './one-voice.css';
import OneVoiceUpload from './OneVoiceUpload.jsx';

const labels = { mixture: 'Both voices', estimate: 'Extracted voice', target: 'Clean target' };


export default function OneVoiceDetails({active: pageActive = true}) {
  const [index, setIndex] = useMotionState(0, '#listen');
  const [track, setTrack] = useMotionState('mixture', '#listen');
  const [modelVersion, setModelVersion] = useMotionState('best', '.one-voice');
  const [latestSnapshot, setLatestSnapshot] = useState(null);
  const [loadingVersion, setLoadingVersion] = useMotionState(false, '.one-voice');
  const snapshot = modelVersion === 'best' ? bestSnapshot : latestSnapshot;
  const [time, setTime] = useState(0);
  const [error, setError] = useMotionState('', '#listen');
  const player = useRef(null), reference = useRef(null);
  const pending = useRef({ time: 0, play: false });
  const item = snapshot.items[index], active = item.tracks[track];
  const duration = active.duration;
  useEffect(() => {
    const a = player.current, b = reference.current;
    return () => { a?.pause(); b?.pause(); };
  }, []);
  useEffect(() => {
    if (!pageActive) {player.current?.pause(); reference.current?.pause(); pending.current.play = false;}
  }, [pageActive]);
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
  async function chooseModel(next) {
    if (next === modelVersion || loadingVersion) return;
    if (next === 'latest' && !latestSnapshot) {
      setLoadingVersion(true);
      try {
        const response = await fetch(releaseAssets.latestListening);
        if (!response.ok) throw new Error('Latest examples unavailable');
        const loaded = await response.json();
        if (!player.current) return;
        setLatestSnapshot(loaded);
      } catch {
        setError('The latest examples could not load. Please try again.');
        return;
      } finally { setLoadingVersion(false); }
    }
    pending.current = { time: player.current.currentTime || 0, play: !player.current.paused };
    player.current.pause(); reference.current.pause();
    setModelVersion(next); setTrack('estimate'); setError('');
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
    <Section className="study-live one-voice" aria-labelledby="one-voice-listen" data-checkpoint={snapshot.checkpointSHA256}>
      <div className="study-section-heading"><Heading id="one-voice-listen">One conversation. Either voice.</Heading><Text>Six conversations · two voices each</Text></div>
      <Text>Listen to the overlapping voices, choose who to keep, then switch to the extraction.</Text>
      <Text className="ov-checkpoint-note">{modelVersion === 'best' ? 'Best full validation · epoch 80 · used for live processing' : 'Latest saved model · epoch 81 + 9 updates · listening comparison'}</Text>
      <div className="ov-controls">
        <fieldset><legend>Model version</legend><div className="ov-switch">
          {['best', 'latest'].map(version => <button key={version} disabled={loadingVersion} aria-pressed={modelVersion === version} onClick={() => chooseModel(version)}>{version === 'best' ? 'Best' : loadingVersion ? 'Loading latest…' : 'Latest'}</button>)}
        </div></fieldset>
        <fieldset><legend>Conversation</legend><div className="ov-switch">
          {[...new Set(snapshot.items.map(example => example.conversation))].map(n => <button key={n} aria-pressed={item.conversation === n} onClick={() => chooseExample((n - 1) * 2 + index % 2)}>Conversation {n}</button>)}
        </div></fieldset>
        <fieldset><legend>Voice to keep</legend><div className="ov-switch">
          {['A', 'B'].map((voice, n) => <button key={voice} aria-pressed={item.voice === voice} onClick={() => chooseExample(Math.floor(index / 2) * 2 + n)}>Voice {voice}</button>)}
        </div></fieldset>
      </div>
      <div className="ov-reference">
        <div className="ov-reference-player">
          <Text tone="ink">Voice {item.voice}’s sample</Text>
          <AudioPlayer ref={reference} src={item.tracks.reference.src} label={`Voice ${item.voice} sample`} onPlay={() => player.current?.pause()}/>
        </div>
        <Text>A separate recording identifies the speaker. It is not the clean answer.</Text>
      </div>
      <div className="ov-waveforms">
        {['mixture', 'estimate'].map(name => <button key={name} className={`ov-wave ov-wave--${name}`} aria-pressed={track === name} onClick={() => chooseTrack(name)}>
          <span><strong>{labels[name]}</strong><small>{name === 'mixture' ? 'Original overlapping speech' : `Model output · voice ${item.voice}`}</small></span>
          <Waveform peaks={item.tracks[name].peaks} progress={track === name ? time / duration : 0} />
          <span className="ov-wave-choice">{track === name ? 'Selected' : 'Listen to this track'} <span aria-hidden="true">↗</span></span>
        </button>)}
      </div>
      <div className="ov-player">
        <Text tone="ink">{labels[track]}</Text>
        <AudioPlayer ref={player} src={active.src} label={labels[track]} onLoadedMetadata={loaded}
          onPlay={() => reference.current?.pause()}
          onSeeking={event => setTime(event.currentTarget.currentTime)}
          onTimeUpdate={event => setTime(event.currentTarget.currentTime)}/>
      </div>
      {error && <Text role="alert">{error}</Text>}
      <Text className="ov-listening-note">Switch tracks while playing to compare the same moment. Playback levels are matched; no extra denoising is applied.</Text>
      <Disclosure className="ov-details" label="Hear the original solo voice">
        <Text>This is the chosen speaker’s original recording before the two voices were mixed. Compare it with the model’s extraction to hear what was preserved or lost. The model never receives this clean answer during extraction.</Text>
        <button aria-pressed={track === 'target'} onClick={() => chooseTrack('target')}>Play original solo voice</button>
        <a href={item.tracks.estimate.raw} download>Download raw model output ↗</a>
        <Text>This example: {item.improvement.toFixed(2)} dB SI-SDR improvement over the mixture. Higher means better separation against the clean target; it is not a listening-quality rating.</Text>
      </Disclosure>
    </Section>
    <OneVoiceUpload active={pageActive}/>
    <TrainingProgress/>
    <Section className="ov-about"><Heading>A sample tells One Voice who to keep.</Heading><Text>Use a separate recording of the person speaking alone. One Voice uses that voice sample to extract their speech from an overlapping conversation. It does not clone voices or generate new speech.</Text><Text>Results can contain distortion or other speakers, especially with noise or unfamiliar recording conditions. Listen to the output before relying on it.</Text><Text className="ov-listening-note">Examples: LibriSpeech / Libri2Mix, <a href="https://www.openslr.org/12/">OpenSLR</a>, <a href="https://creativecommons.org/licenses/by/4.0/">CC BY 4.0</a>. Audio has been mixed and model-processed.</Text><Section className="ov-about ov-research" aria-labelledby="ov-research-title"><Heading id="ov-research-title">Research behind One Voice</Heading><Text>One Voice is an independent implementation informed by published target speaker extraction research. Its reference-conditioned model draws on Junjie Li and colleagues’ <a href="https://arxiv.org/abs/2409.09589">On the effectiveness of enrollment speech augmentation for Target Speaker Extraction</a> (2024), with a smaller configuration for local training. It does not reproduce the paper’s full experiments or reported results.</Text><Text>The separator follows ideas from Yi Luo and Jianwei Yu’s <a href="https://arxiv.org/abs/2209.15174">Music Source Separation with Band-split RNN</a> (2022). <a href="https://github.com/BUTSpeechFIT/speakerbeam">SpeakerBeam</a> and <a href="https://arxiv.org/abs/2004.08326">SpEx</a> informed the target-speaker formulation and speaker supervision. No pretrained weights from these systems are used.</Text><Text>Thanks to the researchers and the <a href="https://www.openslr.org/12/">LibriSpeech</a> and <a href="https://github.com/JorisCos/LibriMix">LibriMix</a> dataset contributors. <a href="https://github.com/Zachshotamartin/target-speaker-extraction/blob/main/docs/SOURCES.md">Full sources and attribution ↗</a></Text></Section>
</Section>
  </>;
}
