import React, {useEffect, useImperativeHandle, useRef, useState} from 'react';
import {useAudioUrl} from '../useAudioUrl.js';
import AudioIcon from './AudioIcon.jsx';
import './audio-player.css';

const clock = seconds => `${Math.floor(seconds / 60)}:${String(Math.floor(seconds % 60)).padStart(2, '0')}`;
const initial = {time: 0, duration: 0, playing: false, muted: false};

/** The forwarded ref stays an HTMLAudioElement for linked playback and editing. */
export default function AudioPlayer({blob, src, label = 'Audio', ref, className = '',
  onPlay, onPause, onEnded, onTimeUpdate, onLoadedMetadata, onDurationChange,
  onVolumeChange, onEmptied, onError, ...props}) {
  const url = useAudioUrl(blob);
  const source = src || url || undefined;
  const player = useRef(null);
  const [media, setMedia] = useState(initial);
  const [error, setError] = useState('');
  useImperativeHandle(ref, () => player.current);
  useEffect(() => {setMedia(initial); setError('');}, [source]);

  function sync(event, callback) {
    callback?.(event);
    const audio = event.currentTarget;
    setMedia({time: audio.currentTime || 0, duration: Number.isFinite(audio.duration) ? audio.duration : 0,
      playing: !audio.paused && !audio.ended, muted: audio.muted || audio.volume === 0});
  }

  async function toggle() {
    const audio = player.current;
    if (!audio.paused) {audio.pause(); return;}
    setError('');
    if (audio.error) audio.load();
    try {await audio.play();}
    catch (failure) {if (failure.name !== 'AbortError') setError('Audio could not play. Try Play again.');}
  }

  function seek(event) {
    const time = Number(event.target.value);
    player.current.currentTime = time;
    setMedia(value => ({...value, time}));
  }

  return <div className={`ui-audio-player ${className}`} role="group" aria-label={label}>
    <audio {...props} ref={player} hidden preload="metadata" src={source}
      onPlay={event => sync(event, onPlay)} onPause={event => sync(event, onPause)}
      onEnded={event => sync(event, onEnded)} onTimeUpdate={event => sync(event, onTimeUpdate)}
      onLoadedMetadata={event => sync(event, onLoadedMetadata)} onDurationChange={event => sync(event, onDurationChange)}
      onVolumeChange={event => sync(event, onVolumeChange)} onEmptied={event => sync(event, onEmptied)}
      onError={event => {setError('Audio could not load. Try Play again.'); onError?.(event);}}/>
    <div className="ui-audio-row">
      <button className="ui-audio-play" type="button" disabled={!source} onClick={toggle}
        aria-label={`${media.playing ? 'Pause' : 'Play'} ${label.toLowerCase()}`} data-playing={media.playing}>
        <span className="ui-audio-play-icon"><AudioIcon name="play"/></span>
        <span className="ui-audio-pause-icon"><AudioIcon name="pause"/></span>
      </button>
      <div className="ui-audio-seek">
        <progress value={media.time} max={media.duration || 1} aria-hidden="true"/>
        <input type="range" min="0" max={media.duration || 0} step="0.01" value={Math.min(media.time, media.duration)}
          disabled={!media.duration} onChange={seek} aria-label={`${label} playback position`}
          aria-valuetext={`${clock(media.time)} of ${clock(media.duration)}`}/>
      </div>
      <span className="ui-audio-time" aria-hidden="true"><span>{clock(media.time)}</span><span className="ui-audio-duration"><span className="ui-audio-time-separator"> / </span>{clock(media.duration)}</span></span>
      <button className="ui-audio-mute" type="button" disabled={!source} aria-label={`${media.muted ? 'Unmute' : 'Mute'} ${label.toLowerCase()}`}
        onClick={() => {player.current.muted = !media.muted; if (player.current.volume === 0) player.current.volume = 1;}}>
        <AudioIcon name={media.muted ? 'muted' : 'volume'}/>
      </button>
    </div>
    {error && <p className="ui-audio-error" role="status">{error}</p>}
  </div>;
}
