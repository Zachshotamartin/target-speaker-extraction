import {useEffect, useRef, useState} from 'react';
import {AudioCapture} from './audioCapture.js';
import {useMotionState} from './motion.js';

export function useAudioRecorder({active, scope, onComplete, onError, onStart}) {
  const [state, setState] = useMotionState({phase: 'idle', target: null}, scope);
  const [elapsed, setElapsed] = useState(0);
  const callbacks = useRef({active, onComplete, onError, onStart});
  callbacks.current = {active, onComplete, onError, onStart};
  const alive = useRef(true);
  const capture = useRef(null);
  if (!capture.current) capture.current = new AudioCapture({
    onState: ({elapsed: seconds, phase, target}) => {
      if (!alive.current) return;
      setElapsed(seconds);
      setState(previous => previous.phase === phase && previous.target === target ? previous : {phase, target});
    },
    onComplete: (target, blob) => { if (alive.current) callbacks.current.onComplete(target, blob); },
    onError: error => { if (alive.current) callbacks.current.onError(error); },
  });
  useEffect(() => {
    alive.current = true;
    const stop = () => capture.current.cancel();
    window.addEventListener('pagehide', stop);
    return () => { alive.current = false; window.removeEventListener('pagehide', stop); stop(); };
  }, []);
  useEffect(() => { if (!active) capture.current.cancel(); }, [active]);
  return {
    ...state, elapsed, busy: state.phase !== 'idle',
    isBusy: () => Boolean(capture.current.session),
    start: (target, limits) => {
      if (!callbacks.current.active || capture.current.session) return;
      callbacks.current.onStart?.();
      document.querySelectorAll('audio').forEach(player => player.pause());
      capture.current.start(target, limits);
    },
    stop: () => capture.current.stop(),
    cancel: () => capture.current.cancel(),
  };
}
