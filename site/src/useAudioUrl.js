import {useEffect} from 'react';
import {useMotionState} from './motion.js';

export function useAudioUrl(blob, scope = '#transcribe') {
  const [url, setUrl] = useMotionState(null, scope);
  useEffect(() => {
    if (!blob) {setUrl(null); return;}
    const value = URL.createObjectURL(blob);
    setUrl(value);
    return () => URL.revokeObjectURL(value);
  }, [blob]);
  return url;
}
