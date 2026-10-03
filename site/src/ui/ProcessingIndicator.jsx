import React, {useEffect, useRef, useState} from 'react';
import './processing-indicator.css';

export default function ProcessingIndicator({active = true}) {
  const canvas = useRef(null);
  const [ready, setReady] = useState(false);
  useEffect(() => {
    if (!active) return;
    let disposed = false, cleanup = () => {};
    const element = canvas.current;
    const preference = window.matchMedia('(prefers-reduced-motion: reduce)');
    setReady(false);
    import('./processingScene.js').then(({createProcessingScene}) => {
      if (disposed) return;
      try {
        cleanup = createProcessingScene(element, preference, () => setReady(false));
        setReady(true);
      } catch {
        // The CSS signal stays visible when WebGL is unavailable.
        setReady(false);
      }
    }).catch(() => {if (!disposed) setReady(false);});
    return () => {disposed = true; cleanup();};
  }, [active]);
  return <span className="processing-indicator" aria-hidden="true" data-ready={ready}>
    <svg className="processing-fallback" viewBox="0 0 48 48"><circle cx="24" cy="24" r="8"/><path d="M24 5v7m0 24v7M5 24h7m24 0h7M10.5 10.5l5 5m17 17 5 5m-27 0 5-5m17-17 5-5"/><circle className="processing-core" cx="24" cy="24" r="3"/></svg>
    <canvas ref={canvas}/>
  </span>;
}
