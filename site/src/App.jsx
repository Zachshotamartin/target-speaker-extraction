import React, {useEffect, useRef, useState} from 'react';
import LandingPage from './LandingPage.jsx';
import TranscriptionWorkspace from './TranscriptionWorkspace.jsx';
import {animateChange, animateDisclosure} from './motion.js';

export default function App() {
  const [hash, setHash] = useState(window.location.hash);
  const route = useRef(hash);
  const transcribing = hash === '#transcribe';

  function scrollToRoute(next, smooth = false) {
    const target = next && document.getElementById(next.slice(1));
    const behavior = smooth && !window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'smooth' : 'instant';
    if (target && next !== '#transcribe') target.scrollIntoView({block: 'start', behavior});
    else window.scrollTo({top: 0, behavior});
  }

  function navigate(next) {
    const wasWorkspace = route.current === '#transcribe';
    const isWorkspace = next === '#transcribe';
    route.current = next;
    document.querySelectorAll('audio').forEach(player => player.pause());
    if (wasWorkspace === isWorkspace) {
      setHash(next);
      scrollToRoute(next, true);
    } else {
      animateChange(() => setHash(next), undefined, () => scrollToRoute(next));
    }
  }

  useEffect(() => {
    const restore = () => { if (window.location.hash !== route.current) navigate(window.location.hash); };
    const frame = requestAnimationFrame(() => scrollToRoute(window.location.hash));
    window.addEventListener('popstate', restore);
    window.addEventListener('hashchange', restore);
    return () => { cancelAnimationFrame(frame); window.removeEventListener('popstate', restore); window.removeEventListener('hashchange', restore); };
  }, []);

  function handleNavigation(event) {
    animateDisclosure(event);
    if (event.defaultPrevented || event.metaKey || event.ctrlKey || event.shiftKey || event.altKey || event.button) return;
    const anchor = event.target.closest('a[href^="#"]');
    if (!anchor) return;
    event.preventDefault();
    const next = anchor.getAttribute('href');
    if (next === '#main-content') {document.getElementById('main-content').focus(); return;}
    const hash = next === '#' ? '' : next;
    if (window.location.hash !== hash) window.history.pushState(null, '', hash || window.location.pathname + window.location.search);
    navigate(hash);
  }

  return <div className={transcribing ? 'app-shell app-shell--workspace' : 'app-shell'} onClickCapture={handleNavigation}>
    <a className="skip" href="#main-content">Skip to content</a>
    <header className="product-nav">
      <a href="#" className="wordmark" aria-label="OneVoice home"><img src="/assets/one-voice/brand/mark.svg" alt=""/>OneVoice</a>
      <nav aria-label="Main">
        <a href="#listen" aria-current={hash === '#listen' ? 'location' : undefined}>Listen</a>
        <a href="#transcribe" aria-current={transcribing ? 'page' : undefined}>Speech to text</a>
        <a href="#ov-upload-title">Try your recording <span aria-hidden="true">↗</span></a>
      </nav>
    </header>
    <main id="main-content" tabIndex={-1}>
      {/* Both stay mounted so page transitions preserve files and active jobs. */}
      <div hidden={transcribing}><LandingPage/></div>
      <div hidden={!transcribing}><TranscriptionWorkspace active={transcribing}/></div>
    </main>
    <footer hidden={transcribing}>
      <a className="wordmark" href="#">OneVoice</a>
      <p>Made by <a href="https://zachsm.com/projects/one-voice">Zach Martin</a></p>
      <a href="https://github.com/Zachshotamartin/target-speaker-extraction">Source ↗</a>
    </footer>
  </div>;
}
