import React, {useEffect, useState} from 'react';
import {createRoot} from 'react-dom/client';
import OneVoiceDetails from './OneVoiceDetails.jsx';
import TranscriptionWorkspace from './TranscriptionWorkspace.jsx';
import './product.css';

function currentView() {
  if (['#listen', '#one-voice-listen'].includes(window.location.hash)) return 'listen';
  if (window.location.hash === '#ov-upload-title') return 'upload';
  if (['#about', '#ov-research-title'].includes(window.location.hash)) return 'about';
  return 'transcribe';
}

function App() {
  const [view, setView] = useState(currentView);
  useEffect(() => {
    const navigate = () => {
      if (window.location.hash === '#main-content') return;
      setView(currentView());
      document.querySelectorAll('audio').forEach(player => player.pause());
      requestAnimationFrame(() => window.scrollTo({top: 0, behavior: 'instant'}));
    };
    window.addEventListener('hashchange', navigate);
    return () => window.removeEventListener('hashchange', navigate);
  }, []);

  return <div className="product-app">
    <a className="skip" href="#main-content">Skip to workspace</a>
    <header className="product-nav">
      <a href="#transcribe" className="wordmark"><img src="/assets/one-voice/brand/mark.svg" alt=""/>OneVoice</a>
      <nav aria-label="Main">
        <a href="#transcribe" aria-current={view === 'transcribe' ? 'page' : undefined}>Transcribe</a>
        <a href="#listen" aria-current={view === 'listen' ? 'page' : undefined}>Listen</a>
        <a href="#ov-upload-title" aria-current={view === 'upload' ? 'page' : undefined}>Isolate audio</a>
      </nav>
      <a className="product-help" href="#about" aria-current={view === 'about' ? 'page' : undefined}>How it works <span aria-hidden="true">↗</span></a>
    </header>
    <main id="main-content" tabIndex={-1}>
      {/* Keep tools mounted so navigation preserves files, results and active jobs. */}
      <div className="page-view" hidden={view !== 'transcribe'}><TranscriptionWorkspace active={view === 'transcribe'}/></div>
      <div className="page-view support-view" hidden={view === 'transcribe'}><OneVoiceDetails view={view}/></div>
    </main>
    <footer className="product-footer">
      <p>OneVoice <span>Local audio workspace</span></p>
      <div><a href="#about">About & research</a><a href="https://github.com/Zachshotamartin/target-speaker-extraction">Source ↗</a><a href="https://zachsm.com/projects/one-voice">Zach Martin</a></div>
    </footer>
  </div>;
}

createRoot(document.getElementById('root')).render(<App/>);
