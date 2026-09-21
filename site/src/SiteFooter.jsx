import React from 'react';
import './footer.css';

export default function SiteFooter({activePage}) {
  return <footer className="site-footer">
    <a className="site-footer-brand" href="#" aria-label="OneVoice home">OneVoice</a>
    <p className="site-footer-credit">Made by <a href="https://zachsm.com/projects/one-voice">Zach Martin</a></p>
    <nav className="site-footer-links" aria-label="Project information">
      <a href="#privacy" aria-current={activePage === 'privacy' ? 'page' : undefined}>Privacy</a>
      <a href="https://github.com/Zachshotamartin/target-speaker-extraction">Source ↗</a>
    </nav>
  </footer>;
}
