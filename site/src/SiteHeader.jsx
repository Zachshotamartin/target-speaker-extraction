import React from 'react';

export default function SiteHeader({activePage}) {
  return <header className="product-nav">
    <a href="#" className="header-brand" aria-label="OneVoice home">
      <img src="/assets/one-voice/brand/mark.svg" alt=""/>OneVoice
    </a>
    <nav aria-label="Main">
      <a href="#" aria-current={activePage === 'overview' ? 'page' : undefined}>Overview</a>
      <a href="#transcribe" aria-current={activePage === 'transcribe' ? 'page' : undefined}>Speech to text</a>
      <a href="#ov-upload-title" className="header-action">Try your recording <span aria-hidden="true">↗</span></a>
    </nav>
  </header>;
}
