import React, {useEffect, useLayoutEffect, useRef, useState} from 'react';
import {initialHeaderScroll, updateHeaderScroll} from './headerScroll.js';

export default function SiteHeader({activePage}) {
  const header = useRef(null);
  const scroll = useRef(initialHeaderScroll());
  const [hidden, setHidden] = useState(false);

  useLayoutEffect(() => {
    scroll.current = initialHeaderScroll(window.scrollY);
    setHidden(false);
  }, [activePage]);

  useEffect(() => {
    let frame = 0;
    const update = () => {
      frame = 0;
      const element = header.current;
      if (!element) return;
      // Clamp elastic overscroll so bouncing at the bottom cannot reverse direction.
      const maxY = Math.max(0, document.documentElement.scrollHeight - window.innerHeight);
      const y = Math.max(0, Math.min(window.scrollY, maxY));
      const keepVisible = Boolean(element.querySelector(':focus-visible'));
      scroll.current = updateHeaderScroll(scroll.current, y, element.offsetHeight, keepVisible);
      setHidden(scroll.current.hidden);
    };
    const onScroll = () => { if (!frame) frame = requestAnimationFrame(update); };
    const onResize = () => {
      scroll.current = initialHeaderScroll(window.scrollY);
      setHidden(false);
    };
    window.addEventListener('scroll', onScroll, {passive: true});
    window.addEventListener('resize', onResize);
    return () => {
      cancelAnimationFrame(frame);
      window.removeEventListener('scroll', onScroll);
      window.removeEventListener('resize', onResize);
    };
  }, []);

  function revealForKeyboard(event) {
    if (!event.target.matches(':focus-visible')) return;
    scroll.current = initialHeaderScroll(window.scrollY);
    setHidden(false);
  }

  return <header ref={header} className="product-nav" data-hidden={hidden} onFocusCapture={revealForKeyboard}>
    <a href="#" className="header-brand" aria-label="OneVoice home">
      <img src="/assets/one-voice/brand/mark.svg" alt=""/>OneVoice
    </a>
    <nav aria-label="Main">
      <a href="#" aria-current={activePage === 'overview' ? 'page' : undefined}>Overview</a>
      <a href="#transcribe" aria-current={activePage === 'transcribe' ? 'page' : undefined}>Speech to text</a>
    </nav>
  </header>;
}
