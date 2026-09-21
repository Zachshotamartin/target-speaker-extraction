import React, {useEffect, useLayoutEffect, useRef, useState} from 'react';
import LandingPage from './LandingPage.jsx';
import SiteHeader from './SiteHeader.jsx';
import SiteFooter from './SiteFooter.jsx';
import TranscriptionWorkspace from './TranscriptionWorkspace.jsx';
import PrivacyPage from './PrivacyPage.jsx';
import {pageForHash} from './pageRoute.js';
import {ownScrollRestoration, scrollToRoute} from './routeNavigation.js';
import {animatePageChange, animateDisclosure} from './motion.js';

export default function App() {
  const [hash, setHash] = useState(window.location.hash);
  const route = useRef(hash);
  const page = pageForHash(hash);
  const transcribing = page === 'transcribe';

  function navigate(next) {
    const previousPage = pageForHash(route.current);
    const nextPage = pageForHash(next);
    route.current = next;
    document.querySelectorAll('audio').forEach(player => player.pause());
    if (previousPage === nextPage) {
      setHash(next);
      scrollToRoute(next, true);
    } else {
      animatePageChange(() => setHash(next), () => {
        if (route.current === next) scrollToRoute(next);
      });
    }
  }

  useLayoutEffect(() => {
    const release = ownScrollRestoration();
    scrollToRoute(window.location.hash);
    return release;
  }, []);

  useEffect(() => {
    const restore = () => { if (window.location.hash !== route.current) navigate(window.location.hash); };
    window.addEventListener('popstate', restore);
    window.addEventListener('hashchange', restore);
    return () => { window.removeEventListener('popstate', restore); window.removeEventListener('hashchange', restore); };
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
    <SiteHeader activePage={page}/>
    <main id="main-content" tabIndex={-1}>
      {/* Views stay mounted so page transitions preserve files and active jobs. */}
      <div hidden={page !== 'overview'}><LandingPage active={page === 'overview'}/></div>
      <div hidden={!transcribing}><TranscriptionWorkspace active={transcribing}/></div>
      <div hidden={page !== 'privacy'}><PrivacyPage/></div>
    </main>
    <SiteFooter activePage={page}/>
  </div>;
}
