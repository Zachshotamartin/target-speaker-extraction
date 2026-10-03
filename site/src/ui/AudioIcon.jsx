import React from 'react';

export default function AudioIcon({name}) {
  return <svg viewBox="0 0 24 24" aria-hidden="true" focusable="false">
    {name === 'play' && <path d="M9 5.5 19 12 9 18.5Z" fill="currentColor" stroke="none"/>}
    {name === 'pause' && <><rect x="7" y="5" width="3.5" height="14" rx="1" fill="currentColor" stroke="none"/><rect x="14" y="5" width="3.5" height="14" rx="1" fill="currentColor" stroke="none"/></>}
    {(name === 'volume' || name === 'muted') && <><path d="M11 5 6 9H3v6h3l5 4Z"/>{name === 'volume' ? <><path d="M15 8a6 6 0 0 1 0 8M18 5a10 10 0 0 1 0 14"/></> : <path d="m16 9 5 6m0-6-5 6"/>}</>}
  </svg>;
}
