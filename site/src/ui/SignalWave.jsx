import React from 'react';
import snapshot from '../snapshot.json';

export default function SignalWave({track}) { const peaks=snapshot.items[0].tracks[track].peaks; const bars=Array.from({length:96},(_,i)=>Math.max(...peaks.slice(Math.floor(i*peaks.length/96),Math.floor((i+1)*peaks.length/96)))); return <svg viewBox="0 0 480 100" aria-hidden="true">{bars.map((p,i)=><line key={i} x1={i*5+2.5} x2={i*5+2.5} y1={50-Math.max(1,p*46)} y2={50+Math.max(1,p*46)}/>)}</svg>; }
