import React from 'react';
import snapshot from './snapshot.json';
import OneVoiceDetails from './OneVoiceDetails.jsx';
function Wave({track}) { const peaks=snapshot.items[0].tracks[track].peaks; const bars=Array.from({length:96},(_,i)=>Math.max(...peaks.slice(Math.floor(i*peaks.length/96),Math.floor((i+1)*peaks.length/96)))); return <svg viewBox="0 0 480 100" aria-hidden="true">{bars.map((p,i)=><line key={i} x1={i*5+2.5} x2={i*5+2.5} y1={50-Math.max(1,p*46)} y2={50+Math.max(1,p*46)}/>)}</svg>; }
function SignalArt(){ return <figure className="signal-art"><figcaption>One conversation. A clearer voice.</figcaption><div className="signal-lane"><div className="signal-caption"><span>01 / THE CONVERSATION</span><span>Two voices</span></div><Wave track="mixture"/></div><div className="signal-lane signal-lane--output"><div className="signal-caption"><span>02 / THE EXTRACTION</span><span>Voice A</span></div><Wave track="estimate"/></div><a href="#listen" className="signal-footnote">Hear this example <span aria-hidden="true">↗</span></a></figure>; }


export default function LandingPage() {
  return <>
    <section className="hero">
      <div>
        <p className="eyebrow">TARGET SPEAKER EXTRACTION</p>
        <h1>Keep the voice<br/>that matters.</h1>
        <p className="intro">Two people talking at once. One voice you want to hear. Give One Voice a sample of that person, and separate their speech from the conversation.</p>
        <div className="hero-actions">
          <a className="primary-link" href="#listen">Hear the difference <span aria-hidden="true">↓</span></a>
          <a className="hero-recording-link" href="#ov-upload-title">Try your recording <span aria-hidden="true">↓</span></a>
        </div>
      </div>
      <SignalArt/>
    </section>
    <section className="steps" aria-label="How to use One Voice">
      <p><b>01</b> Choose the conversation.</p>
      <p><b>02</b> Identify the voice with a sample.</p>
      <p><b>03</b> Listen to the extraction.</p>
    </section>
    <div id="listen" className="experience"><OneVoiceDetails/></div>
  </>;
}
