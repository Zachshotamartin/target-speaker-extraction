import React from 'react';
import SignalWave from './ui/SignalWave.jsx';

export default function SignalArt(){ return <figure className="signal-art"><figcaption>One conversation. A clearer voice.</figcaption><div className="signal-lane"><div className="signal-caption"><span>01 / THE CONVERSATION</span><span>Two voices</span></div><SignalWave track="mixture"/></div><div className="signal-lane signal-lane--output"><div className="signal-caption"><span>02 / THE EXTRACTION</span><span>Voice A</span></div><SignalWave track="estimate"/></div><a href="#listen" className="signal-footnote continuous-underline">Hear this example <span aria-hidden="true">↗</span></a></figure>; }
