import Section from './ui/Section.jsx';
import Text from './ui/Text.jsx';
import Heading from './ui/Heading.jsx';
import React from 'react';
import SignalArt from './SignalArt.jsx';
import OneVoiceDetails from './OneVoiceDetails.jsx';



export default function LandingPage({active = true}) {
  return <>
    <Section className="hero">
      <div>
        <Text className="eyebrow">TARGET SPEAKER EXTRACTION</Text>
        <h1>Keep the voice<br/>that matters.</h1>
        <Text className="intro">Two people talking at once. One voice you want to hear. Give One Voice a sample of that person, and separate their speech from the conversation.</Text>
        <div className="hero-actions">
          <a className="primary-link" href="#listen">Hear the difference <span aria-hidden="true">↓</span></a>
          <a className="hero-recording-link continuous-underline" href="#ov-upload-title">Try your recording <span aria-hidden="true">↓</span></a>
        </div>
      </div>
      <SignalArt/>
    </Section>
    <Section className="steps" aria-label="How to use One Voice">
      <Text><b>01</b> Choose the conversation.</Text>
      <Text><b>02</b> Identify the voice with a sample.</Text>
      <Text><b>03</b> Listen to the extraction.</Text>
    </Section>
    <div id="listen" className="experience"><OneVoiceDetails active={active}/></div>
  </>;
}
