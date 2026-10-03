import Section from './ui/Section.jsx';
import Text from './ui/Text.jsx';
import Heading from './ui/Heading.jsx';
import React from 'react';
import './privacy.css';

const sections = [
  ['demo', 'Using the demo'],
  ['code', 'Original source code'],
  ['audio', 'Datasets and example audio'],
  ['models', 'Models and other components'],
  ['contact', 'Questions and permissions'],
];

export default function TermsPage() {
  return <article className="privacy-page" aria-labelledby="terms-title">
    <header className="privacy-heading">
      <Text className="privacy-updated">Last updated <time dateTime="2026-09-21">September 21, 2026</time></Text>
      <h1 id="terms-title">Terms and licensing</h1>
      <Text className="privacy-intro">Using One Voice and understanding the separate rights in its code, models, and audio.</Text>
    </header>
    <div className="privacy-layout">
      <nav className="privacy-contents" aria-label="Terms and licensing sections">
        <Text>On this page</Text>
        <ol>{sections.map(([id, title], index) => <li key={id}><a className="continuous-underline" href={`#terms-${id}`}><span aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>{title}</a></li>)}</ol>
      </nav>
      <div className="privacy-body">
        <Text className="privacy-scope">One Voice is an independent project by Zachary Martin. This page describes the public demo and the licensing status of project materials. It does not replace licenses that apply to third-party materials or restrict rights you have under applicable law.</Text>
        <Section id="terms-demo" aria-labelledby="terms-demo-title">
          <Heading id="terms-demo-title">Using the demo</Heading>
          <Text>You may use the public interface to try the prepared examples and process recordings you have the right to submit. Obtain any permissions required for recording, processing, and sharing other people’s voices. Do not use the service to violate others’ rights or interfere with its operation.</Text>
          <Text>One Voice is experimental. Extracted audio may contain distortion or other speakers; transcripts, timestamps, and voice-match scores may be wrong. Review results against the original recording. Voice matching is not identity verification, and results should not be treated as an authoritative record or the sole basis for consequential decisions.</Text>
          <Text>Availability and accuracy are not guaranteed. A result does not establish ownership of a recording or grant rights in someone else’s speech. This page does not transfer ownership of recordings you submit to the project owner. Processing and retention are described in the <a href="#privacy">privacy policy</a>.</Text>
        </Section>
        <Section id="terms-code" aria-labelledby="terms-code-title">
          <Heading id="terms-code-title">Original source code</Heading>
          <Text><strong>Original project code: no reuse license granted.</strong> The original One Voice code has not been released under MIT or another open-source license. Public access to its source does not grant a general license to copy, modify, redistribute, or incorporate that code into another product.</Text>
          <Text>This notice is subject to permissions already granted, applicable law, and the terms of the platform hosting the source. For example, GitHub permits viewing and forking public repositories under its <a href="https://docs.github.com/en/site-policy/github-terms/github-terms-of-service#d-user-generated-content">Terms of Service</a>. Third-party material in the repository retains its own license and notices.</Text>
        </Section>
        <Section id="terms-audio" aria-labelledby="terms-audio-title">
          <Heading id="terms-audio-title">Datasets and example audio</Heading>
          <Text>The speech used for the project’s training and prepared examples comes from <a href="https://www.openslr.org/12/">LibriSpeech / OpenSLR 12</a>, credited to Vassil Panayotov, Guoguo Chen, Daniel Povey, and Sanjeev Khudanpur (2015), under <a href="https://creativecommons.org/licenses/by/4.0/">Creative Commons Attribution 4.0 International (CC BY 4.0)</a>.</Text>
          <Text>Prepared examples use Libri2Mix clean mixtures. Audio has been cropped or mixed, playback levels adjusted, and extraction outputs processed by the model. When sharing these recordings or adaptations, retain the applicable attribution, link to the license, and identify modifications. The original-code notice above does not remove these audio permissions or obligations.</Text>
          <Text><a href="https://github.com/JorisCos/LibriMix">LibriMix</a> software and included metadata carry their upstream MIT notice. That notice does not replace the license of the underlying speech recordings. See the project’s <a href="https://github.com/Zachshotamartin/target-speaker-extraction/blob/main/docs/SOURCES.md">sources and attribution</a> for provenance and research references.</Text>
        </Section>
        <Section id="terms-models" aria-labelledby="terms-models-title">
          <Heading id="terms-models-title">Models and other components</Heading>
          <Text>No general redistribution license is granted here for original One Voice model weights. Access to the hosted demo does not provide a license to download or redistribute those weights.</Text>
          <Text>Third-party code and models retain their respective terms. The transcription pipeline includes Whisper and faster-whisper components under MIT, and SpeechBrain ECAPA-TDNN components under Apache 2.0. Other dependencies have their own notices. See the <a href="https://github.com/Zachshotamartin/target-speaker-extraction/blob/main/poc/README.md#sources-and-licenses">component sources and licenses</a> before reusing individual components.</Text>
          <Text>The research workflow also uses locally held SpeakerBeam evaluation mappings from a repository with a <a href="https://github.com/BUTSpeechFIT/speakerbeam/blob/91af02cc617afa35fedfbdbf32533012cd0a8672/LICENSE.txt">custom evaluation license</a>. Those tables are excluded from the public repository. This page grants no permission to redistribute them and makes no claim that their use, or every model artifact, is cleared for unrestricted commercial reuse.</Text>
        </Section>
        <Section id="terms-contact" aria-labelledby="terms-contact-title">
          <Heading id="terms-contact-title">Questions and permissions</Heading>
          <Text>For questions about original project materials or a request for reuse permission, contact <a href="mailto:zachsm@alumni.stanford.edu">zachsm@alumni.stanford.edu</a>. Permission for third-party materials must come from their applicable license or rights holder; the project owner cannot grant rights they do not hold.</Text>
          <Text>Any future license grant will be identified explicitly for the materials it covers. An update to this page does not revoke permissions already granted under an applicable license.</Text>
        </Section>
      </div>
    </div>
  </article>;
}
