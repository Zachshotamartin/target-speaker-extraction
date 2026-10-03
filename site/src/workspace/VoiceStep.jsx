import React from 'react';
import Section from '../ui/Section.jsx';
import Heading from '../ui/Heading.jsx';
import Text from '../ui/Text.jsx';
import Stack from '../ui/Stack.jsx';
import Inline from '../ui/Inline.jsx';
import Button from '../ui/Button.jsx';
import Disclosure from '../ui/Disclosure.jsx';
import SplitLayout from '../ui/SplitLayout.jsx';
import ReferenceTools from './ReferenceTools.jsx';
import RecordingPreview from './RecordingPreview.jsx';
import VoiceReference from './VoiceReference.jsx';
export default function VoiceStep({project,disabled,serviceReady,microphone,savedProfiles,excerpt,setExcerpt,onDiscover,onReference,onExcerpt,onRename,onRemove,onSave,onRetry,onExtract}) {
  const tools = <ReferenceTools {...{project,disabled,serviceReady,microphone,savedProfiles,excerpt,setExcerpt,onDiscover,onReference,onExcerpt}}/>;
  return <Section aria-label="Voice selection" className="workspace-stage"><SplitLayout sidebar={<Stack><Heading size="title">Your recording</Heading><RecordingPreview project={project}/><Text size="small">The full conversation. OneVoice will keep the person identified by your voice sample.</Text></Stack>}><Stack gap="large">
    <Stack gap="small"><Heading>Choose the voice to keep</Heading><Text>{project.references.length ? 'Listen to the sample to check you have the right person.' : 'Use a short solo sample to identify the person you want.'}</Text></Stack>
    {project.references.length ? <><div className="chosen-voices">{project.references.map((reference,index) => <VoiceReference key={reference.id} {...{reference,index,disabled,onRename,onRemove,onSave}}/>)}</div>{project.references.length < 4 && <Disclosure label="Change or add a voice" className="voice-alternatives">{tools}</Disclosure>}</> : tools}
    {!serviceReady && <Inline><Text size="small">Connecting to the processing service…</Text><Button onClick={onRetry}>Retry connection</Button></Inline>}
    <div className="workspace-action-bar"><Text size="small">Original transcript included for comparison.{project.references.length > 2 && ' More than two overlapping speakers is experimental.'}</Text><Button variant="primary" disabled={disabled || microphone.isBusy() || !project.references.length || project.references.some(reference => !reference.label.trim()) || !serviceReady} onClick={onExtract}>Separate {project.references.length > 1 ? `${project.references.length} voices` : 'voice'} &amp; transcribe <span aria-hidden="true">→</span></Button></div>
  </Stack></SplitLayout></Section>;
}
