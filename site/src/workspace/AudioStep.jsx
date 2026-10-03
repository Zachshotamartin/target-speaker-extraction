import React from 'react';
import Section from '../ui/Section.jsx';
import Heading from '../ui/Heading.jsx';
import Text from '../ui/Text.jsx';
import Stack from '../ui/Stack.jsx';
import Inline from '../ui/Inline.jsx';
import Button from '../ui/Button.jsx';
import Field from '../ui/Field.jsx';
import ChoiceList from '../ui/ChoiceList.jsx';
import SplitLayout from '../ui/SplitLayout.jsx';
import AudioCaptureButton from '../AudioCaptureButton.jsx';
import RecordingPreview from './RecordingPreview.jsx';
import ExamplePicker from './ExamplePicker.jsx';
const methods = [{value:'example',label:'Use an example',description:'Ready-made conversations'},{value:'upload',label:'Upload a file',description:'Audio or video'},{value:'record',label:'Record audio',description:'Use your microphone'}];
const fileTypes = 'audio/*,video/*,.wav,.flac,.mp3,.m4a,.mp4,.mov,.webm,.ogg';
export default function AudioStep({project,mode,setMode,demos,demo,setDemo,loadingDemos,onRetry,disabled,microphone,onExample,onUpload,onContinue}) {
  return <Section aria-label="Audio setup" className="workspace-stage"><SplitLayout sidebar={<Stack><Heading size="title">Audio source</Heading><ChoiceList label="Audio source" options={methods} value={mode} onChange={setMode} disabled={disabled || microphone.isBusy()}/></Stack>}>
    <div className="workspace-source" key={mode}>
      {mode === 'example' && <Stack gap="large"><Stack gap="small"><Heading>Choose a conversation</Heading><Text>Two overlapping voices, with a matching sample of the person to keep.</Text></Stack>
        {demos.length ? <><ExamplePicker demos={demos} value={demo} onChange={setDemo} disabled={disabled}/><Inline><Button variant={project.recording ? 'secondary' : 'primary'} disabled={disabled} onClick={onExample}>{project.recording ? 'Replace with this example' : 'Use this example'} <span aria-hidden="true">→</span></Button></Inline></> : <Inline><Text>{loadingDemos ? 'Loading examples…' : 'Examples could not load.'}</Text>{!loadingDemos && <Button onClick={onRetry}>Retry examples</Button>}</Inline>}
      </Stack>}
      {mode === 'upload' && <Stack gap="large"><Stack gap="small"><Heading>Upload your recording</Heading><Text>Choose the conversation you want to transcribe.</Text></Stack><Field label="Audio or video" hint="Up to 10 minutes · 128 MiB. Uploaded for processing only when you start."><input type="file" accept={fileTypes} disabled={disabled || microphone.isBusy()} onChange={event => {const file = event.target.files?.[0]; if (file) onUpload(file,file.name); event.target.value = '';}}/></Field></Stack>}
      {mode === 'record' && <Stack gap="large"><Stack gap="small"><Heading>Record a conversation</Heading><Text>Use your microphone. Choose the voice to keep next.</Text></Stack><AudioCaptureButton target="recording" recorder={microphone} disabled={disabled} label="Your conversation"/></Stack>}
    </div>
    {project.recording && <div className="workspace-current-recording"><div className="workspace-group-label">Current recording</div><RecordingPreview project={project}/><Button variant="primary" onClick={onContinue} disabled={disabled || microphone.isBusy()}>Continue with this audio <span aria-hidden="true">→</span></Button></div>}
  </SplitLayout></Section>;
}
