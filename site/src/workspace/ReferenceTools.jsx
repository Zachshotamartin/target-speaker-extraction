import React from 'react';
import {useMotionState} from '../motion.js';
import Text from '../ui/Text.jsx';
import Stack from '../ui/Stack.jsx';
import Inline from '../ui/Inline.jsx';
import Button from '../ui/Button.jsx';
import Field from '../ui/Field.jsx';
import ChoiceTabs from '../ui/ChoiceTabs.jsx';
import AudioPlayer from '../ui/AudioPlayer.jsx';
import AudioCaptureButton from '../AudioCaptureButton.jsx';

const methods = [{value: 'find', label: 'Find a voice'}, {value: 'reference', label: 'Use a sample'}, {value: 'passage', label: 'Select a passage'}];
const clock = time => `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, '0')}`;

export default function ReferenceTools({project, disabled, serviceReady, microphone, savedProfiles, excerpt, setExcerpt, onDiscover, onReference, onExcerpt}) {
  const [method, setMethod] = useMotionState('find');
  const blocked = disabled || microphone.isBusy();
  return <Stack>
    <ChoiceTabs label="How to identify the voice" options={methods} value={method} onChange={setMethod} disabled={blocked}/>
    <div className="workspace-source" key={method}>
      {method === 'find' && <Stack>
        <Text>Find solo passages in your recording, then listen and choose the person you want.</Text>
        <Inline><Button disabled={blocked || !serviceReady} onClick={onDiscover}>Find voices in recording</Button></Inline>
        {!!project.candidates.length && <Stack className="voice-candidates"><Text size="small">These are suggested samples, not verified identities. Choose a sample where one person speaks alone.</Text>{project.candidates.map(c => <Stack gap="small" key={c.id}>
          <Text tone="ink">{c.label} · {clock(c.start)}</Text><AudioPlayer blob={c.blob} label={`${c.label} suggested sample`}/><Inline><Button disabled={blocked || project.references.length >= 4} onClick={() => onReference(c.blob, c.label)}>Use this voice</Button></Inline>
        </Stack>)}</Stack>}
      </Stack>}
      {method === 'reference' && <Stack>
        <Text>A voice reference is a separate 3–10 second clip of the same person speaking alone. It tells OneVoice who to keep.</Text>
        <Field label="Upload voice reference" hint="Audio up to 4 MiB."><input type="file" accept="audio/*" disabled={blocked} onChange={e => {onReference(e.target.files?.[0], e.target.files?.[0]?.name || 'Voice'); e.target.value = '';}}/></Field>
        <AudioCaptureButton target="reference" minimum={3} maximum={10} recorder={microphone} disabled={disabled} label="Voice reference"/>
        {!!savedProfiles.length && <Field label="Or use a saved voice"><select disabled={blocked} defaultValue="" onChange={e => {const profile = savedProfiles.find(p => p.id === e.target.value); if (profile) onReference(profile.audio, profile.name); e.target.value = '';}}><option value="">Choose a saved voice</option>{savedProfiles.map(p => <option value={p.id} key={p.id}>{p.name}</option>)}</select></Field>}
      </Stack>}
      {method === 'passage' && <Stack>
        <Text>Listen to your recording and choose 3–10 seconds containing only the person you want.</Text>
        <div className="cut-boundaries"><Field label="Start (seconds)"><input disabled={blocked} type="number" step="0.1" min="0" value={excerpt.start} onChange={e => setExcerpt({...excerpt, start: Number(e.target.value)})}/></Field><Field label="End (seconds)"><input disabled={blocked} type="number" step="0.1" min="3" value={excerpt.end} onChange={e => setExcerpt({...excerpt, end: Number(e.target.value)})}/></Field><Button disabled={blocked} onClick={onExcerpt}>Use this passage</Button></div>
      </Stack>}
    </div>
  </Stack>;
}
