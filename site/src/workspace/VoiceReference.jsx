import React from 'react';
import AudioPlayer from '../ui/AudioPlayer.jsx';
import Popover from '../ui/Popover.jsx';
import Field from '../ui/Field.jsx';
import Button from '../ui/Button.jsx';
import Stack from '../ui/Stack.jsx';
export default function VoiceReference({reference, index, disabled, onRename, onRemove, onSave}) {
  return <div className="voice-reference"><span className="voice-reference-mark" aria-hidden="true">{index + 1}</span><div className="voice-reference-content">
    <div className="voice-reference-heading"><div><h3>{reference.label}</h3><p>Voice sample · one speaker</p></div><Popover label="Options" className="voice-options"><Stack gap="small">
      <Field label="Voice name"><input aria-label={`Voice ${index + 1} name`} value={reference.label} maxLength={60} disabled={disabled} onChange={event => onRename(reference.id, event.target.value)}/></Field>
      <Button disabled={disabled} onClick={() => onSave(reference)}>Save voice for later</Button><Button variant="quiet" disabled={disabled} onClick={() => onRemove(reference.id)}>Remove voice</Button>
    </Stack></Popover></div><AudioPlayer blob={reference.blob} label={`Voice reference ${index + 1}`}/></div></div>;
}
