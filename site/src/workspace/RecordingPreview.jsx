import React from 'react';
import AudioPlayer from '../ui/AudioPlayer.jsx';
import Text from '../ui/Text.jsx';
import Stack from '../ui/Stack.jsx';
import {useAudioUrl} from '../useAudioUrl.js';

export default function RecordingPreview({project}) {
  const url = useAudioUrl(project.recording);
  return <Stack gap="tight" className="recording-preview">
    <Text tone="ink" className="recording-filename">{project.name}</Text>
    {project.video ? <video src={url || undefined} controls preload="metadata" aria-label="Imported video"/> : <AudioPlayer blob={project.recording} label="Your recording"/>}
  </Stack>;
}
