import React from 'react';
import Section from '../ui/Section.jsx';
import Heading from '../ui/Heading.jsx';
import Text from '../ui/Text.jsx';
import Stack from '../ui/Stack.jsx';
import Inline from '../ui/Inline.jsx';
import Button from '../ui/Button.jsx';
import ChoiceTabs from '../ui/ChoiceTabs.jsx';
import TranscriptionResults from '../TranscriptionResults.jsx';

export default function ResultsStep({project, voice, setVoice, processing, exportBusy, onExport, onBack, onVideo, onVideoDownload, onEdit, onDelete, onNotice, onError, active}) {
  const selected = project.tracks[voice] || project.tracks[0];
  return <Section aria-label="Transcription results" className="workspace-results"><Stack gap="large">
    {!selected && <Heading id="results-step-title">{processing ? 'Processing your recording' : 'Ready to try again'}</Heading>}
    {project.tracks.length > 1 && <Inline className="recording-result-heading"><Heading id="results-step-title" size="title">Your tracks</Heading><Button disabled={exportBusy} onClick={onExport}>{exportBusy ? 'Preparing tracks…' : 'Download all tracks (.zip)'}</Button></Inline>}
    {!selected ? <Stack><Text>{processing ? 'OneVoice separates the voice, then Whisper transcribes it. Longer recordings take more time. Your project is saved so you can return to it.' : 'Processing did not finish. Your audio and voice samples are still here; return to voice selection to try again.'}</Text>{!processing && <Inline><Button onClick={onBack}>Return to voice selection</Button></Inline>}</Stack> : <>
      {project.tracks.length > 1 && <ChoiceTabs label="Speaker tracks" value={voice} onChange={setVoice} options={project.tracks.map((track, index) => ({value: index, label: track.label}))}/>}
      <div><TranscriptionResults notice={project.notice} key={`${project.id}-${selected.id}-${selected.result.processing_seconds}`} result={selected.result} referenceName={selected.label} jobId={null} active={active}
        audioFiles={{original: project.original, extracted: selected.audio}} savedEdit={project.edits[selected.id]} onEdit={edit => onEdit(selected.id, edit)}
        onRenderVideo={project.video ? options => onVideo({...options, trackId: selected.id}) : undefined} videoBusy={processing}
        onNotice={onNotice} onError={onError} onDelete={() => onDelete(selected.id)}/></div>
    </>}
    {project.videoExport && <Inline><Button onClick={onVideoDownload}>Download last captioned video</Button></Inline>}
  </Stack></Section>;
}
