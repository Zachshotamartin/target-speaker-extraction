import AudioPlayer from '../ui/AudioPlayer.jsx';
import React from 'react';
import Heading from '../ui/Heading.jsx';
import Popover from '../ui/Popover.jsx';
import Text from '../ui/Text.jsx';
import ComparisonPassage from './ComparisonPassage.jsx';
const variants = [['original','Without One Voice'],['extracted','With One Voice']];
export default function ComparisonView({result, rows, audioProps, originalUrl, extractedUrl, playFrom, copy}) {
  if (!result.comparison) return <Text>{result.outcome === 'no_speech' ? 'No speech was detected in the recording.' : 'No comparison is available for this run.'}</Text>;
  return <div className="comparison-workbench"><div className="result-view-heading"><Heading>Compare transcripts</Heading><Popover label="About this comparison"><Text size="small">Both versions use the same Whisper settings. These transcripts are shown before speaker filtering; the Transcript view contains the matched words used for exports. Playback positions stay linked when you switch audio.</Text></Popover></div>
    <div className="comparison-columns">{variants.map(([key,label]) => <section key={key} aria-label={label} className="comparison-column">
      <div className="comparison-column-heading"><h3>{label}</h3><button className="poc-text-button" type="button" aria-label={`Copy ${label.toLowerCase()}`} onClick={() => copy(key === 'original' ? result.comparison.raw.text : result.comparison.one_voice.text, label)}>Copy</button></div>
      <AudioPlayer {...audioProps(key)} src={key === 'original' ? originalUrl : extractedUrl} label={`${label} audio`}/>
    </section>)}</div><div className="comparison-passages">{rows.map(row => <ComparisonPassage key={row.start} row={row} onPlay={playFrom}/>)}</div><Text size="small">Click a passage to listen.</Text>
  </div>;
}
