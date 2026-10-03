import React from 'react';
const clock = time => `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, '0')}`;

export default function TranscriptLines({segments, seek, empty}) {
  return segments.length ? <ol className="transcript-lines">{segments.map((segment, index) =>
    <li key={index}><button type="button" onClick={() => seek(segment.start)} aria-label={`Play from ${clock(segment.start)}`}>
      <time>{clock(segment.start)}</time><span>{segment.text.trim()}</span>
    </button></li>)}</ol> : <p className="transcript-empty">{empty}</p>;
}
