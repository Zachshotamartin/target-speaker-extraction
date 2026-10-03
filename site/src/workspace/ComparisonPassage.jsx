import React from 'react';
const clock = time => `${Math.floor(time / 60)}:${String(Math.floor(time % 60)).padStart(2, '0')}`;
export default function ComparisonPassage({row, onPlay}) {
  return <section className="comparison-passage-row" aria-label={`${clock(row.start)} to ${clock(row.end)}`}><time>{clock(row.start)}–{clock(row.end)}</time>{[['original','Without One Voice'],['extracted','With One Voice']].map(([key,label]) => <div className="comparison-passage-cell" key={key}><span className="comparison-mobile-label">{label}</span>{row[key] ? <button type="button" className="comparison-text" aria-label={`Play ${label.toLowerCase()} from ${clock(row.start)}`} onClick={() => onPlay(key,row.start)}>{row[key]}</button> : <p className="comparison-silence">No speech transcribed.</p>}</div>)}</section>;
}
