import React from 'react';
import Heading from '../ui/Heading.jsx';
import Text from '../ui/Text.jsx';
import Stack from '../ui/Stack.jsx';
const views = [{id:'selected',label:'Transcript',description:'Words matched to your voice'}, {id:'compare',label:'Compare',description:'Original and isolated speech'}, {id:'edit',label:'Edit audio',description:'Cut, correct and export'}];
export default function ResultNavigation({value, onChange, name, duration, words, children}) {
  return <Stack gap="large"><Stack gap="small"><Heading size="title">{name || 'Selected voice'}</Heading><Text size="small">{duration} audio · {words} matched words</Text></Stack>
    <div className="result-navigation" role="tablist" aria-label="Result views" aria-orientation="vertical">{views.map((view,index) => <button type="button" key={view.id} id={`result-tab-${view.id}`} role="tab" aria-selected={value === view.id} aria-controls={`result-${view.id}`} tabIndex={value === view.id ? 0 : -1} onClick={() => onChange(view.id)} onKeyDown={event => {
      const next = ['ArrowDown','ArrowRight'].includes(event.key) ? (index + 1) % views.length : ['ArrowUp','ArrowLeft'].includes(event.key) ? (index + views.length - 1) % views.length : event.key === 'Home' ? 0 : event.key === 'End' ? views.length - 1 : null;
      if (next !== null) {event.preventDefault(); onChange(views[next].id); document.getElementById(`result-tab-${views[next].id}`).focus();}
    }}><span>{view.label}</span><small>{view.description}</small></button>)}</div>{children}
  </Stack>;
}
