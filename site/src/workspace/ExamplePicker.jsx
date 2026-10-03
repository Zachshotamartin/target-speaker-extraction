import React from 'react';
import Field from '../ui/Field.jsx';
import ChoiceTabs from '../ui/ChoiceTabs.jsx';
export default function ExamplePicker({demos, value, onChange, disabled}) {
  const selected = demos.find(demo => demo.id === value) || demos[0];
  const conversations = [...new Set(demos.map(demo => demo.conversation))];
  const voices = demos.filter(demo => demo.conversation === selected.conversation);
  return <div className="example-picker">
    <Field label="Conversation"><select value={selected.conversation} disabled={disabled} onChange={event => {
      const candidates = demos.filter(demo => String(demo.conversation) === event.target.value);
      onChange((candidates.find(demo => demo.voice === selected.voice) || candidates[0]).id);
    }}>{conversations.map(number => <option key={number} value={number}>Conversation {number}</option>)}</select></Field>
    <div className="example-voice"><span>Voice to keep</span><ChoiceTabs label="Example voice" options={voices.map(demo => ({value: demo.id, label: `Voice ${demo.voice}`}))} value={selected.id} onChange={onChange} disabled={disabled}/></div>
  </div>;
}
