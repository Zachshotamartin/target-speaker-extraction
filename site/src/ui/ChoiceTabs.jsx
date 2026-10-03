import React from 'react';
import './ui.css';

export default function ChoiceTabs({label, options, value, onChange, disabled = false}) {
  return <div className="ui-choice-tabs" role="group" aria-label={label}>
    {options.map(option => <button type="button" key={option.value} aria-pressed={value === option.value}
      disabled={disabled} onClick={() => onChange(option.value)}>{option.label}</button>)}
  </div>;
}
