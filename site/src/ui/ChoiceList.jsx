import React from 'react';
import './ui.css';
export default function ChoiceList({label, options, value, onChange, disabled}) {
  return <div className="ui-choice-list" role="group" aria-label={label}>{options.map(option => <button key={option.value} type="button" disabled={disabled} aria-pressed={value === option.value} onClick={() => onChange(option.value)}><span>{option.label}</span><small>{option.description}</small><span className="ui-choice-arrow" aria-hidden="true">↗</span></button>)}</div>;
}
