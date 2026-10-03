import React from 'react';
import './ui.css';

export default function StepNavigation({steps, current, onChange}) {
  return <nav className="ui-steps" aria-label="Transcription steps"><ol>
    {steps.map((step, index) => <li key={step.id}><button type="button" disabled={step.disabled}
      aria-current={current === step.id ? 'step' : undefined} onClick={() => onChange(step.id)}>
      <span className="ui-step-number">{index + 1}</span><span>{step.label}</span>
    </button></li>)}
  </ol></nav>;
}
