import React from 'react';
import './ui.css';

export default function Disclosure({label, className = '', children, ...props}) {
  return <details className={`ui-disclosure ${className}`} {...props}>
    <summary>{label}<svg viewBox="0 0 24 24" aria-hidden="true"><path d="m6 9 6 6 6-6"/></svg></summary>
    <div className="ui-disclosure-body">{children}</div>
  </details>;
}
