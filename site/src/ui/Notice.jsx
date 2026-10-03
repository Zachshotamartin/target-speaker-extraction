import React from 'react';
import './ui.css';

export default function Notice({error = false, children}) {
  return <div className={`ui-notice${error ? ' ui-notice--error' : ''}`} role={error ? 'alert' : 'status'}>{children}</div>;
}
