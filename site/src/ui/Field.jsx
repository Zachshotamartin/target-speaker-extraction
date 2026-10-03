import React from 'react';
import './ui.css';

export default function Field({label, hint, className = '', children}) {
  return <label className={`ui-field ${className}`}><span>{label}</span>{children}{hint && <small>{hint}</small>}</label>;
}
