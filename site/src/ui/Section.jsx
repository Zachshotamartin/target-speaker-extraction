import React from 'react';
import './ui.css';

export default function Section({as: Element = 'section', spacing = 'none', className = '', children, ...props}) {
  return <Element className={`ui-section ui-section--${spacing} ${className}`} {...props}>{children}</Element>;
}
