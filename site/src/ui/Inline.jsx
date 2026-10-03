import React from 'react';
import './ui.css';

export default function Inline({as: Element = 'div', gap = 'small', className = '', children, ...props}) {
  return <Element className={`ui-inline ui-gap--${gap} ${className}`} {...props}>{children}</Element>;
}
