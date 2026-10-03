import React from 'react';
import './ui.css';

export default function Stack({as: Element = 'div', gap = 'medium', className = '', children, ...props}) {
  return <Element className={`ui-stack ui-gap--${gap} ${className}`} {...props}>{children}</Element>;
}
