import React from 'react';
import './ui.css';

export default function Text({as: Element = 'p', size = 'body', tone = 'muted', className = '', children, ...props}) {
  return <Element className={`ui-text ui-text--${size} ui-text--${tone} ${className}`} {...props}>{children}</Element>;
}
