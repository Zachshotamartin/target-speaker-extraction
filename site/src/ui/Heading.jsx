import React from 'react';
import './ui.css';

export default function Heading({as: Element = 'h2', size = 'section', className = '', children, ...props}) {
  return <Element className={`ui-heading ui-heading--${size} ${className}`} {...props}>{children}</Element>;
}
