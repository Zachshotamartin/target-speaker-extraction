import React from 'react';
import './ui.css';

// A reusable surface for content that needs grouping, not a page container.
export default function Card({as: Element = 'div', className = '', children, ...props}) {
  return <Element className={`ui-card ${className}`} {...props}>{children}</Element>;
}
