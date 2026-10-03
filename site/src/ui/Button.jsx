import React from 'react';
import './ui.css';

export default function Button({variant = 'secondary', className = '', children, ...props}) {
  return <button type="button" className={`ui-button ui-button--${variant} ${className}`} {...props}>{children}</button>;
}
