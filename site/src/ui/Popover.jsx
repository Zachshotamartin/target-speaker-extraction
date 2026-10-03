import React, {useRef} from 'react';
import Disclosure from './Disclosure.jsx';
import Card from './Card.jsx';
import {useDismissibleDetails} from '../useDismissibleDetails.js';
import './ui.css';

export default function Popover({ref, label, className = '', children}) {
  const internalRef = useRef(null);
  const details = ref || internalRef;
  useDismissibleDetails(details, '.ui-popover');
  return <Disclosure ref={details} label={label} className={`ui-popover ${className}`}>
    <Card>{children}</Card>
  </Disclosure>;
}
