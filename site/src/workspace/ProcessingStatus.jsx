import React from 'react';
import ProcessingIndicator from '../ui/ProcessingIndicator.jsx';
import Button from '../ui/Button.jsx';
import Text from '../ui/Text.jsx';
import {processingLabel} from './processingLabel.js';

export default function ProcessingStatus({message, active, uploading, canCancel, onCancel}) {
  return <div className="workspace-processing" role="status" aria-live="polite">
    <ProcessingIndicator active={active}/>
    <Text tone="ink" className="workspace-processing-label">{processingLabel(message)}</Text>
    {canCancel && <Button variant="quiet" onClick={onCancel}>{uploading ? 'Pause upload' : 'Cancel processing'}</Button>}
  </div>;
}
