import React from 'react';
import Popover from '../ui/Popover.jsx';
import Text from '../ui/Text.jsx';
import Stack from '../ui/Stack.jsx';
export default function WorkspacePrivacy() {
  return <Popover label="Privacy & storage" className="workspace-privacy"><Stack gap="small">
    <Text size="small">Audio is never used for training. Processing sends your selected files to the service.</Text>
    <Text size="small">Recordings and edits are saved in this browser, not a cloud account. Download important work before clearing browser storage.</Text>
    <Text size="small">Incomplete uploads expire after an hour and completed server results after 15 minutes. Only use recordings you have permission to process.</Text><a href="#privacy">Read the privacy policy</a>
  </Stack></Popover>;
}
