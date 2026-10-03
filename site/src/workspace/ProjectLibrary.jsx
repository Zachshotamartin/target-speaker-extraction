import React, {useRef} from 'react';
import {setDetailsOpen} from '../motion.js';
import Button from '../ui/Button.jsx';
import Popover from '../ui/Popover.jsx';
import Field from '../ui/Field.jsx';
import Text from '../ui/Text.jsx';
import Stack from '../ui/Stack.jsx';
import Inline from '../ui/Inline.jsx';

export default function ProjectLibrary({project, library, saveStatus, disabled, onRename, onOpen, onDelete}) {
  const menu = useRef(null);
  async function open(id) {
    await onOpen(id);
    if (menu.current) setDetailsOpen(menu.current, false, '.project-library');
  }
  return <Popover ref={menu} label={`Saved projects (${library.length})`} className="project-library">
    <Stack>
      {project.recording && <Field label="Current project name"><input value={project.name} maxLength={100} onChange={e => onRename(e.target.value)}/></Field>}
      <Text size="small">{saveStatus || 'Projects are saved in this browser. Download important work before clearing browser storage.'}</Text>
      {library.length ? <ul>{library.map(item => <li key={item.id}>
        <Inline><Button disabled={disabled} onClick={() => open(item.id)}>{item.name}</Button><Text as="span" size="small">{item.tracks} tracks</Text></Inline>
        <Button variant="quiet" disabled={disabled} aria-label={`Delete saved project ${item.name}`} onClick={() => onDelete(item.id)}>Delete</Button>
      </li>)}</ul> : <Text>Your saved projects will appear here.</Text>}
    </Stack>
  </Popover>;
}
