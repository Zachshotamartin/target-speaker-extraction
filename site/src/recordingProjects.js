// Audio and edits stay in this browser's IndexedDB; no account or cloud library.
export function projects(action, value) {
  return new Promise((resolve, reject) => {
    const open = indexedDB.open('one-voice-projects', 1);
    open.onupgradeneeded = () => {open.result.createObjectStore('projects', {keyPath: 'id'}); open.result.createObjectStore('index', {keyPath: 'id'});};
    open.onerror = () => reject(new Error('Local storage is unavailable. Export your audio to keep it.'));
    open.onsuccess = () => {
      const db = open.result, tx = db.transaction(['projects', 'index'], ['list', 'get'].includes(action) ? 'readonly' : 'readwrite');
      let request;
      if (action === 'list') request = tx.objectStore('index').getAll();
      if (action === 'get') request = tx.objectStore('projects').get(value);
      if (action === 'save') {
        request = tx.objectStore('projects').put(value);
        tx.objectStore('index').put({id: value.id, name: value.name, updated: value.updated, tracks: value.tracks?.length || 0});
      }
      if (action === 'delete') {request = tx.objectStore('projects').delete(value); tx.objectStore('index').delete(value);}
      tx.oncomplete = () => {db.close(); resolve(request.result);};
      tx.onabort = tx.onerror = () => {db.close(); reject(new Error('Project could not be saved. Browser storage may be full; download your work.'));};
    };
  });
}
export const newProject = () => ({id: crypto.randomUUID(), name: 'Untitled recording', updated: Date.now(), references: [], candidates: [], tracks: [], edits: {}, uploads: {}});
