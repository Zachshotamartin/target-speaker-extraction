// Only an explicit Save action writes reference audio. Jobs and transcripts are ephemeral.
export async function profiles(action, value) {
  return new Promise((resolve, reject) => {
    const opening = indexedDB.open('one-voice-profiles', 1);
    opening.onupgradeneeded = () => opening.result.createObjectStore('profiles', {keyPath: 'id'});
    opening.onerror = () => reject(new Error('Local profile storage is unavailable. You can still use a temporary reference.'));
    opening.onsuccess = () => {
      const db = opening.result;
      const transaction = db.transaction('profiles', action === 'list' ? 'readonly' : 'readwrite');
      const store = transaction.objectStore('profiles');
      const request = action === 'list' ? store.getAll() : action === 'save' ? store.put(value) : store.delete(value);
      transaction.oncomplete = () => { db.close(); resolve(request.result); };
      transaction.onerror = () => { db.close(); reject(new Error('Could not update local voice profiles.')); };
    };
  });
}
