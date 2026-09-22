import {TRANSCRIPTION_API as API} from './transcriptionApi.js';
export const CHUNK = 1024 * 1024;
export async function request(path, options = {}) {
  const response = await fetch(`${API}${path}`, options);
  if (!response.ok) {
    const payload = await response.json().catch(() => ({}));
    const error = new Error(typeof payload.detail === 'string' ? payload.detail : `Service error (${response.status}). Try again.`);
    error.status = response.status; throw error;
  }
  return response;
}
export const json = (path, options) => request(path, options).then(r => r.status === 204 ? null : r.json());
export const post = (path, body, signal) => json(path, {method: 'POST', headers: {'Content-Type': 'application/json'}, body: JSON.stringify(body), signal});
export async function upload(blob, saved, onProgress, signal) {
  if (!blob?.size || blob.size > 128 * CHUNK) throw new Error('Choose a file smaller than 128 MiB.');
  let state;
  if (saved) state = await json(`/workspace/uploads/${saved}`, {signal}).catch(e => {if (e.status !== 404) throw e;});
  if (!state) state = await post('/workspace/uploads', {size: blob.size}, signal);
  if (state.size !== blob.size) throw new Error('The saved upload belongs to a different file.');
  onProgress(state);
  for (let index = state.chunks; index * CHUNK < blob.size; index++) {
    state = await json(`/workspace/uploads/${state.id}/chunks/${index}`, {method: 'POST', headers: {'Content-Type': 'application/octet-stream'}, body: blob.slice(index * CHUNK, (index + 1) * CHUNK), signal});
    onProgress(state);
  }
  return state.id;
}
export async function asset(job, name, signal) {
  const path = `/workspace/jobs/${job}/assets/${name}`;
  const info = await json(`${path}/info`, {signal});
  if (!Number.isInteger(info.bytes) || info.bytes <= 0 || info.bytes > 128 * CHUNK || info.chunks !== Math.ceil(info.bytes / CHUNK)) throw new Error('Invalid result size.');
  const chunks = [];
  for (let i = 0; i < info.chunks; i++) {
    const data = await request(`${path}/chunks/${i}`, {signal}).then(r => r.arrayBuffer());
    if (data.byteLength !== Math.min(CHUNK, info.bytes - i * CHUNK)) throw new Error('Download interrupted. Try again.');
    chunks.push(data);
  }
  return new Blob(chunks, {type: info.type});
}
