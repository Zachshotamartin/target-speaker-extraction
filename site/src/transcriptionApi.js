const local = ['localhost', '127.0.0.1', '[::1]'].includes(globalThis.location?.hostname);
export const TRANSCRIPTION_API = local ? '/api/poc' : '/api/poc?route=';
