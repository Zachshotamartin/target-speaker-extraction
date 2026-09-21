export function pageForHash(hash) {
  if (hash === '#transcribe') return 'transcribe';
  if (hash === '#privacy' || hash.startsWith('#privacy-')) return 'privacy';
  return 'overview';
}

export function isPageRoot(hash) {
  return !hash || hash === '#' || hash === '#transcribe' || hash === '#privacy';
}
