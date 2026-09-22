export function pageForHash(hash) {
  if (hash === '#transcribe') return 'transcribe';
  if (hash === '#privacy' || hash.startsWith('#privacy-')) return 'privacy';
  if (hash === '#terms' || hash.startsWith('#terms-')) return 'terms';
  return 'overview';
}

export function isPageRoot(hash) {
  return !hash || hash === '#' || hash === '#transcribe' || hash === '#privacy' || hash === '#terms';
}
