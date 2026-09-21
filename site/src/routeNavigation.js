import {isPageRoot} from './pageRoute.js';

export function scrollToRoute(hash, smooth = false) {
  const target = hash && document.getElementById(hash.slice(1));
  const behavior = smooth && !window.matchMedia('(prefers-reduced-motion: reduce)').matches ? 'smooth' : 'instant';
  if (target && !isPageRoot(hash)) {
    target.scrollIntoView({block: 'start', behavior});
    return;
  }
  // A shared footer survives page changes. Move focus off its link before
  // positioning the new page so focus/keyboard navigation cannot pull us back down.
  document.getElementById('main-content')?.focus({preventScroll: true});
  window.scrollTo({top: 0, left: 0, behavior});
}

export function ownScrollRestoration() {
  const previous = window.history.scrollRestoration;
  window.history.scrollRestoration = 'manual';
  return () => { window.history.scrollRestoration = previous; };
}
