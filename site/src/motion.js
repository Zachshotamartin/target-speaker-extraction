import {useCallback, useEffect, useRef, useState} from 'react';
import {flushSync} from 'react-dom';

let queued = [], running = false, scheduled = false;
const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Batch concurrent updates and finish the current transition before capturing the next.
// This keeps rapid clicks and asynchronous model stages from cutting animations short.
export function animateChange(update, scope, after) {
  queued.push({update, scope, after});
  if (!running && !scheduled) {
    scheduled = true;
    queueMicrotask(run);
  }
}

async function run() {
  scheduled = false;
  if (running || !queued.length) return;
  running = true;
  const batch = queued.splice(0);
  const visible = batch.some(({scope}) => !scope || document.querySelector(scope)?.getClientRects().length);
  const commit = () => {
    flushSync(() => batch.forEach(({update}) => update()));
    batch.forEach(({after}) => after?.());
  };
  try {
    if (!visible || reduced() || document.hidden) commit();
    else if (document.startViewTransition) {
      const transition = document.startViewTransition(commit);
      // A browser can skip a visual transition; the update still must complete.
      transition.ready.catch(() => {});
      await transition.finished.catch(() => {});
    } else {
      const page = document.querySelector('main');
      if (page) await page.animate([{opacity: 1}, {opacity: 0}], {duration: 120, fill: 'forwards'}).finished;
      commit();
      if (page) {
        page.getAnimations().forEach(animation => animation.cancel());
        await page.animate([{opacity: 0, transform: 'translateY(6px)'}, {opacity: 1, transform: 'translateY(0)'}], {duration: 260, easing: 'cubic-bezier(.22,1,.36,1)'}).finished;
      }
    }
  } finally {
    running = false;
    if (queued.length) { scheduled = true; queueMicrotask(run); }
  }
}

export function useMotionState(initial, scope = '#transcribe') {
  const [value, setValue] = useState(initial);
  const alive = useRef(true);
  const desired = useRef(value);
  useEffect(() => { alive.current = true; return () => { alive.current = false; }; }, []);
  const set = useCallback(next => {
    const resolved = typeof next === 'function' ? next(desired.current) : next;
    if (Object.is(desired.current, resolved)) return;
    desired.current = resolved;
    animateChange(() => { if (alive.current) setValue(resolved); }, scope);
  }, [scope]);
  return [value, set];
}

export function animateDisclosure(event) {
  const summary = event.target.closest('summary');
  if (!summary || event.target.closest('a, button, input, select')) return;
  const details = summary.parentElement;
  if (details.tagName !== 'DETAILS') return;
  event.preventDefault();
  animateChange(() => { details.open = !details.open; });
}
