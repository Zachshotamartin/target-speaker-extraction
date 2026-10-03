import {useCallback, useEffect, useRef, useState} from 'react';
import {flushSync} from 'react-dom';

let queued = [], running = false, scheduled = false;
const reduced = () => window.matchMedia('(prefers-reduced-motion: reduce)').matches;

// Batch concurrent updates and finish the current transition before capturing the next.
// This keeps rapid clicks and asynchronous model stages from cutting animations short.
export function animateChange(update, scope, after) {
  enqueue({update, scope, after});
}

// Step changes animate only their content; header and step controls stay in place.
export function animateStepChange(update, after) {
  enqueue({update, after, scope: '.workspace-step-content', stepNavigation: true});
}

// Route changes also replace the document's height and scroll destination. Keep
// them out of document snapshots, which capture the footer's old position.
export function animatePageChange(update, after) {
  enqueue({update, after, pageNavigation: true});
}

function enqueue(change) {
  queued.push(change);
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
  const update = () => flushSync(() => batch.forEach(({update}) => update()));
  const position = () => batch.forEach(({after}) => after?.());
  const commit = () => { update(); position(); };
  try {
    if (!visible || reduced() || document.hidden) commit();
    else if (batch.some(({pageNavigation}) => pageNavigation)) {
      await fadePage(update, position);
    }
    else if (batch.some(({stepNavigation}) => stepNavigation)) {
      await fadeStep(update, position);
    }
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

async function fadeStep(update, position) {
  const panel = document.querySelector('.workspace-step-content');
  if (!panel) {update(); position(); return;}
  const before = panel.getBoundingClientRect().height;
  let outgoing, incoming, resize;
  try {
    outgoing = panel.animate([{opacity: 1, transform: 'translateX(0)'}, {opacity: 0, transform: 'translateX(-12px)'}], {duration: 120, fill: 'forwards'});
    await outgoing.finished.catch(() => {});
    update();
    position();
    const after = panel.getBoundingClientRect().height;
    resize = panel.animate([{height: `${before}px`, overflow: 'clip'}, {height: `${after}px`, overflow: 'clip'}], {duration: 300, easing: 'cubic-bezier(.22,1,.36,1)'});
    incoming = panel.animate([{opacity: 0, transform: 'translateX(16px)'}, {opacity: 1, transform: 'translateX(0)'}], {duration: 300, easing: 'cubic-bezier(.22,1,.36,1)'});
    outgoing.cancel();
    await Promise.all([incoming.finished.catch(() => {}), resize.finished.catch(() => {})]);
  } finally {
    outgoing?.cancel(); incoming?.cancel(); resize?.cancel();
  }
}

// Finish layout while the page is faded out, then position it before revealing
// it. The shared header/footer and mounted audio/job state are preserved.
async function fadePage(update, position) {
  const page = document.querySelector('main');
  if (!page) { update(); position(); return; }
  const root = document.documentElement;
  let outgoing, incoming;
  root.classList.add('page-navigation');
  try {
    outgoing = page.animate([{opacity: 1}, {opacity: 0}], {duration: 120, fill: 'forwards'});
    await outgoing.finished.catch(() => {});
    update();
    await layoutFrame();
    position();
    incoming = page.animate([{opacity: 0}, {opacity: 1}], {duration: 260, easing: 'cubic-bezier(.22,1,.36,1)'});
    outgoing.cancel();
    await incoming.finished.catch(() => {});
  } finally {
    outgoing?.cancel();
    incoming?.cancel();
    root.classList.remove('page-navigation');
  }
}

function layoutFrame() {
  if (document.hidden) return Promise.resolve();
  return new Promise(resolve => {
    let frame;
    const done = () => {
      cancelAnimationFrame(frame);
      document.removeEventListener('visibilitychange', hidden);
      resolve();
    };
    const hidden = () => { if (document.hidden) done(); };
    frame = requestAnimationFrame(done);
    document.addEventListener('visibilitychange', hidden);
  });
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

const popoverAnimations = new WeakMap();

export function setDetailsOpen(details, open, scope) {
  if (!details.classList.contains('ui-popover')) {
    animateChange(() => {details.open = open;}, scope);
    return;
  }
  const previous = popoverAnimations.get(details);
  previous?.animation.cancel();
  const body = details.querySelector('.ui-disclosure-body');
  if (!body || reduced() || document.hidden) {details.open = open; popoverAnimations.delete(details); return;}
  details.open = true;
  const frames = open
    ? [{opacity: 0, transform: 'translateY(-6px)'}, {opacity: 1, transform: 'translateY(0)'}]
    : [{opacity: 1, transform: 'translateY(0)'}, {opacity: 0, transform: 'translateY(-4px)'}];
  const animation = body.animate(frames, {duration: open ? 220 : 140, easing: 'cubic-bezier(.22,1,.36,1)', fill: 'forwards'});
  const state = {animation, open};
  popoverAnimations.set(details, state);
  animation.finished.then(() => {
    if (popoverAnimations.get(details) !== state) return;
    details.open = open; animation.cancel(); popoverAnimations.delete(details);
  }).catch(() => {});
}

export function animateDisclosure(event) {
  const summary = event.target.closest('summary');
  if (!summary || event.target.closest('a, button, input, select')) return;
  const details = summary.parentElement;
  if (details.tagName !== 'DETAILS') return;
  event.preventDefault();
  setDetailsOpen(details, !(popoverAnimations.get(details)?.open ?? details.open));
}
