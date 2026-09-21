import assert from 'node:assert/strict';
import {animateChange, animatePageChange} from '../src/motion.js';

let reduced = false;
globalThis.window = {matchMedia: () => ({matches: reduced})};
let transitions = [], effects = [];
globalThis.document = {
  hidden: false,
  querySelector: () => ({getClientRects: () => [1]}),
  startViewTransition(update) {
    update();
    let finish;
    const finished = new Promise(resolve => {finish = resolve;});
    const transition = {ready: Promise.resolve(), finished, finish};
    transitions.push(transition);
    return transition;
  },
};
const tick = () => new Promise(resolve => setTimeout(resolve, 0));

animateChange(() => effects.push('a'));
animateChange(() => effects.push('b'), undefined, () => effects.push('after'));
await tick();
assert.deepEqual(effects, ['a', 'b', 'after']);
assert.equal(transitions.length, 1, 'same-turn updates share a capture');
animateChange(() => effects.push('c'));
await tick();
assert.deepEqual(effects, ['a', 'b', 'after'], 'active animation must finish before the next update');
transitions[0].finish();
await tick();
assert.equal(transitions.length, 2);
assert.equal(effects.at(-1), 'c');
transitions[1].finish();
await tick();

reduced = true;
animateChange(() => effects.push('reduced'));
await tick();
assert.equal(transitions.length, 2, 'reduced motion bypasses captures');
assert.equal(effects.at(-1), 'reduced');
reduced = false;
document.hidden = true;
animateChange(() => effects.push('background'));
await tick();
assert.equal(transitions.length, 2, 'background tabs update without animation');
assert.equal(effects.at(-1), 'background');

// A route must settle its new layout and scroll position before fading in.
// Unlike a component change, it must never capture the old footer in a VT.
document.hidden = false;
let scrollY = 4000, frames = [], fades = [], routeEvents = [];
const classes = new Set();
document.documentElement = {classList: {add: name => classes.add(name), remove: name => classes.delete(name)}};
document.addEventListener = () => {};
document.removeEventListener = () => {};
globalThis.requestAnimationFrame = callback => {frames.push(callback); return frames.length;};
globalThis.cancelAnimationFrame = () => {};
const shell = {
  animate(keyframes) {
    const entering = keyframes[0].opacity === 0;
    if (entering) assert.equal(scrollY, 0, 'The new page must be at the top before it becomes visible');
    routeEvents.push(entering ? 'reveal' : 'hide');
    let finish;
    const finished = new Promise(resolve => {finish = resolve;});
    const fade = {finished, finish, cancel() {}};
    fades.push(fade);
    return fade;
  },
};
document.querySelector = selector => selector === '.app-shell' ? shell : {getClientRects: () => [1]};
animatePageChange(() => {routeEvents.push('render'); scrollY = 5000;}, () => {routeEvents.push('position'); scrollY = 0;});
await tick();
assert.deepEqual(routeEvents, ['hide']);
assert.ok(classes.has('page-navigation'));
assert.equal(transitions.length, 2, 'Page changes do not use a document view transition');
fades[0].finish(); await tick();
assert.deepEqual(routeEvents, ['hide', 'render']);
assert.equal(frames.length, 1, 'Wait for the replacement page layout');
frames.shift()(); await tick();
assert.deepEqual(routeEvents, ['hide', 'render', 'position', 'reveal']);
animateChange(() => effects.push('queued-during-navigation'));
await tick();
assert.notEqual(effects.at(-1), 'queued-during-navigation', 'Component transitions wait until page navigation finishes');
fades[1].finish(); await tick();
assert.equal(effects.at(-1), 'queued-during-navigation');
assert.ok(!classes.has('page-navigation'), 'Release scroll anchoring after navigation');
transitions[2].finish(); await tick();
reduced = true;
animatePageChange(() => effects.push('reduced-navigation'), () => {scrollY = 0;});
await tick();
assert.equal(effects.at(-1), 'reduced-navigation');
assert.equal(fades.length, 2, 'Reduced motion skips navigation fades');
console.log('Motion queue: batching, sequencing, callbacks, navigation layout/scroll before reveal, reduced motion and background updates passed.');
