import assert from 'node:assert/strict';
import {animateChange} from '../src/motion.js';

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
console.log('Motion queue: batching, sequencing, callbacks, reduced motion and background updates passed.');
