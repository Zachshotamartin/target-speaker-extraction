import assert from 'node:assert/strict';
import {ownScrollRestoration, scrollToRoute} from '../src/routeNavigation.js';

let focused = 'footer', top = 4000, reduced = false, events = [];
const main = {focus(options) {assert.equal(options.preventScroll, true); focused = 'main'; events.push('focus');}};
const section = {scrollIntoView(options) {events.push(['section', options]); top = 1200;}};
globalThis.document = {getElementById: id => id === 'main-content' ? main : id === 'privacy-storage' ? section : id === 'transcribe' ? section : null};
globalThis.window = {
  history: {scrollRestoration: 'auto'},
  matchMedia: () => ({matches: reduced}),
  scrollTo(options) {events.push(['scroll', options]); top = options.top; assert.equal(focused, 'main', 'Move focus off the shared footer before scrolling');},
};
const release = ownScrollRestoration();
assert.equal(window.history.scrollRestoration, 'manual', 'History traversal must not restore a stale footer position');
for (const hash of ['#privacy', '#transcribe', '', '#', '#unknown']) {
  focused = 'footer'; top = 4000; events = [];
  scrollToRoute(hash);
  assert.equal(top, 0, `${hash} opens at the top`);
  assert.deepEqual(events, ['focus', ['scroll', {top: 0, left: 0, behavior: 'instant'}]]);
}
events = []; scrollToRoute('#privacy-storage', true);
assert.equal(top, 1200, 'Privacy section links still navigate to their section');
assert.deepEqual(events, [['section', {block: 'start', behavior: 'smooth'}]]);
reduced = true; events = []; scrollToRoute('#privacy-storage', true);
assert.equal(events[0][1].behavior, 'instant');
release(); assert.equal(window.history.scrollRestoration, 'auto', 'Restore browser ownership when the app unmounts');
console.log('Page navigation: root focus and top position, history restoration, section links and reduced motion passed.');
