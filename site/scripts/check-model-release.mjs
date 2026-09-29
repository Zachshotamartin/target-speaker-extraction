import assert from 'node:assert/strict';
import {readFile} from 'node:fs/promises';
import {createHash} from 'node:crypto';

const root = new URL('../', import.meta.url);
const load = async path => JSON.parse(await readFile(new URL(path, root), 'utf8'));
const best = await load('src/snapshot.json');
const latest = await load('src/snapshot-latest.json');
const progress = await load('public/assets/one-voice/training-progress.json');
assert.equal(best.checkpointSHA256, progress.bestFull.checkpoint_sha256);
assert.equal(best.step, progress.bestFull.step);
assert.equal(latest.step, progress.latestStep);
assert.ok(latest.step > progress.latestFull.step);
assert.equal(latest.validation, null, 'Do not attach an earlier validation score to newer weights');
const releaseAssets = await load('src/release-assets.json');
assert.deepEqual(await load('public' + releaseAssets.latestListening), latest);
assert.equal(best.validation.cases, 6000);
assert.equal(progress.bestMonitor.cases, 400);
assert.equal(progress.status, 'paused');
assert.ok(progress.completedEpochs < progress.plannedEpochs);
assert.deepEqual(best.items.map(i => i.caseId), latest.items.map(i => i.caseId));
assert.equal(best.items.length, 12);
const full = progress.history.filter(row => row.kind === 'full');
assert.equal(Math.max(...full.map(row => row.improvement)), best.validation.improvement);
assert.equal(full.at(-1).step, progress.latestFull.step);
for (const snapshot of [best, latest]) {
  for (const [name, digest] of Object.entries(snapshot.files)) {
    const bytes = await readFile(new URL(`public/assets/one-voice/${name}`, root));
    assert.equal(createHash('sha256').update(bytes).digest('hex'), digest, name);
    assert.equal(bytes.subarray(0, 4).toString(), 'RIFF');
  }
  for (const item of snapshot.items) {
    assert.equal(item.tracks.estimate.duration, item.tracks.mixture.duration);
    assert.equal(item.tracks.target.duration, item.tracks.mixture.duration);
    assert.ok(item.tracks.reference.duration >= 3);
    for (const track of Object.values(item.tracks)) assert.equal(track.peaks.length, 192);
  }
}
console.log('Best/latest identities, audio hashes, fixed cases, timing, and validation history verified.');
