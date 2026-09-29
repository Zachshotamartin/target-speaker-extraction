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

const diagnostics = await load('src/validation-diagnostics.json');
const evidence = await load('public' + diagnostics.downloads.report);
assert.equal(evidence.best_step, best.step);
assert.equal(evidence.latest_step, progress.latestFull.step);
assert.ok(evidence.latest_step < latest.step, 'Diagnostics describe validated weights, not the later saved checkpoint');
assert.equal(evidence.paired.length, 6000);
assert.equal(new Set(evidence.paired.map(row => row.case_id)).size, 6000);
assert.equal(evidence.speakers.length, 40);
assert.equal(evidence.speakers.reduce((total, row) => total + row.cases, 0), 6000);
assert.deepEqual(evidence.summary, diagnostics.summary);
assert.equal(evidence.summary.best.mean, best.validation.improvement);
assert.equal(evidence.summary.best.improved / 6000, best.validation.positiveFraction);
assert.equal(evidence.summary.wrong_speaker_cases / 6000, best.validation.confusionFraction);
assert.equal(evidence.summary.paired.improved_over_1db + evidence.summary.paired.within_1db + evidence.summary.paired.worsened_over_1db, 6000);
assert.equal(evidence.failures.at(-1).step, progress.latestFull.step);
for (const path of [...Object.values(diagnostics.charts), ...Object.values(diagnostics.downloads)]) {
  const bytes = await readFile(new URL('public' + path, root));
  const hash = createHash('sha256').update(bytes).digest('hex').slice(0, 12);
  assert.ok(path.includes(`-${hash}.`), `${path}: asset hash mismatch`);
}
console.log('Diagnostic cohorts, paired cases, failure counts, source identity, and chart/data hashes verified.');
