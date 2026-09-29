import progress from './training-progress.json';
import './training-progress.css';

export default function TrainingProgress() {
  return <section className="ov-training" aria-labelledby="ov-training-title">
    <div className="study-section-heading">
      <h2 id="ov-training-title">Training progress</h2>
      <p>Snapshot · September 29, 2026</p>
    </div>
    <p>Paused after {progress.completedEpochs} of {progress.plannedEpochs} planned epochs. The saved run can resume. Live processing uses the best full-validation checkpoint, from epoch {progress.bestFull.epoch}.</p>
    <dl className="ov-training-metrics">
      <div><dt>Best full validation · 6,000 requests</dt><dd>{progress.bestFull.mean_si_sdri_db.toFixed(2)} <span>dB</span></dd></div>
      <div><dt>Latest full validation · epoch {progress.latestFull.epoch}</dt><dd>{progress.latestFull.mean_si_sdri_db.toFixed(2)} <span>dB</span></dd></div>
      <div><dt>Best progress monitor · 400 requests</dt><dd>{progress.bestMonitor.mean_si_sdri_db.toFixed(2)} <span>dB</span></dd></div>
    </dl>
    <figure className="ov-training-figure">
      <img src="/assets/one-voice/training-progress.svg" width="1000" height="430" loading="lazy" alt="Separation improves early and levels off near 11.8 dB. Full validation reaches its best at epoch 80, then falls slightly at epoch 81. The planned schedule extends to epoch 100."/>
      <figcaption>Higher SI-SDR improvement means closer separation against the clean target. These are development results on 40 speakers excluded from training, not a held-out test or transcription accuracy.</figcaption>
    </figure>
    <p>The 400-request monitor and full validation are different suites. The latest saved model is at update {progress.latestStep.toLocaleString('en-US')}, nine updates after the last full validation. Its listening examples are current; that exact checkpoint has no full-suite score.</p>
    <p>Training used {progress.trainingMixtures.toLocaleString('en-US')} Libri2Mix mixtures and {progress.trainingSpeakers} speakers. Both listening versions use the same six conversations, chosen independently of their scores.</p>
    <a href="/assets/one-voice/training-progress.json" download>Download training results</a>
  </section>;
}
