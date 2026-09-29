import {useMotionState} from './motion.js';
import diagnostics from './validation-diagnostics.json';

const options = [
  {id: 'distribution', label: 'Quality spread'},
  {id: 'failures', label: 'Failure rates'},
  {id: 'speakers', label: 'By speaker'},
  {id: 'paired', label: 'Best vs latest'},
];
const percent = (count, total) => `${(100 * count / total).toFixed(1)}%`;
const number = value => value.toLocaleString('en-US');

function descriptions() {
  const s = diagnostics.summary;
  return {
    distribution: {
      title: 'The spread behind the average',
      finding: `The best model’s median is ${s.best.median.toFixed(2)} dB. The weakest 10% of requests score ${s.best.p10.toFixed(2)} dB or less; ${number(s.best.no_improvement)} of ${number(s.cases)} requests (${percent(s.best.no_improvement, s.cases)}) do not improve over the original mixture.`,
      caption: 'Each bar covers a 2 dB range across all validation requests, including negative results. The outline shows the latest full validation. Further right means better separation.',
      alt: 'Histogram comparing the complete separation-improvement distributions at epochs 80 and 81, including the negative tail.',
      table: 'paired',
    },
    failures: {
      title: 'How often extraction fails',
      finding: `At the selected epoch, ${number(s.wrong_speaker_cases)} requests (${percent(s.wrong_speaker_cases, s.cases)}) match the competing speaker more closely than the target. ${percent(s.best.no_improvement, s.cases)} show no improvement. These groups can overlap.`,
      caption: 'Full validation only. A wrong-speaker match means the output’s SI-SDR against the competing voice is more than 3 dB above its target score. No improvement means target SI-SDR improvement is zero or negative. These rates are not added together.',
      alt: 'Two curves show no-improvement and wrong-speaker rates declining during training, then leveling off near five and four percent.',
      table: 'failures',
    },
    speakers: {
      title: 'Which voices remain difficult',
      finding: `The best model’s average improvement ranges from ${s.speaker_mean_range[0].toFixed(2)} to ${s.speaker_mean_range[1].toFixed(2)} dB across ${s.speakers} development speakers. Each speaker has ${s.speaker_case_range[0]}–${s.speaker_case_range[1]} requests.`,
      caption: 'Speakers are ordered by their best-model mean. Each vertical line spans the middle 50% of that speaker’s request scores; it is not a confidence interval. Circles and crosses show the two model means. The dotted line is the overall best-model mean.',
      alt: 'Forty speaker groups ordered from lowest to highest average separation quality, with per-speaker score ranges and both checkpoint means.',
      table: 'speakers',
    },
    paired: {
      title: 'What changed after one more epoch',
      finding: `${percent(s.paired.improved_over_1db, s.cases)} of requests improve by more than 1 dB; ${percent(s.paired.worsened_over_1db, s.cases)} worsen by more than 1 dB. ${percent(s.paired.within_1db, s.cases)} stay within ±1 dB. The average change is ${s.paired.mean_change.toFixed(3)} dB.`,
      caption: 'Each point is the same request evaluated at both epochs. Above the diagonal favors epoch 81; below favors epoch 80. The shaded band is ±1 dB, a descriptive threshold rather than a significance test. Dense points can overlap.',
      alt: 'Paired scatterplot of epoch-80 versus epoch-81 separation scores. Most requests lie close to equal performance, while some improve or regress sharply.',
      table: 'paired',
    },
  };
}

const charts = descriptions();

export default function ValidationDiagnostics() {
  const [selected, setSelected] = useMotionState('distribution', '.ov-diagnostics');
  const chart = charts[selected];
  return <section className="ov-diagnostics" aria-labelledby="ov-diagnostics-title">
    <h3 id="ov-diagnostics-title">Beyond the average</h3>
    <p>Explore all {number(diagnostics.summary.cases)} validation requests—not just the twelve listening examples. These graphs compare epoch {diagnostics.best_epoch} with the last full evaluation at epoch {diagnostics.latest_epoch}.</p>
    <div className="ov-switch ov-diagnostics-controls" role="group" aria-label="Evaluation chart">
      {options.map(option => <button key={option.id} type="button" aria-pressed={selected === option.id} aria-controls="ov-validation-chart" onClick={() => setSelected(option.id)}>{option.label}</button>)}
    </div>
    <div id="ov-validation-chart" className="ov-diagnostics-result">
      <h4>{chart.title}</h4>
      <p className="ov-diagnostics-finding" aria-live="polite">{chart.finding}</p>
      <figure className="ov-training-figure">
        <div className="ov-diagnostics-plot" role="region" aria-label="Scrollable evaluation graph" tabIndex={0}>
          <img src={diagnostics.charts[selected]} width="1000" height="540" loading="lazy" alt={chart.alt}/>
        </div>
        <p className="ov-diagnostics-scroll-hint">Scroll within the graph to see all axes, or download it at full size.</p>
        <figcaption>{chart.caption}</figcaption>
      </figure>
      <nav className="ov-diagnostics-downloads" aria-label="Download evaluation evidence">
        <a href={diagnostics.charts[selected]} download>Download graph</a>
        <a href={diagnostics.downloads[chart.table]} download>Download data · CSV</a>
        <a href={diagnostics.downloads.report} download>Full evaluation report</a>
      </nav>
    </div>
    <p className="ov-diagnostics-scope">Development results used to select the model. They measure separation, not word accuracy or a listening-quality rating. Epoch 81 was evaluated at update 281,475; the latest saved model at update 281,484 has not had this full evaluation.</p>
  </section>;
}
