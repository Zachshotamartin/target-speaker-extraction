/* Static listening artifacts are published independently of the running trainer. */
(() => {
  const get = id => document.getElementById(id);
  const titles = {best: 'Best completed full validation', latest: 'Latest full validation'};
  let signature = '', pending = null, loading = false;
  const node = (tag, text, className) => {
    const value = document.createElement(tag);
    if (text != null) value.textContent = text;
    if (className) value.className = className;
    return value;
  };
  const db = value => Number.isFinite(value) ? `${value.toFixed(2)} dB` : 'Pending';
  const identity = data => JSON.stringify([data.models, data.errors]);
  function track(label, paths, id, metric) {
    const wrapper = node('div', null, 'listening-track');
    const caption = node('label', label);
    caption.htmlFor = id;
    wrapper.append(caption);
    if (!paths) {
      wrapper.append(node('p', 'Exact validation audio is not available yet.', 'listening-caption'));
      return wrapper;
    }
    const audio = node('audio');
    audio.id = id; audio.controls = true; audio.preload = 'metadata';
    audio.dataset.raw = paths.raw; audio.dataset.matched = paths.matched;
    audio.src = get('listening-match').checked ? paths.matched : paths.raw;
    audio.addEventListener('error', () => {
      if (!wrapper.querySelector('.audio-error')) wrapper.append(node('p',
        'This audio could not be loaded. Refresh the samples or check the local server.', 'audio-error listening-caption'));
    });
    wrapper.append(audio);
    if (metric) wrapper.append(node('p', `${db(metric.si_sdri_db)} improvement on this example`, 'listening-caption'));
    return wrapper;
  }
  function render(data) {
    signature = identity(data); pending = null;
    get('listening-update').hidden = true;
    get('listening-models').replaceChildren(); get('listening-cases').replaceChildren();
    for (const role of ['best', 'latest']) {
      const model = data.models[role];
      const block = node('div', null, 'listening-model');
      block.append(node('h3', titles[role]));
      if (model) {
        block.append(node('p', `Update ${model.step.toLocaleString()} · epoch ${model.epoch.toFixed(2)}`));
        block.append(node('p', model.pending
          ? 'Evaluation in progress · overall score pending'
          : `${db(model.validation.mean_si_sdri_db)} overall · ${model.validation.cases.toLocaleString()} requests`));
      } else block.append(node('p', 'Waiting for an exact validation checkpoint.'));
      get('listening-models').append(block);
    }
    const examples = data.models.latest?.items ?? data.models.best?.items ?? [];
    for (const item of examples) {
      const section = node('article', null, 'listening-case');
      section.append(node('h3', `Conversation ${item.mixture_number} · keep voice ${item.speaker}`));
      section.append(node('p', `${item.duration_seconds.toFixed(1)} seconds · the other request for this conversation keeps the other speaker.`, 'listening-caption'));
      const sources = node('div', null, 'listening-tracks');
      for (const [key, label] of [['reference', 'Voice reference · who to keep'], ['mixture', 'Conversation · both voices'], ['target', 'Clean target · ideal result']]) {
        sources.append(track(label, item.tracks[key], `listen-${item.index}-${key}`));
      }
      const estimates = node('div', null, 'listening-tracks listening-estimates');
      for (const role of ['best', 'latest']) {
        const model = data.models[role];
        const example = model?.items.find(value => value.case_id === item.case_id);
        estimates.append(track(`${role === 'best' ? 'Best' : 'Latest'} model estimate${model?.pending ? ' · validation in progress' : ''}`,
          example?.tracks.estimate, `listen-${item.index}-${role}`, example?.metrics));
      }
      section.append(sources, estimates); get('listening-cases').append(section);
    }
  }
  async function refreshListening() {
    if (loading) return;
    loading = true;
    try {
      const [response, workerResponse] = await Promise.all([
        fetch('/gallery/full-validation/index.json', {cache: 'no-store'}),
        fetch('/gallery/full-validation/worker.json', {cache: 'no-store'})
      ]);
      const worker = workerResponse.ok ? await workerResponse.json() : {};
      if (!response.ok) {
        get('listening-status').textContent = worker.status === 'error' ? worker.detail
          : 'Rendering the first listening samples on CPU. Training continues; this section will update automatically.';
        return;
      }
      const data = await response.json();
      if (identity(data) !== signature) {
        if ([...document.querySelectorAll('.validation-listening audio')].some(a => !a.paused)) {
          pending = data; get('listening-update').hidden = false;
        } else render(data);
      }
      const same = data.models.best && data.models.latest && data.models.best.step === data.models.latest.step;
      let message = same ? 'The latest validation is also the best, so both estimates are identical.'
        : 'Best and latest estimates use exactly the same conversations and references.';
      if (pending) message += ' New samples are ready; load them when you finish listening.';
      if (worker.status === 'rendering') message += ` Preparing ${worker.role} samples for update ${worker.step.toLocaleString()}.`;
      if (worker.status === 'error') message += ` Refresh issue: ${worker.detail}. Existing audio is retained.`;
      if (worker.checked_at && Date.now() / 1000 - worker.checked_at > 600) message += ' The listening worker has not checked in recently; these samples may be older.';
      if (data.errors.length) message += ` ${data.errors.join(' ')}`;
      get('listening-status').textContent = message;
    } catch {
      get('listening-status').textContent = 'Listening updates are unavailable. Existing samples remain playable while the server is reachable.';
    } finally { loading = false; }
  }
  document.addEventListener('play', event => {
    if (event.target.tagName === 'AUDIO') document.querySelectorAll('audio').forEach(audio => {
      if (audio !== event.target) audio.pause();
    });
  }, true);
  get('listening-match').addEventListener('change', () => {
    document.querySelectorAll('.validation-listening audio').forEach(audio => {
      const position = audio.ended ? 0 : audio.currentTime;
      audio.pause();
      audio.addEventListener('loadedmetadata', () => { audio.currentTime = Math.min(position, audio.duration); }, {once: true});
      audio.src = get('listening-match').checked ? audio.dataset.matched : audio.dataset.raw;
    });
  });
  get('listening-update').onclick = () => { if (pending) { render(pending); refreshListening(); } };
  refreshListening(); setInterval(refreshListening, 30000);
})();
