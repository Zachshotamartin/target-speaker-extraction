const WINDOW_SECONDS = 5;

// Group both independent ASR outputs by the same audio time, not by sentence index.
export function comparisonRows(comparison, duration) {
  const count = Math.max(1, Math.ceil(duration / WINDOW_SECONDS));
  const rows = Array.from({length: count}, (_, index) => ({
    start: index * WINDOW_SECONDS,
    end: Math.min((index + 1) * WINDOW_SECONDS, duration),
    original: '',
    extracted: '',
  }));
  for (const [source, field] of [['raw', 'original'], ['one_voice', 'extracted']]) {
    const output = comparison?.[source];
    if (!output?.words?.length) {
      rows[0][field] = output?.text?.trim() || '';
      continue;
    }
    for (const word of output.words) {
      const start = Number.isFinite(word.start) ? word.start : 0;
      const index = Math.max(0, Math.min(count - 1, Math.floor(start / WINDOW_SECONDS)));
      rows[index][field] += word.text;
    }
    rows.forEach(row => { row[field] = row[field].trim(); });
  }
  return rows;
}
