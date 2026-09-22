export const emptyEdit = {removed: [], customCuts: [], corrections: {}, decisions: {}, padding: .08};
export function historyStep(history, action) {
  if (action.type === 'undo') return history.past.length ? {past: history.past.slice(0, -1), present: history.past.at(-1), future: [history.present, ...history.future]} : history;
  if (action.type === 'redo') return history.future.length ? {past: [...history.past, history.present], present: history.future[0], future: history.future.slice(1)} : history;
  if (action.type === 'reset') action = {value: emptyEdit};
  const next = {...history.present, ...action.value};
  if (JSON.stringify(next) === JSON.stringify(history.present)) return history;
  return {past: [...history.past.slice(-49), history.present], present: next, future: []};
}
export function naturalPlan(words, state, duration) {
  const deleted = new Set(state.removed), cuts = [];
  for (let i = 0; i < words.length; i++) {
    if (!deleted.has(i)) continue;
    const first = i;
    while (i + 1 < words.length && deleted.has(i + 1)) i++;
    // Preserve breathing room next to retained words without leaking deleted speech.
    cuts.push({start: first === 0 ? 0 : Math.min(words[first].start, words[first - 1].end + state.padding),
      end: i === words.length - 1 ? duration : Math.max(words[i].end, words[i + 1].start - state.padding)});
  }
  let kept = duration > 0 ? [{start: 0, end: duration}] : [];
  for (const cut of [...cuts, ...state.customCuts]) kept = kept.flatMap(clip => cut.end <= clip.start || cut.start >= clip.end ? [clip] : [
    {start: clip.start, end: Math.min(clip.end, cut.start)}, {start: Math.max(clip.start, cut.end), end: clip.end},
  ].filter(c => c.end - c.start >= .01));
  let outputStart = 0;
  return kept.map(clip => {const result = {...clip, outputStart, outputEnd: outputStart + clip.end - clip.start}; outputStart = result.outputEnd; return result;});
}
export function suggestions(words, pause = 1.2, keep = .45) {
  const list = [];
  const clean = text => text.toLowerCase().replace(/[^a-z']/g, '');
  words.forEach((word, index) => {
    if (['um', 'uh', 'erm', 'umm', 'uhh'].includes(clean(word.text))) list.push({label: `Filler: “${word.text}”`, ids: [index], start: word.start, end: word.end});
    if (index && word.start - words[index - 1].end > pause) list.push({label: `Shorten ${(word.start - words[index - 1].end).toFixed(1)}s pause`, start: words[index - 1].end + keep / 2, end: word.start - keep / 2, cut: true});
    if (index >= 3 && index + 2 < words.length) {
      const phrase = words.slice(index, index + 3).map(w => clean(w.text)).join(' ');
      for (let before = Math.max(0, index - 25); before <= index - 3; before++) {
        if (word.start - words[before].start < 25 && words.slice(before, before + 3).map(w => clean(w.text)).join(' ') === phrase) {
          list.push({label: `Repeated phrase: “${words.slice(before, before + 3).map(w => w.text).join(' ')}”`, ids: [before, before + 1, before + 2], start: words[before].start, end: words[before + 2].end}); break;
        }
      }
    }
  });
  return list;
}
export function correctedWords(words, edit) {
  return words.map((word, i) => ({...word, text: edit.corrections[i] ?? word.text, attribution: edit.decisions[i] ?? word.attribution}));
}
export function waveformPeaks(channel, start, end, rate, count = 240) {
  return Array.from({length: count}, (_, i) => {
    const a = Math.floor((start + (end - start) * i / count) * rate), b = Math.min(channel.length, Math.ceil((start + (end - start) * (i + 1) / count) * rate));
    let peak = 0; for (let j = Math.max(0, a); j < b; j++) peak = Math.max(peak, Math.abs(channel[j])); return peak;
  });
}
