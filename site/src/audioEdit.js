// All edit decisions use the source timeline. Rendering and captions share this
// plan so removing speech cannot leave subtitles on the old recording clock.
const clamp = (value, low, high) => Math.min(high, Math.max(low, value));

export function transcriptWords(result) {
  const words = (result.segments || []).flatMap(segment =>
    (segment.words || []).map(word => ({...word, attribution: word.attribution || segment.attribution})));
  const source = words.length ? words : result.comparison?.one_voice?.words || [];
  let end = 0;
  return source.filter(word => Number.isFinite(word.start) && Number.isFinite(word.end) && word.text?.trim())
    .sort((a, b) => a.start - b.start).flatMap(word => {
      const start = clamp(word.start, end, result.duration);
      const stop = clamp(word.end, start, result.duration);
      if (stop <= start) return [];
      end = stop;
      return [{...word, start, end: stop, text: word.text.trim()}];
    });
}

export function editPlan(words, removed, duration) {
  if (!Number.isFinite(duration) || duration <= 0) return [];
  const deleted = new Set(removed);
  // Cut in the gap between words. Limit padding so deleting one word does not
  // unexpectedly remove a long pause. First/last deletions also trim the ends.
  const cuts = [];
  for (let i = 0; i < words.length; i++) {
    if (!deleted.has(i)) continue;
    const first = i;
    while (i + 1 < words.length && deleted.has(i + 1)) i++;
    const start = first === 0 ? 0 : i === words.length - 1
      ? Math.min(words[first].start, words[first - 1].end + 0.04)
      : Math.max(words[first - 1].end, words[first].start - 0.04);
    const end = i === words.length - 1 ? duration : first === 0
      ? Math.max(words[i].end, words[i + 1].start - 0.04)
      : Math.min(words[i + 1].start, words[i].end + 0.04);
    cuts.push([clamp(start, 0, duration), clamp(end, 0, duration)]);
  }
  const clips = [];
  let cursor = 0, output = 0;
  for (const [start, end] of [...cuts, [duration, duration]]) {
    if (start > cursor) {
      clips.push({start: cursor, end: start, outputStart: output, outputEnd: output + start - cursor});
      output += start - cursor;
    }
    cursor = Math.max(cursor, end);
  }
  return clips;
}

export const editedDuration = clips => clips.at(-1)?.outputEnd || 0;

export function sourceToEdited(time, clips) {
  for (const clip of clips) {
    if (time < clip.start) return clip.outputStart;
    if (time < clip.end) return clip.outputStart + time - clip.start;
  }
  return editedDuration(clips);
}

export function editedToSource(time, clips) {
  for (const clip of clips) {
    if (time < clip.outputEnd) return clip.start + Math.max(0, time - clip.outputStart);
  }
  return clips.at(-1)?.end || 0;
}

export function editedWords(words, removed, clips) {
  const deleted = new Set(removed);
  return words.filter((_, index) => !deleted.has(index)).map(word => ({
    ...word, start: sourceToEdited(word.start, clips), end: sourceToEdited(word.end, clips),
  })).filter(word => word.end > word.start);
}

const subtitleTime = seconds => {
  const ms = Math.round(Math.max(0, seconds) * 1000);
  return `${String(Math.floor(ms / 3600000)).padStart(2, '0')}:${String(Math.floor(ms / 60000) % 60).padStart(2, '0')}:${String(Math.floor(ms / 1000) % 60).padStart(2, '0')},${String(ms % 1000).padStart(3, '0')}`;
};

export function captionsSrt(words) {
  const cues = [];
  for (const word of words) {
    const previous = cues.at(-1);
    if (previous && word.end - previous.start <= 3.5 && word.start - previous.end < 0.8 &&
      previous.text.length + word.text.length < 48 && !/[.!?]$/.test(previous.text)) {
      previous.text += ` ${word.text}`;
      previous.end = word.end;
    } else cues.push({...word});
  }
  return cues.map((cue, index) => `${index + 1}\n${subtitleTime(cue.start)} --> ${subtitleTime(cue.end)}\n${cue.text}`).join('\n\n') + (cues.length ? '\n' : '');
}

export function renderEdit(channels, sampleRate, clips, fadeSeconds = 0.005) {
  const spans = clips.map(clip => ({
    start: clamp(Math.round(clip.start * sampleRate), 0, channels[0].length),
    end: clamp(Math.round(clip.end * sampleRate), 0, channels[0].length),
  }));
  const frames = spans.reduce((sum, clip) => sum + clip.end - clip.start, 0);
  return channels.map(channel => {
    const output = new Float32Array(frames);
    let offset = 0;
    spans.forEach(({start, end}) => {
      const length = end - start;
      output.set(channel.subarray(start, end), offset);
      // Fade at newly cut boundaries, without overlapping clips or shifting
      // timestamps. An unedited recording is preserved sample-for-sample.
      const fade = Math.min(Math.round(fadeSeconds * sampleRate), Math.floor(length / 2));
      for (let i = 0; i < fade; i++) {
        const gain = fade > 1 ? i / (fade - 1) : 0;
        if (start > 0) output[offset + i] *= gain;
        if (end < channel.length) output[offset + length - 1 - i] *= gain;
      }
      offset += length;
    });
    return output;
  });
}

export function wavBytes(channels, sampleRate) {
  const frames = channels[0]?.length || 0, count = channels.length;
  if (!frames || !count || !Number.isFinite(sampleRate) || sampleRate <= 0) throw new Error('No audio remains to export.');
  if (channels.some(channel => channel.length !== frames)) throw new Error('Audio channels have different lengths.');
  const bytes = new ArrayBuffer(44 + frames * count * 2), view = new DataView(bytes);
  const text = (offset, value) => [...value].forEach((letter, index) => view.setUint8(offset + index, letter.charCodeAt(0)));
  text(0, 'RIFF'); view.setUint32(4, bytes.byteLength - 8, true); text(8, 'WAVE'); text(12, 'fmt ');
  view.setUint32(16, 16, true); view.setUint16(20, 1, true); view.setUint16(22, count, true);
  view.setUint32(24, sampleRate, true); view.setUint32(28, sampleRate * count * 2, true);
  view.setUint16(32, count * 2, true); view.setUint16(34, 16, true); text(36, 'data'); view.setUint32(40, frames * count * 2, true);
  for (let frame = 0; frame < frames; frame++) for (let channel = 0; channel < count; channel++) {
    const value = clamp(Number.isFinite(channels[channel][frame]) ? channels[channel][frame] : 0, -1, 1);
    view.setInt16(44 + (frame * count + channel) * 2, Math.round(value * (value < 0 ? 32768 : 32767)), true);
  }
  return bytes;
}

export const initialEdits = {past: [], removed: [], future: []};
export function editHistory(state, action) {
  if (action.type === 'undo') return state.past.length ? {
    past: state.past.slice(0, -1), removed: state.past.at(-1), future: [state.removed, ...state.future],
  } : state;
  if (action.type === 'redo') return state.future.length ? {
    past: [...state.past, state.removed], removed: state.future[0], future: state.future.slice(1),
  } : state;
  if (action.type !== 'set') return state;
  const removed = [...new Set(action.removed)].sort((a, b) => a - b);
  if (JSON.stringify(removed) === JSON.stringify(state.removed)) return state;
  return {past: [...state.past.slice(-99), state.removed], removed, future: []};
}
