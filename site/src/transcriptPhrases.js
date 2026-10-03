// Keep verified wording intact while displaying word-level output as readable passages.
export function transcriptPhrases(words) {
  return words.reduce((phrases, word) => {
    const previous = phrases[phrases.length - 1];
    const text = word.text.trim();
    if (!text) return phrases;
    if (!previous || word.start - previous.end > .75 || previous.count >= 28 || (/[.!?]$/.test(previous.text) && previous.count >= 5)) {
      phrases.push({...word, text, count:text.split(/\s+/).length});
    } else {
      previous.text += ` ${text}`;
      previous.end = word.end;
      previous.count += text.split(/\s+/).length;
    }
    return phrases;
  }, []);
}
