export function processingLabel(message = '') {
  if (/frozen|verifying.*model/i.test(message)) return 'Preparing the voice models…';
  if (/saving completed/i.test(message)) return 'Getting your results ready…';
  if (/uploading/i.test(message)) {
    const percent = message.match(/\d+%/);
    return `Uploading your audio${percent ? ` · ${percent[0]}` : '…'}`;
  }
  return message || 'Working on your recording…';
}
