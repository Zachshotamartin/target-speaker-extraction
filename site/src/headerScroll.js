const DIRECTION_THRESHOLD = 12;

export function initialHeaderScroll(y = 0) {
  return {y, travel: 0, direction: 0, hidden: false};
}

export function updateHeaderScroll(previous, y, headerHeight, keepVisible = false) {
  if (keepVisible || y <= headerHeight) return initialHeaderScroll(y);
  const delta = y - previous.y;
  if (!delta) return previous;
  const direction = Math.sign(delta);
  const travel = direction === previous.direction ? previous.travel + Math.abs(delta) : Math.abs(delta);
  return {
    y,
    direction,
    travel,
    hidden: travel >= DIRECTION_THRESHOLD ? direction > 0 : previous.hidden,
  };
}
