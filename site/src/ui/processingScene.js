import {WebGLRenderer, Scene, OrthographicCamera, Group} from 'three';
import {createProcessingMotifs} from './processingMotifs.js';

const SHAPE_HOLD_SECONDS = 3.8;
const SNAP_SECONDS = .14;
const SETTLE_SECONDS = .42;
const CYCLE_SECONDS = SHAPE_HOLD_SECONDS + SNAP_SECONDS + SETTLE_SECONDS;

export function createProcessingScene(canvas, preference, onLost) {
  // Fixed visual dimensions come from CSS; setSize(false) avoids inline styles.
  const size = canvas.parentElement.getBoundingClientRect().width;
  const renderer = new WebGLRenderer({canvas, alpha: true, antialias: true, powerPreference: 'low-power'});
  renderer.setPixelRatio(Math.min(window.devicePixelRatio || 1, 2));
  renderer.setSize(size, size, false);
  const scene = new Scene();
  const camera = new OrthographicCamera(-1, 1, 1, -1, .1, 10);
  camera.position.z = 4;
  const color = getComputedStyle(canvas.parentElement).color;
  const signal = new Group();
  const motifs = createProcessingMotifs(color);
  motifs.forEach(({group}) => signal.add(group));
  scene.add(signal);
  let last = null, phase = 0, spin = 0, contextLost = false;
  function draw(time = 0) {
    if (last !== null && time - last < 1000 / 30) return;
    const delta = last === null ? 0 : Math.min((time - last) / 1000, .1);
    phase += delta;
    last = time;
    const shape = Math.floor(phase / CYCLE_SECONDS) % motifs.length;
    const nextShape = (shape + 1) % motifs.length;
    const changeTime = phase % CYCLE_SECONDS - SHAPE_HOLD_SECONDS;
    const progress = Math.min(1, Math.max(0, changeTime / SNAP_SECONDS));
    const blend = 1 - (1 - progress) ** 4;
    const currentMotion = motifs[shape], nextMotion = motifs[nextShape];
    const radius = currentMotion.radius + (nextMotion.radius - currentMotion.radius) * blend;
    const speed = currentMotion.spin + (nextMotion.spin - currentMotion.spin) * blend;
    // A quick tuck and spin kick punctuate the new shape, then settle without a jump.
    const settling = Math.min(1, Math.max(0, (changeTime - SNAP_SECONDS) / SETTLE_SECONDS));
    const tuck = -.13 * Math.sin(Math.PI * progress);
    const rebound = .08 * Math.sin(Math.PI * settling) * (1 - settling);
    const kick = changeTime < 0 ? 0 : 4 * (1 - settling) ** 3;
    // Integrate velocity: changing speed must never reset the rotation angle.
    spin += delta * (speed + kick);
    motifs.forEach((motif, index) => {
      const weight = index === shape ? 1 - blend : index === nextShape ? blend : 0;
      motif.group.visible = weight > 0;
      if (!motif.group.visible) return;
      motif.group.scale.setScalar(.76 + .24 * weight);
      motif.group.rotation.z = (index === nextShape ? 1 - weight : weight - 1) * .6;
      motif.update(phase, spin, weight);
    });
    signal.scale.setScalar(radius + tuck + rebound + Math.sin(phase * 1.4) * .02);
    renderer.render(scene, camera);
  }
  function sync() {
    renderer.setAnimationLoop(null);
    if (contextLost) return;
    // Resume from the same shape after a hidden tab or motion preference change.
    last = null;
    draw(0);
    last = null;
    if (!preference.matches && !document.hidden) renderer.setAnimationLoop(draw);
  }
  function lost(event) {event.preventDefault(); contextLost = true; renderer.setAnimationLoop(null); onLost();}
  preference.addEventListener('change', sync);
  document.addEventListener('visibilitychange', sync);
  canvas.addEventListener('webglcontextlost', lost);
  sync();
  return () => {
    preference.removeEventListener('change', sync);
    document.removeEventListener('visibilitychange', sync);
    canvas.removeEventListener('webglcontextlost', lost);
    renderer.setAnimationLoop(null);
    motifs.forEach(motif => motif.dispose());
    renderer.dispose();
  };
}
