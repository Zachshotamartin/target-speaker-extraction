"use strict";

const $ = (id) => document.getElementById(id);
const state = {
  ready: false,
  busy: false,
  loadingExample: false,
  files: {},
  buffers: {},
  urls: {},
  previews: {},
  comparison: {},
  selected: "isolated",
  versions: { mixture: 0, reference: 0 },
  context: null,
};

function message(text, kind = "") {
  $("message").textContent = text;
  $("message").className = `message ${kind}`;
}

function clock(seconds) {
  const value = Math.max(0, Number.isFinite(seconds) ? seconds : 0);
  return `${Math.floor(value / 60)}:${Math.floor(value % 60).toString().padStart(2, "0")}`;
}

function updateButton() {
  $("extract-button").disabled = !state.ready || state.busy || state.loadingExample || !state.files.mixture || !state.files.reference;
  $("extract-button").classList.toggle("busy", state.busy);
  $("extract-label").textContent = state.busy ? "Isolating the voice" : "Isolate this voice";
  for (const input of document.querySelectorAll('input[type="file"], .example-button')) input.disabled = state.busy || state.loadingExample;
}

function stopAudio(except = null) {
  for (const audio of [...Object.values(state.previews), ...Object.values(state.comparison)]) {
    if (audio !== except) audio.pause();
  }
}

function clearResult() {
  stopAudio();
  for (const audio of Object.values(state.comparison)) audio.src = "";
  state.comparison = {};
  if (state.urls.output) URL.revokeObjectURL(state.urls.output);
  delete state.urls.output;
  delete state.buffers.output;
  $("result-ready").hidden = true;
  $("result-empty").hidden = false;
  $("download").hidden = true;
  $("download").removeAttribute("href");
}

function draw(canvas, buffer, progress = 0, green = false) {
  const rect = canvas.getBoundingClientRect();
  if (!rect.width || !buffer) return;
  const ratio = window.devicePixelRatio || 1;
  canvas.width = Math.round(rect.width * ratio);
  canvas.height = Math.round(rect.height * ratio);
  const context = canvas.getContext("2d");
  context.scale(ratio, ratio);
  context.clearRect(0, 0, rect.width, rect.height);
  const samples = buffer.getChannelData(0);
  const count = Math.max(1, Math.floor(rect.width / 4));
  const stride = Math.max(1, Math.floor(samples.length / count));
  let peak = 0.01;
  const values = [];
  for (let index = 0; index < count; index++) {
    let maximum = 0;
    const start = index * stride;
    for (let sample = start; sample < Math.min(start + stride, samples.length); sample += 8) maximum = Math.max(maximum, Math.abs(samples[sample]));
    values.push(maximum);
    peak = Math.max(peak, maximum);
  }
  for (let index = 0; index < count; index++) {
    const height = Math.max(2, values[index] / peak * (rect.height - 5));
    context.fillStyle = index / count <= progress && progress > 0 ? "#47634c" : green ? "#a7b99e" : "#c3c8bb";
    context.fillRect(index * rect.width / count, (rect.height - height) / 2, 2, height);
  }
}

async function decode(blob) {
  if (!state.context) state.context = new (window.AudioContext || window.webkitAudioContext)();
  return state.context.decodeAudioData(await blob.arrayBuffer());
}

async function selectFile(kind, file, fromExample = false) {
  if (state.busy || (state.loadingExample && !fromExample) || !file) return;
  const version = ++state.versions[kind];
  clearResult();
  delete state.files[kind];
  updateButton();
  try {
    if (!/\.(wav|flac)$/i.test(file.name)) throw new Error("Choose a WAV or FLAC recording.");
    if (file.size > 24 * 1024 * 1024) throw new Error("The combined upload limit is 24 MiB.");
    const buffer = await decode(file);
    if (version !== state.versions[kind]) return;
    const limits = kind === "reference" ? [3, 10] : [0.05, 60];
    if (buffer.duration < limits[0] || buffer.duration > limits[1] + 0.001) throw new Error(kind === "reference" ? "Use a reference clip between 3 and 10 seconds." : "Use a conversation of up to 60 seconds.");
    const other = state.files[kind === "reference" ? "mixture" : "reference"];
    if (file.size + (other?.size || 0) > 24 * 1024 * 1024 - 4096) throw new Error("Together, the recordings must be smaller than 24 MiB.");
    if (state.urls[kind]) URL.revokeObjectURL(state.urls[kind]);
    if (state.previews[kind]) state.previews[kind].src = "";
    state.files[kind] = file;
    state.buffers[kind] = buffer;
    state.urls[kind] = URL.createObjectURL(file);
    const audio = new Audio(state.urls[kind]);
    state.previews[kind] = audio;
    const button = $(`${kind}-play`);
    const update = () => { button.textContent = audio.paused ? "▶" : "Ⅱ"; button.setAttribute("aria-label", `${audio.paused ? "Play" : "Pause"} ${kind === "mixture" ? "conversation" : "voice sample"}`); };
    audio.addEventListener("play", update);
    audio.addEventListener("pause", update);
    audio.addEventListener("ended", update);
    $(`${kind}-name`).textContent = file.name;
    $(`${kind}-description`).textContent = "Click to choose a different recording";
    $(`${kind}-duration`).textContent = `${buffer.duration.toFixed(1)}s`;
    $(`${kind}-preview`).hidden = false;
    draw($(`${kind}-wave`), buffer, 0, kind === "reference");
    message(state.files.mixture && state.files.reference ? "Both recordings are ready. Isolate the voice when you’re ready to listen." : "Add the other recording to continue.");
  } catch (error) {
    if (version !== state.versions[kind]) return;
    $(`${kind}-preview`).hidden = true;
    $(`${kind}-name`).textContent = kind === "mixture" ? "Choose a recording" : "Choose a voice sample";
    message(error.message || "This browser could not decode the recording. Try a WAV file.", "error");
  }
  updateButton();
}

for (const kind of ["mixture", "reference"]) {
  $(`${kind}-file`).addEventListener("change", (event) => selectFile(kind, event.target.files[0]));
  const zone = $(`${kind}-zone`);
  zone.addEventListener("dragover", (event) => { event.preventDefault(); if (!state.busy) zone.classList.add("dragging"); });
  zone.addEventListener("dragleave", () => zone.classList.remove("dragging"));
  zone.addEventListener("drop", (event) => { event.preventDefault(); zone.classList.remove("dragging"); selectFile(kind, event.dataTransfer.files[0]); });
  $(`${kind}-play`).addEventListener("click", async () => {
    const audio = state.previews[kind];
    if (!audio) return;
    if (!audio.paused) audio.pause();
    else { stopAudio(audio); try { await audio.play(); } catch { message("Playback could not start. Try selecting the file again.", "error"); } }
  });
}

function activeAudio() { return state.comparison[state.selected]; }

function updatePlayback() {
  const audio = activeAudio();
  if (!audio) return;
  $("result-play").textContent = audio.paused ? "▶" : "Ⅱ";
  $("result-play").setAttribute("aria-label", audio.paused ? "Play comparison audio" : "Pause comparison audio");
  const duration = Number.isFinite(audio.duration) ? audio.duration : state.buffers.mixture.duration;
  $("playback-time").textContent = `${clock(audio.currentTime)} / ${clock(duration)}`;
  $("seek").value = duration ? audio.currentTime / duration * 1000 : 0;
  draw($("result-wave"), state.selected === "isolated" ? state.buffers.output : state.buffers.mixture, duration ? audio.currentTime / duration : 0, state.selected === "isolated");
}

async function chooseVersion(version) {
  const previous = activeAudio();
  const time = previous?.currentTime || 0;
  const playing = previous && !previous.paused;
  previous?.pause();
  state.selected = version;
  for (const name of ["original", "isolated"]) {
    $(`select-${name}`).classList.toggle("selected", name === version);
    $(`select-${name}`).setAttribute("aria-pressed", String(name === version));
  }
  const current = activeAudio();
  if (current) {
    current.currentTime = time;
    if (playing) { try { await current.play(); } catch { message("Press play to resume listening."); } }
  }
  updatePlayback();
}

$("select-original").addEventListener("click", () => chooseVersion("original"));
$("select-isolated").addEventListener("click", () => chooseVersion("isolated"));
$("result-play").addEventListener("click", async () => {
  const audio = activeAudio();
  if (!audio) return;
  if (!audio.paused) audio.pause();
  else { stopAudio(audio); try { await audio.play(); } catch { message("Press play again to start the audio."); } }
});
$("seek").addEventListener("input", (event) => {
  const audio = activeAudio();
  if (audio && Number.isFinite(audio.duration)) { audio.currentTime = Number(event.target.value) / 1000 * audio.duration; updatePlayback(); }
});

$("extraction-form").addEventListener("submit", async (event) => {
  event.preventDefault();
  if (state.busy || state.loadingExample || !state.ready || !state.files.mixture || !state.files.reference) return;
  clearResult();
  state.busy = true;
  updateButton();
  $("result-empty").classList.add("processing");
  const start = performance.now();
  message("Isolating the selected voice on this computer…");
  const timer = setInterval(() => message(`Isolating the selected voice… ${Math.floor((performance.now() - start) / 1000)} seconds`), 1000);
  const controller = new AbortController();
  const timeout = setTimeout(() => controller.abort(), 180000);
  try {
    const data = new FormData();
    data.append("mixture", state.files.mixture);
    data.append("reference", state.files.reference);
    const response = await fetch("/extract", { method: "POST", body: data, signal: controller.signal });
    if (!response.ok) {
      const detail = await response.json().catch(() => ({}));
      throw new Error(typeof detail.detail === "string" ? detail.detail : "The recordings could not be processed.");
    }
    const metadata = JSON.parse(response.headers.get("X-TSE-Metadata") || "{}");
    const blob = await response.blob();
    state.buffers.output = await decode(blob);
    state.urls.output = URL.createObjectURL(blob);
    state.comparison = { original: new Audio(state.urls.mixture), isolated: new Audio(state.urls.output) };
    for (const audio of Object.values(state.comparison)) {
      for (const eventName of ["loadedmetadata", "play", "pause", "ended", "timeupdate"]) audio.addEventListener(eventName, updatePlayback);
      audio.preload = "auto";
      audio.load();
    }
    $("result-ready").hidden = false;
    $("result-empty").hidden = true;
    $("download").href = state.urls.output;
    $("download").hidden = false;
    $("result-timing").textContent = `${Number(metadata.processing_seconds).toFixed(2)} seconds to process · 16 kHz WAV · model ${metadata.model_id}`;
    await chooseVersion("isolated");
    message(metadata.playback_gain < 1
      ? "Your audio is ready. Output volume was reduced to prevent playback clipping. Listen to both versions to assess the extraction."
      : "Your audio is ready. Listen to both versions to assess the extraction.", "success");
  } catch (error) {
    message(error.name === "AbortError" ? "Processing took too long. Try a shorter recording." : error.message, "error");
  } finally {
    clearInterval(timer);
    clearTimeout(timeout);
    state.busy = false;
    $("result-empty").classList.remove("processing");
    updateButton();
  }
});

async function initialize() {
  try {
    const response = await fetch("/model");
    const model = await response.json();
    state.ready = response.ok && model.ready;
    $("status-dot").classList.toggle("ready", state.ready);
    $("model-status").textContent = state.ready ? `Local model · ${model.device.toUpperCase()}` : "Model not ready";
    $("model-details").textContent = state.ready ? `Model ${model.model_id}. ${model.parameters.toLocaleString()} parameters, ${model.training_updates.toLocaleString()} updates in this run${model.initialization ? ", following earlier project training" : ""}. ${model.operating_envelope}` : "Start the service with a trained checkpoint to use extraction. The health endpoint and interface are available while no model is loaded.";
    if (!state.ready) message("The local service needs a trained checkpoint before extraction is available.");
  } catch { $("model-status").textContent = "Local service unavailable"; message("The local service is unavailable. Start it and reload this page.", "error"); }
  updateButton();
  try {
    const response = await fetch("/examples");
    if (!response.ok) return;
    const examples = await response.json();
    $("gallery-link").hidden = !examples.gallery_available;
    for (const example of examples.items || []) {
      const button = document.createElement("button");
      button.type = "button";
      button.className = "example-button";
      button.textContent = example.label;
      button.addEventListener("click", async () => {
        if (state.busy || state.loadingExample) return;
        state.loadingExample = true;
        updateButton();
        try {
          message("Loading a public speech example…");
          const audioResponses = await Promise.all([fetch(example.mixture), fetch(example.reference)]);
          if (audioResponses.some((item) => !item.ok)) throw new Error("Example audio is unavailable.");
          const [mixture, reference] = await Promise.all(audioResponses.map((item) => item.blob()));
          await selectFile("mixture", new File([mixture], "example-conversation.wav", { type: "audio/wav" }), true);
          await selectFile("reference", new File([reference], `${example.id}-voice.wav`, { type: "audio/wav" }), true);
        } catch (error) { message(error.message, "error"); }
        finally { state.loadingExample = false; updateButton(); }
      });
      $("example-buttons").append(button);
    }
    $("examples").hidden = !(examples.items || []).length;
  } catch { /* Examples are optional; file upload remains available. */ }
}

const resize = new ResizeObserver(() => {
  for (const kind of ["mixture", "reference"]) if (state.buffers[kind]) draw($(`${kind}-wave`), state.buffers[kind], 0, kind === "reference");
  updatePlayback();
});
for (const canvas of document.querySelectorAll("canvas")) resize.observe(canvas);
initialize();
