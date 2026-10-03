# Shared UI

Use these components for new page layout and controls. Each React component has its own file. Business logic belongs in the feature component or hook, not these primitives.

- `Section`: semantic region; choose `workspace`, `page`, `section`, or `none` spacing.
- `Heading`, `Text`: named type sizes and tones.
- `Stack`, `Inline`: vertical and wrapping horizontal layout; named gap sizes.
- `SplitLayout`, `ChoiceList`: navigation or recording context beside the active task. Keep primary actions in the working area, and supplemental detail in popovers.
- `Button`, `Field`, `ChoiceTabs`: shared control sizing, states, and labels.
- `Disclosure`: optional detail with the app's existing animated disclosure behavior.
- `Popover`: dismissible overlay for secondary controls; never shifts the content below.
- `StepNavigation`: current step, prerequisites, and back navigation.
- `Notice`, `Card`: feedback and optional grouped surfaces. Cards are not page containers.
- `ProcessingIndicator`: lazy-loaded Three.js signal with a CSS fallback. Stops rendering offscreen and for reduced motion, and releases GPU resources on completion.
- `AudioPlayer`, `Waveform`, `SignalWave`: reusable audio presentation.

Spacing, typography, width and control tokens live in `../design-tokens.css`; the palette lives in `../colors.css`. Primitive rules live in `ui.css`; feature files own only their layout composition. Do not add inline style objects or per-instance padding values. Use a named shared token or layout variant. Keep the OneVoice palette and typography.

The transcription controller owns the recording, references, job and edits across Audio → Voice → Results. Feature views in `../workspace/` present only the active step. Source changes do not clear the current recording; replacing it saves the previous project first. Examples load the recording and matching reference together. Back navigation preserves inputs and completed results. The existing motion queue handles structural changes and respects reduced motion.

Use `Dialog` for long supplemental reports. It renders in the browser top layer, caps its height to the viewport, and scrolls its body without extending the document. `Popover` is for short contextual controls.

Colors live only in `../colors.css`. Use its semantic CSS variables in component styles; do not add local hex, RGB/HSL, or inline color declarations. Dark surfaces, text and controls share the header/footer charcoal. Canvas rendering reads the computed CSS color.

`AudioPlayer` is the shared charcoal playback row for recordings, voice samples, comparisons and edits. It accepts a Blob or source URL, forwards the native audio ref/events for synchronized playback, and provides play/pause, seek, time and mute controls. Pill geometry uses the shared `--pill-inset`.
