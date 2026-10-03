# One Voice interface

The landing page retains its hero, signal artwork, warm background, Arial typography and pill controls. All colors live in `src/colors.css`; text and dark controls share the header/footer charcoal. Do not add inline colors or a separate palette.

Speech to Text is a separate, open workspace with Audio, Voice and Results steps. Use a context/sidebar column beside the active task, shared spacing and clear headings. Do not enclose the page in a card or use decorative section dividers. The controller preserves recordings, jobs and edits while step views change. Saved projects opens as a popover without moving the step navigation.

Results open on Transcript, with Compare and Edit audio available in the result navigation. Compare aligns the original and isolated audio with readable text passages in shared time windows; switching audio keeps the current position. Narrow layouts stack each pair together. Compare shows unfiltered Whisper outputs; Transcript contains attributed text used for exports. Edit audio supports word selection, reversible cuts and retimed exports. Run details opens in a viewport-bounded dialog whose body scrolls without extending the page. Keep actions attached to the transcript they affect.

The audio editor uses the same open workspace, controls and spacing tokens in three visible numbered steps: Select words, Make your edit, Listen & download. Clicking words always selects a range; do not reintroduce a Listen/Select mode switch. Play selection auditions the unedited passage, including removed words. Remove selected and Keep only selected explain their different effects. Edits automatically switch the preview to Edited voice; downloads always use the edited output regardless of the preview track. Removed words remain struck through and can be restored. Unconfirmed voice matches keep a dotted underline, independent of edit state. Cut changes animate and keep playback and exports on the same timeline. Do not overwrite the original audio or discard edits when switching result tabs or site pages. The Selected voice tab labels its accepted-word count with “words.”

Privacy is a separate page at `#privacy`, reached from the shared footer and upload notice. It uses the shared header, palette, typography, spacing tokens and page transitions. Keep policy text in a readable column with a section index, without cards or decorative rules. Policy section deep links use `#privacy-*` and must stay on the privacy page. Opening the policy must preserve selected files and active transcription jobs.

`SiteFooter` is rendered once, outside the page views, on every page including Speech to Text. Use `footer.css` for its explicit grid alignment and shared spacing. Keep it compact: brand, author credit and project links. Both the header and footer use full-width charcoal surfaces with their contents aligned to the page gutter and maximum content width. Do not add divider lines, generic `footer` rules or page-specific footer copies.

The shared app shell is a flex column with a minimum height of `100dvh` and a growing main region. This keeps the footer at the viewport bottom on short pages and after the content on long pages. Header and footer retain their natural heights; the footer remains in document flow.

Every audio upload has a Record audio control directly beneath it, including both reference and conversation inputs on Overview and Speech to Text. Use the shared `AudioCaptureButton`, `useAudioRecorder`, and `audioCapture.js` implementation. Reference recordings require 3–10 seconds; conversation recordings allow up to 30 seconds. Preview and replacement stay beside the corresponding input. Only one input records at a time; navigating to another page discards unfinished capture and releases the microphone. WAV conversion keeps recordings compatible with both backends. Recording never submits or saves a profile automatically.

Reserve the root scrollbar gutter in `src/workspace-shell.css` so navigation and expanding content never change the available page width. Older browsers use an always-present vertical scrollbar as the fallback.

The header contains only Overview and Speech to Text, which navigate to the top of their pages. Listening and uploading are sections of Overview, reached through the homepage's Hear the difference, Hear this example, and Try your recording links. Keep those section anchors distinct from page navigation, and retain deep links to the listening and upload sections.

`src/SiteHeader.jsx` is mounted once above both pages. All header layout and responsive rules live in `src/header.css`, using shared design tokens. It has identical padding, typography, and control dimensions on every page, including mobile. Do not add workspace-specific header sizing or header rules to the product/workspace stylesheets; only its active link changes on navigation.

The current page has a light tab with charcoal text against the charcoal header; inactive links use light text without a fill. Do not repeat Overview navigation as a breadcrumb inside Speech to Text.

The sticky header slides away on downward scrolling and returns on upward scrolling. Its space in the page remains constant. A 12px direction threshold prevents jitter; the header stays visible near the top, on page changes, and during keyboard navigation. Reduced motion disables the slide animation.

Do not add horizontal decorative page or section dividers. Keep functional input borders, focus outlines, table rows, chart axes and link underlines.

Use an example is an Audio source option with a conversation selector and Voice A/B choice. It loads a matching reference/recording pair and moves to Voice. The reference is a separate recording of the selected voice, not the clean target from the mixture. Label the speaker and conversation next to their players so this relationship is clear.

Floating About help dismisses on outside pointer interaction, focus moving outside, or Escape; Escape returns focus to the trigger. Keep it in the shared popover layer and its own named view-transition group so settings never paint over it during animation.

Use `src/design-tokens.css` for spacing, type and control dimensions, and `src/colors.css` for colors. Shared UI components live in individual files under `src/ui/`. Workspace secondary text is 14px, controls 15px, body text 16px and transcripts 18px. Put style rules in CSS files; do not add inline styles. Keep technical explanations in Run details or contextual help.

Use the shared `AudioPlayer` for Overview, references, recordings, comparisons and edits. Its timestamps stay on one line at every width. Audio and step pill insets use shared tokens. Evaluation graphs scale proportionally to the available width without internal horizontal scrolling; keep full-size downloads available.

Hero, page, and section headings use shared fluid type tokens, with a mobile range that continues scaling from 700px down to 320px. Keep body and control minimums readable rather than scaling the entire interface down.

The landing hero and transcription workspace share `--page-intro-space` (40–76px) above their content. The hero copy and signal artwork align at their top edges; do not vertically center the text against the artwork.

Structural updates use `useMotionState` or `animateChange` from `src/motion.js`. The queue batches related updates and finishes one transition before starting the next. It covers page navigation, source and track changes, asynchronous pipeline stages, result tabs, notices and disclosures. Do not fabricate intermediate stages or delay backend work to animate them. Keep submission and input-loading guards synchronous even when their visual state is transitioning. Reduced-motion preferences bypass animations. A fade fallback supports browsers without native View Transitions.

Page navigation fades only `main` while its layout and scroll destination change. The shared header and footer remain opaque and mounted, preventing the charcoal surfaces from flashing light during navigation. Do not animate the opacity of the whole app shell.

Validation: `npm run build`, `node scripts/check-motion.mjs`, and browser checks using the public audio examples. Check navigation while a transcription runs, source switching, result tabs, exports, disclosure animation and responsive layouts. Training and inference environments are separate from this UI work.

Implementation references: [View Transition API](https://developer.mozilla.org/en-US/docs/Web/API/Document/startViewTransition) and [React flushSync](https://react.dev/reference/react-dom/flushSync).
