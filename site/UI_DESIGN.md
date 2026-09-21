# One Voice interface

The landing page retains the design from main at aca75b3: #fafaf8 background, #191919 text and controls, Arial typography, pill buttons, the original hero and signal artwork. Do not replace it with the transcription workspace or introduce a new palette.

Speech to Text is a separate, open workspace: the transcript and audio setup sit directly on the page background, separated by a shared column gap and section rules. Do not enclose them in cards, a rounded container, or a contrasting panel. Notices and review sections also use simple rules. Navigation keeps the tools mounted so it does not interrupt jobs or discard selected files.

Results open on Compare when both transcriptions are available. Without One Voice and With One Voice each have their own matching audio and text, aligned into the same five-second sections; switching audio keeps the current position. Narrow layouts stack each pair of sections together. These are the unfiltered Whisper outputs. Selected voice is a separate view for attributed text and exports. Technical information lives in a collapsed Run details disclosure, not a third result tab. Keep actions attached to the transcript they affect.

Privacy is a separate page at `#privacy`, reached from the shared footer and upload notice. It uses the existing shared header and the same palette, typography, spacing tokens, and page transitions. Keep policy text in a readable column with a section index and simple rules, without cards. Policy section deep links use `#privacy-*` and must stay on the privacy page. Opening the policy must preserve selected files and active transcription jobs.

`SiteFooter` is rendered once, outside the page views, on every page including Speech to Text. Use `footer.css` for its explicit grid alignment and shared spacing. Keep it compact: brand, author credit, Privacy, and Source. The divider uses the same page gutter as the header. Do not reintroduce generic `footer` rules or page-specific footer copies.

The shared app shell is a flex column with a minimum height of `100dvh` and a growing main region. This keeps the footer at the viewport bottom on short pages and after the content on long pages. Header and footer retain their natural heights; the footer remains in document flow.

Every audio upload has a Record audio control directly beneath it, including both reference and conversation inputs on Overview and Speech to Text. Use the shared `AudioCaptureButton`, `useAudioRecorder`, and `audioCapture.js` implementation. Reference recordings require 3–10 seconds; conversation recordings allow up to 30 seconds. Preview and replacement stay beside the corresponding input. Only one input records at a time; navigating to another page discards unfinished capture and releases the microphone. WAV conversion keeps recordings compatible with both backends. Recording never submits or saves a profile automatically.

Reserve the root scrollbar gutter in `src/workspace-shell.css` so navigation and expanding content never change the available page width. Older browsers use an always-present vertical scrollbar as the fallback.

The header contains only Overview and Speech to Text, which navigate to the top of their pages. Listening and uploading are sections of Overview, reached through the homepage's Hear the difference, Hear this example, and Try your recording links. Keep those section anchors distinct from page navigation, and retain deep links to the listening and upload sections.

`src/SiteHeader.jsx` is mounted once above both pages. All header layout and responsive rules live in `src/header.css`, using shared design tokens. It has identical padding, typography, and control dimensions on every page, including mobile. Do not add workspace-specific header sizing or header rules to the product/workspace stylesheets; only its active link changes on navigation.

The current page has a dark tab with light text; inactive links stay neutral. Do not repeat Overview navigation as a breadcrumb inside Speech to Text.

The sticky header slides away on downward scrolling and returns on upward scrolling. Its space in the page remains constant. A 12px direction threshold prevents jitter; the header stays visible near the top, on page changes, and during keyboard navigation. Reduced motion disables the slide animation.

The top divider belongs to the shared header and moves with it. Do not duplicate this line on the hero or transcription page.

Use Example selects a reference/recording pair through one selector above both fields. The reference is a separate recording of the selected voice, not the clean target from the mixture. Label the speaker and conversation next to their players so this relationship is clear.

Floating About help dismisses on outside pointer interaction, focus moving outside, or Escape; Escape returns focus to the trigger. Keep it in the shared popover layer and its own named view-transition group so settings never paint over it during animation.

Use `src/design-tokens.css` for spacing, type, control dimensions and colors. Workspace secondary text is 14px, controls 15px, body text 16px and transcripts 18px. Toolbars use a shared 80px row and 44px controls. Put style rules in the CSS files; do not add inline styles. Keep technical explanations in Details or About this tool.

Hero, page, and section headings use shared fluid type tokens, with a mobile range that continues scaling from 700px down to 320px. Keep body and control minimums readable rather than scaling the entire interface down.

Structural updates use `useMotionState` or `animateChange` from `src/motion.js`. The queue batches related updates and finishes one transition before starting the next. It covers page navigation, source and track changes, asynchronous pipeline stages, result tabs, notices and disclosures. Do not fabricate intermediate stages or delay backend work to animate them. Keep submission and input-loading guards synchronous even when their visual state is transitioning. Reduced-motion preferences bypass animations. A fade fallback supports browsers without native View Transitions.

Validation: `npm run build`, `node scripts/check-motion.mjs`, and browser checks using the public audio examples. Check navigation while a transcription runs, source switching, result tabs, exports, disclosure animation and responsive layouts. Training and inference environments are separate from this UI work.

Implementation references: [View Transition API](https://developer.mozilla.org/en-US/docs/Web/API/Document/startViewTransition) and [React flushSync](https://react.dev/reference/react-dom/flushSync).
