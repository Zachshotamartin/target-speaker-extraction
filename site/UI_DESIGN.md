# One Voice interface

The landing page retains the design from main at aca75b3: #fafaf8 background, #191919 text and controls, Arial typography, pill buttons, the original hero and signal artwork. Do not replace it with the transcription workspace or introduce a new palette.

Speech to Text is a separate, open workspace: the transcript and audio setup sit directly on the page background, separated by a shared column gap and section rules. Do not enclose them in cards, a rounded container, or a contrasting panel. Notices and review sections also use simple rules. It has no marketing footer. Navigation keeps the tools mounted so it does not interrupt jobs or discard selected files.

Reserve the root scrollbar gutter in `src/workspace-shell.css` so navigation and expanding content never change the available page width. Older browsers use an always-present vertical scrollbar as the fallback.

The header's Overview and Speech to Text links navigate to the top of their pages. Listening is a section of Overview, reached through the homepage's Hear the difference and Hear this example links. Keep those section anchors distinct from page navigation, and retain deep links to the listening and upload sections.

`src/SiteHeader.jsx` is mounted once above both pages. All header layout and responsive rules live in `src/header.css`, using shared design tokens. It has identical padding, typography, and control dimensions on every page, including mobile. Do not add workspace-specific header sizing or header rules to the product/workspace stylesheets; only its active link changes on navigation.

The current page has a dark tab with light text; inactive links stay neutral and the recording shortcut is outlined. Do not repeat Overview navigation as a breadcrumb inside Speech to Text.

Use `src/design-tokens.css` for spacing, type, control dimensions and colors. Workspace secondary text is 14px, controls 15px, body text 16px and transcripts 18px. Toolbars use a shared 80px row and 44px controls. Put style rules in the CSS files; do not add inline styles. Keep technical explanations in Details or About this tool.

Hero, page, and section headings use shared fluid type tokens, with a mobile range that continues scaling from 700px down to 320px. Keep body and control minimums readable rather than scaling the entire interface down.

Structural updates use `useMotionState` or `animateChange` from `src/motion.js`. The queue batches related updates and finishes one transition before starting the next. It covers page navigation, source and track changes, asynchronous pipeline stages, result tabs, notices and disclosures. Do not fabricate intermediate stages or delay backend work to animate them. Keep submission and input-loading guards synchronous even when their visual state is transitioning. Reduced-motion preferences bypass animations. A fade fallback supports browsers without native View Transitions.

Validation: `npm run build`, `node scripts/check-motion.mjs`, and browser checks using the public audio examples. Check navigation while a transcription runs, source switching, result tabs, exports, disclosure animation and responsive layouts. Training and inference environments are separate from this UI work.

Implementation references: [View Transition API](https://developer.mozilla.org/en-US/docs/Web/API/Document/startViewTransition) and [React flushSync](https://react.dev/reference/react-dom/flushSync).
