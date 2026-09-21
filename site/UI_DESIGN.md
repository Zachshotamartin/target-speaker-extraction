# One Voice interface

The landing page retains the design from main at aca75b3: #fafaf8 background, #191919 text and controls, Arial typography, pill buttons, the original hero and signal artwork. Do not replace it with the transcription workspace or introduce a new palette.

Speech to Text is a separate tool, with a persistent result/editor area and settings. It has no marketing footer. Navigation keeps the tools mounted so it does not interrupt jobs or discard selected files.

Use `src/design-tokens.css` for spacing, type, control dimensions and colors. Workspace secondary text is 14px, controls 15px, body text 16px and transcripts 18px. Toolbars use a shared 80px row and 44px controls. Put style rules in the CSS files; do not add inline styles. Keep technical explanations in Details or About this tool.

Structural updates use `useMotionState` or `animateChange` from `src/motion.js`. The queue batches related updates and finishes one transition before starting the next. It covers page navigation, source and track changes, asynchronous pipeline stages, result tabs, notices and disclosures. Do not fabricate intermediate stages or delay backend work to animate them. Keep submission and input-loading guards synchronous even when their visual state is transitioning. Reduced-motion preferences bypass animations. A fade fallback supports browsers without native View Transitions.

Validation: `npm run build`, `node scripts/check-motion.mjs`, and browser checks using the public audio examples. Check navigation while a transcription runs, source switching, result tabs, exports, disclosure animation and responsive layouts. Training and inference environments are separate from this UI work.

Implementation references: [View Transition API](https://developer.mozilla.org/en-US/docs/Web/API/Document/startViewTransition) and [React flushSync](https://react.dev/reference/react-dom/flushSync).
