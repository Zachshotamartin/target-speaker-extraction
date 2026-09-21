# Privacy policy maintenance

The product page is `site/src/PrivacyPage.jsx`, available at `#privacy`. It describes the current implementation, including the distinction between hosted extraction and local transcription. Review its date and wording whenever data handling or deployment changes.

Verified on September 21, 2026:

- `site/src/OneVoiceUpload.jsx`, `site/server/one-voice.mjs`, and `src/tse/public_api.py`: explicit submission to the extraction service; audio processed in memory; no application result persistence or training.
- `poc/server.py`, `poc/common.py`, and `poc/worker.py`: local-only transcription service, temporary job files, original/extracted playback copies, result JSON, 900-second expiry after completion, explicit cancellation/deletion, shutdown cleanup, and startup cleanup after a crash. Expiry cannot run while the service is stopped.
- `site/src/voiceProfiles.js` and `TranscriptionWorkspace.jsx`: explicit saving of a named reference and its audio in origin-scoped IndexedDB, deletion controls, temporary selected inputs in memory, and microphone permission/capture behavior.
- `site/src/audioCapture.js` and `useAudioRecorder.js`: shared reference and conversation recording on Overview and Speech to Text, browser-side WAV conversion, microphone release on stop/cancel/navigation, and no automatic upload or profile save.
- `poc/models.py`, `poc/pipeline.py`, and `poc/provision.py`: pretrained model downloads during setup, offline local inference, temporary voice embeddings, and saved similarity scores. JSON exports include more than accepted words.
- `site/src` and `site/index.html`: no analytics SDK, tracking pixel, or tracking-cookie integration. Provider infrastructure can still receive request metadata; no provider retention period is promised.

The project owner and contact address are published on [Zach Martin's contact page](https://zachsm.com/contact/message) and [portfolio privacy page](https://zachsm.com/privacy). The contact link opens the user's email application; this page does not submit a message.

The content organization follows the disclosure topics in the [Office of the Privacy Commissioner of Canada's consent guidance](https://www.priv.gc.ca/en/privacy-topics/privacy-laws-in-canada/the-personal-information-protection-and-electronic-documents-act-pipeda/p_principle/principles/p_consent/): what is handled, why, where, and the user's choices. The policy does not assert certification or blanket legal compliance.

If a hosted transcription service, analytics, accounts, a new inference operator, persistent storage, secondary use of audio, or changed retention is introduced, update the policy to name the actual providers and practices before making those features available. A code review alone cannot establish a hosting provider's independent logging or retention settings.
