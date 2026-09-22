import React from 'react';
import './privacy.css';

const sections = [
  ['information', 'Information we process'],
  ['processing', 'Where processing happens'],
  ['storage', 'Storage and deletion'],
  ['choices', 'Your choices'],
  ['website', 'Website and external services'],
  ['contact', 'Questions and updates'],
];

export default function PrivacyPage() {
  return <article className="privacy-page" aria-labelledby="privacy-title">
    <header className="privacy-heading">
      <p className="privacy-updated">Last updated <time dateTime="2026-09-22">September 22, 2026</time></p>
      <h1 id="privacy-title">Privacy policy</h1>
      <p className="privacy-intro">How One Voice handles your recordings, voice references, and transcripts.</p>
    </header>
    <div className="privacy-layout">
      <nav className="privacy-contents" aria-label="Privacy policy sections">
        <p>On this page</p>
        <ol>{sections.map(([id, title], index) => <li key={id}><a className="continuous-underline" href={`#privacy-${id}`}><span aria-hidden="true">{String(index + 1).padStart(2, '0')}</span>{title}</a></li>)}</ol>
      </nav>
      <div className="privacy-body">
        <p className="privacy-scope">One Voice is an independent project by Zach Martin. This policy covers the One Voice website, its voice extraction tool, and its speech-to-text pipeline, both hosted and local. No account is required, and submitted recordings are not used to train models.</p>

        <section id="privacy-information" aria-labelledby="privacy-information-title">
          <h2 id="privacy-information-title">Information we process</h2>
          <p>When you submit a job, the app processes the recording you select and a separate voice reference. A recording can contain other people’s voices and personal information in what they say. Only submit recordings you have permission to use.</p>
          <p>The tools produce isolated audio and, for speech-to-text, words, timestamps, and voice-match scores. Temporary numerical voice features help compare the recording with your reference. They are used for that job, not stored in a persistent identity database or used to verify a person’s identity.</p>
          <p>The purpose is to perform your requested extraction or transcription and let you review the result. One Voice does not sell this information or use submitted audio for advertising or model training.</p>
        </section>

        <section id="privacy-processing" aria-labelledby="privacy-processing-title">
          <h2 id="privacy-processing-title">Where processing happens</h2>
          <h3>Voice extraction on Overview</h3>
          <p>Choosing files does not submit them. When you select <strong>Extract voice</strong>, both recordings are sent to the extraction service. On this public website, recordings pass through Vercel to our extraction service on Hugging Face. Processing happens on the server, not inside your browser. In the local development setup, the extraction service runs on the same computer.</p>
          <h3>Speech-to-text</h3>
          <p>When you select <strong>Find voices</strong> or <strong>Extract selected voice &amp; transcribe</strong> on the public website, your recording and reference pass through Vercel to our service hosted on Hugging Face. One Voice, Whisper, speech detection, and voice matching run in that service. Comparison mode also transcribes the original recording. In the local installation, these models run on your computer.</p>
          <p>The worker uses downloaded model files and does not call OpenAI or a separate transcription provider. Hugging Face hosts the public worker and therefore processes the submitted audio. The local worker does not send audio to a cloud service. Installing the models requires downloads from model distributors; those downloads do not include your recordings.</p>
          <h3>Transcript audio editing</h3>
          <p>After transcription, <strong>Edit audio</strong> downloads the original and isolated audio into the open page. Selecting words, making cuts, and generating edited audio and subtitle exports happen in your browser. Audio edits remain local. Exporting captioned video sends the original video, edited audio, retained time ranges, and caption text to the service for rendering. Suggested speaker samples and labels are approximate, not verified identities.</p>
        </section>

        <section id="privacy-storage" aria-labelledby="privacy-storage-title">
          <h2 id="privacy-storage-title">Storage and deletion</h2>
          <dl className="privacy-retention">
            <div><dt>Extraction uploads</dt><dd>The Overview extraction service processes audio in memory. The application does not save the uploaded recordings or the generated output to a result database. The returned audio remains available in the open page until it is replaced or the page is reloaded or closed.</dd></div>
            <div><dt>Transcription jobs</dt><dd>Uploads are temporarily written to the worker’s disk: on Hugging Face for the public demo, or on your computer for a local installation. Incomplete recording uploads expire after one hour. The worker removes submitted source files when it finishes. A playable copy of the original recording, the isolated audio, and the transcript with its scores remain for 15 minutes after completion while the service is running. Server results expire automatically; cancelling a job also removes its files.</dd></div>
            <div><dt>Interrupted cleanup</dt><dd>A normal service shutdown removes jobs. If the service crashes, leftover files can remain until its next startup cleans them up or the hosting platform discards the container. A hosted restart can also make a temporary result unavailable before the 15-minute limit. The 15-minute timer does not run while the service is stopped.</dd></div>
            <div><dt>Saved voice references</dt><dd>Only choosing <strong>Save reference</strong> stores a reference recording and its profile name in this browser’s local database. It remains until you delete it or clear this site’s browser data. One Voice does not sync saved profiles to an account or cloud database.</dd></div>
            <div><dt>Audio edits</dt><dd>Recording projects automatically save source audio or video, voice references, completed tracks, transcripts, and edit decisions in IndexedDB on this device. They survive a reload; clearing browser data removes them. Use Saved projects → Delete to remove a local project. Downloads remain until you delete them. Undo history is kept only while the editor is open.</dd></div>
            <div><dt>Your files and downloads</dt><dd>Deleting a job does not delete your original files, a saved voice reference, or exports you have downloaded. Selected inputs also remain in the open page until replaced or the page is reloaded or closed. You control copies kept on your device and in your backups.</dd></div>
          </dl>
          <p>The full JSON export includes comparison text and uncertain or excluded words. The unedited audio export contains the complete isolated output. Edit audio exports contain the words and audio you keep, including any unconfirmed matches you leave in. Review these files before sharing them; they can contain more than the selected-voice transcript.</p>
        </section>

        <section id="privacy-choices" aria-labelledby="privacy-choices-title">
          <h2 id="privacy-choices-title">Your choices</h2>
          <ul>
            <li>Use the prepared examples without providing your own recordings.</li>
            <li>Allow microphone access only if you want to record a voice reference or recording. Recording begins after you select <strong>Record audio</strong> and grant permission. References stop at 10 seconds and other recordings at 30 seconds. Leaving the current page or choosing <strong>Discard</strong> stops the microphone and discards the unfinished recording. Completed workspace recordings are saved in this browser and are sent only when you ask to find voices, extract, or export video.</li>
            <li>Keep a reference temporary, or explicitly save it for reuse. To remove a saved reference, select it under <strong>Saved on this device</strong> and choose <strong>Delete</strong>.</li>
            <li>Delete a completed transcription from <strong>Selected voice → Export → Delete this result</strong>, or cancel an active job.</li>
            <li>Revoke microphone permission and clear saved site data through your browser settings.</li>
          </ul>
          <p>Switching between pages preserves selected files and active jobs. Navigating to this policy does not delete them.</p>
        </section>

        <section id="privacy-website" aria-labelledby="privacy-website-title">
          <h2 id="privacy-website-title">Website and external services</h2>
          <p>The One Voice interface does not include visitor analytics, advertising pixels, or tracking cookies. Its saved-reference database is used to provide the feature you choose, not to track browsing activity. Hosted transcription uses an essential, HttpOnly session cookie to keep job access specific to your browser. It expires after one day and is not used for tracking.</p>
          <p>When you visit a hosted version or submit hosted audio, the website and inference infrastructure receive request information such as your network address, browser information, request time, and requested URL. Hosting providers may process this information for delivery, security, and operations under their own practices. The application does not intentionally log recording contents or transcripts.</p>
          <p>Source-code links, research links, and the project owner’s website take you to other services with their own privacy policies. If you email a question, your email address and message are handled by the email providers involved so you can receive a reply. Do not include recordings or transcripts unless they are needed for your request.</p>
          <p>The local transcription service is restricted to local access and marks responses not to be cached. These measures do not make a shared computer, exported file, or recording immune to access by someone who controls that device.</p>
        </section>

        <section id="privacy-contact" aria-labelledby="privacy-contact-title">
          <h2 id="privacy-contact-title">Questions and updates</h2>
          <p>For privacy questions or requests about information held by the project, contact Zach Martin at <a href="mailto:zachsm@alumni.stanford.edu">zachsm@alumni.stanford.edu</a>. Describe the request without including sensitive audio. Files stored only in your local installation or browser must be managed on your device; the project owner cannot remotely retrieve or delete them.</p>
          <p>This page will be updated when the project’s data handling changes, with the revision date shown above. A separately operated or modified deployment may have different practices; check the policy supplied by its operator.</p>
        </section>
      </div>
    </div>
  </article>;
}
