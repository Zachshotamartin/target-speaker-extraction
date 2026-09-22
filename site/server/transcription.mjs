import {createHmac, randomUUID, timingSafeEqual} from 'node:crypto';
import process from 'node:process';
const LIMIT = 4 * 1024 * 1024;
const COOKIE = 'ov_session';
const UUID = '[a-f0-9]{8}-[a-f0-9]{4}-4[a-f0-9]{3}-[89ab][a-f0-9]{3}-[a-f0-9]{12}';
const routes = [
  [/^\/(health|demos)$/, ['GET']],
  [/^\/demos\/\d{1,2}\/(reference|mixture|target|absent|silence)$/, ['GET']],
  [/^\/transcriptions$/, ['POST']],
  [/^\/workspace\/(uploads|jobs)$/, ['POST']],
  [new RegExp(`^/workspace/requests/${UUID}$`), ['GET']],
  [new RegExp(`^/workspace/(?:uploads|jobs)/${UUID}$`), ['GET', 'DELETE']],
  [new RegExp(`^/workspace/uploads/${UUID}/chunks/\\d{1,3}$`), ['POST']],
  [new RegExp(`^/workspace/jobs/${UUID}/assets/(?:(?:original|speaker-[0-3]|reference-[0-3])\\.wav|captioned\\.mp4|report-[0-3]\\.json)/(?:info|chunks/\\d{1,3})$`), ['GET']],
  [new RegExp(`^/transcriptions/${UUID}$`), ['GET', 'DELETE']],
  [new RegExp(`^/transcriptions/${UUID}/(?:audio/(?:original|extracted)|export/(?:txt|srt|json))$`), ['GET']],
];
const signature = (value, secret) => createHmac('sha256', secret).update(value).digest('hex');

export function sessionOwner(cookie, secret, now = Date.now()) {
  const value = String(cookie || '').split(';').map(part => part.trim()).find(part => part.startsWith(`${COOKIE}=`))?.slice(COOKIE.length + 1);
  if (!value) return null;
  const [id, expires, signed] = value.split('.');
  if (!new RegExp(`^${UUID}$`).test(id) || !/^\d{13}$/.test(expires) || Number(expires) <= now || !/^[a-f0-9]{64}$/.test(signed)) return null;
  const expected = signature(`${id}.${expires}`, secret);
  return timingSafeEqual(Buffer.from(signed), Buffer.from(expected)) ? id : null;
}

export default async function transcription(req, res) {
  res.setHeader('Cache-Control', 'private, no-store');
  res.setHeader('X-Content-Type-Options', 'nosniff');
  const fail = (status, detail) => {res.statusCode = status; res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify({detail}));};
  const url = new URL(req.url, 'http://localhost');
  const path = url.searchParams.get('route') || '';
  const validQuery = url.pathname === '/api/poc' && url.searchParams.getAll('route').length === 1 && [...url.searchParams.keys()].every(key => key === 'route');
  const route = routes.find(([pattern]) => pattern.test(path));
  if (!route || !validQuery) return fail(404, 'Not found');
  if (!route[1].includes(req.method)) {res.setHeader('Allow', route[1].join(', ')); return fail(405, 'Method not allowed');}
  const host = String(req.headers.host || '');
  const local = !process.env.VERCEL && /^(localhost|127\.0\.0\.1):\d+$/.test(host);
  const origin = req.headers.origin;
  const allowed = ['https://one-voice.vercel.app'];
  if (local) allowed.push(`http://${host}`);
  if ((origin && !allowed.includes(origin)) || req.headers['sec-fetch-site'] === 'cross-site') return fail(403, 'Use the transcription workspace on this website.');
  const base = process.env.ONE_VOICE_TRANSCRIPTION_URL;
  const secret = process.env.ONE_VOICE_TRANSCRIPTION_TOKEN;
  if (!base || (!local && !secret)) return fail(503, 'Transcription is not configured yet. Please use the listening examples.');
  let upstreamURL;
  try {
    upstreamURL = new URL(base);
    if (!['http:', 'https:'].includes(upstreamURL.protocol) || upstreamURL.username || upstreamURL.password || upstreamURL.search || upstreamURL.hash || (!local && upstreamURL.protocol !== 'https:')) throw new Error();
  } catch {return fail(503, 'The transcription service is not configured.');}
  const signingKey = secret || 'loopback-preview-only';
  let owner = sessionOwner(req.headers.cookie, signingKey);
  const needsOwner = path.startsWith('/transcriptions/') || (path.startsWith('/workspace/') && path !== '/workspace/uploads');
  if (!owner && needsOwner) return fail(404, 'This result expired or belongs to another browser.');
  if (!owner && req.method === 'POST') {
    owner = randomUUID();
    const value = `${owner}.${Date.now() + 24 * 60 * 60 * 1000}`;
    res.setHeader('Set-Cookie', `${COOKIE}=${value}.${signature(value, signingKey)}; Path=/api/poc; HttpOnly; SameSite=Strict; Max-Age=86400${local ? '' : '; Secure'}`);
  }
  try {
    let body;
    if (req.method === 'POST') {
      const workspace = path.startsWith('/workspace/');
      const contentType = String(req.headers['content-type'] || '');
      const expected = workspace ? (path.includes('/chunks/') ? 'application/octet-stream' : 'application/json') : 'multipart/form-data;';
      const limit = workspace ? (path.includes('/chunks/') ? 1024 * 1024 : 256 * 1024) : LIMIT;
      if (!contentType.startsWith(expected)) return fail(415, 'Unsupported request format.');
      if (Number(req.headers['content-length']) > limit) return fail(413, 'Combined upload must be under 4 MiB.');
      let size = 0; const chunks = [];
      for await (const chunk of req) {size += chunk.length; if (size > limit) return fail(413, 'Combined upload must be under 4 MiB.'); chunks.push(Buffer.from(chunk));}
      body = Buffer.concat(chunks);
    }
    const upstream = await fetch(`${base.replace(/\/$/, '')}${path}`, {
      method: req.method, body, redirect: 'error', signal: AbortSignal.timeout(90000),
      headers: {...(body ? {'Content-Type': req.headers['content-type']} : {}), ...(secret ? {Authorization: `Bearer ${secret}`} : {}), ...(owner ? {'X-OneVoice-Owner': owner} : {})},
    });
    // Audio is at most 30 seconds; do not buffer an unbounded upstream response.
    let size = 0; const chunks = [];
    for await (const chunk of upstream.body || []) {size += chunk.length; if (size > 8 * 1024 * 1024) throw new Error(); chunks.push(Buffer.from(chunk));}
    res.statusCode = upstream.status;
    res.setHeader('Content-Type', upstream.headers.get('content-type') || 'application/json');
    if (upstream.headers.has('content-disposition')) res.setHeader('Content-Disposition', upstream.headers.get('content-disposition'));
    let output = Buffer.concat(chunks);
    if (path === '/health' && upstream.ok) output = Buffer.from(JSON.stringify({...JSON.parse(output.toString()), processing_location: local ? 'local' : 'hosted'}));
    res.end(output);
  } catch {return fail(503, 'The model service may be waking up. Please retry shortly.');}
}
