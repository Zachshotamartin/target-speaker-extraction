import process from 'node:process';
const LIMIT = 4 * 1024 * 1024;

export default async function oneVoice(req, res) {
  const action = new URL(req.url, 'http://localhost').pathname.split('/').filter(Boolean).at(-1);
  res.setHeader('Cache-Control', 'no-store');
  const fail = (code, detail) => {res.statusCode = code; res.setHeader('Content-Type', 'application/json'); res.end(JSON.stringify({detail}));};
  if (!['model', 'extract'].includes(action)) return fail(404, 'Not found');
  if (req.method !== (action === 'model' ? 'GET' : 'POST')) return fail(405, 'Method not allowed');
  const origin = req.headers.origin;
  const publicOrigin = process.env.ONE_VOICE_PUBLIC_ORIGIN;
  if (origin && origin !== publicOrigin && !['https://zachsm.com', 'https://www.zachsm.com', 'http://127.0.0.1:5291', 'http://localhost:5291', 'http://127.0.0.1:4175'].includes(origin)) return fail(403, 'This upload must come from the portfolio.');
  const base = process.env.ONE_VOICE_SERVICE_URL;
  if (!base) return fail(503, 'Uploads are not available yet. You can listen to the prepared examples above.');
  try {
    let body;
    if (action === 'extract') {
      if (!String(req.headers['content-type']).startsWith('multipart/form-data;')) return fail(415, 'Choose two audio files.');
      const chunks = []; let size = 0;
      for await (const chunk of req) {size += chunk.length; if (size > LIMIT) return fail(413, 'Combined upload must be under 4 MiB.'); chunks.push(chunk);}
      body = Buffer.concat(chunks);
    }
    const upstream = await fetch(`${base.replace(/\/$/, '')}/${action}`, {method: req.method, body, headers: {...(body ? {'Content-Type': req.headers['content-type']} : {}), ...(process.env.ONE_VOICE_SERVICE_TOKEN ? {Authorization: `Bearer ${process.env.ONE_VOICE_SERVICE_TOKEN}`} : {})}, signal: AbortSignal.timeout(110000)});
    res.statusCode = upstream.status;
    res.setHeader('Content-Type', upstream.headers.get('content-type') || 'application/json');
    for (const name of ['x-checkpoint-sha256','content-disposition']) if (upstream.headers.has(name)) res.setHeader(name, upstream.headers.get(name));
    res.end(Buffer.from(await upstream.arrayBuffer()));
  } catch { fail(503, 'The extraction service is unavailable. Please try again shortly.'); }
}
