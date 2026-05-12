/** Base URL: Vite dev proxies /api; production uses same origin. */
export async function getJson(path) {
  const r = await fetch(path);
  if (!r.ok) throw new Error(`${path} ${r.status}`);
  return r.json();
}
