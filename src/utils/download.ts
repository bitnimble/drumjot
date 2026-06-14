/**
 * Trigger a client-side download of in-memory text as a file. Standard
 * Blob + object-URL + synthetic anchor-click; no server round-trip.
 *
 * The object URL is revoked on the next tick rather than immediately,
 * since some engines cancel the download if the URL is freed during the
 * same task that dispatched the click.
 */
export function downloadTextFile(
  filename: string,
  text: string,
  mime = 'text/plain;charset=utf-8',
): void {
  const url = URL.createObjectURL(new Blob([text], { type: mime }));
  const a = document.createElement('a');
  a.href = url;
  a.download = filename;
  a.rel = 'noopener';
  document.body.appendChild(a);
  a.click();
  a.remove();
  setTimeout(() => URL.revokeObjectURL(url), 0);
}
