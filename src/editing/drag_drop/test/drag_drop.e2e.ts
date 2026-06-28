import { expect, test, type Page } from '@playwright/test';

// Two distinct valid jots so a load is detectable by its title.
const JOT_A =
  '{{ bpm: 120, time: "4/4", title: "Alpha", instrumentMapping: { k: { name: "Kick" } } }}\n| k . s . k . s . |';
const JOT_B =
  '{{ bpm: 90, time: "4/4", title: "Bravo", instrumentMapping: { k: { name: "Kick" } } }}\n| k . k . k . k . |';

/** Dispatch a synthetic file drag-and-drop onto the app-shell container.
 *  Builds a real `DataTransfer` with in-page `File`s so the drop routes
 *  through the production handlers exactly as an OS drag would. */
async function dropFiles(
  page: Page,
  files: { name: string; content: string; type?: string }[],
  phase: 'enter' | 'drop' = 'drop'
) {
  await page.evaluate(
    ({ files, phase }) => {
      const dt = new DataTransfer();
      for (const f of files) {
        dt.items.add(new File([f.content], f.name, { type: f.type ?? 'text/plain' }));
      }
      const target = document.querySelector('[data-testid="app-container"]')!;
      const fire = (kind: string) =>
        target.dispatchEvent(new DragEvent(kind, { bubbles: true, dataTransfer: dt }));
      fire('dragenter');
      fire('dragover');
      if (phase === 'drop') fire('drop');
    },
    { files, phase }
  );
}

async function loadedTitle(page: Page): Promise<string | undefined> {
  return page.evaluate(() => (window as any).drumjot.jotEditorStore.jot?.title);
}

/** Build a minimal stored (uncompressed) zip with the given entries, as a
 *  Node Buffer suitable for Playwright's `setInputFiles`. Mirrors the
 *  unit-test builder; enough to exercise the zip auto-detect path. */
function makeStoredZip(entries: { name: string; content: string }[]): Buffer {
  const locals: Buffer[] = [];
  const central: Buffer[] = [];
  let offset = 0;
  for (const { name, content } of entries) {
    const nameBuf = Buffer.from(name);
    const data = Buffer.from(content);
    const local = Buffer.alloc(30 + nameBuf.length + data.length);
    local.writeUInt32LE(0x04034b50, 0);
    local.writeUInt16LE(0, 8); // stored
    local.writeUInt32LE(data.length, 18);
    local.writeUInt32LE(data.length, 22);
    local.writeUInt16LE(nameBuf.length, 26);
    nameBuf.copy(local, 30);
    data.copy(local, 30 + nameBuf.length);

    const cen = Buffer.alloc(46 + nameBuf.length);
    cen.writeUInt32LE(0x02014b50, 0);
    cen.writeUInt16LE(0, 10); // stored
    cen.writeUInt32LE(data.length, 20);
    cen.writeUInt32LE(data.length, 24);
    cen.writeUInt16LE(nameBuf.length, 28);
    cen.writeUInt32LE(offset, 42);
    nameBuf.copy(cen, 46);

    locals.push(local);
    central.push(cen);
    offset += local.length;
  }
  const cdSize = central.reduce((n, c) => n + c.length, 0);
  const eocd = Buffer.alloc(22);
  eocd.writeUInt32LE(0x06054b50, 0);
  eocd.writeUInt16LE(entries.length, 8);
  eocd.writeUInt16LE(entries.length, 10);
  eocd.writeUInt32LE(cdSize, 12);
  eocd.writeUInt32LE(offset, 16);
  return Buffer.concat([...locals, ...central, eocd]);
}

test.beforeEach(async ({ page }) => {
  await page.goto('/');
  await page.waitForFunction(() => typeof (window as any).drumjot?.loadDsl === 'function');
});

test('shows the drop overlay while a file drag hovers', async ({ page }) => {
  await expect(page.getByTestId('file-drop-overlay')).toHaveCount(0);
  await dropFiles(page, [{ name: 'x.jot', content: JOT_A }], 'enter');
  await expect(page.getByTestId('file-drop-overlay')).toBeVisible();
});

test('dropping a .jot onto the empty state loads it without a confirm', async ({ page }) => {
  await expect(page.locator('[data-testid^="instrument-track-"]').first()).toHaveCount(0);
  await dropFiles(page, [{ name: 'alpha.jot', content: JOT_A }]);
  await page.waitForSelector('[data-testid^="instrument-track-"]');
  await expect(page.getByTestId('drop-confirm-modal')).toHaveCount(0);
  expect(await loadedTitle(page)).toBe('Alpha');
});

test('dropping a .jot over a loaded score confirms before replacing', async ({ page }) => {
  await page.evaluate((src) => (window as any).drumjot.loadDsl(src), JOT_A);
  await page.waitForSelector('[data-testid^="instrument-track-"]');

  // Cancel keeps the original score.
  await dropFiles(page, [{ name: 'bravo.jot', content: JOT_B }]);
  await expect(page.getByTestId('drop-confirm-modal')).toBeVisible();
  await page.getByTestId('drop-confirm-cancel').click();
  await expect(page.getByTestId('drop-confirm-modal')).toHaveCount(0);
  expect(await loadedTitle(page)).toBe('Alpha');

  // Replace swaps it in.
  await dropFiles(page, [{ name: 'bravo.jot', content: JOT_B }]);
  await page.getByTestId('drop-confirm-replace').click();
  await expect.poll(() => loadedTitle(page)).toBe('Bravo');
});

test('dropping an unrecognised file surfaces an error toast', async ({ page }) => {
  await dropFiles(page, [{ name: 'notes.pdf', content: 'nope' }]);
  await expect(page.getByText(/Don't know how to load/)).toBeVisible();
});

test('toolbar "Load zip" auto-detects a zipped .jot and loads it', async ({ page }) => {
  // A zip whose only entry is a valid .jot routes through classifyZip →
  // extract → loadJotFile, the same flow a drop uses.
  const buffer = makeStoredZip([{ name: 'groove.jot', content: JOT_A }]);
  await page.getByTestId('load-zip-input').setInputFiles({
    name: 'pack.zip',
    mimeType: 'application/zip',
    buffer,
  });
  await page.waitForSelector('[data-testid^="instrument-track-"]');
  expect(await loadedTitle(page)).toBe('Alpha');
});

test('empty-state "Open zip" auto-detects a zipped .jot and loads it', async ({ page }) => {
  // The welcome screen's zip picker routes through the same auto-detect
  // flow; no confirm since nothing is loaded yet.
  const buffer = makeStoredZip([{ name: 'groove.jot', content: JOT_A }]);
  await page.getByTestId('empty-state-open-zip-input').setInputFiles({
    name: 'pack.zip',
    mimeType: 'application/zip',
    buffer,
  });
  await page.waitForSelector('[data-testid^="instrument-track-"]');
  await expect(page.getByTestId('drop-confirm-modal')).toHaveCount(0);
  expect(await loadedTitle(page)).toBe('Alpha');
});
