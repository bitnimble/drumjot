import { expect, test } from '@playwright/test';

/**
 * End-to-end round-trip for the mutable `.jot` save format, exercising the
 * three things that have to hold together: lossless document serialisation,
 * the wholesale session reset on every load, and editor-metadata restore.
 *
 * The whole flow runs in one page.evaluate (single JS context), so the
 * `toMutableBytes()` result is handed straight to `loadMutableBytes()` with
 * no cross-boundary serialisation. The save bundle is built entirely
 * client-side; no transcriber backend is involved.
 */

const JOT_A = `{{ bpm: 120, time: "4/4", title: "Save Load A",
  instrumentMapping: { h: { name: "HiHat" }, s: { name: "Snare" }, k: { name: "Kick" } } }}
${Array.from({ length: 4 }, () => '(k+h s+h k+h s+h)').join('\n')}
`;

const JOT_B = `{{ bpm: 90, time: "4/4", title: "Save Load B",
  instrumentMapping: { s: { name: "Snare" }, k: { name: "Kick" } } }}
${Array.from({ length: 2 }, () => '(k s k s)').join('\n')}
`;

test('mutable .jot save round-trips the document + editor metadata, and a load resets', async ({
  page,
}) => {
  await page.goto('/');
  await page.waitForFunction(() => typeof (window as any).drumjot?.loadDsl === 'function');

  const result = await page.evaluate(
    async ({ a, b }) => {
      const dj = (window as any).drumjot;

      // Load song A, capture its document shape, then set a piece of editor
      // metadata the DSL can't carry (the drum-section master mute).
      dj.loadDsl(a);
      const snapA = dj.jotEditorStore.snapshot();
      const titleA = snapA?.title as string | undefined;
      const elemCountA = Object.keys(snapA?.elements ?? {}).length;
      dj.mixerPresenter.toggleDrumMasterMute();
      const mutedBeforeSave = dj.mixer.drumMasterMuted as boolean;

      // Serialise the session to mutable-format bytes.
      const bytes: Uint8Array = await dj.toMutableBytes();
      const magic = Array.from(bytes.slice(0, 4));

      // Loading a different song must wholesale-reset: the master mute from A
      // can't leak onto B.
      dj.loadDsl(b);
      const afterB = {
        title: dj.jotEditorStore.snapshot()?.title as string | undefined,
        muted: dj.mixer.drumMasterMuted as boolean,
      };

      // Restore the saved session: document (lossless) + the saved mute.
      await dj.loadMutableBytes(bytes);
      const snapRestore = dj.jotEditorStore.snapshot();
      const afterRestore = {
        title: snapRestore?.title as string | undefined,
        elemCount: Object.keys(snapRestore?.elements ?? {}).length,
        muted: dj.mixer.drumMasterMuted as boolean,
      };

      return { titleA, elemCountA, mutedBeforeSave, magic, afterB, afterRestore };
    },
    { a: JOT_A, b: JOT_B }
  );

  // Sanity: A loaded and the mute took.
  expect(result.titleA).toBe('Save Load A');
  expect(result.elemCountA).toBeGreaterThan(0);
  expect(result.mutedBeforeSave).toBe(true);

  // The bytes carry the mutable-format magic header ("DJOT").
  expect(result.magic).toEqual([0x44, 0x4a, 0x4f, 0x54]);

  // Loading B reset the session: B's document, and A's mute is gone.
  expect(result.afterB.title).toBe('Save Load B');
  expect(result.afterB.muted).toBe(false);

  // Restoring the save brings back A's document losslessly AND its editor
  // metadata (the master mute).
  expect(result.afterRestore.title).toBe('Save Load A');
  expect(result.afterRestore.elemCount).toBe(result.elemCountA);
  expect(result.afterRestore.muted).toBe(true);
});
