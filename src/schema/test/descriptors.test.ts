import { describe, expect, it } from 'bun:test';
import { z } from 'zod';
import { idMap, movableList, record } from 'src/schema/descriptors';

describe('record() normalization', () => {
  it('wraps a bare zod field into a reg descriptor', () => {
    const s = record({ title: z.string() });
    expect(s.fields.title.kind).toBe('reg');
  });

  it('leaves a higher-order container descriptor as-is', () => {
    const Note = record({ lane: z.string() });
    const s = record({ notes: idMap(Note), bars: movableList(Note) });
    expect(s.fields.notes.kind).toBe('idMap');
    expect(s.fields.bars.kind).toBe('movableList');
  });
});
