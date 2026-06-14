import type { Meta, StoryObj } from '@storybook/react-vite';
import React from 'react';
import { RenderedJot } from 'src/jot/resolved_jot';
import { createJotView } from 'src/jot_view/jot_view';
import { fromMidi } from 'src/midi/from_midi';
import { parse } from 'src/parser/parser';
import { rockJot, tripletJot } from 'src/fakes/fakes';
import type { Jot } from 'src/dsl/dsl';

/**
 * Library sandbox for the front-end loaders, the DSL parser
 * (`src/parser`) and the MIDI→Jot converter (`src/midi/from_midi`).
 *
 * Pick a `.jot` / `.txt` (DSL) or `.mid` / `.midi` file, or load one of
 * the built-in examples; the harness runs it through the real conversion
 * path and shows (a) the resulting Jot as pretty-printed text and (b) a
 * live JotView rendered from it. The point is to exercise one specific
 * part of the product (the loader/converter) in isolation, without the
 * rest of the app's chrome getting in the way.
 *
 * This is NOT a production component, it lives only in Storybook.
 */
const JotLoaderSandbox = () => {
  // One real JotView instance for the lifetime of the story. createJotView
  // wires up the stores + presenters; we drive it via documentPresenter.
  const view = React.useMemo(() => createJotView({}), []);
  const { View, document: documentStore, documentPresenter } = view;
  const [jotText, setJotText] = React.useState<string>('');
  const [error, setError] = React.useState<string | undefined>();

  const loadJot = React.useCallback(
    (jot: Jot, label: string) => {
      setError(undefined);
      try {
        setJotText(JSON.stringify(jot, null, 2));
        documentPresenter.setJot(new RenderedJot(jot, documentStore.viewConfig));
      } catch (e) {
        setError(`${label}: ${e instanceof Error ? e.message : String(e)}`);
      }
    },
    [documentPresenter, documentStore],
  );

  const onFile = async (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0];
    e.target.value = '';
    if (!file) return;
    try {
      const isMidi = /\.midi?$/i.test(file.name);
      const jot = isMidi ? fromMidi(await file.arrayBuffer()) : parse(await file.text());
      loadJot(jot, file.name);
    } catch (err) {
      setError(`${file.name}: ${err instanceof Error ? err.message : String(err)}`);
      setJotText('');
    }
  };

  return (
    <div style={{ display: 'flex', flexDirection: 'column', gap: 12, height: '100vh' }}>
      <div style={{ display: 'flex', gap: 8, alignItems: 'center', flex: '0 0 auto' }}>
        <input type="file" accept=".jot,.txt,.mid,.midi,text/plain" onChange={onFile} />
        <button type="button" onClick={() => loadJot(rockJot, 'rockJot')}>
          Load rock example
        </button>
        <button type="button" onClick={() => loadJot(tripletJot, 'tripletJot')}>
          Load triplet example
        </button>
      </div>
      {error && <pre style={{ color: 'crimson', flex: '0 0 auto', margin: 0 }}>{error}</pre>}
      <div style={{ display: 'flex', gap: 12, flex: '1 1 auto', minHeight: 0 }}>
        <pre
          style={{
            flex: '0 0 360px',
            overflow: 'auto',
            margin: 0,
            padding: 8,
            fontSize: 11,
            background: 'var(--color-bg-neutral, #fafaf7)',
            border: '1px solid var(--color-border, #ddd)',
            borderRadius: 6,
          }}
        >
          {jotText || '// Load a .jot / .mid file or an example to see the parsed Jot here.'}
        </pre>
        <div style={{ flex: '1 1 auto', minWidth: 0, border: '1px solid var(--color-border, #ddd)' }}>
          <View />
        </div>
      </div>
    </div>
  );
};

const meta = {
  title: 'Sandboxes/Jot loader',
  component: JotLoaderSandbox,
  parameters: { layout: 'fullscreen' },
} satisfies Meta<typeof JotLoaderSandbox>;
export default meta;

type Story = StoryObj<typeof meta>;

export const Sandbox: Story = {};
