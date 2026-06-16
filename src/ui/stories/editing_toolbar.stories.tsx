import type { Meta, StoryObj } from '@storybook/react-vite';
import { runInAction } from 'mobx';
import React from 'react';
import { EditingStoreContext, EditingPresenterContext } from 'src/editing/editing_contexts';
import type { EditingPresenter } from 'src/editing/editing_presenter';
import { EditingStore, type EditMode } from 'src/editing/editing_store';
import { EditingToolbar } from 'src/editing/editing_toolbar';
import { Gallery, Variant } from 'src/ui/stories/_variants';

/**
 * The vertical floating mode toolbar (select / insert) pinned to the right edge
 * of the editor. Rendered here against a real (observable) {@link EditingStore}
 * plus a stand-in presenter whose only job is the `setMode` mutation the toolbar
 * calls, so the stories are live: clicking a button toggles the active state.
 *
 * The toolbar is `position: absolute`, so each instance is dropped into a
 * relatively-positioned frame for it to anchor within.
 */
const meta: Meta = {
  title: 'Editing/FloatingToolbar',
};
export default meta;

type Story = StoryObj;

/** A real store + a minimal presenter stand-in (only `setMode` is exercised). */
function ToolbarHarness({ initialMode }: { initialMode: EditMode }) {
  const store = React.useMemo(() => {
    const s = new EditingStore();
    runInAction(() => {
      s.mode = initialMode;
    });
    return s;
  }, [initialMode]);
  const presenter = React.useMemo(
    () =>
      ({
        setMode: (mode: EditMode) =>
          runInAction(() => {
            store.mode = mode;
          }),
      }) as unknown as EditingPresenter,
    [store]
  );
  return (
    <div
      style={{
        position: 'relative',
        width: 220,
        height: 140,
        border: '1px dashed var(--color-border)',
        borderRadius: 8,
      }}
    >
      <EditingStoreContext.Provider value={store}>
        <EditingPresenterContext.Provider value={presenter}>
          <EditingToolbar />
        </EditingPresenterContext.Provider>
      </EditingStoreContext.Provider>
    </div>
  );
}

/**
 * Both starting modes side by side. Each toolbar is interactive, clicking a
 * button switches the active mode and updates the highlight live.
 */
export const All: Story = {
  render: () => (
    <Gallery>
      <Variant label="Select mode active">
        <ToolbarHarness initialMode="select" />
      </Variant>
      <Variant label="Insert mode active">
        <ToolbarHarness initialMode="insert" />
      </Variant>
    </Gallery>
  ),
};
