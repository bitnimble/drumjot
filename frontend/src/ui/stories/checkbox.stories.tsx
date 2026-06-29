import type { Meta, StoryObj } from '@storybook/react-vite';
import React from 'react';
import { fn } from 'storybook/test';
import { Checkbox } from 'src/ui/checkbox/checkbox';
import { Gallery, Variant } from 'src/ui/stories/_variants';

const meta: Meta = {
  title: 'Components/Checkbox',
};
export default meta;

type Story = StoryObj;

/** Interactive instance: holds its own state so the box toggles; the
 *  change still reports to the Actions panel. */
function InteractiveCheckbox() {
  const [on, setOn] = React.useState(false);
  const onChange = fn((e: React.ChangeEvent<HTMLInputElement>) => setOn(e.target.checked));
  return <Checkbox checked={on} onChange={onChange} />;
}

/** Every Checkbox state in one place. */
export const All: Story = {
  render: () => (
    <Gallery>
      <Variant label="Unchecked">
        <Checkbox checked={false} onChange={fn()} />
      </Variant>
      <Variant label="Checked">
        <Checkbox checked onChange={fn()} />
      </Variant>
      <Variant label="Disabled">
        <Checkbox checked disabled onChange={fn()} />
      </Variant>
      <Variant label="Interactive (toggles)">
        <InteractiveCheckbox />
      </Variant>
    </Gallery>
  ),
};
