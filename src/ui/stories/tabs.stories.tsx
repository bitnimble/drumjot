import type { Meta, StoryObj } from '@storybook/react-vite';
import React from 'react';
import { fn } from 'storybook/test';
import { Tabs } from 'src/ui/tabs/tabs';
import { Gallery, Variant } from 'src/ui/stories/_variants';

/** Generic WAI-ARIA tab strip (today: the Transcribe dropdown's New ↔
 *  Resume switch). Each instance holds its own selection; changes also
 *  report to the Actions panel. Typed loosely because `Tabs` is generic. */
const meta: Meta = {
  title: 'Components/Tabs',
};
export default meta;

type Story = StoryObj;

function TabsDemo({
  options,
  initial,
}: {
  options: React.ComponentProps<typeof Tabs>['options'];
  initial: string;
}) {
  const [value, setValue] = React.useState(initial);
  const onChange = fn((v: string) => setValue(v));
  return <Tabs ariaLabel="Example tabs" value={value} onChange={onChange} options={options} />;
}

/** Every Tabs variant in one place; each one actually switches. */
export const All: Story = {
  render: () => (
    <Gallery>
      <Variant label="Two tabs">
        <TabsDemo
          initial="new"
          options={[
            { value: 'new', label: 'New' },
            { value: 'resume', label: 'Resume' },
          ]}
        />
      </Variant>
      <Variant label="With a disabled tab">
        <TabsDemo
          initial="a"
          options={[
            { value: 'a', label: 'Available' },
            { value: 'b', label: 'Disabled', disabled: true, title: 'Not available yet' },
            { value: 'c', label: 'Also available' },
          ]}
        />
      </Variant>
    </Gallery>
  ),
};
