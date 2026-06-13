import type { Meta, StoryObj } from '@storybook/react-vite';
import React from 'react';
import { fn } from 'storybook/test';
import { Checkbox } from '../checkbox';

const meta = {
  title: 'Components/Checkbox',
  component: Checkbox,
  args: { onChange: fn() },
} satisfies Meta<typeof Checkbox>;
export default meta;

type Story = StoryObj<typeof meta>;

export const Unchecked: Story = { args: { checked: false } };
export const Checked: Story = { args: { checked: true } };
export const Disabled: Story = { args: { checked: true, disabled: true } };

/** Interactive: holds its own state so the box actually toggles; the
 *  change still reports to the Actions panel. */
export const Interactive: Story = {
  render: (args) => {
    const [on, setOn] = React.useState(false);
    return (
      <Checkbox
        {...args}
        checked={on}
        onChange={(e) => {
          setOn(e.target.checked);
          args.onChange?.(e);
        }}
      />
    );
  },
};
