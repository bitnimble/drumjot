import type { Meta, StoryObj } from '@storybook/react-vite';
import React from 'react';
import { fn } from 'storybook/test';
import { NumberStepper } from '../number_stepper';

/** Compact [−][input][+] numeric control (used by the playback offsets +
 *  the lyrics offset). Controlled, so each story holds local state; the
 *  committed value also reports to the Actions panel. */
const meta = {
  title: 'Components/NumberStepper',
  component: NumberStepper,
  args: { step: 1, ariaLabel: 'Example value', onChange: fn() },
  render: (args) => {
    const [v, setV] = React.useState(args.value ?? 0);
    return (
      <NumberStepper
        {...args}
        value={v}
        onChange={(n) => {
          setV(n);
          args.onChange(n);
        }}
      />
    );
  },
} satisfies Meta<typeof NumberStepper>;
export default meta;

type Story = StoryObj<typeof meta>;

export const Default: Story = { args: { value: 4 } };
export const Decimal: Story = { args: { value: 0.25, step: 0.01, precision: 2 } };
export const Clamped: Story = {
  args: { value: 0, step: 1, min: 0, max: 10 },
};
export const Disabled: Story = { args: { value: 3, disabled: true } };
