import type { Meta, StoryObj } from '@storybook/react-vite';
import React from 'react';
import { fn } from 'storybook/test';
import { NumberStepper } from 'src/ui/number_stepper/number_stepper';
import { Gallery, Variant } from 'src/ui/stories/_variants';

/** Compact [−][input][+] numeric control (used by the playback offsets +
 *  the lyrics offset). Controlled, so each instance holds local state; the
 *  committed value also reports to the Actions panel. */
const meta: Meta = {
  title: 'Components/NumberStepper',
};
export default meta;

type Story = StoryObj;

type DemoProps = Omit<React.ComponentProps<typeof NumberStepper>, 'onChange'>;

/** Self-contained interactive stepper for one gallery cell. */
function StepperDemo(props: DemoProps) {
  const [v, setV] = React.useState(props.value);
  const onChange = fn((n: number) => setV(n));
  return <NumberStepper {...props} value={v} onChange={onChange} />;
}

/** Every NumberStepper variant in one place; each one actually steps. */
export const All: Story = {
  render: () => (
    <Gallery>
      <Variant label="Default (step 1)">
        <StepperDemo value={4} step={1} ariaLabel="Example value" />
      </Variant>
      <Variant label="Decimal (step 0.01, precision 2)">
        <StepperDemo value={0.25} step={0.01} precision={2} ariaLabel="Decimal value" />
      </Variant>
      <Variant label="Clamped (min 0, max 10)">
        <StepperDemo value={0} step={1} min={0} max={10} ariaLabel="Clamped value" />
      </Variant>
      <Variant label="Disabled">
        <StepperDemo value={3} step={1} disabled ariaLabel="Disabled value" />
      </Variant>
    </Gallery>
  ),
};
