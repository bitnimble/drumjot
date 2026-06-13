import type { Meta, StoryObj } from '@storybook/react-vite';
import { fn } from 'storybook/test';
import { ClearButton, IconButton, MuteButton, SoloButton } from '../icon_button';

/**
 * Compact 18×18 icon button shared by the mixer-row controls, plus its
 * specialised Mute / Solo / Clear wrappers. Every handler is routed to a
 * spy (`fn()`) so clicks show up in the Actions panel.
 */
const meta = {
  title: 'Components/IconButton',
  component: IconButton,
  args: { onClick: fn(), children: 'M' },
} satisfies Meta<typeof IconButton>;
export default meta;

type Story = StoryObj<typeof meta>;

export const Plain: Story = {};

export const Mute: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: 8 }}>
      <MuteButton active={false} onToggle={fn()} offTitle="Mute kick" onTitle="Unmute kick" />
      <MuteButton active onToggle={fn()} offTitle="Mute kick" onTitle="Unmute kick" />
    </div>
  ),
};

export const Solo: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: 8 }}>
      <SoloButton active={false} onToggle={fn()} offTitle="Solo kick" onTitle="Unsolo kick" />
      <SoloButton active onToggle={fn()} offTitle="Solo kick" onTitle="Unsolo kick" />
    </div>
  ),
};

export const Clear: Story = {
  render: () => <ClearButton onClear={fn()} label="Remove the kick audio track" />,
};
