import type { Meta, StoryObj } from '@storybook/react-vite';
import { fn } from 'storybook/test';
import { ClearButton, IconButton, MuteButton, SoloButton } from 'src/ui/icon_button/icon_button';
import { Gallery, Variant } from 'src/ui/stories/_variants';

/**
 * Compact 18×18 icon button shared by the mixer-row controls, plus its
 * specialised Mute / Solo / Clear wrappers. Every handler is routed to a
 * spy (`fn()`) so clicks show up in the Actions panel.
 */
const meta: Meta = {
  title: 'Components/IconButton',
};
export default meta;

type Story = StoryObj;

/** Every icon-button variant + its on/off states in one place. */
export const All: Story = {
  render: () => (
    <Gallery>
      <Variant label="Plain">
        <IconButton onClick={fn()}>M</IconButton>
      </Variant>
      <Variant label="Mute (off / on)">
        <MuteButton active={false} onToggle={fn()} offTitle="Mute kick" onTitle="Unmute kick" />
        <MuteButton active onToggle={fn()} offTitle="Mute kick" onTitle="Unmute kick" />
      </Variant>
      <Variant label="Solo (off / on)">
        <SoloButton active={false} onToggle={fn()} offTitle="Solo kick" onTitle="Unsolo kick" />
        <SoloButton active onToggle={fn()} offTitle="Solo kick" onTitle="Unsolo kick" />
      </Variant>
      <Variant label="Clear">
        <ClearButton onClear={fn()} label="Remove the kick audio track" />
      </Variant>
    </Gallery>
  ),
};
