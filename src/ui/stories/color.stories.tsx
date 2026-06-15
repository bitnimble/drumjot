import type { Meta, StoryObj } from '@storybook/react-vite';
import React from 'react';
import { fn } from 'storybook/test';
import { ColorPot } from 'src/ui/color_pot/color_pot';
import { ColorPicker } from 'src/ui/color_picker/color_picker';
import { Gallery, Variant } from 'src/ui/stories/_variants';

const PALETTE = ['#e4572e', '#f3a712', '#669bbc', '#3a7d44', '#8e44ad', '#2b2d42'];

/**
 * Colour chips + the reusable colour-picker popover used by the mixer's
 * per-instrument colour override.
 */
const meta: Meta = {
  title: 'Components/Color',
};
export default meta;

type Story = StoryObj;

/** The full picker (palette row + HSL wheel + hex + Reset), floated in a
 *  portal anchored to its swatch. Holds the live colour + open state and
 *  routes every commit to the Actions panel. */
function PickerDemo() {
  const [color, setColor] = React.useState('#669bbc');
  const [open, setOpen] = React.useState(false);
  const [hasOverride, setHasOverride] = React.useState(false);
  const [anchor, setAnchor] = React.useState<{ top: number; left: number } | null>(null);
  const ref = React.useRef<HTMLButtonElement>(null);
  const onChange = fn((hex: string) => {
    setColor(hex);
    setHasOverride(true);
  });
  const onReset = fn(() => {
    setColor('#669bbc');
    setHasOverride(false);
  });
  return (
    <>
      <ColorPot
        ref={ref}
        color={color}
        selected={open}
        ariaHasPopup="dialog"
        ariaExpanded={open}
        ariaLabel="Pick a colour"
        onClick={() => {
          const r = ref.current?.getBoundingClientRect();
          if (r) setAnchor({ top: r.bottom + 6, left: r.left });
          setOpen((o) => !o);
        }}
      />
      <ColorPicker
        open={open}
        anchor={anchor}
        value={color}
        palette={PALETTE}
        hasOverride={hasOverride}
        onChange={onChange}
        onReset={onReset}
        onClose={() => setOpen(false)}
      />
    </>
  );
}

/** Colour chips + the interactive picker, in one place. */
export const All: Story = {
  render: () => (
    <Gallery>
      <Variant label="Pots (first selected)">
        {PALETTE.map((c, i) => (
          <ColorPot key={c} color={c} selected={i === 0} onClick={fn()} ariaLabel={`Colour ${c}`} />
        ))}
      </Variant>
      <Variant label="Picker (click the swatch to open)">
        <PickerDemo />
      </Variant>
    </Gallery>
  ),
};
