import type { Meta, StoryObj } from '@storybook/react-vite';
import React from 'react';
import { fn } from 'storybook/test';
import { ColorPot } from '../color_pot';
import { ColorPicker } from '../color_picker';

const PALETTE = ['#e4572e', '#f3a712', '#669bbc', '#3a7d44', '#8e44ad', '#2b2d42'];

/**
 * Colour chips + the reusable colour-picker popover used by the mixer's
 * per-instrument colour override. The picker is fully controlled, so the
 * interactive story holds the live colour + open state and routes every
 * commit to the Actions panel.
 */
const meta: Meta = {
  title: 'Components/Color',
};
export default meta;

type Story = StoryObj;

export const Pots: Story = {
  render: () => (
    <div style={{ display: 'flex', gap: 8 }}>
      {PALETTE.map((c, i) => (
        <ColorPot key={c} color={c} selected={i === 0} onClick={fn()} ariaLabel={`Colour ${c}`} />
      ))}
    </div>
  ),
};

/** Click the swatch to open the picker (palette row + HSL wheel + hex +
 *  Reset); it floats in a portal anchored to the swatch. */
export const Picker: Story = {
  render: () => {
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
  },
};
