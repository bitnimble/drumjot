import type { Meta, StoryObj } from '@storybook/react-vite';
import React from 'react';
import { fn } from 'storybook/test';
import { Tabs } from '../tabs';

/** Generic WAI-ARIA tab strip (today: the Transcribe dropdown's
 *  New ↔ Resume switch). Interactive: holds its own selection; changes
 *  also report to the Actions panel. Typed loosely because `Tabs` is a
 *  generic component, each story drives it through `render`. */
const meta: Meta = {
  title: 'Components/Tabs',
};
export default meta;

type Story = StoryObj;

export const TwoTabs: Story = {
  render: () => {
    const [value, setValue] = React.useState('new');
    const onChange = fn((v: string) => setValue(v));
    return (
      <Tabs
        ariaLabel="Transcribe mode"
        value={value}
        onChange={onChange}
        options={[
          { value: 'new', label: 'New' },
          { value: 'resume', label: 'Resume' },
        ]}
      />
    );
  },
};

export const WithDisabledTab: Story = {
  render: () => {
    const [value, setValue] = React.useState('a');
    const onChange = fn((v: string) => setValue(v));
    return (
      <Tabs
        ariaLabel="Example"
        value={value}
        onChange={onChange}
        options={[
          { value: 'a', label: 'Available' },
          { value: 'b', label: 'Disabled', disabled: true, title: 'Not available yet' },
          { value: 'c', label: 'Also available' },
        ]}
      />
    );
  },
};
