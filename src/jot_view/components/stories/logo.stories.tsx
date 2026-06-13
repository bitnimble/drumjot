import type { Meta, StoryObj } from '@storybook/react-vite';
import { Logo } from '../logo';

const meta = {
  title: 'Components/Logo',
  component: Logo,
  args: { size: 56 },
} satisfies Meta<typeof Logo>;
export default meta;

type Story = StoryObj<typeof meta>;

export const Default: Story = {};
export const Small: Story = { args: { size: 24 } };
export const Large: Story = { args: { size: 128, title: 'Drumjot' } };
