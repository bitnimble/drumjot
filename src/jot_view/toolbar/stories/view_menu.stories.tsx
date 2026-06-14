import type { Meta, StoryObj } from '@storybook/react-vite';
import { ThemeSection } from '../view_menu';

/**
 * The theme picker from the toolbar's View dropdown. Renders as radio-
 * style menu items bound to the global `themeStore`, so it works in
 * isolation, selecting a mode flips the live `data-theme` (and thus the
 * Storybook canvas) the same way it does in the app. Shown here outside
 * its usual dropdown panel.
 */
const meta: Meta = {
  title: 'Toolbar/ThemeSection',
  component: ThemeSection,
};
export default meta;

type Story = StoryObj<typeof ThemeSection>;

export const Default: Story = {};
