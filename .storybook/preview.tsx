import React from 'react';
import type { Decorator, Preview } from '@storybook/react-vite';
// Global design tokens (the `:root` custom-property palette). Plain global
// stylesheet, exactly as the app loads it from src/index.tsx, so every
// story's `var(--token)` resolves. The dark palette lives behind
// `:root[data-theme='dark']`; the Theme toolbar toggle below sets that
// attribute on the preview <html>, mirroring src/settings/theme.ts.
import 'src/design_tokens.css';

/**
 * Apply the Theme toolbar choice the same way the app does: a `data-theme`
 * attribute on the iframe's <html>. design_tokens.css re-maps every color
 * token under `:root[data-theme='dark']`, and preview-head.html's body
 * bg/text read those tokens, so flipping the attribute recolors the whole
 * canvas, no per-component dark CSS needed (same contract as the app).
 */
const withTheme: Decorator = (Story, context) => {
  const theme = (context.globals.theme as string) || 'light';
  React.useEffect(() => {
    document.documentElement.dataset.theme = theme;
  }, [theme]);
  return <Story />;
};

const preview: Preview = {
  globalTypes: {
    theme: {
      description: 'Color theme',
      toolbar: {
        title: 'Theme',
        icon: 'contrast',
        items: [
          { value: 'light', title: 'Light', icon: 'sun' },
          { value: 'dark', title: 'Dark', icon: 'moon' },
        ],
        dynamicTitle: true,
      },
    },
  },
  initialGlobals: { theme: 'light' },
  decorators: [withTheme],
  parameters: {
    controls: {
      matchers: { color: /(background|color)$/i, date: /Date$/i },
    },
    layout: 'centered',
  },
};

export default preview;
