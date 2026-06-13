import type { Preview } from '@storybook/react-vite';
// Global design tokens (the `:root` custom-property palette). Plain global
// stylesheet, exactly as the app loads it from src/index.tsx, so every
// story's `var(--token)` resolves. Defaults to the light palette (the
// `:root` values); the dark overrides live behind `[data-theme]` and
// aren't needed for stories.
import 'src/design_tokens.css';

const preview: Preview = {
  parameters: {
    controls: {
      matchers: { color: /(background|color)$/i, date: /Date$/i },
    },
    layout: 'centered',
  },
};

export default preview;
