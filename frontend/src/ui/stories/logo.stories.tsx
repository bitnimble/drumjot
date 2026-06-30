import type { Meta, StoryObj } from '@storybook/react-vite';
import { Logo } from 'src/ui/logo/logo';
import { Gallery, Variant } from 'src/ui/stories/_variants';

const meta: Meta = {
  title: 'Components/Logo',
};
export default meta;

type Story = StoryObj;

/** The logo at the sizes it's used, in one place. */
export const All: Story = {
  render: () => (
    <Gallery>
      <Variant label="Small (24)">
        <Logo size={24} />
      </Variant>
      <Variant label="Medium (56)">
        <Logo size={56} />
      </Variant>
      <Variant label="Large (128, with title)">
        <Logo size={128} title="Drumjot" />
      </Variant>
    </Gallery>
  ),
};
