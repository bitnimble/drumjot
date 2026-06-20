import { FileText, Layers, type LucideIcon } from 'lucide-react';
import React from 'react';
import type { SidebarPanelId } from '../sidebar_store';
import { LayersPanel } from './layers_panel';
import { NotePropertiesPanel } from './note_properties_panel';

/**
 * One entry per sidebar panel: its id, the rail icon + label, and the panel
 * body to render when active. The rail's buttons and the panel area are both
 * driven off this single list (see `sidebar.tsx`), so adding a panel is one
 * entry here (plus a {@link SidebarPanelId} member and the store/context wiring
 * at the editor composition root). The `satisfies` below ties each entry's id
 * to the store-owned union, so a typo'd or unknown id is a compile error.
 */
export type SidebarPanelDef = {
  id: SidebarPanelId;
  /** Rail tooltip + accessible name. */
  label: string;
  /** Rail icon (lucide). */
  Icon: LucideIcon;
  /** The panel body shown when this panel is active. */
  render: () => React.ReactNode;
};

export const SIDEBAR_PANELS = [
  { id: 'layers', label: 'Layers', Icon: Layers, render: () => <LayersPanel /> },
  {
    id: 'note_properties',
    label: 'Note properties',
    Icon: FileText,
    render: () => <NotePropertiesPanel />,
  },
] as const satisfies readonly SidebarPanelDef[];
