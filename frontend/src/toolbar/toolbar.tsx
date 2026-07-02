import { ChevronDown } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { Logo } from 'src/ui/logo/logo';
import { EditMenu } from './edit_menu';
import { FileMenu } from './file_menu';
import { PlaybackMenu } from './playback_menu';
import styles from './toolbar.module.css';
import { DrumLoadingIndicator, LyricsAlignBusyPill, TranscribeBusyPill } from './toolbar_status';
import { ViewMenu } from './view_menu';

/**
 * Toolbar dropdown trigger label with a trailing caret indicator. The
 * shared {@link DropdownButton} no longer renders the caret itself
 * (overflow-icon callers like the mixer don't want one); toolbar
 * triggers compose it via this helper instead.
 */
export const ToolbarDropdownLabel = ({ children }: { children: React.ReactNode }) => (
  <span className={styles.toolbarDropdownLabel}>
    {children}
    <ChevronDown size={14} aria-hidden="true" />
  </span>
);

/**
 * The editor's top menu bar. Each dropdown is a self-contained `observer`
 * (File / View / Edit / Playback) that reads its own domain's stores +
 * presenters off context, so the Toolbar itself carries only the handful of
 * genuinely view-level callbacks those menus can't source from a store:
 * `onNewJot` (gates a React-local confirm dialog), `onOpenSettings` (toggles
 * the Settings modal), `onLoadZip` (window drag-and-drop auto-load), and
 * `onSetZoom` (DOM-anchored centered zoom). Everything else lives in the menu
 * components. The right-aligned status pills read their stores off context too.
 */
export const Toolbar = observer(
  ({
    onNewJot,
    onOpenSettings,
    onLoadZip,
    onSetZoom,
  }: {
    onNewJot: () => void;
    onOpenSettings: () => void;
    onLoadZip: (file: File) => void;
    onSetZoom: (z: number) => void;
  }) => (
    <div className={styles.toolbar}>
      <Logo size={28} title="Drumjot" />
      <FileMenu onNewJot={onNewJot} onOpenSettings={onOpenSettings} onLoadZip={onLoadZip} />
      <span className={styles.toolbarDivider} aria-hidden="true" />
      <ViewMenu onSetZoom={onSetZoom} />
      <EditMenu />
      <PlaybackMenu />
      <div className={styles.toolbarSpacer} aria-hidden="true" />
      <DrumLoadingIndicator />
      <LyricsAlignBusyPill />
      <TranscribeBusyPill />
    </div>
  )
);
