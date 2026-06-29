import classNames from 'classnames';
import { X } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import modal from 'src/ui/modal/modal.module.css';
import { CapabilityList } from 'src/desktop/capability_list';
import { HardwareInfo } from 'src/desktop/hardware_info';
import styles from './settings_dialog.module.css';

type SettingsTab = 'capabilities' | 'hardware';

const TABS: ReadonlyArray<{ value: SettingsTab; label: string }> = [
  { value: 'capabilities', label: 'Capabilities' },
  { value: 'hardware', label: 'Hardware' },
];

/**
 * The File → Settings dialog: a left tab rail + right content. For now two
 * desktop tabs, Capabilities (the shared install list) and Hardware (read-only
 * accelerator info). The menu entry is desktop-gated, so it never opens in web.
 */
export const SettingsDialog = observer(function SettingsDialog({
  open,
  onClose,
}: {
  open: boolean;
  onClose: () => void;
}) {
  const [tab, setTab] = React.useState<SettingsTab>('capabilities');
  if (!open) return null;

  return (
    <div
      className={modal.backdrop}
      role="dialog"
      aria-modal="true"
      aria-label="Settings"
      onMouseDown={(e) => {
        if (e.target === e.currentTarget) onClose();
      }}
      onKeyDown={(e) => {
        if (e.key === 'Escape') onClose();
      }}
    >
      <div className={styles.panel}>
        <div className={modal.header}>
          <h2 className={modal.title}>Settings</h2>
          <button className={styles.close} type="button" onClick={onClose} aria-label="Close">
            <X size={18} aria-hidden="true" />
          </button>
        </div>
        <div className={styles.layout}>
          <nav className={styles.rail} role="tablist" aria-label="Settings sections">
            {TABS.map((t) => (
              <button
                key={t.value}
                type="button"
                role="tab"
                aria-selected={tab === t.value}
                className={classNames(styles.tab, tab === t.value && styles.tabActive)}
                onClick={() => setTab(t.value)}
                data-testid={`settings-tab-${t.value}`}
              >
                {t.label}
              </button>
            ))}
          </nav>
          <div className={styles.content} role="tabpanel">
            {tab === 'capabilities' && (
              <>
                <p className={styles.intro}>
                  Optional features download what they need the first time you use them.
                </p>
                <CapabilityList />
              </>
            )}
            {tab === 'hardware' && <HardwareInfo />}
          </div>
        </div>
      </div>
    </div>
  );
});
