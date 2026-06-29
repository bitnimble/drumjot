import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { Modal, ModalFooter, ModalHeader, modalStyles } from 'src/ui/modal/modal';
import { formatBytes } from 'src/desktop/capability_manifest';
import { CapabilityTree, useCapabilityInstall } from 'src/desktop/capability_install';
import { HardwareInfo } from 'src/desktop/hardware_info';
import panelStyles from 'src/desktop/capability_panel.module.css';
import styles from './settings_dialog.module.css';

type SettingsTab = 'capabilities' | 'hardware';

const TABS: ReadonlyArray<{ value: SettingsTab; label: string }> = [
  { value: 'capabilities', label: 'Capabilities' },
  { value: 'hardware', label: 'Hardware' },
];

/**
 * The File → Settings dialog: a left tab rail + right content, on the shared
 * Modal primitive. For now two desktop tabs, Capabilities (the shared install
 * picker, with a cumulative-size install footer) and Hardware (read-only
 * accelerator info). The menu entry is desktop-gated, so it never opens in web.
 */
export const SettingsDialog = observer(function SettingsDialog({
  onClose,
}: {
  onClose: () => void;
}) {
  const [tab, setTab] = React.useState<SettingsTab>('capabilities');
  const install = useCapabilityInstall();

  return (
    <Modal
      open
      onClose={onClose}
      ariaLabel="Settings"
      width={720}
      panelClassName={styles.panel}
      testId="settings-dialog"
    >
      <ModalHeader title="Settings" onClose={onClose} />
      <div className={styles.layout}>
        <nav className={styles.rail} role="tablist" aria-label="Settings sections">
          {TABS.map((t) => (
            <button
              key={t.value}
              id={`settings-tab-${t.value}`}
              type="button"
              role="tab"
              aria-selected={tab === t.value}
              aria-controls="settings-tabpanel"
              className={classNames(styles.tab, tab === t.value && styles.tabActive)}
              onClick={() => setTab(t.value)}
              data-testid={`settings-tab-${t.value}`}
            >
              {t.label}
            </button>
          ))}
        </nav>
        <div
          className={styles.content}
          role="tabpanel"
          id="settings-tabpanel"
          aria-labelledby={`settings-tab-${tab}`}
        >
          {tab === 'capabilities' && (
            <>
              <p className={panelStyles.intro}>
                Optional features download what they need the first time you use them.
              </p>
              <CapabilityTree controller={install} />
            </>
          )}
          {tab === 'hardware' && <HardwareInfo />}
        </div>
      </div>
      {tab === 'capabilities' && (
        <ModalFooter>
          <span className={panelStyles.total}>
            {install.totalBytes > 0
              ? `Total: ${formatBytes(install.totalBytes)}`
              : 'Nothing selected'}
          </span>
          <button
            type="button"
            className={modalStyles.primaryButton}
            disabled={install.totalBytes === 0 || install.installing}
            onClick={install.install}
          >
            {install.installing ? 'Installing…' : 'Install'}
          </button>
        </ModalFooter>
      )}
    </Modal>
  );
});
