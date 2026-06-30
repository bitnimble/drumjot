import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { Modal, ModalFooter, ModalHeader, modalStyles } from 'src/ui/modal/modal';
import { formatBytes } from 'src/desktop/capability_manifest';
import { CapabilityTree, useCapabilityInstall } from 'src/desktop/capability_install';
import { HardwareInfo } from 'src/desktop/hardware_info';
import panelStyles from 'src/desktop/capability_panel.module.css';
import styles from './settings_dialog.module.css';

type SettingsTab = 'capabilities' | 'hardware' | 'about';

const ALL_TABS: ReadonlyArray<{ value: SettingsTab; label: string; desktopOnly: boolean }> = [
  { value: 'capabilities', label: 'Capabilities', desktopOnly: true },
  { value: 'hardware', label: 'Hardware', desktopOnly: true },
  { value: 'about', label: 'About', desktopOnly: false },
];

/** Third-party projects + models Drumjot is built on. Surfacing them here is the
 *  visible-acknowledgement obligation for the attribution-required ones (JMdict's
 *  CC-BY-SA, the NAIST dictionary); full texts live in THIRD_PARTY_LICENSES.md. */
const ACKNOWLEDGEMENTS: ReadonlyArray<{ name: string; detail: string }> = [
  { name: 'Beat-Transformer', detail: 'Beat tracking. MIT, © 2022 Zhao Jingwei.' },
  { name: 'ADTOF', detail: 'Drum onset detection. CC-BY-NC-SA (non-commercial).' },
  { name: 'MERT', detail: 'Learned-onset encoder (optional). CC-BY-NC (non-commercial).' },
  {
    name: 'audio-separator / UVR models, MMS aligner',
    detail: 'Stem separation + lyric alignment. Community / CC-BY-NC models (non-commercial).',
  },
  { name: 'Beat This!', detail: 'Beat / downbeat tracking. MIT, © 2024 Foscarin et al.' },
  { name: 'ctc-forced-aligner', detail: 'Lyric forced alignment. BSD.' },
  { name: 'signalsmith-stretch', detail: 'Audio time-stretch. MIT, © Signalsmith Audio Ltd.' },
  {
    name: 'kuromoji + IPADIC dictionary',
    detail: 'Japanese tokeniser + dictionary. Apache-2.0 / NAIST, © 2000–2003 NAIST.',
  },
  {
    name: 'JMdict / JmdictFurigana',
    detail: 'Furigana data. CC-BY-SA, © James William Breen and the EDRDG.',
  },
  { name: 'uv', detail: 'Python package manager. MIT / Apache-2.0, © Astral.' },
];

function AboutTab(): React.ReactElement {
  return (
    <div>
      <p className={panelStyles.intro}>
        Drumjot is distributed for non-commercial use. It is built on the
        open-source projects and models below; full license texts are in
        THIRD_PARTY_LICENSES.md.
      </p>
      <div className={styles.about}>
        {ACKNOWLEDGEMENTS.map((a) => (
          <div key={a.name} className={styles.aboutEntry}>
            <span className={panelStyles.name}>{a.name}</span>
            <span className={panelStyles.desc}>{a.detail}</span>
          </div>
        ))}
      </div>
    </div>
  );
}

/**
 * The File → Settings dialog: a left tab rail + right content, on the shared
 * Modal primitive. Always shows the About tab (licenses + acknowledgements);
 * Capabilities (the install picker + cumulative-size footer) and Hardware
 * (accelerator info) are desktop-only and hidden in the web build.
 */
export const SettingsDialog = observer(function SettingsDialog({
  onClose,
}: {
  onClose: () => void;
}) {
  const install = useCapabilityInstall();
  const tabs = ALL_TABS.filter((t) => install.available || !t.desktopOnly);
  const [tab, setTab] = React.useState<SettingsTab>(install.available ? 'capabilities' : 'about');

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
          {tabs.map((t) => (
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
                Optional features download their dependencies + models when you install them.
              </p>
              <CapabilityTree controller={install} />
            </>
          )}
          {tab === 'hardware' && <HardwareInfo />}
          {tab === 'about' && <AboutTab />}
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
