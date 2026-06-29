import { observer } from 'mobx-react-lite';
import { Modal, ModalBody, ModalFooter, ModalHeader, modalStyles } from 'src/ui/modal/modal';
import { capabilityById, formatBytes } from './capability_manifest';
import { desktopCapabilities } from './desktop_services';

/**
 * Point-of-use install prompt. Shows when something (e.g. a transcribe) needs a
 * capability that isn't installed yet (`store.pendingGate`): confirm to download
 * + install with live progress, or cancel. Desktop-only.
 */
export const CapabilityGate = observer(function CapabilityGate() {
  const deps = desktopCapabilities();
  if (deps == null) return null;
  const { store, presenter } = deps;
  const id = store.pendingGate;
  if (id == null) return null;

  const cap = capabilityById(id);
  const installing = store.statusOf(id) === 'installing';
  const error = store.errors.get(id);
  const size = formatBytes(presenter.incrementalBytes([id]));
  // Don't let Escape / backdrop dismiss the prompt mid-install (the buttons are
  // already disabled then); the install would keep running with no UI.
  const close = (): void => {
    if (!installing) presenter.cancelGate();
  };

  return (
    <Modal open onClose={close} ariaLabel={`Install ${cap.name}?`} width={440} testId="capability-gate">
      <ModalHeader title={`Install ${cap.name}?`} onClose={close} closeLabel="Cancel" />
      <ModalBody>
        <p>{cap.description}</p>
        {installing ? (
          <p className={modalStyles.note}>Installing… {store.installLog.get(id) ?? ''}</p>
        ) : error != null ? (
          <p className={modalStyles.note}>Install failed: {error}</p>
        ) : (
          <p className={modalStyles.note}>Downloads about {size} once, then runs offline.</p>
        )}
      </ModalBody>
      <ModalFooter align="end">
        <button
          type="button"
          className={modalStyles.secondaryButton}
          disabled={installing}
          onClick={() => presenter.cancelGate()}
        >
          Cancel
        </button>
        <button
          type="button"
          className={modalStyles.primaryButton}
          disabled={installing}
          onClick={() => void presenter.confirmGate()}
        >
          {error != null ? 'Retry' : `Download · ${size}`}
        </button>
      </ModalFooter>
    </Modal>
  );
});
