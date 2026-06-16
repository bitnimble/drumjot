import classNames from 'classnames';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { PICKER_PALETTE } from 'src/editing/tracks/tracks';
import { ColorPickerMenuRow } from 'src/ui/color_picker_menu_row/color_picker_menu_row';
import { DropdownButton } from 'src/ui/dropdown/dropdown';
import { LayersPresenterContext, LayersStoreContext } from './layers_contexts';
import type { LayersLayerView, LayersTrackView } from './layers_store';
import type { LayersPresenter } from './layers_presenter';
import styles from './layers_view.module.css';

/** The hex the colour-picker wheel opens at when a layer has no usable
 *  colour (the transparent default); a neutral grey. */
const PICKER_FALLBACK = '#7e7e7e';

/** The per-layer ⋯ menu: rename + band colour, writing through the presenter. */
const LayerMenu = observer(function LayerMenu({
  layer,
  fallbackName,
  presenter,
}: {
  layer: LayersLayerView;
  fallbackName: string;
  presenter: LayersPresenter;
}) {
  return (
    <DropdownButton label="⋯" className={styles.layerMore} title={`Options for ${layer.name ?? fallbackName}`}>
      {(close) => (
        <div className={styles.layerMenu} onClick={(e) => e.stopPropagation()}>
          <label className={styles.renameRow}>
            <span>Name</span>
            <input
              className={styles.renameInput}
              defaultValue={layer.name ?? ''}
              placeholder={fallbackName}
              data-testid="layer-rename-input"
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  presenter.setLayerName(layer.id, (e.target as HTMLInputElement).value);
                  close();
                }
              }}
              onBlur={(e) => presenter.setLayerName(layer.id, e.target.value)}
            />
          </label>
          <ColorPickerMenuRow
            label="Colour"
            value={layer.color ?? PICKER_FALLBACK}
            palette={PICKER_PALETTE}
            hasOverride={layer.hasColorOverride}
            onChange={(hex) => presenter.setLayerColor(layer.id, hex)}
            onReset={() => presenter.setLayerColor(layer.id, undefined)}
            ariaLabel={`Band colour for ${layer.name ?? fallbackName}`}
          />
        </div>
      )}
    </DropdownButton>
  );
});

/** DataTransfer MIME tags identifying what's being dragged in the panel. */
const TRACK_MIME = 'application/x-drumjot-layers-track';
const LAYER_MIME = 'application/x-drumjot-layers-layer';
const GROUP_MIME = 'application/x-drumjot-layers-group';

/** The per-group ⋯ menu: rename + colour + ungroup. */
const GroupMenu = observer(function GroupMenu({
  groupId,
  name,
  color,
  presenter,
}: {
  groupId: string;
  name: string;
  color?: string;
  presenter: LayersPresenter;
}) {
  return (
    <DropdownButton label="⋯" className={styles.layerMore} title={`Options for ${name}`}>
      {(close) => (
        <div className={styles.layerMenu} onClick={(e) => e.stopPropagation()}>
          <label className={styles.renameRow}>
            <span>Name</span>
            <input
              className={styles.renameInput}
              defaultValue={name}
              data-testid="group-rename-input"
              onKeyDown={(e) => {
                if (e.key === 'Enter') {
                  presenter.setGroupName(groupId, (e.target as HTMLInputElement).value);
                  close();
                }
              }}
              onBlur={(e) => presenter.setGroupName(groupId, e.target.value)}
            />
          </label>
          <ColorPickerMenuRow
            label="Colour"
            value={color ?? PICKER_FALLBACK}
            palette={PICKER_PALETTE}
            hasOverride={color !== undefined}
            onChange={(hex) => presenter.setGroupColor(groupId, hex)}
            onReset={() => presenter.setGroupColor(groupId, undefined)}
            ariaLabel={`Colour for ${name}`}
          />
          <button
            type="button"
            className={styles.menuButton}
            data-testid="group-ungroup"
            onClick={() => {
              presenter.ungroup(groupId);
              close();
            }}
          >
            Ungroup
          </button>
        </div>
      )}
    </DropdownButton>
  );
});

/**
 * Layers tree: renders the {@link LayersStore} read-model as layer bands
 * (tinted by their data colour) containing groups (heading + indent) and track
 * rows, mirroring the score's row layout, and lets the user **drag** to
 * rearrange, reorder a track, move it to another layer, pull it into/out of a
 * group (drop before a track in/out of a group), or reorder whole layers (drag
 * the layer header). All writes go through {@link LayersPresenter}, the same
 * source of truth the score reads, so the two stay in sync.
 */
export const LayersView = observer(function LayersView() {
  const store = React.useContext(LayersStoreContext);
  const presenter = React.useContext(LayersPresenterContext);
  // Transient drag state (React-local, not a store): the track id being
  // dragged, the track to drop before (indicator), and the layer-reorder pair.
  const [dragTrack, setDragTrack] = React.useState<string | null>(null);
  const [dropBefore, setDropBefore] = React.useState<string | null>(null);
  const [dropAppendLayer, setDropAppendLayer] = React.useState<string | null>(null);
  const [dropLayerIdx, setDropLayerIdx] = React.useState<number | null>(null);
  const resetDrag = () => {
    setDragTrack(null);
    setDropBefore(null);
    setDropAppendLayer(null);
    setDropLayerIdx(null);
  };
  if (!store) return null;
  const layout = store.layout;

  const trackLabel = (t: LayersTrackView): string => {
    if (t.kind === 'instrument') {
      return store.instrumentName(t.lane) ?? `Lane ${t.lane.toUpperCase()}`;
    }
    return t.kind === 'audio' ? 'Audio track' : 'Lyrics';
  };

  const renderTrack = (t: LayersTrackView, layerId: string) => (
    <div
      key={t.id}
      className={classNames(
        styles.track,
        dragTrack === t.id && styles.trackDragging,
        dropBefore === t.id && styles.trackDropBefore
      )}
      data-testid="layers-track"
      data-track-id={t.id}
      data-track-kind={t.kind}
      draggable={presenter !== null}
      onDragStart={(e) => {
        e.dataTransfer.setData(TRACK_MIME, t.id);
        e.dataTransfer.effectAllowed = 'move';
        setDragTrack(t.id);
      }}
      onDragEnd={resetDrag}
      onDragOver={(e) => {
        const types = e.dataTransfer.types;
        if (!types.includes(TRACK_MIME) && !types.includes(GROUP_MIME)) return;
        e.preventDefault();
        e.stopPropagation(); // don't let the band treat this as an append
        e.dataTransfer.dropEffect = 'move';
        setDropBefore(t.id);
        setDropAppendLayer(null);
      }}
      onDrop={(e) => {
        const types = e.dataTransfer.types;
        if (!types.includes(TRACK_MIME) && !types.includes(GROUP_MIME)) return;
        e.preventDefault();
        e.stopPropagation();
        if (types.includes(GROUP_MIME)) {
          presenter?.moveGroup(e.dataTransfer.getData(GROUP_MIME), layerId, t.id);
        } else {
          const id = e.dataTransfer.getData(TRACK_MIME);
          if (id && id !== t.id) presenter?.moveTrack(id, layerId, t.id);
        }
        resetDrag();
      }}
    >
      <span className={styles.grip} aria-hidden>
        ⠿
      </span>
      <span className={styles.swatch} />
      {trackLabel(t)}
      <span className={styles.trackKind}>{t.kind}</span>
      {presenter && (
        <button
          type="button"
          className={styles.newGroupBtn}
          data-testid="layers-new-group"
          title={`Group ${trackLabel(t)}`}
          onClick={() => presenter.createGroup(t.id)}
        >
          ⊕
        </button>
      )}
    </div>
  );

  if (layout.length === 0) {
    return (
      <p className={styles.empty} data-testid="layers-empty">
        No layers yet.
      </p>
    );
  }

  return (
    <div className={styles.root} data-testid="layers-tree">
      {layout.map((layer, i) => (
        <div
          key={layer.id}
          className={classNames(
            styles.layer,
            dropAppendLayer === layer.id && styles.layerDropTarget,
            dropLayerIdx === i && styles.layerDropTarget
          )}
          data-testid="layers-layer"
          data-layer-id={layer.id}
          style={
            layer.color
              ? { background: `color-mix(in srgb, ${layer.color} 14%, transparent)` }
              : undefined
          }
          onDragOver={(e) => {
            const types = e.dataTransfer.types;
            if (types.includes(TRACK_MIME) || types.includes(GROUP_MIME)) {
              // Over the band but not a specific row -> append to this layer.
              e.preventDefault();
              e.dataTransfer.dropEffect = 'move';
              setDropAppendLayer(layer.id);
              setDropBefore(null);
            } else if (types.includes(LAYER_MIME)) {
              e.preventDefault();
              e.dataTransfer.dropEffect = 'move';
              setDropLayerIdx(i);
            }
          }}
          onDrop={(e) => {
            const types = e.dataTransfer.types;
            if (types.includes(GROUP_MIME)) {
              e.preventDefault();
              presenter?.moveGroup(e.dataTransfer.getData(GROUP_MIME), layer.id, null);
            } else if (types.includes(TRACK_MIME)) {
              e.preventDefault();
              const id = e.dataTransfer.getData(TRACK_MIME);
              if (id) presenter?.moveTrack(id, layer.id, null);
            } else if (types.includes(LAYER_MIME)) {
              e.preventDefault();
              const from = parseInt(e.dataTransfer.getData(LAYER_MIME), 10);
              if (Number.isFinite(from)) presenter?.reorderLayer(from, i);
            }
            resetDrag();
          }}
        >
          <div
            className={styles.layerHeader}
            draggable={presenter !== null}
            onDragStart={(e) => {
              e.dataTransfer.setData(LAYER_MIME, String(i));
              e.dataTransfer.effectAllowed = 'move';
            }}
            onDragEnd={resetDrag}
          >
            <span className={styles.grip} aria-hidden>
              ⠿
            </span>
            <span
              className={styles.swatch}
              style={layer.color ? { background: layer.color } : undefined}
            />
            <span className={styles.layerName}>{layer.name ?? `Layer ${i + 1}`}</span>
            {presenter && (
              <LayerMenu layer={layer} fallbackName={`Layer ${i + 1}`} presenter={presenter} />
            )}
          </div>
          {layer.slots.map((slot, si) =>
            slot.kind === 'group' ? (
              <div
                key={slot.id}
                className={styles.group}
                data-testid="layers-group"
                data-group-id={slot.id}
              >
                <div
                  className={styles.groupHeader}
                  draggable={presenter !== null}
                  onDragStart={(e) => {
                    e.dataTransfer.setData(GROUP_MIME, slot.id);
                    e.dataTransfer.effectAllowed = 'move';
                  }}
                  onDragEnd={resetDrag}
                >
                  <span className={styles.grip} aria-hidden>
                    ⠿
                  </span>
                  <span className={styles.groupName}>{slot.name}</span>
                  {presenter && (
                    <GroupMenu
                      groupId={slot.id}
                      name={slot.name}
                      color={slot.color}
                      presenter={presenter}
                    />
                  )}
                </div>
                {slot.tracks.map((t) => renderTrack(t, layer.id))}
              </div>
            ) : (
              <React.Fragment key={`loose-${si}`}>
                {slot.tracks.map((t) => renderTrack(t, layer.id))}
              </React.Fragment>
            )
          )}
        </div>
      ))}
    </div>
  );
});
