import classNames from 'classnames';
import { AudioWaveform, Captions, Drum, Group } from 'lucide-react';
import { observer } from 'mobx-react-lite';
import React from 'react';
import { jotPlayer } from 'src/editing/playback/player';
import { PICKER_PALETTE } from 'src/editing/tracks/tracks';
import { ColorPickerMenuRow } from 'src/ui/color_picker_menu_row/color_picker_menu_row';
import { DropdownButton, dropdownStyles } from 'src/ui/dropdown/dropdown';
import { LayersPresenterContext, LayersStoreContext } from './layers_contexts';
import type { LayersLayerView, LayersTrackView } from './layers_store';
import type { LayersPresenter } from './layers_presenter';
import styles from './layers_view.module.css';

/** The hex the colour-picker wheel opens at when a layer has no usable
 *  colour (the transparent default); a neutral grey. */
const PICKER_FALLBACK = '#7e7e7e';

/** A labelled rename field inside an overflow menu, styled to match the shared
 *  ColorPickerMenuRow so the whole panel reads like one menu. Commits on blur
 *  and on Enter (which also dismisses the menu). */
function MenuRenameRow({
  label,
  defaultValue,
  placeholder,
  testId,
  onCommit,
  onEnter,
}: {
  label: string;
  defaultValue: string;
  placeholder?: string;
  testId?: string;
  onCommit: (value: string) => void;
  onEnter: () => void;
}) {
  return (
    <label
      className={styles.menuRow}
      onClick={(e) => e.stopPropagation()}
      onMouseDown={(e) => e.stopPropagation()}
    >
      <span>{label}</span>
      <input
        className={styles.menuInput}
        defaultValue={defaultValue}
        placeholder={placeholder}
        data-testid={testId}
        onKeyDown={(e) => {
          if (e.key === 'Enter') {
            onCommit((e.target as HTMLInputElement).value);
            onEnter();
          }
        }}
        onBlur={(e) => onCommit(e.target.value)}
      />
    </label>
  );
}

/** The per-layer ⋯ menu: rename + band colour. Uses the shared dropdown menu
 *  primitives so it's visually identical to the header dropdowns. */
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
    <DropdownButton
      label="⋯"
      className={styles.layerMore}
      title={`Options for ${layer.name ?? fallbackName}`}
    >
      {(close) => (
        <>
          <MenuRenameRow
            label="Name"
            defaultValue={layer.name ?? ''}
            placeholder={fallbackName}
            testId="layer-rename-input"
            onCommit={(value) => presenter.setLayerName(layer.id, value)}
            onEnter={close}
          />
          <span className={dropdownStyles.dropdownDivider} aria-hidden="true" />
          <ColorPickerMenuRow
            label="Colour"
            value={layer.color ?? PICKER_FALLBACK}
            palette={PICKER_PALETTE}
            hasOverride={layer.hasColorOverride}
            onChange={(hex) => presenter.setLayerColor(layer.id, hex)}
            onReset={() => presenter.setLayerColor(layer.id, undefined)}
            ariaLabel={`Band colour for ${layer.name ?? fallbackName}`}
          />
        </>
      )}
    </DropdownButton>
  );
});

/** DataTransfer MIME tags identifying what's being dragged in the panel. */
const TRACK_MIME = 'application/x-drumjot-layers-track';
const LAYER_MIME = 'application/x-drumjot-layers-layer';
const GROUP_MIME = 'application/x-drumjot-layers-group';

/** Which part of a row a track is being dropped on: the top/bottom edges
 *  reorder before/after the target; the centre groups it with the target. */
type DropZone = 'before' | 'after' | 'into';

/** The track-kind badge: a Lucide icon + hover/aria tooltip, replacing the
 *  former "Audio"/"Instrument" text. */
function TrackKindIcon({ kind }: { kind: LayersTrackView['kind'] }) {
  const { Icon, label } =
    kind === 'audio'
      ? { Icon: AudioWaveform, label: 'Audio' }
      : kind === 'lyrics'
        ? { Icon: Captions, label: 'Lyrics' }
        : { Icon: Drum, label: 'Instrument' };
  return (
    <span className={styles.trackKindIcon} title={label} aria-label={label}>
      <Icon size={14} aria-hidden />
    </span>
  );
}

/** The per-group ⋯ menu: rename + colour + ungroup/delete. Uses the shared
 *  dropdown menu primitives so it matches the header dropdowns. */
const GroupMenu = observer(function GroupMenu({
  groupId,
  name,
  color,
  isEmpty,
  presenter,
}: {
  groupId: string;
  name: string;
  color?: string;
  isEmpty: boolean;
  presenter: LayersPresenter;
}) {
  return (
    <DropdownButton label="⋯" className={styles.layerMore} title={`Options for ${name}`}>
      {(close) => (
        <>
          <MenuRenameRow
            label="Name"
            defaultValue={name}
            testId="group-rename-input"
            onCommit={(value) => presenter.setGroupName(groupId, value)}
            onEnter={close}
          />
          <span className={dropdownStyles.dropdownDivider} aria-hidden="true" />
          <ColorPickerMenuRow
            label="Colour"
            value={color ?? PICKER_FALLBACK}
            palette={PICKER_PALETTE}
            hasOverride={color !== undefined}
            onChange={(hex) => presenter.setGroupColor(groupId, hex)}
            onReset={() => presenter.setGroupColor(groupId, undefined)}
            ariaLabel={`Colour for ${name}`}
          />
          <span className={dropdownStyles.dropdownDivider} aria-hidden="true" />
          <button
            type="button"
            className={dropdownStyles.dropdownItem}
            role="menuitem"
            data-testid={isEmpty ? 'group-delete' : 'group-ungroup'}
            onClick={() => {
              if (isEmpty) presenter.deleteGroup(groupId);
              else presenter.ungroup(groupId);
              close();
            }}
          >
            {isEmpty ? 'Delete' : 'Ungroup'}
          </button>
        </>
      )}
    </DropdownButton>
  );
});

/**
 * Layers tree: renders the {@link LayersStore} read-model as layer bands
 * (tinted by their data colour) containing groups and track rows, mirroring the
 * score's row layout. Every row is a draggable tile, and so is each group's
 * header (so the whole group reads as one draggable unit). Dropping a track on
 * another row's **top/bottom edge** reorders it before/after; the **centre**
 * groups the two; a between-rows move shows a full-width line in the gap.
 * Dragging a row/group to a **different** layer lights up that whole target
 * layer. Whole groups and layers reorder by dragging their headers. All writes
 * go through {@link LayersPresenter}, the source of truth the score reads.
 */
export const LayersView = observer(function LayersView() {
  const store = React.useContext(LayersStoreContext);
  const presenter = React.useContext(LayersPresenterContext);
  // Transient drag state (React-local, not a store): the dragged track id (or
  // group id, for a group-header drag), the source layer (to tell same-layer
  // reorder from cross-layer move), the same-layer drop target(s), the
  // cross-layer target (whole-layer highlight) and the layer-reorder index.
  const [dragTrack, setDragTrack] = React.useState<string | null>(null);
  const [dragGroup, setDragGroup] = React.useState<string | null>(null);
  const [dragSourceLayer, setDragSourceLayer] = React.useState<string | null>(null);
  const [dropTarget, setDropTarget] = React.useState<{ id: string; zone: DropZone } | null>(null);
  const [dropBefore, setDropBefore] = React.useState<string | null>(null);
  const [dropAppendLayer, setDropAppendLayer] = React.useState<string | null>(null);
  const [dropOverLayer, setDropOverLayer] = React.useState<string | null>(null);
  const [dropLayerIdx, setDropLayerIdx] = React.useState<number | null>(null);
  const resetDrag = () => {
    setDragTrack(null);
    setDragGroup(null);
    setDragSourceLayer(null);
    setDropTarget(null);
    setDropBefore(null);
    setDropAppendLayer(null);
    setDropOverLayer(null);
    setDropLayerIdx(null);
  };
  /** Clear every same-layer gap cue (used when switching to a cross-layer or
   *  whole-layer indicator). */
  const clearGapCues = () => {
    setDropTarget(null);
    setDropBefore(null);
    setDropAppendLayer(null);
  };
  if (!store) return null;
  const layout = store.layout;
  const isCrossLayer = (layerId: string) =>
    dragSourceLayer !== null && layerId !== dragSourceLayer;

  const trackLabel = (t: LayersTrackView): string => {
    if (t.kind === 'instrument') {
      return store.instrumentName(t.lane) ?? `Lane ${t.lane.toUpperCase()}`;
    }
    if (t.kind === 'audio') {
      // Mirror the gutter/trackhead label: the audio track's filename with its
      // extension stripped (falling back to the raw filename, then a generic).
      const audio = jotPlayer.audioTracks.get(t.audioId);
      if (audio) return audio.filename.replace(/\.[^./\\]+$/, '') || audio.filename;
      return 'Audio track';
    }
    return 'Lyrics';
  };

  // Drop-zone handlers for a track drag (the three overlay zones per tile). In
  // the same layer, the top/bottom edges reorder before/after and the centre
  // groups; over a different layer, the whole target layer highlights instead.
  const zoneOver = (e: React.DragEvent, id: string, zone: DropZone, layerId: string) => {
    if (!e.dataTransfer.types.includes(TRACK_MIME)) return;
    e.preventDefault();
    e.stopPropagation();
    e.dataTransfer.dropEffect = 'move';
    if (isCrossLayer(layerId)) {
      setDropOverLayer(layerId);
      clearGapCues();
    } else {
      setDropTarget({ id, zone });
      setDropOverLayer(null);
      setDropBefore(null);
      setDropAppendLayer(null);
    }
  };
  const zoneDrop = (e: React.DragEvent, t: LayersTrackView, layerId: string, zone: DropZone) => {
    if (!e.dataTransfer.types.includes(TRACK_MIME)) return;
    e.preventDefault();
    e.stopPropagation();
    const id = e.dataTransfer.getData(TRACK_MIME);
    if (id && id !== t.id) {
      // Cross-layer is always a plain move into that layer (the indicator was a
      // whole-layer highlight, not a precise gap); same-layer respects the zone.
      if (isCrossLayer(layerId)) presenter?.moveTrack(id, layerId, t.id);
      else if (zone === 'into') presenter?.groupTracks(id, t.id);
      else if (zone === 'before') presenter?.moveTrack(id, layerId, t.id);
      else presenter?.moveTrackAfter(id, layerId, t.id);
    }
    resetDrag();
  };

  const renderTrack = (
    t: LayersTrackView,
    layerId: string,
    slotGroupId: string | null,
    indexInSlot: number
  ) => {
    const isLoose = slotGroupId === null;
    // The between-rows move line: a track drag's before/after edge, or a group
    // drag landing before a *loose* track (group-before-a-group lines render on
    // the group header instead). Cross-layer drags clear these cues.
    const beforeLine =
      (dropTarget?.id === t.id && dropTarget.zone === 'before') ||
      (dragGroup !== null && dropBefore === t.id && isLoose);
    const afterLine = dropTarget?.id === t.id && dropTarget.zone === 'after';
    const intoHighlight = dropTarget?.id === t.id && dropTarget.zone === 'into';
    // A group drag may only land at a top-level boundary: before a loose track
    // (any position) or before the first track of *another* group. Hovering a
    // non-first track inside a group, or any track of the dragged group itself,
    // is not a legal target (no nesting) -> no indicator, no drop.
    const groupDropOk =
      dragGroup !== null &&
      dragGroup !== slotGroupId &&
      (isLoose || indexInSlot === 0);
    return (
      <div
        key={t.id}
        className={classNames(
          styles.track,
          !isLoose && styles.trackInGroup,
          dragTrack === t.id && styles.trackDragging,
          intoHighlight && styles.trackDropInto
        )}
        data-testid="layers-track"
        data-track-id={t.id}
        data-track-kind={t.kind}
        draggable={presenter !== null}
        onDragStart={(e) => {
          e.dataTransfer.setData(TRACK_MIME, t.id);
          e.dataTransfer.effectAllowed = 'move';
          setDragTrack(t.id);
          setDragSourceLayer(layerId);
        }}
        onDragEnd={resetDrag}
        onDragOver={(e) => {
          // Track drags are handled by the overlay zones below; the row itself
          // only fields group drags.
          if (!e.dataTransfer.types.includes(GROUP_MIME)) return;
          if (!groupDropOk) {
            // Inside another group (or the dragged group's own rows): swallow so
            // the band doesn't show an append cue here, and show nothing -> the
            // drop is rejected (no preventDefault). Disallows group nesting.
            e.stopPropagation();
            return;
          }
          e.preventDefault();
          e.stopPropagation();
          e.dataTransfer.dropEffect = 'move';
          if (isCrossLayer(layerId)) {
            setDropOverLayer(layerId);
            clearGapCues();
          } else {
            setDropBefore(t.id);
            setDropOverLayer(null);
            setDropTarget(null);
            setDropAppendLayer(null);
          }
        }}
        onDrop={(e) => {
          if (!e.dataTransfer.types.includes(GROUP_MIME)) return;
          e.stopPropagation();
          if (!groupDropOk) return; // nesting drop rejected
          e.preventDefault();
          presenter?.moveGroup(e.dataTransfer.getData(GROUP_MIME), layerId, t.id);
          resetDrag();
        }}
      >
        {beforeLine && <div className={styles.dropLineBefore} aria-hidden />}
        <span className={styles.trackName}>{trackLabel(t)}</span>
        <TrackKindIcon kind={t.kind} />
        {afterLine && <div className={styles.dropLineAfter} aria-hidden />}
        {/* Drop zones, mounted only while another panel track is dragged so they
            never intercept normal pointer interaction (menu, hover). */}
        {dragTrack !== null && dragTrack !== t.id && (
          <>
            <div
              className={styles.dropZoneBefore}
              onDragOver={(e) => zoneOver(e, t.id, 'before', layerId)}
              onDrop={(e) => zoneDrop(e, t, layerId, 'before')}
            />
            <div
              className={styles.dropZoneInto}
              onDragOver={(e) => zoneOver(e, t.id, 'into', layerId)}
              onDrop={(e) => zoneDrop(e, t, layerId, 'into')}
            />
            <div
              className={styles.dropZoneAfter}
              onDragOver={(e) => zoneOver(e, t.id, 'after', layerId)}
              onDrop={(e) => zoneDrop(e, t, layerId, 'after')}
            />
          </>
        )}
      </div>
    );
  };

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
            (dropOverLayer === layer.id || dropLayerIdx === i) && styles.layerDropTarget
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
              // Over the band but not on a row: same layer -> append at the end
              // (the end-of-layer line); different layer -> highlight the whole
              // target layer.
              e.preventDefault();
              e.dataTransfer.dropEffect = 'move';
              if (isCrossLayer(layer.id)) {
                setDropOverLayer(layer.id);
                clearGapCues();
              } else {
                setDropAppendLayer(layer.id);
                setDropOverLayer(null);
                setDropBefore(null);
                setDropTarget(null);
              }
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
            <span
              className={styles.swatch}
              style={layer.color ? { background: layer.color } : undefined}
            />
            <span className={styles.layerName}>{layer.name ?? `Layer ${i + 1}`}</span>
            {presenter && (
              <LayerMenu layer={layer} fallbackName={`Layer ${i + 1}`} presenter={presenter} />
            )}
          </div>
          {layer.slots.map((slot, si) => {
            if (slot.kind !== 'group') {
              return (
                <React.Fragment key={`loose-${si}`}>
                  {slot.tracks.map((t, ti) => renderTrack(t, layer.id, null, ti))}
                </React.Fragment>
              );
            }
            const firstMemberId = slot.tracks[0]?.id;
            // A group drag landing before this group draws the move line above
            // the group's *header* (its draggable handle), not its first member.
            const headerBeforeLine = dragGroup !== null && dropBefore === firstMemberId;
            const ownGroupDrag = dragGroup === slot.id;
            return (
              <div
                key={slot.id}
                className={styles.group}
                data-testid="layers-group"
                data-group-id={slot.id}
              >
                <div
                  className={classNames(styles.groupHeader, ownGroupDrag && styles.trackDragging)}
                  draggable={presenter !== null}
                  onDragStart={(e) => {
                    e.dataTransfer.setData(GROUP_MIME, slot.id);
                    e.dataTransfer.effectAllowed = 'move';
                    setDragGroup(slot.id);
                    setDragSourceLayer(layer.id);
                  }}
                  onDragEnd={resetDrag}
                  onDragOver={(e) => {
                    // Only a *different* group may drop before this one; track
                    // drags bubble to the band.
                    if (!e.dataTransfer.types.includes(GROUP_MIME) || ownGroupDrag) {
                      if (ownGroupDrag) e.stopPropagation();
                      return;
                    }
                    e.preventDefault();
                    e.stopPropagation();
                    e.dataTransfer.dropEffect = 'move';
                    if (isCrossLayer(layer.id)) {
                      setDropOverLayer(layer.id);
                      clearGapCues();
                    } else {
                      setDropBefore(firstMemberId ?? null);
                      setDropOverLayer(null);
                      setDropTarget(null);
                      setDropAppendLayer(null);
                    }
                  }}
                  onDrop={(e) => {
                    if (!e.dataTransfer.types.includes(GROUP_MIME)) return;
                    e.stopPropagation();
                    if (ownGroupDrag) return;
                    e.preventDefault();
                    presenter?.moveGroup(
                      e.dataTransfer.getData(GROUP_MIME),
                      layer.id,
                      firstMemberId ?? null
                    );
                    resetDrag();
                  }}
                >
                  {headerBeforeLine && <div className={styles.dropLineBefore} aria-hidden />}
                  <span className={styles.groupIcon} aria-hidden>
                    <Group size={14} />
                  </span>
                  <span className={styles.groupName}>{slot.name}</span>
                  {presenter && (
                    <GroupMenu
                      groupId={slot.id}
                      name={slot.name}
                      color={slot.color}
                      isEmpty={slot.tracks.length === 0}
                      presenter={presenter}
                    />
                  )}
                </div>
                {slot.tracks.map((t, ti) => renderTrack(t, layer.id, slot.id, ti))}
              </div>
            );
          })}
          {/* End-of-layer append cue: a full-width line below the last row when
              a same-layer append is pending. Non-interactive so the band owns
              the drop. */}
          {(dragTrack !== null || dragGroup !== null) && (
            <div
              className={classNames(
                styles.endZone,
                dropAppendLayer === layer.id && styles.endZoneActive
              )}
              data-testid="layers-end-zone"
              aria-hidden
            />
          )}
        </div>
      ))}
    </div>
  );
});
