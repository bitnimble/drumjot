/**
 * The tempo/timeline domain: per-bar tempo segments, the dominant
 * bpm/time-signature summary, and the audio-time timeline used by the
 * playhead. Thin presenter over the tempo + timeline modules
 * (`buildBarTempos` / `pickDominantBpmAndTime` / `buildTimeline`), reading
 * a live {@link LaidOutJot} (the structural presenter) so the timeline it
 * builds re-reads current pixel positions during zoom.
 */
import { computed, makeObservable } from 'mobx';
import { TimeSignature } from 'src/schema/dsl/dsl';
import { LEAD_IN_BAR_ID, toTempoBars } from 'src/editing/structure/structure_store';
import { BarTempos, buildBarTempos, initialBpm } from 'src/schema/dsl/tempo';
import {
  buildTimeline,
  type JotTimeline,
  type LaidOutJot,
  pickDominantBpmAndTime,
} from './timeline';

/**
 * A gradual tempo change (`BpmTransition`) ready to render in the timeline
 * gutter: a solid bar spanning `[startBeat, endBeat)` in the same lead-in-
 * inclusive global-beat space the header positions ticks against
 * (`--bar-start-beat`), labelled `startBpm -> endBpm`. `duration` is in
 * quarter-note beats, so `endBeat = startBeat + duration` with no meter walk.
 */
export type TempoRamp = {
  startBeat: number;
  endBeat: number;
  startBpm: number;
  endBpm: number;
};

export class TempoPresenter {
  constructor(private readonly jot: LaidOutJot) {
    makeObservable(this, {
      timeline: computed,
      barTempos: computed,
      tempoRamps: computed,
      dominantBpmAndTime: computed,
    });
  }

  get timeline(): JotTimeline {
    return buildTimeline(this.jot);
  }

  get barTempos(): readonly BarTempos[] {
    // View bars (incl. the virtual lead-in); `toTempoBars` flags the
    // synthetic bar so tempo-event anchoring stays aligned to the source.
    const bars = this.jot.layers[0]?.bars;
    if (!bars) return [];
    return buildBarTempos(this.jot.tempoSource, toTempoBars(bars));
  }

  /**
   * The gradual tempo changes (`bpm` events whose value is a
   * `BpmTransition`) as renderable ramps. Empty unless the jot carries
   * transition tempo events. Anchored in the same global-beat space as
   * {@link barTempos} / the header ticks: the synthetic lead-in bar is
   * walked for its beats but consumes no source-indexed tempo event.
   */
  get tempoRamps(): readonly TempoRamp[] {
    const bars = this.jot.layers[0]?.bars;
    const source = this.jot.tempoSource;
    const events = source.tempoEvents ?? [];
    if (!bars || events.length === 0) return [];

    // Map each SOURCE bar index -> its global start-beat in the lead-in-
    // inclusive space the header positions against (`--bar-start-beat`).
    // The synthetic lead-in bar advances the beat cursor but isn't a
    // source bar, mirroring `buildBarTempos`'s anchoring.
    const sourceBarStartBeat: number[] = [];
    let cumBeats = 0;
    for (const bar of bars) {
      if (bar.id !== LEAD_IN_BAR_ID) sourceBarStartBeat.push(cumBeats);
      cumBeats += bar.beats;
    }

    // Walk events in canonical order, tracking the running tempo so a
    // transition with no explicit `start` ramps from the tempo in force.
    const sorted = [...events].sort((a, b) =>
      a.barIndex !== b.barIndex ? a.barIndex - b.barIndex : a.beat - b.beat
    );
    let currentBpm = initialBpm(source);
    const ramps: TempoRamp[] = [];
    for (const ev of sorted) {
      const bpm = ev.bpm;
      if (typeof bpm === 'object') {
        const base = sourceBarStartBeat[ev.barIndex];
        // Skip (don't advance the running tempo for) an event whose
        // barIndex can't be placed, only possible from a corrupt jot, but
        // advancing anyway would skew a later ramp's implicit start.
        if (base === undefined) continue;
        const startBeat = base + ev.beat;
        ramps.push({
          startBeat,
          endBeat: startBeat + bpm.duration,
          startBpm: bpm.start ?? currentBpm,
          endBpm: bpm.end,
        });
        // A ramp leaves the tempo at its `end`, so a following ramp with
        // an implicit `start` resumes from there (musical semantics; not
        // the flattened `resolveBpm` value playback currently uses).
        currentBpm = bpm.end;
      } else if (bpm > 0) {
        currentBpm = bpm;
      }
    }
    return ramps;
  }

  get dominantBpmAndTime(): {
    dominantBpm: number | undefined;
    dominantTime: TimeSignature | undefined;
  } {
    return pickDominantBpmAndTime(this.jot);
  }
}
