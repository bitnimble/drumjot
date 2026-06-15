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
import { BarTempos, buildBarTempos } from 'src/schema/dsl/tempo';
import {
  buildTimeline,
  type JotTimeline,
  type LaidOutJot,
  pickDominantBpmAndTime,
} from './timeline';

export class TempoPresenter {
  constructor(private readonly jot: LaidOutJot) {
    makeObservable(this, {
      timeline: computed,
      barTempos: computed,
      dominantBpmAndTime: computed,
    });
  }

  get timeline(): JotTimeline {
    return buildTimeline(this.jot);
  }

  get barTempos(): readonly BarTempos[] {
    const bars = this.jot.voices[0]?.bars;
    if (!bars) return [];
    return buildBarTempos(this.jot.source, bars);
  }

  get dominantBpmAndTime(): {
    dominantBpm: number | undefined;
    dominantTime: TimeSignature | undefined;
  } {
    return pickDominantBpmAndTime(this.jot);
  }
}
