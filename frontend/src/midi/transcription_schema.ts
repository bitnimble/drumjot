/**
 * Runtime schema for the transcriber's `transcription.json` sidecar, the
 * Drumjot-native container that rides alongside `prediction.mid` in a debug
 * bundle and, when present, supplies tempo (and bar drift) at higher fidelity
 * than the MIDI tempo track. See `transcriber/app/pipeline/transcription.py`.
 *
 * It's parsed from a *downloaded* bundle, so we validate it rather than trust
 * a cast: `TranscriptionSchema.safeParse` fails closed (→ the loader falls
 * back to the MIDI tempo track) on a malformed or unknown-format payload. The
 * inferred types are the single source of truth for the shape on the TS side.
 */
import { z } from 'zod';

/** A flat tempo change: the song plays at `bpm` from `tick` onward. */
const TempoStepEventSchema = z.object({
  /** Absolute MIDI tick of the change (same PPQ as `prediction.mid`). */
  tick: z.number(),
  bpm: z.number(),
});

/** A gradual tempo change from `start` at `tick` to `end` at `end_tick`. */
const TempoRampEventSchema = z.object({
  /** Absolute MIDI tick the ramp starts at. */
  tick: z.number(),
  bpm: z.object({
    start: z.number(),
    end: z.number(),
    /** Absolute MIDI tick the ramp reaches `end` at. */
    end_tick: z.number(),
  }),
  /**
   * Easing of the gradual tempo change. ONLY `"linear"` is supported today:
   * linear-in-time tempo, i.e. BPM rises at a constant rate per second,
   * equivalently `bpm²` is linear in beat (this is the closed form
   * `src/schema/dsl/tempo.ts` integrates for a `BpmTransition`). The schema is
   * intentionally strict here, a non-`"linear"` value fails the parse and the
   * loader falls back to the MIDI tempo track, rather than silently
   * mis-rendering a curve we can't draw.
   *
   * Two other easings are planned (TODO); each would relax this to an enum,
   * add a matching closed-form integral in `tempo.ts` + a fit/model-selection
   * in the transcriber's segmenter, and bump `format`:
   *
   * - `"exponential"`: constant *proportional* change (a fixed BPM ratio per
   *   unit time). Arguably the most perceptually natural accelerando, since
   *   tempo perception is roughly logarithmic (a 60->70 change reads as bigger
   *   than 160->170), so equal ratios feel like equal steps.
   * - `"logarithmic"`: front-loaded change that eases off (fast at first,
   *   then flattening); the inverse easing of `"exponential"`.
   */
  shape: z.literal('linear').optional(),
});

/** Step OR ramp; discriminated by whether `bpm` is a number or an object. */
const TempoMapEventSchema = z.union([TempoRampEventSchema, TempoStepEventSchema]);

const TempoMapSchema = z.object({
  /** Tempo in force before the first event (the pre-first-event default). */
  initial_bpm: z.number(),
  events: z.array(TempoMapEventSchema),
});

export const TranscriptionSchema = z.object({
  /**
   * Container format version (mirrors
   * `transcriber/app/pipeline/transcription.py::TRANSCRIPTION_FORMAT`). A
   * `z.literal` so any newer/unknown format fails the parse and the loader
   * cleanly falls back to the MIDI tempo track instead of misreading it.
   */
  format: z.literal(1),
  tempoMap: TempoMapSchema,
  /**
   * Per-(drum-)bar performance drift in seconds: how far each real downbeat
   * sits past the clean uniform grid (`beats.BarInfo.drift_sec`), so the
   * editor can keep a uniform tempo display yet align bar lines + waveform to
   * the recording. Optional for bundles produced before it existed; absent
   * means no drift. Indexed by drum bar (maps to `layers[0].bars[leadBars+i]`).
   */
  barDrift: z.array(z.number()).optional(),
});

export type Transcription = z.infer<typeof TranscriptionSchema>;
export type TranscriptionTempoMap = z.infer<typeof TempoMapSchema>;
