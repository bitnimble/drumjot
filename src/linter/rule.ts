/**
 * Rule shape, ESLint-style. Each rule is a small module that exports a
 * `Rule` object with metadata (id, default severity, kind) and a `check`
 * function. The rule registry collects all rules; the linter runs them.
 *
 * Rules accept a pre-computed `LintContext` (the Jot, a flat list of
 * resolved notes with their effective instrument kinds and source ranges,
 * etc.) so individual rules don't repeat the AST walk.
 */
import { Bar, Element, Group, Jot, Modifier, Note, SourceRange, Voice } from 'src/schema/dsl/dsl';
import {
  DrumInstrumentKind,
  LimbCategory,
  defaultKindForPitch,
  effectiveLimbCategory,
} from 'src/instruments/instruments';
import { LintDiagnostic, LintKind, LintSeverity } from './diagnostics';

export type RuleSeverityConfig = {
  enabled: boolean;
  severity: LintSeverity;
};

export type Rule = {
  id: string;
  /** Default behaviour — can be overridden via lint config. */
  defaultSeverity: LintSeverity;
  /** Categorisation surfaced on every diagnostic this rule emits. */
  kind: LintKind;
  /** Short human-readable description. Used in CLI / prompt output. */
  description: string;
  /**
   * Walk the context and emit diagnostics. The runtime severity (after
   * config) is passed in so rules don't have to read configuration
   * themselves.
   */
  check: (ctx: LintContext, severity: LintSeverity) => LintDiagnostic[];
};

/**
 * Resolved note: a single Note from the AST flattened to include its
 * containing voice/bar indices, the effective instrument kind looked up
 * via the active mapping, and (when available) a source range.
 *
 * Pre-flattening like this lets each rule operate on a list rather than
 * recursing into Groups/Simultaneities itself. Groups don't appear here
 * directly; they're preserved separately on the context for rules that
 * actually care about group-level behaviour (e.g. roll-on-group).
 */
export type ResolvedNote = {
  note: Note;
  pitch: string;
  kind: DrumInstrumentKind;
  modifiers: ReadonlySet<Modifier>;
  /** Hand vs foot vs either, after applying foot-modifier overrides. */
  limbCategory: LimbCategory;
  voiceIndex: number;
  barIndex: number;
  /** Onset bucket within the bar — notes inside the same Simultaneity share an id. */
  simulId: number;
  /** Best-effort source range; undefined if the note was hand-built. */
  range?: SourceRange;
};

/** Group-level reference; preserved separately from `ResolvedNote`. */
export type ResolvedGroup = {
  group: Group;
  voiceIndex: number;
  barIndex: number;
  /** Each child note's `kind` (so roll-on-multi-instrument can be detected). */
  childKinds: Set<DrumInstrumentKind>;
  range?: SourceRange;
};

export type LintContext = {
  jot: Jot;
  source: string;
  notes: ResolvedNote[];
  groups: ResolvedGroup[];
  /**
   * Pre-computed bucket of resolved notes by simulId. The "too many hands"
   * and "same-hand conflict" rules read this directly to avoid re-grouping.
   */
  notesBySimul: ReadonlyMap<number, ResolvedNote[]>;
};

// ---------- Context builder ----------

/**
 * Flatten the Jot AST into the shape rules want to read. Simultaneity
 * children share a simulId; sequential notes / standalone notes each get a
 * fresh one. Patterns are not expanded — the linter operates on the
 * unexpanded AST so diagnostics point at the source the user wrote.
 */
export function buildLintContext(jot: Jot, source: string): LintContext {
  const notes: ResolvedNote[] = [];
  const groups: ResolvedGroup[] = [];
  const mapping = jot.globalMetadata.instrumentMapping ?? {};
  let simulCounter = 0;

  const kindFor = (pitch: string): DrumInstrumentKind => {
    const entry = mapping[pitch];
    if (entry) return entry.kind;
    return defaultKindForPitch(pitch);
  };

  const visit = (
    el: Element,
    voiceIndex: number,
    barIndex: number,
    simulId: number | null
  ): void => {
    if (el.kind === 'note') {
      const id = simulId ?? simulCounter++;
      const mods = new Set<Modifier>(el.modifiers ?? []);
      const kind = kindFor(el.pitch);
      notes.push({
        note: el,
        pitch: el.pitch,
        kind,
        modifiers: mods,
        limbCategory: effectiveLimbCategory(kind, mods),
        voiceIndex,
        barIndex,
        simulId: id,
        range: el.range,
      });
      return;
    }
    if (el.kind === 'rest') return;
    if (el.kind === 'simul') {
      const id = simulCounter++;
      for (const child of el.elements) {
        visit(child, voiceIndex, barIndex, id);
      }
      return;
    }
    if (el.kind === 'group') {
      const childKinds = new Set<DrumInstrumentKind>();
      const collect = (inner: Element): void => {
        if (inner.kind === 'note') childKinds.add(kindFor(inner.pitch));
        else if (inner.kind === 'simul') inner.elements.forEach(collect);
        else if (inner.kind === 'group') inner.elements.forEach(collect);
      };
      el.elements.forEach(collect);
      groups.push({
        group: el,
        voiceIndex,
        barIndex,
        childKinds,
        range: el.range,
      });
      for (const child of el.elements) visit(child, voiceIndex, barIndex, simulId);
      return;
    }
    if (el.kind === 'patternRef') {
      // Patterns are referenced rather than inlined — the rules linting
      // the pattern's own notes will fire when the pattern is *defined*
      // (via the patterns dict below). Pattern usages aren't double-linted.
      return;
    }
  };

  const visitBar = (bar: Bar, voiceIndex: number, barIndex: number) => {
    for (const el of bar.elements) visit(el, voiceIndex, barIndex, null);
  };

  jot.voices.forEach((voice: Voice, voiceIndex: number) => {
    if (voice.anacrusis) {
      for (const el of voice.anacrusis) visit(el, voiceIndex, -1, null);
    }
    voice.bars.forEach((bar, barIndex) => visitBar(bar, voiceIndex, barIndex));
  });

  if (jot.patterns) {
    for (const [name, pat] of Object.entries(jot.patterns)) {
      // Synthetic voice/bar indices for pattern notes so they don't collide
      // with real-voice indices in downstream grouping. The negative
      // ranges are enough for rules that key off voice/bar identity.
      const voiceIndex = -1 - name.length;
      for (const el of pat.elements) visit(el, voiceIndex, -1, null);
    }
  }

  const notesBySimul = new Map<number, ResolvedNote[]>();
  for (const n of notes) {
    const arr = notesBySimul.get(n.simulId);
    if (arr) arr.push(n);
    else notesBySimul.set(n.simulId, [n]);
  }

  return { jot, source, notes, groups, notesBySimul };
}
