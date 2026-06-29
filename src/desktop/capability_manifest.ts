import { type AcceleratorKind } from './desktop_bridge';

export type CapabilityId = 'transcription' | 'lyrics' | 'lyrics.japanese' | 'ai-assist';

/** How a capability is satisfied: by downloading deps, by configuring
 *  credentials (no download), or by a non-installable system prereq. */
export type CapabilityKind = 'deps' | 'credentials' | 'system';

export type WeightSpec = { repoId: string; revision: string; approxBytes: number };

export type CapabilitySpec = {
  id: CapabilityId;
  name: string;
  /** First-run UI copy: what this capability does. */
  description: string;
  kind: CapabilityKind;
  /** uv dependency-group names this capability adds. */
  groups: string[];
  /** HF weights pulled lazily on first use (content-addressed). */
  weights: WeightSpec[];
  /** Prerequisite capabilities (the install DAG). */
  requires: CapabilityId[];
  /** Whether installing pulls the shared torch / accelerator tier. */
  accelerator: 'required' | 'none';
  /** This capability's own incremental bytes (weights + its unique non-torch
   *  deps), excluding the shared accelerator tier counted separately. */
  ownApproxBytes: number;
};

const GB = 1_000_000_000;
const MB = 1_000_000;

/**
 * Approximate *download* (compressed) size of the shared torch / accelerator
 * tier per hardware variant. Counted once across all accelerator-needing
 * capabilities. Estimates for pre-detection UI copy; the real shown number is
 * the resolver's diff vs the current venv.
 */
export const ACCELERATOR_TIER_BYTES: Record<AcceleratorKind, number> = {
  cuda: 4.5 * GB,
  rocm: 4 * GB,
  directml: 350 * MB,
  mps: 250 * MB,
  cpu: 600 * MB,
};

export const CAPABILITIES: readonly CapabilitySpec[] = [
  {
    id: 'transcription',
    name: 'Local transcription',
    description:
      'Transcribe drums from audio on this machine: stem separation, beat tracking, and the learned onset model.',
    kind: 'deps',
    groups: ['transcription'],
    weights: [
      { repoId: 'm-a-p/MERT-v1-330M', revision: 'main', approxBytes: 1.3 * GB },
    ],
    requires: [],
    accelerator: 'required',
    ownApproxBytes: 1.9 * GB,
  },
  {
    id: 'lyrics',
    name: 'Lyrics alignment',
    description: 'Align lyrics to the audio timeline using a forced aligner.',
    kind: 'deps',
    groups: ['lyrics'],
    weights: [
      { repoId: 'MahmoudAshraf/mms-300m-1130-forced-aligner', revision: 'main', approxBytes: 1.2 * GB },
    ],
    requires: [],
    accelerator: 'required',
    ownApproxBytes: 1.2 * GB,
  },
  {
    id: 'lyrics.japanese',
    name: 'Japanese lyrics',
    description: 'Contextual Japanese romanization for lyric alignment (adds a bundled dictionary).',
    kind: 'deps',
    groups: ['lyrics-ja'],
    weights: [],
    requires: ['lyrics'],
    accelerator: 'none',
    ownApproxBytes: 250 * MB,
  },
  {
    id: 'ai-assist',
    name: 'AI assist',
    description: 'LLM cleanup of transcriptions. Needs an Anthropic API key and a network connection.',
    kind: 'credentials',
    groups: [],
    weights: [],
    requires: [],
    accelerator: 'none',
    ownApproxBytes: 0,
  },
];

const BY_ID: ReadonlyMap<CapabilityId, CapabilitySpec> = new Map(
  CAPABILITIES.map((c) => [c.id, c]),
);

export function capabilityById(id: CapabilityId): CapabilitySpec {
  const spec = BY_ID.get(id);
  if (spec == null) {
    throw new Error(`unknown capability: ${id}`);
  }
  return spec;
}
