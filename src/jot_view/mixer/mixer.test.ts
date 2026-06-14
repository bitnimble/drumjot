/**
 * Enable-matrix tests for the audio-track overflow menu. Each role
 * should produce the correct enable state for the two split items;
 * `unknown` defaults to enabled (ad-hoc loads where we couldn't
 * classify the audio).
 */
import { describe, expect, it } from 'bun:test';
import { AudioTrackRole } from 'src/jot_view/playback/audio_tracks';
import { splitDrumPiecesState, splitFromMixState } from './overflow_menus';

type Row = [AudioTrackRole | undefined, boolean, boolean];

const MATRIX: Row[] = [
  // role,         splitFromMix, splitDrumPieces
  ['full-mix',     true,         false],
  ['drums',        false,        true],
  ['no-drums',     false,        false],
  ['drum-piece',   false,        false],
  ['unknown',      true,         true],
  [undefined,      true,         true],
];

describe('audio-track overflow-menu enable matrix', () => {
  for (const [role, mixEnabled, piecesEnabled] of MATRIX) {
    const tag = role ?? '<undefined>';
    it(`role=${tag} → splitFromMix.enabled=${mixEnabled}, splitDrumPieces.enabled=${piecesEnabled}`, () => {
      expect(splitFromMixState(role).enabled).toBe(mixEnabled);
      expect(splitDrumPiecesState(role).enabled).toBe(piecesEnabled);
    });
  }

  it('every state has a non-empty reason string for tooltip rendering', () => {
    const roles: (AudioTrackRole | undefined)[] = [
      'full-mix', 'drums', 'no-drums', 'drum-piece', 'unknown', undefined,
    ];
    for (const role of roles) {
      expect(splitFromMixState(role).reason.length).toBeGreaterThan(0);
      expect(splitDrumPiecesState(role).reason.length).toBeGreaterThan(0);
    }
  });
});
