import { describe, expect, it } from 'bun:test';
import type { SampleLoadProgress } from 'src/jot_view/playback';
import {
  formatMb,
  formatStageLabel,
  samplePct,
  sampleProgressLabel,
  sampleProgressWidth,
} from './toolbar_status';

const MB = 1024 * 1024;
const progress = (over: Partial<SampleLoadProgress>): SampleLoadProgress => ({
  loaded: 0,
  total: 0,
  fromCache: false,
  ...over,
});

describe('samplePct', () => {
  it('rounds loaded/total to a percentage', () => {
    expect(samplePct(progress({ loaded: 15 * MB, total: 30 * MB }))).toBe(50);
    expect(samplePct(progress({ loaded: 1, total: 3 }))).toBe(33);
  });

  it('clamps to 100 when loaded exceeds total', () => {
    expect(samplePct(progress({ loaded: 40 * MB, total: 30 * MB }))).toBe(100);
  });
});

describe('formatMb', () => {
  it('formats bytes as MB to one decimal', () => {
    expect(formatMb(30 * MB)).toBe('30.0 MB');
    expect(formatMb(1.5 * MB)).toBe('1.5 MB');
    expect(formatMb(0)).toBe('0.0 MB');
  });
});

describe('sampleProgressWidth', () => {
  it('pins to 100% while decoding regardless of byte counts', () => {
    expect(sampleProgressWidth('decoding', undefined)).toBe('100%');
    expect(sampleProgressWidth('decoding', progress({ loaded: 1, total: 100 }))).toBe('100%');
  });

  it('shows a working sliver while connecting or with no progress object', () => {
    expect(sampleProgressWidth('connecting', undefined)).toBe('8%');
    expect(sampleProgressWidth('downloading', undefined)).toBe('8%');
  });

  it('shows full when served from cache', () => {
    expect(sampleProgressWidth('downloading', progress({ fromCache: true }))).toBe('100%');
  });

  it('reflects the download percentage when a total is known', () => {
    expect(sampleProgressWidth('downloading', progress({ loaded: 15 * MB, total: 30 * MB }))).toBe(
      '50%',
    );
  });

  it('falls back to a fixed sliver when the total is unknown', () => {
    expect(sampleProgressWidth('downloading', progress({ loaded: 5 * MB, total: 0 }))).toBe('40%');
  });
});

describe('sampleProgressLabel', () => {
  it('describes the connecting / no-progress state', () => {
    expect(sampleProgressLabel('connecting', undefined)).toBe('Drums · waiting for server…');
    expect(sampleProgressLabel('downloading', undefined)).toBe('Drums · waiting for server…');
  });

  it('describes decoding', () => {
    expect(sampleProgressLabel('decoding', progress({ loaded: 1, total: 1 }))).toBe(
      'Drums · decoding samples…',
    );
  });

  it('describes a cache hit', () => {
    expect(sampleProgressLabel('downloading', progress({ fromCache: true }))).toBe(
      'Drums · loading from cache',
    );
  });

  it('shows loaded / total while downloading with a known total', () => {
    expect(
      sampleProgressLabel('downloading', progress({ loaded: 15 * MB, total: 30 * MB })),
    ).toBe('Drums · downloading 15.0 MB / 30.0 MB');
  });

  it('shows only loaded while downloading with an unknown total', () => {
    expect(sampleProgressLabel('downloading', progress({ loaded: 5 * MB, total: 0 }))).toBe(
      'Drums · downloading 5.0 MB',
    );
  });
});

describe('formatStageLabel', () => {
  it('maps every pipeline stage to friendly wording', () => {
    expect(formatStageLabel('stems_all')).toBe('separating drums');
    expect(formatStageLabel('stems_per')).toBe('separating drum pieces');
    expect(formatStageLabel('beats')).toBe('tracking beats');
    expect(formatStageLabel('onsets')).toBe('detecting onsets');
    expect(formatStageLabel('filter')).toBe('filtering artifact onsets');
    expect(formatStageLabel('quantise')).toBe('quantising onsets');
    expect(formatStageLabel('transcribe')).toBe('rendering MIDI');
  });
});
