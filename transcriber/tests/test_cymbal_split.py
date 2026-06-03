"""Unit tests for the voice-based cymbal classifier.

Covers the deterministic helpers that don't require librosa / anthropic:

  * `_coerce_index_set`; out-of-range clamping and non-int ignore.
  * `_band_flatness` / `_band_crest_db`; band-restricted timbre features
    (pure-numpy cores, fed synthetic spectra; no audio fixture needed).
  * `_envelope_db` / `_decay_metrics`; the decay-shape / intrinsic-decay
    measures (pure numpy).
  * `_cluster_voices` / `_label_voices`; the deterministic voice split and
    the crash-default ride/crash labelling.
  * `_discard_fallback`; the no-LLM discard pass keeps everything.

The `_discard_llm` and `_measure` paths require external dependencies
(anthropic and librosa respectively) and are exercised end-to-end, not
here.
"""
from __future__ import annotations

import numpy as np

from app.pipeline.cymbal_split import (
    _CRASH_LABEL,
    _ENV_OFFSETS_S,
    _RIDE_ACCENT_LOWMID_MARGIN_DB,
    _RIDE_LABEL,
    _RIDE_STREAM_GAP_S,
    _band_crest_db,
    _band_flatness,
    _cluster_voices,
    _coerce_index_set,
    _decay_metrics,
    _demote_ride_accents,
    _discard_fallback,
    _envelope_db,
    _Feat,
    _label_voices,
)

_STREAM_GAP = _RIDE_STREAM_GAP_S - 0.05   # comfortably in-stream
_SPARSE_GAP = _RIDE_STREAM_GAP_S + 1.0    # comfortably isolated


def _f(
    *, decay: float = 0.0, gap: float = 0.0, centroid: float = 0.0,
    tonal: float = 0.0, flat: float = 0.0, rate: float = 0.0,
    low_mid: float = 0.0,
) -> _Feat:
    return _Feat(
        decay_s=decay, flatness=flat, centroid_hz=centroid, gap_s=gap,
        tonal_db=tonal, decay_rate_db_s=rate, low_mid_db=low_mid,
    )


def test_coerce_index_set_clamps_and_dedupes() -> None:
    # Out-of-range, negative, duplicates, and non-int entries all filtered.
    out = _coerce_index_set([0, 2, 2, 4, 9, -1, "x", None, 1.5], n=5)
    assert out == {0, 1, 2, 4}


def test_discard_fallback_keeps_everything() -> None:
    # No LLM -> discard nothing; the deterministic split still produced the
    # lanes, so "keep everything" is the safe degraded behaviour.
    assert _discard_fallback() == set()


# --- _band_flatness -----------------------------------------------------
#
# The bug these guard against: full-range flatness collapses toward zero
# for cymbals because the stem/source is dead above ~14 kHz and below
# ~1 kHz, so a third of the FFT bins are empty and crush the geometric
# mean. Restricting to the occupied band must lift the number off the
# floor while still reading low for a genuine tone.

def _spec_with_band_energy(
    lo_hz: float, hi_hz: float, *, sr: int = 44100, n_fft: int = 2048
):
    """A flat (noise-like) power spectrum confined to [lo, hi] Hz; bins
    outside that band hold a tiny floor. Returns (power_spec, freqs)."""
    freqs = np.fft.rfftfreq(n_fft, 1.0 / sr)
    band = (freqs >= lo_hz) & (freqs <= hi_hz)
    col = np.full(freqs.shape, 1e-8)
    col[band] = 1.0
    return col[:, None], freqs  # single frame


def test_band_flatness_lifts_off_floor_for_bandlimited_noise() -> None:
    # Energy is flat across the occupied band and absent elsewhere (the
    # real cymbal case: dead above ~14 kHz and below ~1.5 kHz). Full-range
    # flatness is crushed by the empty bins; restricting to the occupied
    # band recovers it.
    spec, freqs = _spec_with_band_energy(1500.0, 14000.0)
    full = _band_flatness(spec, freqs, 0.0, freqs[-1])
    band = _band_flatness(spec, freqs, 1500.0, 14000.0)
    assert full < 0.05
    assert band > 0.5
    assert band > full * 10


def test_band_flatness_low_for_a_tone_even_band_limited() -> None:
    # A single dominant bin inside the band is tonal: flatness stays near
    # zero regardless of band restriction (the metric still works).
    freqs = np.fft.rfftfreq(2048, 1.0 / 44100)
    col = np.full(freqs.shape, 1e-8)
    col[np.searchsorted(freqs, 5000.0)] = 1.0
    spec = col[:, None]
    assert _band_flatness(spec, freqs, 1500.0, 14000.0) < 0.05


def test_band_flatness_empty_band_is_zero() -> None:
    spec, freqs = _spec_with_band_energy(2000.0, 9000.0)
    # A band above Nyquist holds no FFT bins: return 0.0, don't blow up.
    assert _band_flatness(spec, freqs, 30000.0, 31000.0) == 0.0


# --- _band_crest_db -----------------------------------------------------
#
# The low-band crest is the perception-matching ride/crash cue the
# flatness band misses: a ride has a tall narrow partial (its pitched
# "ping") in ~200-1500 Hz; a crash is flat broadband noise there. Crest =
# peak-to-mean power in dB over the band, so a partial reads high and noise
# reads near 0 dB.

def test_band_crest_high_for_a_tonal_partial() -> None:
    freqs = np.fft.rfftfreq(2048, 1.0 / 44100)
    col = np.full(freqs.shape, 0.01)
    col[np.searchsorted(freqs, 500.0)] = 10.0  # a dominant partial
    crest = _band_crest_db(col[:, None], freqs, 200.0, 1500.0)
    assert crest > 15.0


def test_band_crest_near_zero_for_flat_noise() -> None:
    # Perfectly flat band: peak == mean, so crest is 0 dB. (Real broadband
    # noise sits a little above 0; a crash's low band is near-flat.)
    freqs = np.fft.rfftfreq(2048, 1.0 / 44100)
    spec = np.full((freqs.size, 3), 0.5)
    assert _band_crest_db(spec, freqs, 200.0, 1500.0) < 1.0


def test_band_crest_empty_band_is_zero() -> None:
    freqs = np.fft.rfftfreq(2048, 1.0 / 44100)
    spec = np.full((freqs.size, 1), 0.5)
    assert _band_crest_db(spec, freqs, 30000.0, 31000.0) == 0.0


# --- _envelope_db -------------------------------------------------------

def _rms_track(values, hop: int = 512, sr: int = 44100):
    """Build an (rms, rms_t) pair from a list of per-frame RMS values."""
    rms = np.asarray(values, dtype=float)
    rms_t = np.arange(len(values)) * (hop / sr)
    return rms, rms_t


def test_envelope_db_decaying_is_monotonic_and_starts_near_zero() -> None:
    # An exponential decay from a peak at t=0: each sampled offset should
    # read progressively further below the peak, first sample near 0 dB.
    t = np.arange(0, 1.0, 512 / 44100)
    rms = np.exp(-t / 0.15)  # ~ -29 dB/0.1s time-constant
    rms_t = t
    env = _envelope_db(rms, rms_t, peak_time=0.0, peak=1.0, win_end=10.0)
    assert len(env) == len(_ENV_OFFSETS_S)
    assert env[0] > -5.0  # +50ms still near the peak
    assert env == sorted(env, reverse=True)  # monotonically decreasing
    assert env[-1] < -15.0  # +800ms well down


def test_envelope_db_sustained_stays_near_zero() -> None:
    # A crash-like sustain: RMS holds near the peak for the whole window.
    rms, rms_t = _rms_track([1.0] * 200)
    env = _envelope_db(rms, rms_t, peak_time=0.0, peak=1.0, win_end=10.0)
    assert len(env) == len(_ENV_OFFSETS_S)
    assert all(v > -3.0 for v in env)


def test_envelope_db_truncates_at_next_onset() -> None:
    # When the next onset sits 0.25s out, only offsets <= 0.25s survive,
    # so the model sees a short list (the tail was cut, as with `decay_s`).
    rms, rms_t = _rms_track([1.0] * 200)
    env = _envelope_db(rms, rms_t, peak_time=0.0, peak=1.0, win_end=0.25)
    expected = [o for o in _ENV_OFFSETS_S if o <= 0.25]
    assert len(env) == len(expected)


def test_envelope_db_zero_peak_is_empty() -> None:
    rms, rms_t = _rms_track([0.0] * 50)
    assert _envelope_db(rms, rms_t, peak_time=0.0, peak=0.0, win_end=10.0) == []


# --- _decay_metrics -----------------------------------------------------
#
# The truncation-robust intrinsic-decay measure: a RATE (dB/s), so it
# separates a washy crash (shallow) from an articulate ride (steep) even
# when the window is short.

def test_decay_metrics_fast_decay_high_rate() -> None:
    # RMS drops 20 dB in 0.1s -> ~200 dB/s, sustain well below 0.
    t = np.arange(0, 0.3, 512 / 44100)
    rr = np.array([1.0 if x == 0 else 10 ** (-(x / 0.1)) for x in t])
    sustain, rate = _decay_metrics(rr, t, 0, 1.0, horizon_end=0.5)
    assert sustain < -15.0
    assert rate > 100.0


def test_decay_metrics_sustained_low_rate() -> None:
    # A crash wall: energy barely falls across the window -> shallow rate.
    t = np.arange(0, 0.3, 512 / 44100)
    rr = np.full(t.shape, 1.0)
    rr[len(rr) // 2:] = 0.85  # ~ -1.4 dB
    sustain, rate = _decay_metrics(rr, t, 0, 1.0, horizon_end=0.5)
    assert sustain > -3.0
    assert rate < 20.0


def test_decay_metrics_zero_peak() -> None:
    t = np.arange(0, 0.1, 512 / 44100)
    assert _decay_metrics(np.zeros_like(t), t, 0, 0.0, horizon_end=0.5) == (0.0, 0.0)


# --- _cluster_voices ----------------------------------------------------

def _voice_feats(centroids, tonals):
    return [
        _f(centroid=c, tonal=tn, flat=0.1)
        for c, tn in zip(centroids, tonals, strict=True)
    ]


def test_cluster_voices_single_voice_when_uniform() -> None:
    # All onsets share a fingerprint -> one voice, ids all 0.
    feats = _voice_feats([7000.0] * 12, [10.0] * 12)
    ids = _cluster_voices(feats)
    assert set(ids) == {0}


def test_cluster_voices_splits_two_separated_voices() -> None:
    # A bright ride cluster and a darker crash cluster, well separated.
    ride = _voice_feats([12000.0] * 8, [14.0] * 8)
    crash = _voice_feats([5000.0] * 8, [4.0] * 8)
    ids = _cluster_voices(ride + crash)
    assert len(set(ids)) == 2
    # Each contiguous block landed in one voice (membership is consistent).
    assert len(set(ids[:8])) == 1
    assert len(set(ids[8:])) == 1
    assert ids[0] != ids[-1]


def test_cluster_voices_too_few_onsets_is_single() -> None:
    feats = _voice_feats([12000.0, 5000.0, 12000.0], [14.0, 4.0, 14.0])
    assert set(_cluster_voices(feats)) == {0}


# --- _label_voices ------------------------------------------------------

def test_label_voices_sparse_voice_is_crash() -> None:
    # A voice that is mostly isolated accents -> crash (the default), even
    # though a couple of its hits happen to be closely spaced.
    feats = [_f(gap=_SPARSE_GAP)] * 6 + [_f(gap=_STREAM_GAP)] * 2
    assert _label_voices(feats, [0] * 8) == {0: _CRASH_LABEL}


def test_label_voices_dense_stream_is_ride() -> None:
    # A voice that is predominantly a dense, evenly-spaced stream -> ride.
    feats = [_f(gap=_STREAM_GAP)] * 10
    assert _label_voices(feats, [0] * 10) == {0: _RIDE_LABEL}


def test_label_voices_small_voice_never_ride() -> None:
    # Below `_VOICE_MIN_SIZE` onsets -> crash even if all in-stream (too
    # little evidence to call a timekeeping ride).
    feats = [_f(gap=_STREAM_GAP)] * 3
    assert _label_voices(feats, [0, 0, 0]) == {0: _CRASH_LABEL}


def test_label_voices_independent_per_voice() -> None:
    # Voice 0 a dense stream (ride), voice 1 sparse accents (crash).
    feats = [_f(gap=_STREAM_GAP)] * 6 + [_f(gap=_SPARSE_GAP)] * 6
    labels = _label_voices(feats, [0] * 6 + [1] * 6)
    assert labels == {0: _RIDE_LABEL, 1: _CRASH_LABEL}


# --- _demote_ride_accents -----------------------------------------------

_RIDE_LOWMID = -15.0   # a typical ride: strong low fundamental
_CRASH_LOWMID = _RIDE_LOWMID - _RIDE_ACCENT_LOWMID_MARGIN_DB - 5.0  # mid wash


def test_demote_ride_accents_relabels_crash_timbred_hit() -> None:
    # A ride stream (high low/mid) with one crash-timbred hit (low low/mid:
    # a mid wash, no fundamental): only that hit -> crash. Timbre-only, so
    # the in-stream gap of the crash-timbred hit doesn't matter.
    feats = [
        _f(gap=_STREAM_GAP, low_mid=_RIDE_LOWMID),
        _f(gap=_STREAM_GAP, low_mid=_RIDE_LOWMID),
        _f(gap=_STREAM_GAP, low_mid=_CRASH_LOWMID),  # embedded crash wash
        _f(gap=_STREAM_GAP, low_mid=_RIDE_LOWMID),
        _f(gap=_STREAM_GAP, low_mid=_RIDE_LOWMID),
    ]
    prov = [_RIDE_LABEL] * 5
    out = _demote_ride_accents(prov, feats, [0] * 5)
    assert out == [_RIDE_LABEL, _RIDE_LABEL, _CRASH_LABEL, _RIDE_LABEL, _RIDE_LABEL]


def test_demote_ride_accents_spares_ride_timbred_hit() -> None:
    # A hit whose timbre MATCHES the ride stream (high low/mid: it has the
    # fundamental) is a real ride note, not a crash -> kept ride, even if
    # isolated. This is the false-positive the timbre gate prevents.
    feats = [
        _f(gap=_STREAM_GAP, low_mid=_RIDE_LOWMID),
        _f(gap=_STREAM_GAP, low_mid=_RIDE_LOWMID),
        _f(gap=10.0, low_mid=_RIDE_LOWMID + 0.5),  # isolated but ride timbre
        _f(gap=_STREAM_GAP, low_mid=_RIDE_LOWMID),
        _f(gap=_STREAM_GAP, low_mid=_RIDE_LOWMID),
    ]
    prov = [_RIDE_LABEL] * 5
    out = _demote_ride_accents(prov, feats, [0] * 5)
    assert out == [_RIDE_LABEL] * 5


def test_demote_ride_accents_leaves_crash_voices_untouched() -> None:
    # No ride labels -> nothing to demote, even for crash-timbred hits.
    feats = [_f(gap=10.0, low_mid=_CRASH_LOWMID)] * 3
    prov = [_CRASH_LABEL] * 3
    assert _demote_ride_accents(prov, feats, [0] * 3) == prov
