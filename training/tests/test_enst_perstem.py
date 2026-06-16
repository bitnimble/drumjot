import importlib.util
from pathlib import Path

import numpy as np
import pytest

from drumjot_training import enst
from drumjot_training.lanes import LANES


def test_perstem_to_lanes_covers_all_lanes_no_overlap():
    flat = [ln for lanes in enst.PERSTEM_TO_LANES.values() for ln in lanes]
    assert set(flat) == set(LANES)            # every lane has a stem
    assert len(flat) == len(set(flat))         # hard routing: no lane in two stems


def test_restricted_onsets_keeps_only_stem_lanes(tmp_path):
    ann = tmp_path / "t1.txt"
    ann.write_text(
        "0.10 bd\n0.20 sd\n0.25 cs\n0.30 cr\n0.40 rc\n0.50 ch\n0.60 cb\n"
    )
    # cymbals stem -> only ride/crash kept; china (ch) dropped, everything else emptied
    o = enst.restricted_onsets(ann, "c")
    assert set(o) == set(LANES)
    assert o["cr"] == [0.30] and o["rd"] == [0.40]
    assert "mc" not in o  # china dropped (mc lane removed)
    assert o["k"] == [] and o["s"] == []
    # snare stem -> snare + side stick (cs) ride along, nothing else
    os_ = enst.restricted_onsets(ann, "s")
    assert os_["s"] == [0.20] and os_["ss"] == [0.25]
    assert sum(len(v) for v in os_.values()) == 2
    # kick stem -> only k
    ok = enst.restricted_onsets(ann, "k")
    assert ok["k"] == [0.10] and sum(len(v) for v in ok.values()) == 1


def test_perstem_index_pairs_present_stems_and_infers_drummer(tmp_path):
    base = tmp_path / "drummer_2"
    (base / "annotation").mkdir(parents=True)
    (base / "annotation" / "t1.txt").write_text("0.1 bd\n")
    for pitch in ("k", "c"):  # only 2 of the 5 stems produced
        d = base / "audio" / "perstem" / pitch
        d.mkdir(parents=True)
        (d / "t1.flac").write_bytes(b"")
    clips = enst.perstem_index(tmp_path)
    assert len(clips) == 2
    assert {c.pitch for c in clips} == {"k", "c"}
    assert all(c.drummer == "drummer_2" for c in clips)


def test_index_pairs_flac_for_sep_drum_and_wav_still_works(tmp_path):
    base = tmp_path / "drummer_1"
    (base / "annotation").mkdir(parents=True)
    (base / "annotation" / "t1.txt").write_text("0.1 bd\n")
    (base / "audio" / "sep_drum").mkdir(parents=True)
    (base / "audio" / "sep_drum" / "t1.flac").write_bytes(b"")
    (base / "audio" / "wet_mix").mkdir(parents=True)
    (base / "audio" / "wet_mix" / "t1.wav").write_bytes(b"")
    sep = enst.index(tmp_path, mix="sep_drum")
    assert len(sep) == 1 and sep[0].audio_path.suffix == ".flac"
    wet = enst.index(tmp_path, mix="wet_mix")
    assert len(wet) == 1 and wet[0].audio_path.suffix == ".wav"


def test_perstem_for_split_holds_out_drummer(tmp_path):
    for d in ("drummer_1", "drummer_3"):
        base = tmp_path / d
        (base / "annotation").mkdir(parents=True)
        (base / "annotation" / "t.txt").write_text("0.1 bd\n")
        pd = base / "audio" / "perstem" / "k"
        pd.mkdir(parents=True)
        (pd / "t.flac").write_bytes(b"")
    clips = enst.perstem_index(tmp_path)
    tr = enst.perstem_for_split(clips, "train")            # drummer_3 held out
    va = enst.perstem_for_split(clips, "validation")
    assert {c.drummer for c in tr} == {"drummer_1"}
    assert {c.drummer for c in va} == {"drummer_3"}


# --- combine_mix (separate_enst_dataset.py) --------------------------------


def _load_sep_module():
    path = Path(__file__).resolve().parent.parent / "scripts" / "separate_enst_dataset.py"
    spec = importlib.util.spec_from_file_location("separate_enst_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_combine_mix_sums_and_clip_guards(tmp_path):
    sf = pytest.importorskip("soundfile")
    sep = _load_sep_module()
    sr, n = 44100, 1000
    wet = np.zeros((n, 2), dtype=np.float32); wet[:, 0] = 0.6
    acc = np.zeros((n, 2), dtype=np.float32); acc[:, 0] = 0.6
    wp, ap = tmp_path / "wet.wav", tmp_path / "acc.wav"
    sf.write(str(wp), wet, sr, subtype="FLOAT")
    sf.write(str(ap), acc, sr, subtype="FLOAT")
    y, out_sr = sep.combine_mix(wp, ap)
    assert out_sr == sr
    assert abs(float(np.abs(y).max()) - 1.0) < 1e-4   # 0.6+0.6=1.2 -> scaled to 1.0
    assert y[:, 0].max() > y[:, 1].max()              # balance preserved (ch1 silent)


def test_combine_mix_silent_accompaniment_passthrough(tmp_path):
    sf = pytest.importorskip("soundfile")
    sep = _load_sep_module()
    sr, n = 44100, 500
    wet = (np.random.RandomState(0).randn(n, 2) * 0.1).astype(np.float32)
    acc = np.zeros((n, 2), dtype=np.float32)          # silent backing (most ENST takes)
    wp, ap = tmp_path / "wet.wav", tmp_path / "acc.wav"
    sf.write(str(wp), wet, sr, subtype="FLOAT")
    sf.write(str(ap), acc, sr, subtype="FLOAT")
    y, _ = sep.combine_mix(wp, ap)
    assert np.allclose(y, wet, atol=1e-5)


def test_combine_mix_missing_accompaniment_passthrough(tmp_path):
    sf = pytest.importorskip("soundfile")
    sep = _load_sep_module()
    sr, n = 44100, 300
    wet = (np.ones((n, 2)) * 0.2).astype(np.float32)
    wp = tmp_path / "wet.wav"
    sf.write(str(wp), wet, sr, subtype="FLOAT")
    y, _ = sep.combine_mix(wp, tmp_path / "does_not_exist.wav")
    assert np.allclose(y, wet, atol=1e-5)
