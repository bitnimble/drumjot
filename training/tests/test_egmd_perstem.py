import importlib.util
from pathlib import Path

import numpy as np
import pytest

from drumjot_training import egmd
from drumjot_training.lanes import LANES


def test_perstem_to_lanes_covers_all_lanes_no_overlap():
    flat = [ln for lanes in egmd.PERSTEM_TO_LANES.values() for ln in lanes]
    assert set(flat) == set(LANES)
    assert len(flat) == len(set(flat))


def test_restricted_onsets_keeps_only_stem_lanes(monkeypatch):
    full = {ln: [] for ln in LANES}
    full.update({"k": [0.1], "s": [0.2], "ss": [0.25], "rd": [0.4], "cr": [0.3], "mc": [0.5], "t": [0.6]})
    monkeypatch.setattr("drumjot_training.midi_labels.onsets_from_path", lambda _p: full)
    # cymbal stem -> ride/crash/misc-cymbal only
    o = egmd.restricted_onsets("ignored.midi", "c")
    assert set(o) == set(LANES)
    assert o["rd"] == [0.4] and o["cr"] == [0.3] and o["mc"] == [0.5]
    assert o["k"] == [] and o["s"] == [] and o["t"] == []
    # snare stem -> snare + side stick
    os_ = egmd.restricted_onsets("ignored.midi", "s")
    assert os_["s"] == [0.2] and os_["ss"] == [0.25]
    assert sum(len(v) for v in os_.values()) == 2
    # kick stem -> only k
    ok = egmd.restricted_onsets("ignored.midi", "k")
    assert ok["k"] == [0.1] and sum(len(v) for v in ok.values()) == 1


def test_perstem_index_pairs_stems_via_csv(tmp_path):
    (tmp_path / "annotation").mkdir()
    (tmp_path / "annotation" / "d1__s__c1.midi").write_bytes(b"")
    for pitch in ("k", "c"):  # only 2 of 5 stems produced
        d = tmp_path / "audio" / "perstem" / pitch
        d.mkdir(parents=True)
        (d / "d1__s__c1.flac").write_bytes(b"")
    (tmp_path / "e-gmd-v1.0.0.csv").write_text(
        "audio_filename,midi_filename,split,duration,bpm\n"
        "audio/sep_drum/d1__s__c1.flac,annotation/d1__s__c1.midi,train,12.5,120\n"
    )
    clips = egmd.perstem_index(tmp_path)
    assert len(clips) == 2
    assert {c.pitch for c in clips} == {"k", "c"}
    assert all(c.split == "train" for c in clips)
    assert all(c.midi_path.name == "d1__s__c1.midi" for c in clips)


# --- pure helpers in separate_egmd_dataset.py ------------------------------


def _load_sep():
    pytest.importorskip("soundfile")  # the module imports soundfile at top
    path = Path(__file__).resolve().parent.parent / "scripts" / "separate_egmd_dataset.py"
    spec = importlib.util.spec_from_file_location("separate_egmd_dataset", path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def test_uid_for():
    sed = _load_sep()
    assert sed.uid_for("drummer1/eval_session/1_funk-groove1_138_beat_4-4_1.wav") == \
        "drummer1__eval_session__1_funk-groove1_138_beat_4-4_1"


def test_greedy_select_is_balanced_and_deterministic():
    sed = _load_sep()
    # lanes=2: A,B,D are common-lane-only; C is the rare-lane clip
    counts = np.array([[100, 0], [100, 0], [1, 5], [100, 0]])
    durs = np.array([1.0, 1.0, 1.0, 1.0])
    order_all = sed.greedy_select(counts, durs, 4.0)
    assert order_all == sed.greedy_select(counts, durs, 4.0)      # deterministic
    assert order_all[0] == 0                                      # highest-onset first
    assert 2 in order_all[:2]                                     # rare clip pulled in early
    # duration cap stops selection
    capped = sed.greedy_select(counts, durs, 2.0)
    assert len(capped) == 2 and 2 in capped                       # rare clip beats redundant commons


def test_plan_batches_groups_by_buffer():
    sed = _load_sep()
    # gap 2s, buffer 25s: 10 + (2+10)=22 fit; +(2+10)=34 over -> new batch
    assert sed.plan_batches([10.0, 10.0, 10.0], buffer_sec=25.0, gap_sec=2.0) == [[0, 1], [2]]
    # a clip longer than the buffer becomes its own batch
    assert sed.plan_batches([40.0, 3.0], buffer_sec=25.0, gap_sec=2.0) == [[0], [1]]


def test_slice_seconds():
    sed = _load_sep()
    y = np.arange(100).reshape(100, 1)
    s = sed.slice_seconds(y, sr=10, start_s=2.0, len_s=3.0)
    assert s.shape == (30, 1) and int(s[0, 0]) == 20 and int(s[-1, 0]) == 49


def test_build_buffer_offsets_and_pad(tmp_path):
    sed = _load_sep()
    import soundfile as sf

    sr = 8000
    pa, pb = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(str(pa), np.ones((sr, 1), dtype="float32"), sr)          # 1.0 s
    sf.write(str(pb), np.full((sr // 2, 1), 0.5, dtype="float32"), sr)  # 0.5 s
    batch = [{"uid": "a", "src_audio": pa, "midi": pa}, {"uid": "b", "src_audio": pb, "midi": pb}]
    valid, buf, bsr, offsets = sed.build_buffer(batch, gap_sec=0.25, log=lambda s: None)
    assert bsr == sr and [e["uid"] for e in valid] == ["a", "b"]
    assert offsets[0] == (0.0, 1.0)
    assert offsets[1] == (1.25, 0.5)  # clip a (1.0) + gap (0.25)
    assert buf.shape[0] == int(sed.MIN_SEP_SECONDS * sr)  # undersized buffer padded


def test_build_buffer_skips_sr_mismatch_without_desync(tmp_path):
    sed = _load_sep()
    import soundfile as sf

    pa, pb = tmp_path / "a.wav", tmp_path / "b.wav"
    sf.write(str(pa), np.ones((8000, 1), dtype="float32"), 8000)
    sf.write(str(pb), np.ones((4000, 1), dtype="float32"), 16000)  # different sr -> dropped
    batch = [{"uid": "a", "src_audio": pa, "midi": pa}, {"uid": "b", "src_audio": pb, "midi": pb}]
    valid, _buf, sr, offsets = sed.build_buffer(batch, gap_sec=0.0, log=lambda s: None)
    assert sr == 8000 and [e["uid"] for e in valid] == ["a"] and len(offsets) == 1  # aligned 1:1


def test_write_outputs_slices_all_five_stems_silence_filling_absent(tmp_path):
    sed = _load_sep()
    import soundfile as sf

    out = tmp_path / "out"
    dsr = 100
    dy = np.ones((5 * dsr, 1), dtype="float32") * 0.3   # 5 s drum stem
    pys = {"k": (np.ones((5 * dsr, 1), dtype="float32") * 0.2, dsr)}  # only kick present
    midi = tmp_path / "x.midi"
    midi.write_bytes(b"MThd-fake")
    valid = [{"uid": "clip1", "midi": midi}]
    n = sed.write_outputs(valid, [(1.0, 2.0)], dy, dsr, pys, out)  # slice [1s, 3s)
    assert n == 1
    p = sed.out_paths(out, "clip1")
    assert sed.is_done(p)  # drum + all 5 perstem + midi all landed
    drum, _ = sf.read(str(p["drum"]), always_2d=True)
    assert drum.shape[0] == int(round(2.0 * dsr))  # sliced to len_s
    silent, _ = sf.read(str(p["per"]["s"]), always_2d=True)  # snare absent -> silence
    assert silent.shape[0] == int(round(2.0 * dsr)) and float(np.abs(silent).max()) == 0.0
