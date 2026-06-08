import drumjot_training.egmd as egmd

CSV = (
    "drummer,session,id,style,bpm,beat_type,time_signature,"
    "midi_filename,audio_filename,duration,split\n"
    "drummer1,s1,1,funk/groove,120,beat,4-4,drummer1/1.mid,drummer1/1.wav,10.5,train\n"
    "drummer1,s1,2,rock/groove,,beat,4-4,drummer1/2.mid,drummer1/2.wav,20.0,test\n"
)


def _write_csv(tmp_path):
    f = tmp_path / "e-gmd-v1.0.0.csv"
    f.write_text(CSV)
    return f


def test_read_index_pairs_audio_and_midi(tmp_path):
    clips = egmd.read_index(_write_csv(tmp_path), tmp_path)
    assert len(clips) == 2
    assert clips[0].audio_path == tmp_path / "drummer1/1.wav"
    assert clips[0].midi_path == tmp_path / "drummer1/1.mid"
    assert clips[0].split == "train"
    assert clips[0].duration == 10.5
    assert clips[0].bpm == 120.0


def test_missing_bpm_is_none(tmp_path):
    clips = egmd.read_index(_write_csv(tmp_path), tmp_path)
    assert clips[1].bpm is None


def test_read_index_accepts_str_paths(tmp_path):
    csvf = _write_csv(tmp_path)
    clips = egmd.read_index(str(csvf), str(tmp_path))  # str, not Path
    assert clips[0].audio_path == tmp_path / "drummer1/1.wav"


def test_filter_by_split(tmp_path):
    clips = egmd.read_index(_write_csv(tmp_path), tmp_path)
    train = egmd.for_split(clips, "train")
    assert [c.split for c in train] == ["train"]


def test_take_duration_caps_total_seconds(tmp_path):
    clips = egmd.read_index(_write_csv(tmp_path), tmp_path)
    # 10.5 s fits under a 15 s cap; adding the 20 s clip would exceed it.
    capped = egmd.take_duration(clips, 15.0)
    assert len(capped) == 1
    assert capped[0].duration == 10.5
