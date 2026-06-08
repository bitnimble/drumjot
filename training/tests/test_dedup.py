import hashlib

import drumjot_training.dedup as dedup


def test_sha1_of_file(tmp_path):
    f = tmp_path / "a.bin"
    f.write_bytes(b"hello world")
    assert dedup.sha1_of_file(f) == hashlib.sha1(b"hello world").hexdigest()


def test_identical_patterns_share_signature():
    a = {"k": [0.0, 0.5], "s": [0.25]}
    b = {"k": [0.0, 0.5], "s": [0.25]}
    assert dedup.onset_signature(a) == dedup.onset_signature(b)


def test_small_jitter_within_rounding_is_same():
    a = {"k": [0.500]}
    b = {"k": [0.503]}  # within a 10 ms bin
    assert dedup.onset_signature(a, round_s=0.01) == dedup.onset_signature(b, round_s=0.01)


def test_jitter_past_a_bin_differs():
    a = {"k": [0.500]}
    b = {"k": [0.520]}  # two bins away
    assert dedup.onset_signature(a, round_s=0.01) != dedup.onset_signature(b, round_s=0.01)


def test_different_lane_structure_differs():
    a = {"k": [0.0]}
    b = {"k": [0.0], "s": [0.5]}
    assert dedup.onset_signature(a) != dedup.onset_signature(b)


def test_signature_is_order_independent_within_a_lane():
    a = {"k": [0.5, 0.0, 0.25]}
    b = {"k": [0.0, 0.25, 0.5]}
    assert dedup.onset_signature(a) == dedup.onset_signature(b)
