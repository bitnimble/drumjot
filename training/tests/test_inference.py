import drumjot_training.inference as inference
from drumjot_training.lanes import LANES


def test_lane_to_pitch_covers_every_lane():
    assert set(inference.LANE_TO_PITCH) == set(LANES)


def test_lane_to_pitch_is_injective():
    # every trained class maps to a distinct pitch (no merging down)
    pitches = list(inference.LANE_TO_PITCH.values())
    assert len(pitches) == len(set(pitches))


def test_to_pitch_preserves_all_classes():
    lane_onsets = {
        "k": [0.5],
        "s": [0.6],
        "ss": [0.7],  # side stick stays its own pitch
        "t": [1.1],
        "hc": [0.1],  # closed -> h
        "hp": [0.2],  # pedal stays its own pitch (not folded into h)
        "ho": [0.4],  # open -> H
        "rd": [1.0],  # ride -> d
        "cr": [2.0],  # crash -> c
        "mc": [2.5],  # misc cymbals stays its own pitch
        "mp": [9.0],  # misc percussion stays its own pitch (not dropped)
    }
    p = inference.to_pitch_onsets(lane_onsets)
    assert p["k"] == [0.5]
    assert p["s"] == [0.6]
    assert p["ss"] == [0.7]
    assert p["t"] == [1.1]
    assert p["h"] == [0.1]
    assert p["hp"] == [0.2]
    assert p["H"] == [0.4]
    assert p["d"] == [1.0]
    assert p["c"] == [2.0]
    assert p["mc"] == [2.5]
    assert p["mp"] == [9.0]
    assert len(p) == 11  # all classes preserved
