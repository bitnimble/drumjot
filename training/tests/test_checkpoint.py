import json

import drumjot_training.checkpoint as checkpoint
from drumjot_training.config import Config


def test_run_metadata_has_inference_fields():
    meta = checkpoint.run_metadata(Config(), {"k": 0.6, "s": 0.5})
    assert meta["lanes"] == list(Config().lanes)
    assert meta["encoder"] == Config().encoder
    assert meta["encoder_layer"] == Config().encoder_layer
    assert meta["in_dim"] == 1024  # MERT hidden dim
    assert meta["head_hidden"] == Config().head_hidden


def test_run_metadata_roundtrips_via_json():
    meta = checkpoint.run_metadata(Config(), {"k": 0.6, "ho": 0.3})
    back = json.loads(json.dumps(meta))
    assert back["thresholds"]["k"] == 0.6
    assert back["thresholds"]["ho"] == 0.3
    assert back["lanes"] == list(Config().lanes)


def test_run_metadata_thresholds_are_floats():
    # tuned thresholds may arrive as numpy/py floats; metadata must be plain
    meta = checkpoint.run_metadata(Config(), {"k": 1})
    assert isinstance(meta["thresholds"]["k"], float)
