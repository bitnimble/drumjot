"""Per-lane-layer routing: Config resolution, model per-head routing, collate
batch shape, and checkpoint round-trip. The single-layer path must stay identical
(routing is fully opt-in via cfg.lane_layers / meta["lane_layers"])."""
import numpy as np
import pytest

from drumjot_training.config import Config

# ---- Config layer resolution (pure) ----

def test_lane_layer_map_defaults_to_encoder_layer():
    cfg = Config(lanes=("k", "s", "rd"), encoder_layer=10)
    assert cfg.lane_layer_map() == {"k": 10, "s": 10, "rd": 10}
    assert cfg.distinct_layers() == [10]
    assert not cfg.is_multilayer()


def test_lane_layer_map_applies_overrides_and_falls_back():
    cfg = Config(lanes=("k", "s", "rd", "cr"), encoder_layer=10,
                 lane_layers=(("k", 1), ("s", 4)))
    assert cfg.lane_layer_map() == {"k": 1, "s": 4, "rd": 10, "cr": 10}
    assert cfg.distinct_layers() == [1, 4, 10]
    assert cfg.is_multilayer()


def test_not_multilayer_when_all_overrides_equal():
    cfg = Config(lanes=("k", "s"), encoder_layer=10, lane_layers=(("k", 10), ("s", 10)))
    assert cfg.distinct_layers() == [10]
    assert not cfg.is_multilayer()


def test_config_stays_hashable_with_lane_layers():
    cfg = Config(lane_layers=(("k", 1), ("rd", 10)))
    hash(cfg)  # frozen dataclass: a TUPLE lane_layers field keeps it hashable
    assert {cfg: 1}[cfg] == 1


def test_parse_lane_layers():
    from drumjot_training.train import _parse_lane_layers
    assert _parse_lane_layers(None) is None
    assert _parse_lane_layers("") is None
    assert _parse_lane_layers("k:1,s:4,rd:10") == (("k", 1), ("s", 4), ("rd", 10))
    assert _parse_lane_layers(" k:1 , s:4 ") == (("k", 1), ("s", 4))
    with pytest.raises(SystemExit):
        _parse_lane_layers("k")  # missing :layer


# ---- model routing (needs torch) ----

def test_model_routes_each_head_to_its_assigned_layer():
    import torch

    from drumjot_training.model import MultiLaneHeads
    torch.manual_seed(0)
    m = MultiLaneHeads(in_dim=8, hidden=4, num_layers=1, lane_names=("a", "b"),
                       lane_layers={"a": 1, "b": 7})
    m.eval()
    fe1, fe7 = torch.randn(2, 5, 8), torch.randn(2, 5, 8)
    with torch.no_grad():
        out = m({1: fe1, 7: fe7})  # (B, 2, T)
        assert torch.allclose(out[:, 0], m.heads["a"](fe1), atol=1e-6)
        assert torch.allclose(out[:, 1], m.heads["b"](fe7), atol=1e-6)
        # swapping which layer feeds which head changes the result -> routing is real
        swapped = m({1: fe7, 7: fe1})
    assert not torch.allclose(swapped[:, 0], out[:, 0], atol=1e-4)


def test_model_single_layer_path_unchanged():
    import torch

    from drumjot_training.model import MultiLaneHeads
    torch.manual_seed(0)
    m = MultiLaneHeads(in_dim=8, hidden=4, num_layers=1, lane_names=("a", "b"))  # no routing
    assert m.lane_layers is None
    x = torch.randn(2, 5, 8)
    with torch.no_grad():
        out = m(x)
        ref = torch.stack([m.heads["a"](x), m.heads["b"](x)], dim=1)
    assert torch.allclose(out, ref, atol=1e-6)


def test_model_dict_input_without_routing_asserts():
    import torch

    from drumjot_training.model import MultiLaneHeads
    m = MultiLaneHeads(in_dim=8, hidden=4, num_layers=1, lane_names=("a",))  # lane_layers None
    with pytest.raises(AssertionError):
        m({1: torch.randn(1, 5, 8)})


# ---- collate (needs torch) ----

def _clip(feat_by_layer, n_lanes=2):
    from drumjot_training.train import Clip
    anchor = next(iter(feat_by_layer.values())) if feat_by_layer is not None else None
    return Clip(
        features=anchor if anchor is not None else np.zeros((5, 8), np.float32),
        targets=np.zeros((n_lanes, 5), np.float32), onsets_by_lane={},
        feat_by_layer=feat_by_layer,
    )


def test_collate_multilayer_yields_per_layer_dict_batch():
    import torch

    from drumjot_training.train import collate_clips
    mk = lambda: {1: np.random.randn(5, 8).astype(np.float32),
                  7: np.random.randn(5, 8).astype(np.float32)}
    clips = [_clip(mk()), _clip(mk())]
    X, _Y, _Yw, _A, _mask = collate_clips(clips)
    assert isinstance(X, dict) and sorted(X) == [1, 7]
    assert X[1].shape == (2, 5, 8) and X[7].shape == (2, 5, 8)
    assert torch.allclose(X[1][0], torch.as_tensor(clips[0].feat_by_layer[1]))
    assert torch.allclose(X[7][1], torch.as_tensor(clips[1].feat_by_layer[7]))


def test_collate_single_layer_yields_tensor():
    import torch

    from drumjot_training.train import collate_clips
    c = _clip(None)
    c.features = np.random.randn(5, 8).astype(np.float32)
    X, *_ = collate_clips([c, c])
    assert isinstance(X, torch.Tensor) and X.shape == (2, 5, 8)


# ---- checkpoint round-trip (needs torch) ----

def test_checkpoint_roundtrips_lane_layers(tmp_path):
    from drumjot_training import checkpoint
    from drumjot_training.embeddings import feat_dim
    from drumjot_training.model import MultiLaneHeads
    lanes = ("k", "s", "rd", "cr")
    cfg = Config(lanes=lanes, encoder_layer=10, lane_layers=(("k", 1), ("s", 4)),
                 head_hidden=4, head_layers=1)
    in_dim = feat_dim(cfg.high_band)
    m = MultiLaneHeads(in_dim=in_dim, hidden=4, num_layers=1, lane_names=lanes,
                       lane_layers=cfg.lane_layer_map())
    checkpoint.save(tmp_path, m, cfg, {ln: 0.5 for ln in lanes}, in_dim=in_dim)
    m2, meta = checkpoint.load(tmp_path)
    assert meta["lane_layers"] == {"k": 1, "s": 4, "rd": 10, "cr": 10}
    assert m2.lane_layers == {"k": 1, "s": 4, "rd": 10, "cr": 10}


def test_checkpoint_single_layer_lane_layers_is_none(tmp_path):
    from drumjot_training import checkpoint
    from drumjot_training.embeddings import feat_dim
    from drumjot_training.model import MultiLaneHeads
    cfg = Config(lanes=("k", "s"), encoder_layer=10, head_hidden=4, head_layers=1)
    in_dim = feat_dim(cfg.high_band)
    m = MultiLaneHeads(in_dim=in_dim, hidden=4, num_layers=1, lane_names=("k", "s"))
    checkpoint.save(tmp_path, m, cfg, {"k": 0.5, "s": 0.5}, in_dim=in_dim)
    m2, meta = checkpoint.load(tmp_path)
    assert meta["lane_layers"] is None
    assert m2.lane_layers is None
