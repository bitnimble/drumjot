import numpy as np
import torch

from drumjot_training.train import Clip, collate_clips, masked_bce


def _clip(t, dim=4, n_lanes=3, val=1.0):
    return Clip(
        features=np.full((t, dim), val, dtype=np.float32),
        targets=np.zeros((n_lanes, t), dtype=np.float32),
        onsets_by_lane={},
    )


def test_collate_pads_and_masks():
    X, Y, Yw, A, mask = collate_clips([_clip(5), _clip(3)])
    assert X.shape == (2, 5, 4)
    assert Y.shape == (2, 3, 5)
    assert torch.equal(Yw, Y)  # no weight_targets set -> falls back to targets
    assert float(A.sum()) == 0.0  # no activity targets set -> zeros
    assert mask.sum().item() == 8  # 5 + 3 valid frames
    assert mask[1, 3:].sum().item() == 0  # second clip padded after frame 3
    assert torch.all(X[1, 3:] == 0)  # padded feature region zeroed


def test_collate_upcasts_fp16_features():
    # fp16 cache files must feed the model as float32 (collate upcasts)
    c = Clip(
        features=np.ones((4, 3), dtype=np.float16),
        targets=np.zeros((2, 4), dtype=np.float32),
        onsets_by_lane={},
    )
    X, *_ = collate_clips([c])
    assert X.dtype == torch.float32


def test_masked_bce_ignores_padding():
    # The per-frame masked mean for a clip must not change when it sits in a
    # batch next to a longer clip (padding + mask must zero out the rest).
    c = _clip(6, val=0.5)
    pw = torch.ones(3, 1)

    _, Ya, _, _, ma = collate_clips([c])
    logits_a = torch.randn_like(Ya)
    loss_a = masked_bce(logits_a, Ya, ma, pw)

    _, Yb, _, _, _ = collate_clips([c, _clip(10)])
    logits_b = torch.randn(2, 3, 10)
    logits_b[0, :, :6] = logits_a[0]
    only_clip0 = torch.zeros(2, 10)
    only_clip0[0, :6] = 1.0
    loss_b = masked_bce(logits_b, Yb, only_clip0, pw)

    assert torch.allclose(loss_a, loss_b, atol=1e-6)


class _StubEncoder:
    # only .name/.layer are read on a cache hit (no audio/model needed)
    def __init__(self, cfg):
        self.name = cfg.encoder
        self.layer = cfg.encoder_layer


def test_materialize_and_cached_clips_stream_from_disk(tmp_path):
    from drumjot_training import embeddings
    from drumjot_training.config import Config
    from drumjot_training.train import CachedClips, materialize

    cfg = Config()
    audio, T = "/fake/song.flac", 200
    onsets = {"k": [0.1, 100.0], "s": [0.2]}  # 100.0s is past the 30s cap
    # pre-populate the cache so materialize takes the cache-hit path
    key = embeddings.cache_key(audio, cfg.encoder, cfg.encoder_layer, 30.0)
    cache = tmp_path / "_cache_mert"
    cache.mkdir()
    np.save(cache / f"{key}.npy", np.zeros((T, embeddings.MERT_DIM), dtype=np.float32))

    ds = materialize([(audio, onsets)], _StubEncoder(cfg), cfg, cache, 30.0, "t", log=lambda s: None)
    assert isinstance(ds, CachedClips)
    assert len(ds) == 1

    # iter_targets must NOT need the feature file (uses the stored frame count)
    targets = list(ds.iter_targets())
    assert targets[0].shape == (len(cfg.lanes), T)

    clip = ds[0]  # __getitem__ streams the features back from disk
    assert clip.features.shape == (T, embeddings.MERT_DIM)
    assert clip.onsets_by_lane["k"] == [0.1]  # 100.0s capped out


def test_train_loop_runs_batched_over_variable_lengths():
    # Exercises shuffle + chunking + per-batch averaging with a batch that
    # spans clips of different lengths (so padding/masking is in play).
    from drumjot_training.config import Config
    from drumjot_training.model import MultiLaneHeads
    from drumjot_training.train import train_loop

    cfg = Config(encoder_fps=100.0)
    nl = len(cfg.lanes)
    clips = [_clip(t, dim=8, n_lanes=nl) for t in (30, 20, 25, 15, 18)]
    model = MultiLaneHeads(in_dim=8, hidden=8, num_layers=1, lane_names=cfg.lanes)
    logs: list[str] = []
    hist = train_loop(model, clips, cfg, epochs=3, batch_size=2, log=logs.append)
    assert len(hist["train_loss"]) == 3
    assert all(np.isfinite(v) for v in hist["train_loss"])
    # progress lines must carry per-epoch timing + ETA so throughput is visible
    epoch_lines = [s for s in logs if s.startswith("epoch")]
    assert len(epoch_lines) == 3  # first 3 epochs all print
    assert "s/ep" in epoch_lines[-1]
    assert "eta" in epoch_lines[-1]


def test_train_loop_writes_periodic_checkpoint(tmp_path):
    from drumjot_training.config import Config
    from drumjot_training.model import MultiLaneHeads
    from drumjot_training.train import train_loop

    cfg = Config(encoder_fps=100.0)
    nl = len(cfg.lanes)
    clips = [_clip(20, dim=8, n_lanes=nl) for _ in range(4)]
    model = MultiLaneHeads(in_dim=8, hidden=8, num_layers=1, lane_names=cfg.lanes)
    out = tmp_path / "ckpt"
    train_loop(
        model, clips, cfg, epochs=3, batch_size=2,
        out_dir=str(out), checkpoint_every=1, log=lambda s: None,
    )
    assert (out / "model.pt").exists()  # periodic save fired mid-run
    assert (out / "meta.json").exists()


def test_checkpoint_reloads_into_fresh_model_for_resume(tmp_path):
    # --resume warm-start: a saved state_dict must load cleanly into an
    # identically-built model and restore the exact weights.
    from drumjot_training import checkpoint
    from drumjot_training.config import Config
    from drumjot_training.model import MultiLaneHeads

    cfg = Config()
    m1 = MultiLaneHeads(in_dim=8, hidden=8, num_layers=1)
    checkpoint.save(tmp_path, m1, cfg, {ln: 0.5 for ln in cfg.lanes})
    m2 = MultiLaneHeads(in_dim=8, hidden=8, num_layers=1)
    m2.load_state_dict(torch.load(tmp_path / "model.pt", map_location="cpu"))
    for p1, p2 in zip(m1.parameters(), m2.parameters(), strict=True):
        assert torch.equal(p1, p2)


def test_masked_bce_applies_per_lane_pos_weight():
    # pos_weight (n_lanes, 1) must broadcast over (B, n_lanes, T): weighting a
    # lane's positives up must raise the loss when that lane has a positive.
    targets = torch.zeros(1, 2, 4)
    targets[0, 0, 0] = 1.0  # one positive in lane 0
    logits = torch.zeros(1, 2, 4)
    mask = torch.ones(1, 4)
    base = masked_bce(logits, targets, mask, torch.ones(2, 1))
    up = masked_bce(logits, targets, mask, torch.tensor([[10.0], [1.0]]))
    assert up > base
