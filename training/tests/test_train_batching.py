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
    from drumjot_training.train import CachedClips, _window_specs, materialize

    cfg = Config()
    audio, T = "/fake/song.flac", 200
    onsets = {"k": [0.1, 100.0], "s": [0.2]}  # 100.0s is past the 30s cap
    # pre-populate the cache so materialize takes the cache-hit path
    key = embeddings.cache_key(audio, cfg.encoder, cfg.encoder_layer, 30.0)
    cache = tmp_path / "_cache_mert"
    cache.mkdir()
    np.save(cache / f"{key}.npy", np.zeros((T, embeddings.MERT_DIM), dtype=np.float32))

    specs = _window_specs([(audio, onsets)], 30.0, 3.0, 1)  # legacy single window
    ds = materialize(specs, _StubEncoder(cfg), cfg, cache, 30.0, "t", log=lambda s: None)
    assert isinstance(ds, CachedClips)
    assert len(ds) == 1

    # iter_targets must NOT need the feature file (uses the stored frame count)
    targets = list(ds.iter_targets())
    assert targets[0].shape == (len(cfg.lanes), T)

    clip = ds[0]  # __getitem__ streams the features back from disk
    assert clip.features.shape == (T, embeddings.MERT_DIM)
    assert clip.onsets_by_lane["k"] == [0.1]  # 100.0s capped out


def test_materialize_frame_index_backfills_then_avoids_npy(tmp_path):
    from drumjot_training import embeddings
    from drumjot_training.config import Config
    from drumjot_training.train import _load_feature_index, _window_specs, materialize

    cfg = Config()
    audio, T = "/fake/song.flac", 200
    onsets = {"k": [0.1], "s": [0.2]}
    variant = embeddings.feat_variant(cfg.high_band)
    key = embeddings.cache_key(audio, cfg.encoder, cfg.encoder_layer, 30.0, variant)
    cache = tmp_path / "_cache_mert"
    cache.mkdir()
    np.save(cache / f"{key}.npy", np.zeros((T, embeddings.MERT_DIM), dtype=np.float32))

    specs = _window_specs([(audio, onsets)], 30.0, 3.0, 1)
    # 1st pass: index empty -> backfills the frame count from the .npy header.
    materialize(specs, _StubEncoder(cfg), cfg, cache, 30.0, "t", log=lambda s: None)
    assert _load_feature_index(cache).get(key) == T
    assert (cache / "_feature_index.json").exists()

    # 2nd pass with the .npy DELETED: the index hit supplies the frame count, so
    # materialize + iter_targets work WITHOUT reading the feature file at all.
    (cache / f"{key}.npy").unlink()
    ds = materialize(specs, _StubEncoder(cfg), cfg, cache, 30.0, "t", log=lambda s: None)
    assert len(ds) == 1
    assert list(ds.iter_targets())[0].shape == (len(cfg.lanes), T)


def test_feature_index_version_mismatch_rebuilds(tmp_path):
    import json

    from drumjot_training.train import (
        FEATURE_INDEX_VERSION,
        _load_feature_index,
        _save_feature_index,
    )

    _save_feature_index(tmp_path, {"abc": 123})
    assert _load_feature_index(tmp_path) == {"abc": 123}  # roundtrip
    p = tmp_path / "_feature_index.json"
    d = json.loads(p.read_text())
    d["v"] = FEATURE_INDEX_VERSION + 99  # stale format version
    p.write_text(json.dumps(d))
    assert _load_feature_index(tmp_path) == {}  # discarded -> rebuild


def test_train_loop_resume_continues_from_checkpoint(tmp_path):
    # A run stopped after 2 epochs resumes (into a fresh model) and finishes the
    # rest, continuing the SAME history instead of restarting from epoch 0.
    import os

    from drumjot_training.config import Config
    from drumjot_training.model import MultiLaneHeads
    from drumjot_training.train import train_loop

    cfg = Config(encoder_fps=100.0)
    nl = len(cfg.lanes)
    clips = [_clip(t, dim=8, n_lanes=nl) for t in (30, 20, 25, 15, 18)]
    rp = str(tmp_path / "r.pt")

    # "stop" after 2 epochs (a full-state checkpoint is written each epoch)
    m1 = MultiLaneHeads(in_dim=8, hidden=8, num_layers=1, lane_names=cfg.lanes)
    h1 = train_loop(m1, clips, cfg, epochs=2, batch_size=2, resume_path=rp, log=lambda s: None)
    assert len(h1["train_loss"]) == 2
    assert os.path.exists(rp)  # checkpoint left behind for resume

    # resume for 5 epochs total: loads state, trains only epochs 2,3,4
    logs: list[str] = []
    m2 = MultiLaneHeads(in_dim=8, hidden=8, num_layers=1, lane_names=cfg.lanes)
    h2 = train_loop(m2, clips, cfg, epochs=5, batch_size=2, resume_path=rp, log=logs.append)
    assert len(h2["train_loss"]) == 5                  # 2 restored + 3 new, NOT restarted
    assert h2["train_loss"][:2] == h1["train_loss"]    # restored history carried forward
    assert any("resumed @ epoch 2" in s for s in logs)
    trained = {int(s.split()[1]) for s in logs if s.startswith("epoch")}
    assert trained == {2, 3, 4}  # only the remaining epochs ran this session


def test_train_loop_skips_nonfinite_batches(tmp_path):
    # A batch with nan features -> nan loss/grad. The step must be SKIPPED so the
    # weights stay finite and the run continues, instead of one nan poisoning every
    # later epoch (val -> 0). Batch size 1 isolates the bad clip to its own batch.
    import torch

    from drumjot_training.config import Config
    from drumjot_training.model import MultiLaneHeads
    from drumjot_training.train import train_loop

    cfg = Config(encoder_fps=100.0)
    nl = len(cfg.lanes)
    clips = [_clip(20, dim=8, n_lanes=nl) for _ in range(4)]
    clips[0].features[:] = np.nan  # poison one clip's features
    logs: list[str] = []
    model = MultiLaneHeads(in_dim=8, hidden=8, num_layers=1, lane_names=cfg.lanes)
    hist = train_loop(model, clips, cfg, epochs=3, batch_size=1, log=logs.append)

    assert all(np.isfinite(v) for v in hist["train_loss"])  # nan batch didn't pollute the avg
    assert all(torch.isfinite(p).all() for p in model.parameters())  # weights never poisoned
    assert any("skipped" in s for s in logs)  # the skip is reported


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


def test_train_loop_keep_best_restores_per_lane_peak(monkeypatch):
    # keep_best is PER-LANE: each head is restored from the epoch where THAT lane's
    # val F1 peaked (lanes overfit at different times -- here k peaks @1, s @3).
    from drumjot_training import train
    from drumjot_training.config import Config
    from drumjot_training.model import MultiLaneHeads

    cfg = Config(encoder_fps=100.0, lanes=("k", "s"), aux_act_weight=0.0)
    clips = [_clip(20, dim=8, n_lanes=2) for _ in range(4)]
    val = [train.Clip(
        features=np.zeros((20, 8), dtype=np.float32),
        targets=np.zeros((2, 20), dtype=np.float32),
        onsets_by_lane={"k": [0.1], "s": [0.2]},  # both lanes present every epoch
    )]
    model = MultiLaneHeads(in_dim=8, hidden=8, num_layers=1, lane_names=("k", "s"))

    sched = iter([{"k": 0.1, "s": 0.1}, {"k": 0.9, "s": 0.3},   # k peaks @ epoch 1
                  {"k": 0.5, "s": 0.5}, {"k": 0.4, "s": 0.9}])   # s peaks @ epoch 3
    snaps: list[dict] = []

    def fake_eval(m, c, cf, th=None):  # 1 val clip -> called once per epoch
        snaps.append({k: v.detach().clone() for k, v in m.state_dict().items()})
        return next(sched)

    monkeypatch.setattr(train, "evaluate_clip", fake_eval)
    hist = train.train_loop(model, clips, cfg, epochs=4, batch_size=2, val_clips=val,
                            keep_best=True, log=lambda s: None)
    assert hist["best_epoch_by_lane"] == [1.0, 3.0]  # k@1, s@3 (cfg.lanes order)
    assert hist["vf1_k"] == [0.1, 0.9, 0.5, 0.4] and hist["vf1_s"] == [0.1, 0.3, 0.5, 0.9]
    # each head restored from its OWN best epoch: k's params from ep1, s's from ep3
    sd = model.state_dict()
    for k in sd:
        src = snaps[1] if k.startswith("heads.k.") else snaps[3]
        assert torch.equal(sd[k], src[k])


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


def test_mean_f1_counts_only_lanes_with_reference_onsets(monkeypatch):
    # per-stem clips carry onsets for one instrument; mean_f1 must average only
    # the lanes that have reference onsets, not score the empty lanes as 0.
    import numpy as np

    from drumjot_training import train
    from drumjot_training.config import Config

    cfg = Config()
    clip = train.Clip(
        features=np.zeros((4, 4), dtype=np.float32),
        targets=np.zeros((len(cfg.lanes), 4), dtype=np.float32),
        onsets_by_lane={"k": [0.1]},  # only kick present (a kick stem)
    )
    monkeypatch.setattr(
        train, "evaluate_clip",
        lambda m, c, cf, th=None: {ln: (0.9 if ln == "k" else 0.0) for ln in cf.lanes},
    )
    # == kick's F1, NOT 0.9/len(lanes); empty lanes are skipped
    assert abs(train.mean_f1(None, [clip], cfg) - 0.9) < 1e-9


def test_report_scores_output_lanes_only(monkeypatch):
    # _report must aggregate over cfg.lanes only -- iterating onsets_by_lane keys
    # would index the per-lane F1 dict with any stray non-output key and KeyError.
    import numpy as np

    from drumjot_training import train
    from drumjot_training.config import Config

    cfg = Config()
    clip = train.Clip(
        features=np.zeros((4, 4), dtype=np.float32),
        targets=np.zeros((len(cfg.lanes), 4), dtype=np.float32),
        onsets_by_lane={"k": [0.1], "zz": [0.2]},  # stray non-output key
    )
    monkeypatch.setattr(
        train, "evaluate_clip", lambda m, c, cf, th=None: {ln: 0.5 for ln in cf.lanes}
    )
    train._report(None, [clip], cfg, {})  # must not raise KeyError on `zz`
