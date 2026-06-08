"""Unit tests for the opt-in LarsNet Stage-2 separator integration.

These cover the wiring (lane mapping, checkpoint layout, missing-weights
error, provisioning idempotency, runner dispatch) WITHOUT loading the
real ~590 MB CC-BY-NC weights or touching a GPU - the actual separation
quality is validated by hand on real songs, not here.
"""
from __future__ import annotations

from pathlib import Path

import pytest

from app.pipeline import larsnet, provision
from app.pipeline.runner import PipelineContext, PipelineOptions, _do_stems_per
from app.pipeline.separate import STEM_NAME_TO_PITCH, StemsPerResult


def test_stem_to_pitch_aligns_with_mdx23c_lanes() -> None:
    # LarsNet must emit the same five DSL pitch lanes as MDX23C so it's a
    # drop-in; ride+crash arrive merged as `cymbals` -> `c`.
    assert larsnet.STEM_TO_PITCH == {
        "kick": "k",
        "snare": "s",
        "toms": "t",
        "hihat": "h",
        "cymbals": "c",
    }
    # Every LarsNet output pitch is a pitch MDX23C's mapping also produces,
    # so cymbal_split / hihat_split / onsets treat both identically.
    for pitch in larsnet.STEM_TO_PITCH.values():
        assert pitch in set(STEM_NAME_TO_PITCH.values())


def test_checkpoint_path_layout(tmp_path: Path) -> None:
    p = larsnet.checkpoint_path(tmp_path, "kick")
    assert p == tmp_path / "larsnet" / "kick" / "pretrained_kick_unet.pth"


def test_load_models_missing_raises(tmp_path: Path) -> None:
    # No checkpoints provisioned -> fail loud (mapped to HTTP 500 upstream).
    with pytest.raises(FileNotFoundError, match="LarsNet checkpoint missing"):
        larsnet.load_models(tmp_path, "cpu")


def test_provision_larsnet_idempotent_and_size_pinned(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    # Shrink the integrity pin so the fake download can write a tiny file.
    monkeypatch.setattr(provision, "_LARSNET_CKPT_SIZE", 4)

    downloaded: list[str] = []

    def fake_download(url: str, dest: Path) -> None:
        downloaded.append(url)
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"OKAY")  # exactly 4 bytes -> passes the pin

    monkeypatch.setattr(provision, "_download", fake_download)

    # Fresh provision: all five stems fetched.
    provision._provision_larsnet(tmp_path)
    assert len(downloaded) == 5
    for stem in larsnet.STEMS:
        assert larsnet.checkpoint_path(tmp_path, stem).exists()

    # Idempotent: correct-size files already present -> no re-download.
    downloaded.clear()
    provision._provision_larsnet(tmp_path)
    assert downloaded == []

    # Wrong-size file -> refetched (corruption / truncation recovery).
    larsnet.checkpoint_path(tmp_path, "kick").write_bytes(b"BAD")  # 3 bytes
    downloaded.clear()
    provision._provision_larsnet(tmp_path)
    assert downloaded == [
        f"{provision._LARSNET_HF_BASE}/kick/pretrained_kick_unet.pth"
    ]


def test_provision_larsnet_rejects_wrong_download_size(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.setattr(provision, "_LARSNET_CKPT_SIZE", 4)

    def short_download(url: str, dest: Path) -> None:
        dest.parent.mkdir(parents=True, exist_ok=True)
        dest.write_bytes(b"XX")  # 2 bytes != pinned 4

    monkeypatch.setattr(provision, "_download", short_download)
    with pytest.raises(RuntimeError, match="integrity pin"):
        provision._provision_larsnet(tmp_path)


class _FakeSeparator:
    """Records which Stage-2 method the runner dispatched to."""

    def __init__(self) -> None:
        self.called: str | None = None

    def run_stems_per(self, drum_stem: Path, work_dir: Path) -> StemsPerResult:
        self.called = "mdx23c"
        return StemsPerResult(per_instrument={}, residual=None)

    def run_stems_per_larsnet(
        self, drum_stem: Path, work_dir: Path
    ) -> StemsPerResult:
        self.called = "larsnet"
        return StemsPerResult(per_instrument={}, residual=None)


@pytest.mark.parametrize("choice", ["mdx23c", "larsnet"])
def test_do_stems_per_dispatches_on_option(
    tmp_path: Path, choice: str
) -> None:
    drum_stem = tmp_path / "drum_stem.wav"
    drum_stem.write_bytes(b"\x00")  # must merely exist for the guard
    ctx = PipelineContext(audio_path=tmp_path / "in.wav", work_dir=tmp_path)
    ctx.drum_stem = drum_stem
    sep = _FakeSeparator()
    options = PipelineOptions(drum_separator=choice)  # type: ignore[arg-type]

    _do_stems_per(ctx, sep, options, output_sink=None)

    assert sep.called == choice
