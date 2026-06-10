"""ENST-Drums dataset loader.

ENST-Drums (Gillet & Richard, ISMIR 2006; Telecom Paris) is REAL acoustic
drum-kit audio: 3 drummers on 3 different kits, multi-miked and recorded, with
hand-aligned onset annotations. Unlike STAR (synthetic audio, ADT-generated
labels) and E-GMD (drum-module audio), ENST is genuine acoustic-drum
recordings, so it's the closest training match to our inference domain
(separated real drums) -- at the cost of being small (~1.5 h total, 3 kits).

Both 2025 ADT SOTA systems trained on it: MIROS (in the YourMT3+ mix) and N2N
used E-GMD; ENST is the real-recording complement to those.

Expected layout (the public release; point `DRUMJOT_ENST` or
`data_paths.toml[enst]` at `<root>`):

    <root>/drummer_{1,2,3}/
        annotation/<take>.txt           # "<time_seconds> <label>" per onset
        audio/
            wet_mix/<take>.wav          # drums-only kit mix (close+overheads) <- default
            dry_mix/<take>.wav          # drums-only, close mics only
            accompaniment/<take>.wav    # drums + minus-one backing (NOT used)
            <channel>/<take>.wav        # individual mics

We train on `wet_mix` (realistic isolated-kit audio) by default; the
`accompaniment` mixes add backing music and are skipped. ENST has no official
split, so we hold out a whole drummer/kit (drummer 3 by default) for a
genuinely song- and kit-disjoint eval.

Label vocabulary: ENST uses its own abbreviations (bd/sd/chh/ohh/rc/cr/...)
with numbered and rim variants (rc1, c3, ltr, sd-). We normalise (strip
trailing digits/punctuation) and fold to our 11 lanes. NOTE: ENST has
per-drummer label quirks and this table is assembled from the paper + secondary
sources -- VERIFY it against the real annotation files when the data lands
(unmapped labels, incl. the `sticks` count-in, drop to None like STAR's
out-of-kit classes).
"""
from __future__ import annotations

from collections.abc import Iterable
from dataclasses import dataclass
from pathlib import Path

from drumjot_training.lanes import LANES

# Normalised ENST label -> our lane (see `_normalize`: rc1->rc, c3->c, sd- ->sd).
_ENST_TO_LANE: dict[str, str] = {
    "bd": "k",                                    # bass drum
    "sd": "s", "rs": "s",                         # snare; rim shot = full snare hit
    "cs": "ss",                                   # cross stick = side stick
    "chh": "hc", "hh": "hc",                      # closed hi-hat (bare hh -> closed)
    "phh": "hp",                                  # pedal hi-hat (rare in ENST)
    "ohh": "ho",                                  # open hi-hat
    "lt": "t", "mt": "t", "lmt": "t", "lft": "t",  # toms
    "ltr": "t", "mtr": "t", "lftr": "t",          # tom rim hits -> tom
    "rc": "rd",                                   # ride cymbal
    "cr": "cr", "c": "cr",                         # crash; bare/other cymbal -> crash
    "ch": "mc", "spl": "mc", "rb": "mc",           # china / splash / ride-bell -> misc cymbal
    "cb": "mp",                                    # cowbell -> misc perc
}

_TRAIL = "0123456789-_ "


def _normalize(label: str) -> str:
    """Reduce a raw ENST label to the base token in `_ENST_TO_LANE` by stripping
    trailing variant markers (numbers / rim `-` / spaces): rc1->rc, c3->c."""
    return label.strip().lower().rstrip(_TRAIL)


def lane_for_enst_class(label: str) -> str | None:
    """Fold an ENST label to our lane, or None if out-of-kit / a count-in."""
    return _ENST_TO_LANE.get(_normalize(label))


def onsets_by_lane(annotation_path: str | Path) -> dict[str, list[float]]:
    """Parse an ENST `.txt` annotation into per-lane onset times (seconds).

    Lines are whitespace-separated `<time> <label>`; out-of-kit labels (and
    malformed lines) are dropped. Always returns all 11 lanes (empty lists for
    absent ones), each sorted ascending."""
    out: dict[str, list[float]] = {lane: [] for lane in LANES}
    with open(annotation_path) as f:
        for line in f:
            parts = line.split()
            if len(parts) < 2:
                continue
            lane = lane_for_enst_class(parts[1])
            if lane is None:
                continue
            try:
                out[lane].append(float(parts[0]))
            except ValueError:
                continue  # header / non-numeric time
    for ts in out.values():
        ts.sort()
    return out


_DRUMMER_DIRS = ("drummer_1", "drummer_2", "drummer_3")
DEFAULT_VAL_DRUMMER = "drummer_3"


@dataclass(frozen=True)
class EnstClip:
    audio_path: Path
    annotation_path: Path
    drummer: str


def _drummer_of(path: Path) -> str:
    for p in path.parts:
        if p.lower() in _DRUMMER_DIRS:
            return p.lower()
    return "unknown"


def index(root: str | Path, mix: str = "wet_mix") -> list[EnstClip]:
    """Pair every `drummer_*/annotation/<take>.txt` with `audio/<mix>/<take>.wav`.

    `mix`: "wet_mix" (default; realistic isolated kit), "dry_mix" (close mics
    only), or "accompaniment" (drums + backing -- not for drum-only training).
    Annotations with no matching audio file are skipped."""
    root = Path(root)
    clips: list[EnstClip] = []
    for ann in sorted(root.rglob("annotation/*.txt")):
        audio = ann.parent.parent / "audio" / mix / f"{ann.stem}.wav"
        if audio.exists():
            clips.append(EnstClip(audio_path=audio, annotation_path=ann, drummer=_drummer_of(ann)))
    return clips


def for_split(
    clips: Iterable[EnstClip], split: str, val_drummer: str = DEFAULT_VAL_DRUMMER
) -> list[EnstClip]:
    """Drummer-held-out split (ENST has no official one): "train" = every drummer
    except `val_drummer`; "validation"/"test" = `val_drummer` only, so eval is on
    a genuinely unseen kit + player."""
    if split == "train":
        return [c for c in clips if c.drummer != val_drummer]
    return [c for c in clips if c.drummer == val_drummer]
