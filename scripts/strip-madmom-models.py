"""Strip madmom's CC-BY-NC-SA pretrained model weights (madmom/models/**/*.pkl)
from a built wheel.

Drumjot tracks beats with Beat-Transformer and never loads madmom's RNN models,
so shipping ~25 MB of non-commercial weights is dead weight (and would block a
commercial build). The tiny madmom/models/{__init__.py,LICENSE,README.rst} are
kept so `import madmom.models` still resolves. RECORD lines for the removed files
are dropped so the wheel stays valid.

Usage: python strip-madmom-models.py <wheel.whl>
"""
import sys
import zipfile
from pathlib import Path


def _is_weight(name: str) -> bool:
    return name.startswith("madmom/models/") and name.endswith(".pkl")


def main(wheel: str) -> int:
    src = Path(wheel)
    tmp = src.with_name(src.name + ".stripped")
    removed = 0
    with zipfile.ZipFile(src) as zin:
        record = next((n for n in zin.namelist() if n.endswith(".dist-info/RECORD")), None)
        if record is None:
            # Removing files without fixing RECORD would leave dangling hash
            # entries (an invalid wheel); fail loud instead.
            print(f"strip-madmom-models: no RECORD in {src.name}; refusing to strip", file=sys.stderr)
            return 1
        with zipfile.ZipFile(tmp, "w", zipfile.ZIP_DEFLATED) as zout:
            for info in zin.infolist():
                if _is_weight(info.filename):
                    removed += 1
                    continue
                data = zin.read(info.filename)
                if info.filename == record:
                    kept = [ln for ln in data.decode().splitlines() if not _is_weight(ln.split(",")[0])]
                    data = ("\n".join(kept) + "\n").encode()
                zout.writestr(info, data)
    tmp.replace(src)
    print(f"strip-madmom-models: removed {removed} weight file(s) from {src.name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1]))
