"""Fetch + cache every model the training/eval code needs, ONCE, so the rest of
the pipeline runs fully offline.

`drumjot_training` forces Hugging Face offline by default (HF_HUB_OFFLINE), so the
trainer/probes never touch the network -- reproducible, and immune to the
unauthenticated-Hub rate-limiting that intermittently killed batch jobs. The cost
is that models must already be in the local cache; this script is the one place
that is allowed to download them. Run it once after setup (and again only if a
model id changes or the HF cache is wiped):

    python training/scripts/fetch_models.py

It is idempotent (already-cached models are a fast no-op), and it CRASHES on any
download failure -- so connectivity problems surface here, not mid-training.

NOTE: this covers the TRAINING stack (the frozen MERT encoder). The transcriber's
separator/aligner models are provisioned by their own mechanism (the container /
app.pipeline.provision); that subsystem is intentionally not touched here.
"""
import os

# MUST run before any transformers / huggingface_hub / drumjot_training import:
# re-enable the network for THIS script only (the package init otherwise pins
# offline, and these libs read the flags at import time). setdefault is wrong
# here -- we must override the package default.
os.environ["HF_HUB_OFFLINE"] = "0"
os.environ["TRANSFORMERS_OFFLINE"] = "0"

import subprocess  # noqa: E402
import sys  # noqa: E402

_HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(_HERE, ".."))  # training/

from drumjot_training import embeddings  # noqa: E402


def fetch_mert(name: str) -> None:
    """Download the MERT feature extractor + model + its trust_remote_code
    modules. Calling from_pretrained (not just snapshot_download) is deliberate:
    it also populates the transformers_modules cache the remote-code model needs
    to load offline later. CPU-only (no .to(device)) -- this is a cache warm, not
    a run."""
    from transformers import AutoModel, Wav2Vec2FeatureExtractor

    print(f"fetching {name}: feature extractor ...", flush=True)
    Wav2Vec2FeatureExtractor.from_pretrained(name, trust_remote_code=True)
    print(f"fetching {name}: model weights + remote code ...", flush=True)
    AutoModel.from_pretrained(name, trust_remote_code=True)
    print(f"  cached {name}", flush=True)


def _verify_offline(name: str) -> None:
    """Spawn a fresh OFFLINE process and load the feature extractor from cache,
    proving a real (offline) run will find it."""
    code = (
        "from transformers import Wav2Vec2FeatureExtractor as W; "
        f"W.from_pretrained({name!r}, trust_remote_code=True); print('ok')"
    )
    env = {**os.environ, "HF_HUB_OFFLINE": "1", "TRANSFORMERS_OFFLINE": "1"}
    r = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True)
    if r.returncode != 0 or "ok" not in r.stdout:
        raise RuntimeError(
            f"offline verification FAILED for {name} after fetch:\n{r.stderr[-600:]}"
        )
    print(f"  verified {name} loads offline", flush=True)


# Add other internet-fetched models here as the training stack grows (each must
# end up in a local cache the offline run can find).
FETCHERS = [(embeddings.MERT_NAME, fetch_mert)]


def main() -> None:
    for name, fetch in FETCHERS:
        fetch(name)
        _verify_offline(name)
    print("\nDONE. All training models cached; the trainer runs fully offline.", flush=True)


if __name__ == "__main__":
    main()
