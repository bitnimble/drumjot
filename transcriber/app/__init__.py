"""Drumjot transcriber backend.

Pipeline:
    audio bytes
        -> Demucs v4 (htdemucs_ft)            (full mix -> drum stem)
        -> Jarredou MDX23C 6-stem DrumSep      (drum stem -> per-instrument stems)
        -> librosa peak picker per stem        (per-stem onset candidates)
        -> tempo + downbeat estimation         (madmom-style via librosa)
        -> grid quantizer                      (snap onsets to 1/16 by default)
        -> LLM (Claude)                        (candidates + grid -> Drumjot DSL)
        -> Drumjot DSL string returned to client.
"""
