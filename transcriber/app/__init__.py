"""Drumjot transcriber backend.

Pipeline (beat-aware, no fixed grid):
    audio bytes
        -> Demucs v4 (htdemucs_ft)            (full mix -> drum stem)
        -> Jarredou MDX23C 6-stem DrumSep     (drum stem -> per-instrument stems)
        -> librosa peak picker per stem       (per-stem onset candidates)
        -> madmom RNN+DBN downbeat tracker    (per-beat anchors, downbeats,
                                               per-bar time signature + feel)
        -> attach (bar, beat_in_bar) positions to each onset
        -> LLM (Claude) with per-bar listings (-> Drumjot DSL)
        -> optional multi-level refinement loop (score-gated revisions)
        -> Drumjot DSL string returned to client.

The grid quantizer that used to live between onset detection and the LLM
was removed when the pipeline went beat-aware: triplets, tempo changes
and time-signature changes are now first-class because every onset
carries a beat-relative position rather than a fixed-grid slot index.
"""
