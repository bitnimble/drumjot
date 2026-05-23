"""Drumjot transcriber backend.

Pipeline (beat-aware, no fixed grid):
    audio bytes
        -> BS-Roformer SW                       (full mix -> drum stem)
        -> Jarredou MDX23C 6-stem DrumSep       (drum stem -> per-instrument stems)
        -> ADTOF Frame_RNN per stem             (per-stem onset candidates)
        -> madmom RNN+DBN downbeat tracker      (per-beat anchors, downbeats,
                                                 per-bar time signature + feel)
        -> attach (bar, beat_in_bar) positions to each onset
        -> Claude filter (per instrument) → rejects artifact onsets
        -> render kept onsets to MIDI (prediction.mid)
        -> client receives the prediction MIDI URL + a per-note debug
           provenance sidecar; the frontend converts the MIDI to a
           Drumjot Jot via src/midi/from_midi.ts.

The legacy DSL-output pathway (LLM-emitted Drumjot DSL + F1-gated
refinement) and the librosa onset backend were removed in May 2026; see
docs/ai-midi-to-jot-notes.md for the techniques captured from them.
"""
