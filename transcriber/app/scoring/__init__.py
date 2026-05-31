"""Drum-chart alignment scoring.

Scores an external drum chart (a ParaDB `.rlrr` map or a MIDI file) against
the real drum audio's detected onsets, producing a 0-100 quality number for
corpus filtering. See
`docs/superpowers/specs/2026-05-30-midi-scoring-utility-design.md` and the
algorithm reference `research/midi-audio-alignment-score.md`.
"""
