# Beat Transformer checkpoint

Drop a `fold_N_trf_param.pt` from
<https://github.com/zhaojw1998/Beat-Transformer/tree/main/checkpoint>
here, renamed to `beat_transformer.pt`. The `docker/Dockerfile` copies
this directory into the image at build time, and
`pipeline/beat_transformer.py` loads `/app/checkpoints/beat_transformer.pt`
at first inference.

Override the path via the `BEAT_TRANSFORMER_CHECKPOINT` env var if you
want to point at a different file.

To enable BT instead of madmom, set `BEAT_TRACKER=beat_transformer` in
`docker/docker-compose.*.yml` (or the `.env` file).
