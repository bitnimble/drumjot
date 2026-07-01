"""Torch-free STFT / iSTFT in numpy, bit-compatible with `torch.stft` /
`torch.istft` for the separators' settings (periodic Hann, `center=True`,
`normalized=False`, one-sided).

This is the building block for a torch-free ONNX inference path: the model
bodies run via onnxruntime and the STFT prep/post run here, so inference needs
no torch at all (torch is used only at build time to export the `.onnx`).

Forward transform is exact vs torch.stft to fp32 rounding; the inverse uses the
same weighted-overlap-add + window-envelope normalization torch.istft does, so
a round trip reconstructs the interior to fp32 rounding. Edge samples within
`n_fft` of the boundary match torch's normalization; the models call istft with
no explicit `length`, taking the natural length both implementations agree on.
"""

from __future__ import annotations

import numpy as np


def hann_window(n: int) -> np.ndarray:
    """`torch.hann_window(n, periodic=True)` in fp32."""
    return (0.5 - 0.5 * np.cos(2.0 * np.pi * np.arange(n) / n)).astype(np.float32)


def stft(x: np.ndarray, n_fft: int, hop: int, window: np.ndarray, center: bool = True) -> np.ndarray:
    """`(b, t)` real -> `(b, n_fft//2+1, frames)` complex64. Matches torch.stft."""
    if center:
        p = n_fft // 2
        x = np.pad(x, ((0, 0), (p, p)), mode="reflect")
    n_frames = 1 + (x.shape[1] - n_fft) // hop
    idx = np.arange(n_fft)[None, :] + hop * np.arange(n_frames)[:, None]
    frames = x[:, idx].astype(np.float32) * window[None, None, :]
    spec = np.fft.rfft(frames, n=n_fft, axis=-1).astype(np.complex64)
    return np.transpose(spec, (0, 2, 1))


def istft(
    spec: np.ndarray,
    n_fft: int,
    hop: int,
    window: np.ndarray,
    center: bool = True,
    length: int | None = None,
) -> np.ndarray:
    """`(b, n_fft//2+1, frames)` complex -> `(b, samples)` real fp32. Matches
    torch.istft (weighted overlap-add / window-envelope normalization)."""
    n_frames = spec.shape[-1]
    b = spec.shape[0]
    spec = np.transpose(spec, (0, 2, 1))
    frames = np.fft.irfft(spec, n=n_fft, axis=-1).astype(np.float32) * window[None, None, :]
    out_len = n_fft + hop * (n_frames - 1)
    w2 = (window**2).astype(np.float32)
    if n_fft % hop == 0:
        # Vectorized overlap-add: split each n_fft frame into K = n_fft//hop
        # hop-blocks; block j of frame i lands in output block i+j, so the whole
        # add is a K-iteration column sweep instead of an n_frames Python loop
        # (K is ~4-16; n_frames is 512-1101). Same values, only the add order
        # differs (fp32-negligible; still bit-compatible with torch to rounding).
        k = n_fft // hop
        n_blocks = n_frames + k - 1  # == out_len // hop
        fb = frames.reshape(b, n_frames, k, hop)
        yb = np.zeros((b, n_blocks, hop), np.float32)
        eb = np.zeros((n_blocks, hop), np.float32)
        w2b = w2.reshape(k, hop)
        for j in range(k):
            yb[:, j : j + n_frames] += fb[:, :, j]
            eb[j : j + n_frames] += w2b[j]
        y = yb.reshape(b, out_len)
        env = eb.reshape(out_len)
    else:
        y = np.zeros((b, out_len), dtype=np.float32)
        env = np.zeros(out_len, dtype=np.float32)
        for i in range(n_frames):
            s = i * hop
            y[:, s : s + n_fft] += frames[:, i, :]
            env[s : s + n_fft] += w2
    env = np.where(env > 1e-11, env, 1.0)
    y = y / env[None, :]
    if center:
        p = n_fft // 2
        y = y[:, p:-p]
    if length is not None:
        y = y[:, :length] if y.shape[1] >= length else np.pad(y, ((0, 0), (0, length - y.shape[1])))
    return y
