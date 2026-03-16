"""
Convert synth output (audio, VowelMeta) to the labeller's LabelledSample dict
format so that synthetic and real data can be used interchangeably.
"""

from __future__ import annotations

from typing import Optional

import numpy as np

from labeller.config import (
    LabellerConfig,
    N_FORMANTS,
    _window_centres,
    _window_params,
)
from .synth import VowelMeta


def _loudness_rms_dbfs(audio: np.ndarray) -> float:
    """RMS loudness in dBFS (full-scale = 1.0)."""
    rms = np.sqrt(np.mean(audio.astype(np.float64) ** 2))
    if rms <= 0:
        return -100.0
    return float(20.0 * np.log10(rms))


def synth_to_labelled(
    audio: np.ndarray,
    meta: VowelMeta,
    cfg: Optional[LabellerConfig] = None,
) -> dict:
    """Convert a single ``(audio, VowelMeta)`` pair to a labelled sample dict.

    Per-window arrays are created by broadcasting the (stationary) ground-truth
    values.  This lets synth output pass through ``flag_outliers``,
    ``build_metadata``, ``save_dataset`` etc. unchanged.
    """
    if cfg is None:
        cfg = LabellerConfig()

    sr = meta.sr
    n_samples = len(audio)

    # Window geometry — reuse the shared helper
    win_samples, hop_samples, effective_win_ms = _window_params(
        n_samples, sr, cfg.win_ms, cfg.hop_ms, cfg.min_f0_hz,
    )
    t_centres = _window_centres(
        n_samples, sr, cfg.win_ms, cfg.hop_ms, cfg.min_f0_hz,
    )
    W = len(t_centres)

    # Pad formants from (n,) to (N_FORMANTS,) with NaN
    n_orig = len(meta.formant_hz)
    pad = N_FORMANTS - n_orig
    if pad > 0:
        formant_hz = np.concatenate([meta.formant_hz,
                                     np.full(pad, np.nan, dtype=np.float32)])
        formant_bw_hz = np.concatenate([meta.formant_bw_hz,
                                        np.full(pad, np.nan, dtype=np.float32)])
    else:
        formant_hz = meta.formant_hz[:N_FORMANTS].copy()
        formant_bw_hz = meta.formant_bw_hz[:N_FORMANTS].copy()

    # Broadcast to (W, N_FORMANTS)
    formant_hz_2d = np.tile(formant_hz, (W, 1)).astype(np.float32)
    formant_bw_hz_2d = np.tile(formant_bw_hz, (W, 1)).astype(np.float32)

    # Scalar → (W,)
    def _broadcast(val: float) -> np.ndarray:
        return np.full(W, val, dtype=np.float32)

    loudness = meta.loudness_dbfs if meta.loudness_dbfs is not None else _loudness_rms_dbfs(audio)
    slope = meta.spectral_slope if meta.spectral_slope is not None else 0.0

    # Ground truth sub-dict
    ground_truth = {
        "vtl_mm":       meta.vtl_mm,
        "delta_f_hz":   meta.delta_f_hz,
        "f0_hz":        meta.f0_hz,
        "formant_hz":   meta.formant_hz.copy(),
        "formant_bw_hz": meta.formant_bw_hz.copy(),
        "fn_norm":      meta.fn_norm.copy(),
        "bn_norm":      meta.bn_norm.copy(),
    }
    # Include optional KL metadata if present
    for key in ("register", "effort", "phonation_type", "glottal_gap_frac"):
        val = getattr(meta, key, None)
        if val is not None:
            ground_truth[key] = val

    return {
        "speaker":          "synth",
        "speaker_meta":     None,
        "group":            "unknown",
        "label":            "",
        "source":           "synth",
        "modality":         "spoken",
        "dialect":          "unknown",

        "audio":            audio.astype(np.float32),
        "sr":               sr,
        "window_length_ms": effective_win_ms,
        "hop_ms":           cfg.hop_ms,

        "t_centre_s":       t_centres,
        "loudness_dbfs":    _broadcast(loudness),
        "spectral_slope":   _broadcast(slope),
        "f0_hz":            _broadcast(meta.f0_hz),
        "periodicity":      _broadcast(1.0),
        "f0_praat_hz":      _broadcast(meta.f0_hz),
        "f0_pyin_hz":       _broadcast(meta.f0_hz),

        "formant_hz":       formant_hz_2d,
        "formant_bw_hz":    formant_bw_hz_2d,
        "vtl_sample_mm":    _broadcast(meta.vtl_mm),

        "ground_truth":     ground_truth,
    }


def synth_dataset_to_labelled(
    dataset: list[tuple[np.ndarray, VowelMeta]],
    cfg: Optional[LabellerConfig] = None,
) -> list[dict]:
    """Convert a list of ``(audio, VowelMeta)`` pairs to labelled sample dicts."""
    return [synth_to_labelled(audio, meta, cfg) for audio, meta in dataset]
