"""
Acoustic feature extraction helpers: F0, loudness, periodicity, spectral tilt.
"""

from __future__ import annotations

import warnings

import numpy as np
import parselmouth

from .config import LabellerConfig


def _loudness_dbfs(frame: np.ndarray) -> float:
    rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
    return 20.0 * np.log10(rms) if rms >= 1e-10 else -np.inf


def _periodicity(frame: np.ndarray, sr: int, f0_hz: float) -> float:
    if not np.isfinite(f0_hz) or f0_hz <= 0:
        return np.nan
    lag = int(round(sr / f0_hz))
    if lag <= 0 or lag >= len(frame):
        return np.nan
    n     = len(frame) - lag
    denom = np.sqrt(np.sum(frame[:n] ** 2) * np.sum(frame[lag:lag + n] ** 2))
    return float(np.sum(frame[:n] * frame[lag:lag + n]) / denom) if denom >= 1e-12 else np.nan


def _praat_f0(snd: parselmouth.Sound, cfg: LabellerConfig) -> float:
    try:
        pitch  = snd.to_pitch_ac(time_step=None,
                                  pitch_floor=cfg.min_f0_hz,
                                  pitch_ceiling=cfg.max_f0_hz)
        voiced = pitch.selected_array["frequency"]
        voiced = voiced[voiced > 0]
        return float(np.median(voiced)) if len(voiced) else np.nan
    except Exception:
        return np.nan


def _praat_f0_all(audio: np.ndarray, sr: int, cfg: LabellerConfig,
                  t_centres: np.ndarray) -> np.ndarray:
    """Run Praat pitch once on the whole sample, return F0 at each frame centre."""
    out = np.full(len(t_centres), np.nan, dtype=np.float32)
    try:
        snd = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
        pitch = snd.to_pitch_ac(time_step=None,
                                pitch_floor=cfg.min_f0_hz,
                                pitch_ceiling=cfg.max_f0_hz)
        for i, t in enumerate(t_centres):
            f0 = pitch.get_value_at_time(float(t))
            try:
                f0 = float(f0)
                if np.isfinite(f0) and f0 > 0:
                    out[i] = f0
            except (TypeError, ValueError):
                pass
    except Exception:
        pass
    return out


def _slope_to_alpha(slope: float, sr: int) -> float:
    """Convert spectral slope (dB/oct) to matched pre-emphasis alpha."""
    target       = -slope
    desired_diff = target * np.log2(4000.0 / 500.0)
    alphas       = np.linspace(0.70, 0.99, 300)
    w12          = 2.0 * np.pi * np.array([500.0, 4000.0]) / sr
    # Vectorised: compute H(w) for all alphas at once
    # alphas[:, None] * exp(-j*w12)[None, :] → (300, 2)
    H    = np.abs(1.0 - alphas[:, None] * np.exp(-1j * w12[None, :]))
    diff = 20.0 * np.log10(H[:, 1] / H[:, 0])
    return float(alphas[np.argmin(np.abs(diff - desired_diff))])


def _estimate_spectral_tilt_alpha(frame: np.ndarray, sr: int) -> tuple[float, float]:
    """
    Estimate spectral tilt (dB/octave) and matched first-order pre-emphasis
    alpha by fitting log-magnitude vs log2-frequency in 300-4000 Hz.
    Returns (slope, alpha); both NaN on failure.
    """
    try:
        w     = np.hanning(len(frame))
        mag   = np.abs(np.fft.rfft(frame * w)) + 1e-12
        freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)
        mask  = (freqs >= 300) & (freqs <= 4000)
        if mask.sum() < 4:
            return np.nan, np.nan

        x     = np.log2(freqs[mask])
        y     = 20.0 * np.log10(mag[mask])
        # Closed-form slope: cov(x,y) / var(x)
        x_mean = x.mean()
        slope  = float(np.sum((x - x_mean) * (y - y.mean())) /
                        np.sum((x - x_mean) ** 2))
        alpha  = _slope_to_alpha(slope, sr)
        return slope, alpha
    except Exception:
        return np.nan, np.nan


def _apply_preemphasis(x: np.ndarray, alpha: float) -> np.ndarray:
    y     = np.empty_like(x)
    y[0]  = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y
