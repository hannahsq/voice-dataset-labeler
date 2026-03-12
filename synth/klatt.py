# synth/klatt.py
"""
Klatt-like cascade resonator synthesis backend.

Extracted from synth.py — implements the source-filter model:
  Source : band-limited impulse train (raised-cosine glottal pulse)
  Filter : cascade of 2nd-order all-pole resonators, one per formant
"""

from __future__ import annotations

import warnings

import numpy as np

from synth.synth import (
    SynthConfig, VowelMeta,
    SPEED_OF_SOUND_MM_S, BW_BASE_HZ, BW_SLOPE, BW_NOISE_HZ,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _delta_f(vtl_mm: float) -> float:
    """ΔF = c/(4L) — the uniform tube half-spacing (Hz)."""
    return SPEED_OF_SOUND_MM_S / (4.0 * vtl_mm)


def _formant_hz_from_norm(fn_norm: np.ndarray, vtl_mm: float) -> np.ndarray:
    """Convert normalised formant values to Hz."""
    return (fn_norm * _delta_f(vtl_mm)).astype(np.float32)


def _bandwidth_hz(formant_hz: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    """
    Generate bandwidths from a simple empirical model with additive noise.
    Always returns strictly positive values.
    """
    bw = BW_BASE_HZ + BW_SLOPE * formant_hz
    bw = bw + rng.normal(0.0, BW_NOISE_HZ, size=bw.shape).astype(np.float32)
    return np.clip(bw, 20.0, None).astype(np.float32)


def _resonator_coeffs(
    f_hz: float,
    bw_hz: float,
    sr: int,
) -> tuple[float, float, float]:
    """
    Compute IIR coefficients for a 2nd-order resonator (all-pole bandpass).

    Transfer function:  H(z) = 1 / (1 - 2R*cos(θ)*z⁻¹ + R²*z⁻²)
    where θ = 2π*f/sr,  R = exp(-π*bw/sr).

    Returns (b0, a1, a2) for  y[n] = b0*x[n] + a1*y[n-1] + a2*y[n-2]
    with b0 chosen so DC gain = 1 (normalised).
    """
    theta = 2.0 * np.pi * f_hz / sr
    R     = np.exp(-np.pi * bw_hz / sr)
    a1    =  2.0 * R * np.cos(theta)
    a2    = -R * R
    # DC gain = 1 / (1 - a1 - a2); normalise so resonator has unit DC gain
    dc    = 1.0 - a1 - a2
    b0    = dc if abs(dc) > 1e-12 else 1.0
    return b0, a1, a2


def _apply_resonator(
    x: np.ndarray,
    f_hz: float,
    bw_hz: float,
    sr: int,
) -> np.ndarray:
    """Apply a single 2nd-order resonator in-place via direct-form II."""
    b0, a1, a2 = _resonator_coeffs(f_hz, bw_hz, sr)
    y   = np.zeros_like(x)
    y1  = 0.0
    y2  = 0.0
    for i, xn in enumerate(x):
        yn   = b0 * xn + a1 * y1 + a2 * y2
        y[i] = yn
        y2   = y1
        y1   = yn
    return y


def _glottal_source(
    n_samples:    int,
    sr:           int,
    f0_hz:        float,
    jitter_frac:  float,
    shimmer_frac: float,
    rng:          np.random.Generator,
) -> np.ndarray:
    """
    Generate a band-limited impulse train approximating a glottal pulse train.

    Each period is a raised-cosine pulse (smoother than a Dirac, avoids
    aliasing at high F0) with per-period jitter (timing) and shimmer
    (amplitude).
    """
    x        = np.zeros(n_samples, dtype=np.float64)
    period   = sr / f0_hz
    t        = 0.0
    amp      = 1.0

    # Width of raised-cosine pulse: 1/3 of period, minimum 3 samples
    pulse_width = max(3, int(round(period / 3.0)))

    half = pulse_width // 2
    pulse_template = 0.5 * (1.0 - np.cos(
        2.0 * np.pi * np.arange(pulse_width) / (pulse_width - 1)
    ))

    while t < n_samples:
        pos = int(round(t))
        if pos >= n_samples:
            break

        # Write pulse centred at pos, clipped to array bounds
        start = pos - half
        end   = start + pulse_width
        a0    = max(0, -start)
        a1    = pulse_width - max(0, end - n_samples)
        s0    = max(0, start)
        s1    = s0 + (a1 - a0)
        x[s0:s1] += amp * pulse_template[a0:a1]

        # Advance by one period with jitter
        jitter = rng.normal(0.0, jitter_frac * period)
        t     += period + jitter

        # Shimmer
        amp *= 1.0 + rng.normal(0.0, shimmer_frac)
        amp  = np.clip(amp, 0.2, 2.0)

    return x.astype(np.float32)


# ---------------------------------------------------------------------------
# Klatt synthesis entry point
# ---------------------------------------------------------------------------

def generate_vowel_klatt(
    cfg: SynthConfig,
    rng: np.random.Generator,
    f0_hz: float,
    vtl_mm: float,
    df: float,
    fn_norm: np.ndarray,
    formant_hz: np.ndarray,
    bw_hz: np.ndarray,
) -> tuple[np.ndarray, VowelMeta]:
    """
    Synthesise a vowel using the Klatt-like cascade resonator model.

    Parameters are pre-sampled by the caller (generate_vowel in synth.py).
    """
    bn_norm = (bw_hz / df).astype(np.float32)

    n_samples = int(round(cfg.sr * cfg.duration_s))
    source    = _glottal_source(n_samples, cfg.sr, f0_hz,
                                cfg.jitter_frac, cfg.shimmer_frac, rng)

    # Cascade resonators (source -> F1 -> F2 -> ... -> FN)
    signal = source.copy()
    for i in range(cfg.n_formants):
        signal = _apply_resonator(signal, float(formant_hz[i]),
                                  float(bw_hz[i]), cfg.sr)

    # Normalise amplitude
    peak = np.max(np.abs(signal))
    if peak > 1e-9:
        signal = (signal / peak * cfg.amplitude).astype(np.float32)
    else:
        warnings.warn("Synthesised signal has near-zero amplitude; check parameters.")
        signal = signal.astype(np.float32)

    meta = VowelMeta(
        sr            = cfg.sr,
        f0_hz         = f0_hz,
        vtl_mm        = vtl_mm,
        delta_f_hz    = df,
        formant_hz    = formant_hz,
        formant_bw_hz = bw_hz,
        fn_norm       = fn_norm,
        bn_norm       = bn_norm,
        duration_s    = n_samples / cfg.sr,
    )

    return signal, meta
