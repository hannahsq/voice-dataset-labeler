# synth.py
"""
Synthetic vowel generator for TCN formant estimator training data.

Generates voiced vowel-like signals using a simple but physically grounded
source-filter model:

  Source : band-limited impulse train (glottal pulse approximation)
           with per-period jitter and shimmer for naturalness
  Filter : cascade of 2nd-order resonators, one per formant (all-pole)
  Output : float32 audio at a specified sample rate

The generator covers the full physiologically plausible F1/F2 space uniformly
rather than targeting any particular dialect or speaker group.  This makes it
a good complement to real-data fine-tuning: broad coverage + exact ground truth.

Ground-truth parameterisation
------------------------------
Formants are parameterised as normalised values relative to delta_F = c/(4L),
so fn_norm[n] = Fn / delta_F ≈ (2n-1) for a neutral vowel.  The generator
accepts either Hz or normalised values (see SynthParams).

The same delta_F convention as labeller.py is used:
    delta_F = SPEED_OF_SOUND_MM_S / (4 * vtl_mm)
    Fn_uniform = (2n-1) * delta_F  →  fn_norm = Fn / delta_F = 2n-1

Bandwidth model
---------------
Bandwidths are generated from a simple empirical model adapted from
Fant (1960) and Hawks & Miller (1995):
    Bn ≈ base_bw * (1 + Fn / bw_slope_hz)
with added per-formant noise.  Bandwidths are always positive and are
also returned normalised as bn_norm = Bn / delta_F.

Usage
-----
    from synth import SynthConfig, generate_vowel, generate_dataset

    cfg    = SynthConfig(sr=16000)
    audio, meta = generate_vowel(cfg)           # one random vowel
    dataset     = generate_dataset(cfg, n=5000) # list of (audio, meta) pairs
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Constants — keep in sync with labeller.py
# ---------------------------------------------------------------------------

SPEED_OF_SOUND_MM_S: float = 350_000.0   # mm/s, warm humid vocal tract

# Physiological VTL bounds (mm)
VTL_MIN_MM: float = 110.0   # small child
VTL_MAX_MM: float = 195.0   # large adult male

# Normalised formant bounds (Fn / delta_F).
# These define the full plausible vowel space rather than any dialect.
# Neutral tube: fn_norm = [1, 3, 5, 7, 9]
# Rows: [min, max] for each of F1-F5.
FN_NORM_BOUNDS: np.ndarray = np.array([
    [0.30, 2.20],   # F1  (low: /u/ schwa-range; high: /a/)
    [1.20, 5.50],   # F2  (front/back dimension)
    [3.50, 8.00],   # F3
    [5.50, 11.0],   # F4
    [8.00, 14.0],   # F5
], dtype=np.float32)

# Bandwidth model: Bn ≈ base_bw_hz + slope * Fn
# Adapted from Hawks & Miller (1995) Table II medians.
BW_BASE_HZ:  float = 50.0
BW_SLOPE:    float = 0.06     # dimensionless; 6% of Fn
BW_NOISE_HZ: float = 25.0    # std of additive Gaussian noise on bandwidth

# F0 bounds
F0_MIN_HZ: float =  60.0
F0_MAX_HZ: float = 900.0     # covers soprano singing

# Jitter / shimmer — mild defaults for naturalness without pathology
JITTER_FRAC:  float = 0.005   # period-to-period pitch variation (fraction)
SHIMMER_FRAC: float = 0.03    # period-to-period amplitude variation (fraction)


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class SynthConfig:
    sr:             int   = 16_000   # output sample rate (Hz)
    n_formants:     int   = 5        # must be <= len(FN_NORM_BOUNDS)
    duration_s:     float = 0.5      # clip duration
    f0_hz:          Optional[float] = None   # None = uniform random
    vtl_mm:         Optional[float] = None   # None = uniform random
    fn_norm:        Optional[np.ndarray] = None  # (n_formants,) override
    bw_hz:          Optional[np.ndarray] = None  # (n_formants,) bandwidth override
    jitter_frac:    float = JITTER_FRAC
    shimmer_frac:   float = SHIMMER_FRAC
    amplitude:      float = 0.5      # peak amplitude of output (0–1)
    rng_seed:       Optional[int] = None

    def __post_init__(self):
        if self.n_formants > len(FN_NORM_BOUNDS):
            raise ValueError(
                f"n_formants={self.n_formants} exceeds "
                f"available bounds ({len(FN_NORM_BOUNDS)})"
            )
        if not (0.0 < self.amplitude <= 1.0):
            raise ValueError(f"amplitude={self.amplitude} must be in (0, 1]")


# ---------------------------------------------------------------------------
# Synthesis helpers
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
) -> tuple[float, float, float, float, float]:
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
# Public API
# ---------------------------------------------------------------------------

@dataclass
class VowelMeta:
    """Ground-truth metadata for a synthetic vowel clip."""
    sr:           int
    f0_hz:        float
    vtl_mm:       float
    delta_f_hz:   float             # c/(4L)
    formant_hz:   np.ndarray        # (n_formants,) float32
    formant_bw_hz: np.ndarray       # (n_formants,) float32
    fn_norm:      np.ndarray        # formant_hz / delta_f_hz
    bn_norm:      np.ndarray        # formant_bw_hz / delta_f_hz
    duration_s:   float


def generate_vowel(
    cfg: SynthConfig,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, VowelMeta]:
    """
    Generate one synthetic vowel clip.

    Returns
    -------
    audio : (T,) float32 in [-amplitude, +amplitude]
    meta  : VowelMeta with exact ground-truth formant parameters
    """
    if rng is None:
        seed = cfg.rng_seed if cfg.rng_seed is not None else None
        rng  = np.random.default_rng(seed)

    # --- Sample F0 and VTL ---
    f0_hz  = cfg.f0_hz  if cfg.f0_hz  is not None else float(rng.uniform(F0_MIN_HZ, F0_MAX_HZ))
    vtl_mm = cfg.vtl_mm if cfg.vtl_mm is not None else float(rng.uniform(VTL_MIN_MM, VTL_MAX_MM))
    df     = _delta_f(vtl_mm)

    # --- Sample normalised formants ---
    if cfg.fn_norm is not None:
        fn_norm = np.asarray(cfg.fn_norm, dtype=np.float32)
    else:
        fn_norm = np.array([
            rng.uniform(float(FN_NORM_BOUNDS[i, 0]),
                        float(FN_NORM_BOUNDS[i, 1]))
            for i in range(cfg.n_formants)
        ], dtype=np.float32)

        # Enforce ordering: each formant must be above the previous
        for i in range(1, cfg.n_formants):
            min_val = fn_norm[i - 1] + 0.4   # minimum separation in norm units
            if fn_norm[i] < min_val:
                fn_norm[i] = float(rng.uniform(
                    min_val,
                    max(min_val + 0.1, float(FN_NORM_BOUNDS[i, 1]))
                ))

    formant_hz = _formant_hz_from_norm(fn_norm, vtl_mm)

    # --- Sample bandwidths ---
    if cfg.bw_hz is not None:
        bw_hz = np.asarray(cfg.bw_hz, dtype=np.float32)
    else:
        bw_hz = _bandwidth_hz(formant_hz, rng)

    bn_norm = (bw_hz / df).astype(np.float32)

    # --- Synthesise ---
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


def generate_dataset(
    cfg:  SynthConfig,
    n:    int = 5000,
    seed: Optional[int] = None,
) -> list[tuple[np.ndarray, VowelMeta]]:
    """
    Generate n synthetic vowel clips with independent random parameters.

    Parameters
    ----------
    cfg  : SynthConfig  (f0_hz / vtl_mm / fn_norm overrides are ignored;
                         all are sampled randomly per clip)
    n    : number of clips to generate
    seed : master RNG seed for reproducibility

    Returns
    -------
    List of (audio, meta) pairs.
    """
    # Override any fixed parameters so each clip is independently random
    free_cfg = SynthConfig(
        sr           = cfg.sr,
        n_formants   = cfg.n_formants,
        duration_s   = cfg.duration_s,
        jitter_frac  = cfg.jitter_frac,
        shimmer_frac = cfg.shimmer_frac,
        amplitude    = cfg.amplitude,
    )

    rng     = np.random.default_rng(seed)
    dataset = []
    for _ in range(n):
        audio, meta = generate_vowel(free_cfg, rng=rng)
        dataset.append((audio, meta))

    return dataset


# ---------------------------------------------------------------------------
# Quick smoke test
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    import time

    cfg = SynthConfig(sr=16_000, n_formants=5, duration_s=0.5)

    print("Generating single vowel...")
    audio, meta = generate_vowel(cfg, rng=np.random.default_rng(42))
    print(f"  audio shape : {audio.shape}  dtype={audio.dtype}")
    print(f"  f0          : {meta.f0_hz:.1f} Hz")
    print(f"  vtl         : {meta.vtl_mm:.1f} mm")
    print(f"  delta_F     : {meta.delta_f_hz:.1f} Hz")
    print(f"  formants Hz : {np.round(meta.formant_hz, 1)}")
    print(f"  bandwidths  : {np.round(meta.formant_bw_hz, 1)}")
    print(f"  fn_norm     : {np.round(meta.fn_norm, 3)}  (neutral ≈ [1,3,5,7,9])")
    print(f"  bn_norm     : {np.round(meta.bn_norm, 3)}")

    print("\nGenerating dataset of 200 clips...")
    t0 = time.time()
    ds = generate_dataset(cfg, n=200, seed=0)
    dt = time.time() - t0
    print(f"  Generated {len(ds)} clips in {dt:.2f}s ({1000*dt/len(ds):.1f} ms/clip)")

    f0s   = np.array([m.f0_hz      for _, m in ds])
    vtls  = np.array([m.vtl_mm     for _, m in ds])
    f1n   = np.array([m.fn_norm[0] for _, m in ds])
    f2n   = np.array([m.fn_norm[1] for _, m in ds])
    print(f"\n  F0  range : [{f0s.min():.0f}, {f0s.max():.0f}] Hz  "
          f"mean={f0s.mean():.0f}")
    print(f"  VTL range : [{vtls.min():.1f}, {vtls.max():.1f}] mm  "
          f"mean={vtls.mean():.1f}")
    print(f"  F1_norm   : [{f1n.min():.2f}, {f1n.max():.2f}]  "
          f"mean={f1n.mean():.2f}  (neutral=1.0)")
    print(f"  F2_norm   : [{f2n.min():.2f}, {f2n.max():.2f}]  "
          f"mean={f2n.mean():.2f}  (neutral=3.0)")
