# synth.py
"""
Synthetic vowel generator for TCN formant estimator training data.

Generates voiced vowel-like signals using pluggable backends:

  "klatt"          — cascade of 2nd-order resonators (source-filter)
  "kelly_lochbaum" — Kelly-Lochbaum waveguide with source-filter coupling

The generator covers the full physiologically plausible F1/F2 space uniformly
rather than targeting any particular dialect or speaker group.  This makes it
a good complement to real-data fine-tuning: broad coverage + exact ground truth.

Ground-truth parameterisation
------------------------------
Formants are parameterised as normalised values relative to delta_F = c/(4L),
so fn_norm[n] = Fn / delta_F ≈ (2n-1) for a neutral vowel.  The same
delta_F convention as labeller.py is used:
    delta_F = SPEED_OF_SOUND_MM_S / (4 * vtl_mm)
    Fn_uniform = (2n-1) * delta_F  →  fn_norm = Fn / delta_F = 2n-1

Usage
-----
    from synth import SynthConfig, generate_vowel, generate_dataset

    cfg    = SynthConfig(sr=16000)
    audio, meta = generate_vowel(cfg)           # one random vowel
    dataset     = generate_dataset(cfg, n=5000) # list of (audio, meta) pairs
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional

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
    [10.0, 15.0],   # F6
    [12.0, 17.0],   # F7
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

_VALID_BACKENDS = ("klatt", "kelly_lochbaum")


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
    backend:        str   = "klatt"            # "klatt" | "kelly_lochbaum"
    backend_cfg:    Optional[Any] = None       # KLConfig | None

    def __post_init__(self):
        if self.n_formants > len(FN_NORM_BOUNDS):
            raise ValueError(
                f"n_formants={self.n_formants} exceeds "
                f"available bounds ({len(FN_NORM_BOUNDS)})"
            )
        if not (0.0 < self.amplitude <= 1.0):
            raise ValueError(f"amplitude={self.amplitude} must be in (0, 1]")
        if self.backend not in _VALID_BACKENDS:
            raise ValueError(
                f"backend={self.backend!r} must be one of {_VALID_BACKENDS}"
            )


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

@dataclass
class VowelMeta:
    """Ground-truth metadata for a synthetic vowel clip."""
    sr:              int
    f0_hz:           float
    vtl_mm:          float
    delta_f_hz:      float             # c/(4L)
    formant_hz:      np.ndarray        # (n_formants,) float32
    formant_bw_hz:   np.ndarray        # (n_formants,) float32
    fn_norm:         np.ndarray        # formant_hz / delta_f_hz
    bn_norm:         np.ndarray        # formant_bw_hz / delta_f_hz
    duration_s:      float
    # Optional KL-specific metadata
    register:        Optional[str]   = None
    effort:          Optional[str]   = None
    phonation_type:  Optional[str]   = None
    glottal_gap_frac: Optional[float] = None
    spectral_slope:  Optional[float] = None
    loudness_dbfs:   Optional[float] = None


def _delta_f(vtl_mm: float) -> float:
    """ΔF = c/(4L) — the uniform tube half-spacing (Hz)."""
    return SPEED_OF_SOUND_MM_S / (4.0 * vtl_mm)


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

    # --- KL backend handles its own parameter sampling ---
    if cfg.backend == "kelly_lochbaum":
        from synth.kl import generate_vowel_kl
        return generate_vowel_kl(cfg, rng)

    # --- Klatt path: sample F0, VTL, formants, bandwidths, then delegate ---
    f0_hz  = cfg.f0_hz  if cfg.f0_hz  is not None else float(rng.uniform(F0_MIN_HZ, F0_MAX_HZ))
    vtl_mm = cfg.vtl_mm if cfg.vtl_mm is not None else float(rng.uniform(VTL_MIN_MM, VTL_MAX_MM))
    df     = _delta_f(vtl_mm)

    # Sample normalised formants
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

    from synth.klatt import _formant_hz_from_norm, _bandwidth_hz

    formant_hz = _formant_hz_from_norm(fn_norm, vtl_mm)
    print(formant_hz)

    # Sample bandwidths
    if cfg.bw_hz is not None:
        bw_hz = np.asarray(cfg.bw_hz, dtype=np.float32)
    else:
        bw_hz = _bandwidth_hz(formant_hz, rng)

    from synth.klatt import generate_vowel_klatt
    return generate_vowel_klatt(cfg, rng, f0_hz, vtl_mm, df,
                                fn_norm, formant_hz, bw_hz)


def save_vowel_wav(
    path: str,
    cfg: SynthConfig,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, "VowelMeta"]:
    """
    Generate a synthetic vowel and save it as a WAV file.

    Parameters
    ----------
    path : destination file path (e.g. "out/vowel.wav")
    cfg  : SynthConfig
    rng  : optional RNG (seeded from cfg.rng_seed if omitted)

    Returns
    -------
    audio, meta — same as generate_vowel
    """
    import os
    import wave
    audio, meta = generate_vowel(cfg, rng=rng)
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(cfg.sr)
        wf.writeframes(pcm.tobytes())
    return audio, meta

def save_glottal_source_wav(
    path: str,
    cfg: SynthConfig,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, "VowelMeta"]:
    """
    Generate a synthetic glottal source waveform and save it as a WAV file.

    Parameters
    ----------
    path : destination file path (e.g. "out/vowel.wav")
    cfg  : SynthConfig
    rng  : optional RNG (seeded from cfg.rng_seed if omitted)

    Returns
    -------
    audio — raw glottal source waveform
    """
    import os
    import wave
    from synth.klatt import _glottal_source
    n_samples = int(round(cfg.sr * cfg.duration_s))
    audio = _glottal_source(n_samples, cfg.sr, cfg.f0_hz, cfg.jitter_frac if cfg.jitter_frac is not None else 0, cfg.shimmer_frac if cfg.shimmer_frac is not None else 0, np.random.default_rng())
    os.makedirs(os.path.dirname(os.path.abspath(path)), exist_ok=True)
    pcm = (audio * 32767).clip(-32768, 32767).astype(np.int16)
    with wave.open(path, "w") as wf:
        wf.setnchannels(1)
        wf.setsampwidth(2)
        wf.setframerate(cfg.sr)
        wf.writeframes(pcm.tobytes())
    return audio


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
        backend      = cfg.backend,
        backend_cfg  = cfg.backend_cfg,
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
