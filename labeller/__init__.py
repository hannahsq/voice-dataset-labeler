"""
Acoustic feature labeller for vowel recordings.

Takes dataset dicts produced by dataset.py / datasets_hf.py and returns
one labelled dict per source sample, with all per-window measurements
stored as numpy arrays.  This is the native internal format — no
flat-list intermediate or post-hoc transposition step is involved.

Formant index correction
------------------------
Parselmouth's Burg tracker returns however many poles it finds in a frame
without regard for which anatomical resonance each corresponds to.  At high
F0 or in noisy frames a resonance can be missed entirely, causing all higher
formant indices to shift down by one.

This module corrects that via a Bayesian convergence-based pipeline:

  Seed pass:
    For each sample, extract formants at up to 20 intensity-gated positions
    and take the median VTL.  Seed the Bayesian estimator before the
    convergence loop begins.

  Convergence loop:
    1. Extract all windows using current speaker VTL posteriors for ceilings
    2. Fit per-sample GMM to separate signal from noise in VTL observations
    3. Update Normal-Normal conjugate posteriors (group → speaker)
    4. Check convergence; repeat until group μ stabilises

  Final pass:
    Extract with converged speaker VTL posterior means.  vtl_sample_mm in
    the output is the unsmoothed per-window value from these final assigned
    formants — this is what you train on.

Bayesian VTL estimation
-----------------------
    Group prior:   N(μ_literature, σ_prior²)
    Speaker post:  conjugate update from group prior + GMM observations
    Per-window:    vtl_sample_mm is unsmoothed (no blending)

Adaptive pre-emphasis
---------------------
When cfg.adaptive_preemphasis is True (default), spectral tilt is estimated
per frame (dB/oct) and a matched first-order alpha is applied before Praat
analysis.  Praat's own internal pre-emphasis is disabled.  The slope is
stored as spectral_slope.  Set adaptive_preemphasis=False to use a fixed
preemphasis_from_hz instead.

Speed of sound
--------------
350 m/s (warm, humid vocal tract interior).

Output schema  (one dict per source sample)
-------------------------------------------
    speaker              : str
    group                : str
    label                : str
    source               : str
    modality             : str
    dialect              : str
    audio                : (T,)    float32   original audio
    window_length_ms     : float
    hop_ms               : float

    t_centre_s           : (N,)    float32
    loudness_dbfs        : (N,)    float32
    spectral_slope       : (N,)    float32   dB/oct; NaN if adaptive_preemphasis=False
    f0_hz                : (N,)    float32   NaN where unvoiced
    periodicity          : (N,)    float32
    f0_praat_hz          : (N,)    float32
    formant_hz           : (N, 7)  float32   F1-F7 frequencies
    formant_bw_hz        : (N, 7)  float32   F1-F7 bandwidths
    vtl_sample_mm        : (N,)    float32   unsmoothed per-window VTL
    vtl_posterior_mu_mm   : float             speaker posterior mean VTL
    vtl_posterior_sigma_mm: float             speaker posterior σ
"""

# --- config ---
from .config import (
    SPEED_OF_SOUND_MM_S,
    VTL_PRIOR_MM,
    N_FORMANTS,
    FORMANT_MATCH_TOLERANCE,
    VALID_VTL_CLASS,
    VALID_AGE,
    VALID_MODALITY,
    VALID_DIRECTION,
    _AGE_CHILD_MAX,
    _AGE_TEEN_MAX,
    LabellerConfig,
    OutlierConfig,
)

# --- types ---
from .types import Modality, SpeakerMeta

# --- acoustic ---
from .acoustic import (
    _loudness_dbfs,
    _periodicity,
    _praat_f0,
    _estimate_spectral_tilt_alpha,
    _apply_preemphasis,
)

# --- formants ---
from .formants import (
    _extract_raw_formants,
    _vtl_from_formants,
    _assign_formant_indices,
)

# --- bayesian vtl ---
from .bayesian_vtl import (
    NormalPosterior,
    BayesianVTLEstimator,
    fit_vtl_mixture,
    fit_vtl_single,
)

# --- extractors ---
from extractors import FormantExtractor, PraatFormantExtractor, TVLPFormantExtractor

# --- pipeline ---
from .pipeline import (
    _extract_sample,
    _worker_extract,
    _worker_seed_vtl,
    _parallel_map,
    _resolve_workers,
    _make_worker_args,
    _estimator_from_labelled,
    label_dataset,
    label_incremental,
    probe_raw_formants,
    probe_full,
)

# --- postprocess ---
from .postprocess import (
    flag_outliers,
    build_metadata,
    save_dataset,
    load_dataset,
)
