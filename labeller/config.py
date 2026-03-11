"""
Configuration constants and dataclasses for the labeller package.
"""

from __future__ import annotations

from dataclasses import dataclass


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEED_OF_SOUND_MM_S = 350_000.0   # mm/s — warm, humid vocal tract interior

# VTL size classes — used for smoothing prior only, decoupled from demographics.
# Boundaries (mm):  small <135 | medium 135–162 | large >162
# Based on Fitch & Giedd (1999) / Story (2005); roughly child / women / men.
VALID_VTL_CLASS: frozenset = frozenset({"small", "medium", "large", "unknown"})

VTL_PRIOR_MM: dict[str, float] = {
    "small":   128.0,   # representative: children
    "medium":  148.0,   # representative: adult women
    "large":   174.0,   # representative: adult men
    "unknown": 148.0,   # conservative fallback
}

# Age labels — used in SpeakerMeta; coerced from numeric age at construction.
# Thresholds: <12 child | 12–15 teen | >=16 adult
VALID_AGE: frozenset = frozenset({"adult", "teen", "child", "unknown"})
_AGE_CHILD_MAX  = 12
_AGE_TEEN_MAX   = 16

# Modality labels
VALID_MODALITY:  frozenset = frozenset({"spoken", "sung"})
VALID_DIRECTION: frozenset = frozenset({"exhale", "inhale"})

FORMANT_MATCH_TOLERANCE = 0.45   # +/- fraction of delta-F for pole assignment

N_FORMANTS = 7


# ---------------------------------------------------------------------------
# Configuration dataclasses
# ---------------------------------------------------------------------------

@dataclass
class LabellerConfig:
    win_ms:                 float = 10.0
    hop_ms:                 float = 5.0
    min_f0_hz:              float = 50.0
    max_f0_hz:              float = 600.0
    max_formant_hz:         float = 5500.0
    n_praat_formants:       int   = N_FORMANTS  # upper bound; Nyquist-clipped per window
    vtl_prior_strength:     float = 10.0    # n0 for uncertainty-weighted blend
    vtl_sample_alpha:       float = 0.30    # fixed sample-level blend fraction [0, 0.9]
    voicing_threshold_dbfs: float = -40.0
    f0_agreement_threshold: float = 0.10
    n_jobs:                 int   = -1      # -1 = all CPUs, 1 = serial
    adaptive_preemphasis:   bool  = True
    intensity_threshold_db: float = 40.0
    preemphasis_from_hz:    float = 50.0    # used only when adaptive_preemphasis=False

    def __post_init__(self):
        if not (0.0 <= self.vtl_sample_alpha <= 0.9):
            raise ValueError(
                f"vtl_sample_alpha={self.vtl_sample_alpha} is outside [0.0, 0.9]. "
                f"Values above 0.9 risk cascading formant misassignment from "
                f"degenerate per-window VTL estimates."
            )


@dataclass
class OutlierConfig:
    # Per-group F1 bounds (Hz).  Groups not listed fall back to the
    # _default entry.  Ceilings are set generously to cover the open
    # vowels (/a/, /ɑ/) while still catching tracker misfires.
    f1_bounds_hz: dict = None

    def __post_init__(self):
        if self.f1_bounds_hz is None:
            self.f1_bounds_hz = {
                "large":   (200.0,  950.0),
                "medium":  (200.0, 1100.0),
                "small":   (200.0, 1250.0),
                "unknown": (200.0, 1250.0),
            }

    # VTL: flag windows outside speaker_mean +/- vtl_std_threshold * speaker_std
    vtl_std_threshold: float = 3.0

    # Formant gap: flag windows where any adjacent assigned formant pair
    # exceeds gap_threshold * expected_delta_F (suggests a missed formant
    # slipped through the assignment step)
    formant_gap_threshold: float = 2.2

    # Flag unvoiced windows (NaN f0) — not "bad" data but often useful to
    # mask separately at training time
    flag_unvoiced: bool = True
