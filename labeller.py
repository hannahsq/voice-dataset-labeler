# labeller.py
"""
Acoustic feature labeller for vowel recordings.

Takes dataset dicts produced by dataset.py / datasets_hf.py and returns
per-window acoustic labels including loudness, F0, periodicity, spectral
slope, and formant frequencies + bandwidths with physiologically-corrected
indices.

Formant index correction
------------------------
Parselmouth's Burg tracker returns however many poles it finds in a frame
without awareness of which anatomical resonance each pole corresponds to.
At high F0 or in noisy frames a resonance can be missed entirely, causing
all higher formant indices to shift down by one — a silent but serious error.

This module corrects for that via a two-pass hierarchical VTL estimation:

  Pass 0 (raw):
    Extract formants naively. Estimate per-sample VTL from raw delta-F.
    Blend: literature prior -> group mean -> speaker mean -> sample VTL.
    Use smoothed sample VTL + uniform-tube model to assign formant indices
    via minimum-cost matching (greedy over 7 formants).  -> firstpass_formants

  Pass 1 (refinement):
    Re-estimate group/speaker VTL means from firstpass_formants.
    Smooth again: literature prior -> group mean -> speaker mean.
    Re-assign formant indices using refined per-speaker prior.  -> formants

VTL blending
------------
At group and speaker level, blending weight is:
    alpha = n / (n + n0)
where n = samples seen for that entity, n0 = prior_strength (default 10).

At sample level a fixed configurable alpha (default 0.3) blends the
smoothed speaker VTL toward the raw sample VTL estimate:
    vtl_sample = (1 - alpha_s) * vtl_speaker_smooth + alpha_s * vtl_raw_sample

Adaptive pre-emphasis
---------------------
When cfg.adaptive_preemphasis is True (default), spectral tilt is estimated
for each frame by regressing log-magnitude against log-frequency in the
300-4000 Hz band. A matched first-order pre-emphasis coefficient is derived
and applied before Praat analysis, with Praat's own internal pre-emphasis
disabled (pre_emphasis_from set above Nyquist). The spectral slope
(dB/octave) is stored as a labelled feature. Set adaptive_preemphasis=False
to use a fixed pre_emphasis_from_hz instead.

Speed of sound
--------------
350 m/s is used -- appropriate for the warm (~35 degC), humid vocal tract
interior. 343 m/s (dry air at 20 degC) underestimates VTL by ~2%.

Voicing / unvoiced frames
--------------------------
A frame is considered unvoiced if its loudness is below
voicing_threshold_dbfs (default -40 dBFS). Unvoiced frames are kept in
the output -- F0 and periodicity are NaN, but formants and loudness are
still estimated. This preserves the full timeline for windowed local
standard deviation computation.

F0 estimation
-------------
Both Praat's autocorrelation pitch tracker and librosa PYIN are run.
If the two estimates agree within 10% their mean is returned.
Otherwise the modality-based preference wins: Praat for "spoken*",
PYIN for "sung*".

VTL seeding
-----------
The initial VTL estimate for each sample uses 20 intensity-gated time
positions spread across the full file (gated by cfg.intensity_threshold_db)
rather than a single centre-window, giving a more representative prior
before the hierarchical blending begins.

Output
------
label_dataset() returns a flat list of per-window dicts.
transpose_to_samples() converts this to one dict per original sample with
numpy arrays for each acoustic parameter, and attaches the original audio.
build_metadata() derives unsmoothed speaker/group VTL statistics.
save_dataset() / load_dataset() persist the full set.

Per-window dict schema
----------------------
    speaker          : str
    group            : str
    label            : str           IPA vowel label
    source           : str
    modality         : str
    dialect          : str
    window_length_ms : float
    t_centre_s       : float

    loudness_dbfs    : float
    spectral_slope   : float         dB/octave; NaN if adaptive_preemphasis=False
    f0_hz            : float | NaN   (NaN if unvoiced)
    periodicity      : float | NaN
    f0_praat_hz      : float | NaN
    f0_pyin_hz       : float | NaN

    f1_hz ... f7_hz  : float | NaN
    b1_hz ... b7_hz  : float | NaN

    vtl_sample_mm    : float   VTL from assigned formants (unsmoothed)
    vtl_raw_mm       : float   VTL from raw Praat poles (unsmoothed)

    _sample_idx      : int     index into original samples list (stripped on transpose)

Transposed sample dict schema
------------------------------
    speaker          : str
    group            : str
    label            : str
    source           : str
    modality         : str
    dialect          : str
    audio            : (T,) float32        original audio from dataset loader
    window_length_ms : float
    hop_ms           : float               inferred from t_centre_s spacing
    t_centre_s       : (N,) float32
    loudness_dbfs    : (N,) float32
    spectral_slope   : (N,) float32
    f0_hz            : (N,) float32        NaN where unvoiced
    periodicity      : (N,) float32
    f0_praat_hz      : (N,) float32
    f0_pyin_hz       : (N,) float32
    formant_hz       : (N, 7) float32      F1-F7 frequencies
    formant_bw_hz    : (N, 7) float32      F1-F7 bandwidths
    vtl_sample_mm    : (N,) float32        per-window unsmoothed VTL
"""

from __future__ import annotations

import json
import os
import warnings
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass
from typing import Optional

import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

# 350 m/s -- warm (~35 degC), humid vocal tract interior.
# 343 m/s (dry air at 20 degC) underestimates VTL by ~2%.
SPEED_OF_SOUND_MM_S = 350_000.0

# Literature VTL priors (mm) -- from Fitch & Giedd (1999) / Story (2005)
VTL_PRIOR_MM: dict[str, float] = {
    "men":      174.0,
    "women":    148.0,
    "boys":     128.0,
    "girls":    123.0,
    "_default": 148.0,
}

# Tolerance for formant-to-expected-position matching, as a fraction of delta-F.
FORMANT_MATCH_TOLERANCE = 0.45   # +/- 45% of delta-F

N_FORMANTS = 5


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LabellerConfig:
    # Window parameters
    win_ms: float = 10.0
    hop_ms: float = 5.0
    min_f0_hz: float = 50.0
    max_f0_hz: float = 600.0

    # Formant extraction
    max_formant_hz: float = 9000.0
    n_praat_formants: int = N_FORMANTS + 2

    # VTL blending
    vtl_prior_strength: float = 10.0
    vtl_sample_alpha: float = 0.30

    # Voicing
    voicing_threshold_dbfs: float = -40.0

    # F0 agreement threshold (fractional difference)
    f0_agreement_threshold: float = 0.10

    # Parallelism -- -1 = all CPUs, 1 = serial (good for debugging)
    n_jobs: int = -1

    # Adaptive pre-emphasis (adapted from vtl.py).
    # When True, spectral tilt is estimated per-frame and a matched
    # first-order pre-emphasis is applied before Praat analysis.
    # The spectral slope (dB/oct) is stored as a feature regardless.
    # When False, Praat's fixed pre_emphasis_from_hz is used instead.
    adaptive_preemphasis: bool = True

    # Intensity gate for VTL seeding (dB SPL, Praat scale)
    intensity_threshold_db: float = 40.0

    # Fixed pre-emphasis floor -- only used when adaptive_preemphasis=False
    preemphasis_from_hz: float = 50.0


# ---------------------------------------------------------------------------
# Low-level acoustic helpers
# ---------------------------------------------------------------------------

def _loudness_dbfs(frame: np.ndarray) -> float:
    """RMS loudness in dBFS. Returns -inf for silent frames."""
    rms = np.sqrt(np.mean(frame.astype(np.float64) ** 2))
    if rms < 1e-10:
        return -np.inf
    return 20.0 * np.log10(rms)


def _periodicity(frame: np.ndarray, sr: int, f0_hz: float) -> float:
    """
    Normalised autocorrelation at the fundamental period lag.
    Returns NaN if f0 is NaN/zero.
    """
    if not np.isfinite(f0_hz) or f0_hz <= 0:
        return np.nan
    lag = int(round(sr / f0_hz))
    if lag <= 0 or lag >= len(frame):
        return np.nan
    n = len(frame) - lag
    denom = np.sqrt(np.sum(frame[:n] ** 2) * np.sum(frame[lag:lag + n] ** 2))
    if denom < 1e-12:
        return np.nan
    return float(np.sum(frame[:n] * frame[lag:lag + n]) / denom)


def _praat_f0(snd: parselmouth.Sound, cfg: LabellerConfig) -> float:
    """Praat autocorrelation F0 estimate for a short Sound object."""
    try:
        pitch = snd.to_pitch_ac(
            time_step=None,
            pitch_floor=cfg.min_f0_hz,
            pitch_ceiling=cfg.max_f0_hz,
        )
        values = pitch.selected_array["frequency"]
        voiced = values[values > 0]
        if len(voiced) == 0:
            return np.nan
        return float(np.median(voiced))
    except Exception:
        return np.nan


def _pyin_f0(frame: np.ndarray, sr: int, cfg: LabellerConfig) -> float:
    """librosa PYIN F0 estimate."""
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0, voiced_flag, _ = librosa.pyin(
                frame.astype(np.float32),
                fmin=cfg.min_f0_hz,
                fmax=cfg.max_f0_hz,
                sr=sr,
            )
        f0_voiced = f0[voiced_flag.astype(bool)] if voiced_flag is not None else f0
        f0_voiced = f0_voiced[np.isfinite(f0_voiced)]
        if len(f0_voiced) == 0:
            return np.nan
        return float(np.median(f0_voiced))
    except Exception:
        return np.nan


def _blend_f0(
    f0_praat: float,
    f0_pyin: float,
    modality: str,
    cfg: LabellerConfig,
) -> float:
    """
    If both estimates agree within threshold, return their mean.
    Otherwise return the modality-preferred estimate.
    """
    both_finite = np.isfinite(f0_praat) and np.isfinite(f0_pyin)
    if both_finite:
        mean = 0.5 * (f0_praat + f0_pyin)
        frac_diff = abs(f0_praat - f0_pyin) / mean
        if frac_diff <= cfg.f0_agreement_threshold:
            return mean
    if modality.startswith("sung"):
        return f0_pyin if np.isfinite(f0_pyin) else f0_praat
    else:
        return f0_praat if np.isfinite(f0_praat) else f0_pyin


# ---------------------------------------------------------------------------
# Spectral tilt / adaptive pre-emphasis  (adapted from vtl.py)
# ---------------------------------------------------------------------------

def _estimate_spectral_tilt_alpha(
    frame: np.ndarray,
    sr: int,
) -> tuple[float, float]:
    """
    Estimate spectral tilt (dB/octave) and a matched first-order
    pre-emphasis coefficient alpha.

    Fits a linear regression of log-magnitude (dB) against log2-frequency
    in the 300-4000 Hz speech band, then finds the first-order filter
    alpha in [0.70, 0.99] whose gain slope best cancels that tilt.

    Returns
    -------
    (slope_db_per_oct, alpha)
    Both NaN if estimation fails or the band contains fewer than 4 bins.
    """
    try:
        w     = np.hanning(len(frame))
        X     = np.fft.rfft(frame * w)
        mag   = np.abs(X) + 1e-12
        freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)

        mask = (freqs >= 300) & (freqs <= 4000)
        if mask.sum() < 4:
            return np.nan, np.nan

        f_band = freqs[mask]
        m_db   = 20.0 * np.log10(mag[mask])
        x_log  = np.log2(f_band)
        A      = np.vstack([x_log, np.ones_like(x_log)]).T
        slope, _ = np.linalg.lstsq(A, m_db, rcond=None)[0]

        # Find alpha that best cancels the tilt
        target       = -slope
        f1, f2       = 500.0, 4000.0
        w12          = 2.0 * np.pi * np.array([f1, f2]) / sr
        desired_diff = target * np.log2(f2 / f1)

        best_alpha, best_err = 0.95, 1e9
        for alpha in np.linspace(0.70, 0.99, 300):
            H    = np.abs(1.0 - alpha * np.exp(-1j * w12))
            diff = 20.0 * np.log10(H[1] / H[0])
            err  = abs(diff - desired_diff)
            if err < best_err:
                best_err   = err
                best_alpha = alpha

        return float(slope), float(best_alpha)
    except Exception:
        return np.nan, np.nan


def _apply_preemphasis_array(x: np.ndarray, alpha: float) -> np.ndarray:
    """First-order pre-emphasis: y[n] = x[n] - alpha * x[n-1]."""
    y     = np.empty_like(x)
    y[0]  = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y


# ---------------------------------------------------------------------------
# Praat formant + bandwidth extraction (raw poles, no relabelling yet)
# ---------------------------------------------------------------------------

def _extract_raw_formants(
    frame: np.ndarray,
    sr: int,
    cfg: LabellerConfig,
    win_s: float,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Run Praat's FormantPath tracker on a frame array.

    When cfg.adaptive_preemphasis is True, spectral tilt is estimated and a
    matched first-order pre-emphasis is applied before analysis. Praat's own
    internal pre-emphasis is then disabled (pre_emphasis_from set above
    Nyquist). When False, Praat's fixed pre_emphasis_from_hz is used.

    Returns
    -------
    freqs          : (n_praat_formants,) float32
    bws            : (n_praat_formants,) float32
    spectral_slope : float  (dB/octave; NaN if adaptive_preemphasis=False)
    """
    spectral_slope = np.nan
    analysis_frame = frame.astype(np.float32)

    if cfg.adaptive_preemphasis:
        slope, alpha   = _estimate_spectral_tilt_alpha(frame, sr)
        spectral_slope = slope
        if np.isfinite(alpha):
            analysis_frame = _apply_preemphasis_array(frame, alpha).astype(np.float32)
        # Disable Praat's internal pre-emphasis -- we've already done it
        praat_preemph = 20_000.0
    else:
        praat_preemph = cfg.preemphasis_from_hz

    snd = parselmouth.Sound(analysis_frame.astype(np.float64), sampling_frequency=sr)

    try:
        formant_path = call(
            snd,
            "To FormantPath (burg)...",
            win_s / 2.0,          # time step (must be > 0)
            cfg.n_praat_formants,
            cfg.max_formant_hz,
            win_s,
            praat_preemph,
            0.05,                 # ceiling step fraction
            4,                    # steps on either side
        )
        formant = call(formant_path, "Extract Formant")
    except Exception:
        try:
            formant = snd.to_formant_burg(
                time_step=win_s / 2.0,
                max_number_of_formants=cfg.n_praat_formants,
                maximum_formant=cfg.max_formant_hz,
                window_length=win_s,
                pre_emphasis_from=praat_preemph,
            )
        except Exception:
            return (
                np.full(cfg.n_praat_formants, np.nan, dtype=np.float32),
                np.full(cfg.n_praat_formants, np.nan, dtype=np.float32),
                spectral_slope,
            )

    t_mid = snd.duration / 2.0
    freqs = np.full(cfg.n_praat_formants, np.nan, dtype=np.float32)
    bws   = np.full(cfg.n_praat_formants, np.nan, dtype=np.float32)
    for i in range(cfg.n_praat_formants):
        fval = formant.get_value_at_time(i + 1, t_mid)
        bval = formant.get_bandwidth_at_time(i + 1, t_mid)
        if fval is not None and np.isfinite(fval) and fval > 0:
            freqs[i] = fval
        if bval is not None and np.isfinite(bval) and bval > 0:
            bws[i] = bval

    return freqs, bws, spectral_slope


# ---------------------------------------------------------------------------
# VTL estimation from formant frequencies
# ---------------------------------------------------------------------------

def _vtl_from_formants(freqs: np.ndarray) -> float:
    """
    Estimate vocal tract length from formant frequencies using the
    uniform-tube formula: Fn = (2n-1)*c/(4L) => L = (2n-1)*c/(4*Fn)

    Returns the median estimate across all valid formants.
    Returns NaN if no valid formants.
    """
    estimates = []
    for i, f in enumerate(freqs[1:]):
        if np.isfinite(f) and f > 0:
            n = i + 1
            L_mm = (2 * n - 1) * SPEED_OF_SOUND_MM_S / (4.0 * f)
            estimates.append(L_mm)
    if not estimates:
        return np.nan
    return float(np.median(estimates))


# ---------------------------------------------------------------------------
# Formant index assignment via minimum-cost matching
# ---------------------------------------------------------------------------

def _assign_formant_indices(
    raw_freqs: np.ndarray,
    raw_bws: np.ndarray,
    vtl_mm: float,
    n_out: int = N_FORMANTS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Assign raw Praat poles to anatomically correct formant slots using
    greedy minimum-distance matching against uniform-tube expected positions.

    Expected: Fn = (2n-1)*c/(4L) for n = 1 ... n_out
    Tolerance: +/-FORMANT_MATCH_TOLERANCE * delta-F  (delta-F = c/2L)
    Unmatched slots -> NaN.
    """
    if not np.isfinite(vtl_mm) or vtl_mm <= 0:
        out_f   = np.full(n_out, np.nan, dtype=np.float32)
        out_b   = np.full(n_out, np.nan, dtype=np.float32)
        valid   = raw_freqs[np.isfinite(raw_freqs)][:n_out]
        valid_b = raw_bws[np.isfinite(raw_bws)][:n_out]
        out_f[:len(valid)]   = valid
        out_b[:len(valid_b)] = valid_b
        return out_f, out_b

    delta_f  = SPEED_OF_SOUND_MM_S / (2.0 * vtl_mm)
    expected = np.array(
        [(2 * n - 1) * SPEED_OF_SOUND_MM_S / (4.0 * vtl_mm)
         for n in range(1, n_out + 1)],
        dtype=np.float64,
    )
    tol = FORMANT_MATCH_TOLERANCE * delta_f

    poles = [
        (float(raw_freqs[i]),
         float(raw_bws[i]) if np.isfinite(raw_bws[i]) else np.nan,
         i)
        for i in range(len(raw_freqs))
        if np.isfinite(raw_freqs[i]) and raw_freqs[i] > 0
    ]

    out_f          = np.full(n_out, np.nan, dtype=np.float32)
    out_b          = np.full(n_out, np.nan, dtype=np.float32)
    assigned_poles = set()

    for slot_idx, exp_f in enumerate(expected):
        best_dist     = np.inf
        best_pole_idx = -1
        for k, (pf, pb, _) in enumerate(poles):
            if k in assigned_poles:
                continue
            dist = abs(pf - exp_f)
            if dist < best_dist and dist <= tol:
                best_dist     = dist
                best_pole_idx = k
        if best_pole_idx >= 0:
            pf, pb, _ = poles[best_pole_idx]
            out_f[slot_idx] = pf
            out_b[slot_idx] = pb
            assigned_poles.add(best_pole_idx)

    return out_f, out_b


# ---------------------------------------------------------------------------
# Hierarchical VTL smoother
# ---------------------------------------------------------------------------

class VTLSmoother:
    """
    Maintains running group/speaker VTL statistics and blends them with the
    literature prior using uncertainty-weighted alphas.

    Blending weight: alpha = n / (n + n0)
    where n = samples seen for that entity, n0 = prior_strength.
    """

    def __init__(self, prior_strength: float = 10.0, sample_alpha: float = 0.30):
        self.n0           = prior_strength
        self.sample_alpha = sample_alpha
        self._group_vtls:   dict[str, list[float]] = {}
        self._speaker_vtls: dict[str, list[float]] = {}

    def update(self, group: str, speaker: str, vtl_mm: float) -> None:
        if not np.isfinite(vtl_mm):
            return
        self._group_vtls.setdefault(group, []).append(vtl_mm)
        self._speaker_vtls.setdefault(speaker, []).append(vtl_mm)

    def _group_prior(self, group: str) -> float:
        return VTL_PRIOR_MM.get(group, VTL_PRIOR_MM["_default"])

    def smoothed_group_vtl(self, group: str) -> float:
        vals = self._group_vtls.get(group, [])
        n    = len(vals)
        if n == 0:
            return self._group_prior(group)
        alpha = n / (n + self.n0)
        return (1.0 - alpha) * self._group_prior(group) + alpha * float(np.mean(vals))

    def smoothed_speaker_vtl(self, group: str, speaker: str) -> float:
        vals      = self._speaker_vtls.get(speaker, [])
        n         = len(vals)
        group_vtl = self.smoothed_group_vtl(group)
        if n == 0:
            return group_vtl
        alpha = n / (n + self.n0)
        return (1.0 - alpha) * group_vtl + alpha * float(np.mean(vals))

    def smooth_vtl(self, group: str, speaker: str, raw_sample_vtl: float) -> float:
        """Blend hierarchy: literature -> group -> speaker -> sample."""
        speaker_vtl = self.smoothed_speaker_vtl(group, speaker)
        if not np.isfinite(raw_sample_vtl):
            return speaker_vtl
        return ((1.0 - self.sample_alpha) * speaker_vtl
                + self.sample_alpha * raw_sample_vtl)


# ---------------------------------------------------------------------------
# Per-sample feature extractor
# ---------------------------------------------------------------------------

def _extract_sample_windows(
    sample: dict,
    cfg: LabellerConfig,
    vtl_smoother: VTLSmoother,
    pass_label: str = "raw",
) -> list[dict]:
    """
    Slice a sample into overlapping windows and extract acoustic features.

    vtl_smoother is used read-only (frozen snapshot from the calling pass).
    Unvoiced windows are retained with NaN for F0/periodicity so the full
    timeline is preserved for downstream windowed std computation.
    """
    audio    = sample["audio"].astype(np.float32)
    sr       = sample.get("sr", 16000)
    speaker  = sample.get("speaker", sample.get("group", "unknown"))
    group    = sample.get("group", speaker)
    modality = sample.get("modality", "spoken")

    min_win_ms  = 3000.0 / cfg.min_f0_hz
    win_ms      = max(cfg.win_ms, min_win_ms)
    win_s       = win_ms / 1000.0
    win_samples = int(round(sr * win_s))
    hop_samples = max(1, int(round(sr * cfg.hop_ms / 1000.0)))
    if win_samples > len(audio):
        win_samples = len(audio)

    windows = []
    start   = 0
    while start + win_samples <= len(audio):
        frame      = audio[start : start + win_samples]
        t_centre_s = (start + win_samples / 2.0) / sr

        # --- Loudness & voicing ---
        loudness = _loudness_dbfs(frame)
        voiced   = loudness >= cfg.voicing_threshold_dbfs

        # --- F0 (unvoiced frames keep NaN) ---
        snd = parselmouth.Sound(frame.astype(np.float64), sampling_frequency=sr)
        if voiced:
            f0_praat = _praat_f0(snd, cfg)
            f0_pyin  = _pyin_f0(frame, sr, cfg)
            f0       = _blend_f0(f0_praat, f0_pyin, modality, cfg)
        else:
            f0_praat = f0_pyin = f0 = np.nan

        # --- Periodicity ---
        period = _periodicity(frame, sr, f0)

        # --- Formants + spectral slope (all windows, voiced or not) ---
        raw_freqs, raw_bws, spectral_slope = _extract_raw_formants(
            frame, sr, cfg, win_s
        )

        # --- Raw VTL from unassigned poles ---
        vtl_raw_mm = _vtl_from_formants(raw_freqs)

        # --- Smoothed VTL for formant index assignment ---
        smoothed_vtl = vtl_smoother.smooth_vtl(group, speaker, vtl_raw_mm)

        # --- Assign formant indices ---
        freqs_out, bws_out = _assign_formant_indices(
            raw_freqs, raw_bws, smoothed_vtl, n_out=N_FORMANTS
        )

        # --- Unsmoothed sample VTL from assigned formants ---
        vtl_sample_mm = _vtl_from_formants(freqs_out)

        w = {
            "speaker":          speaker,
            "group":            group,
            "label":            sample.get("label", ""),
            "source":           sample.get("source", ""),
            "modality":         modality,
            "dialect":          sample.get("dialect", "unknown"),
            "window_length_ms": win_ms,
            "t_centre_s":       t_centre_s,
            "loudness_dbfs":    loudness,
            "spectral_slope":   spectral_slope,
            "f0_hz":            f0,
            "periodicity":      period,
            "f0_praat_hz":      f0_praat,
            "f0_pyin_hz":       f0_pyin,
            "vtl_sample_mm":    vtl_sample_mm,
            "vtl_raw_mm":       vtl_raw_mm,
        }
        for i in range(N_FORMANTS):
            w[f"f{i+1}_hz"] = float(freqs_out[i]) if np.isfinite(freqs_out[i]) else np.nan
            w[f"b{i+1}_hz"] = float(bws_out[i])   if np.isfinite(bws_out[i])   else np.nan

        windows.append(w)
        start += hop_samples

    return windows


# ---------------------------------------------------------------------------
# Top-level picklable workers  (must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _worker_extract(
    args: tuple[dict, LabellerConfig, VTLSmoother, str],
) -> list[dict]:
    """Extract windows for one sample (parallel worker)."""
    sample, cfg, smoother, pass_label = args
    return _extract_sample_windows(sample, cfg, smoother, pass_label)


def _worker_seed_vtl(
    args: tuple[dict, LabellerConfig],
) -> tuple[str, str, float]:
    """
    Compute a robust VTL estimate for one sample for smoother seeding.

    Uses up to 20 intensity-gated time positions spread across the full file
    rather than a single centre-window, for a more representative prior.
    Returns (group, speaker, median_vtl_mm).
    """
    s, cfg   = args
    audio    = s["audio"].astype(np.float32)
    speaker  = s.get("speaker", s.get("group", "unknown"))
    group    = s.get("group", speaker)
    sr       = s.get("sr", 16000)
    win_s    = max(cfg.win_ms, 3000.0 / cfg.min_f0_hz) / 1000.0
    duration = len(audio) / sr

    # Candidate positions -- stay at least half a window from the edges
    margin    = win_s / 2.0
    positions = np.linspace(margin, duration - margin, 20)
    positions = positions[positions > 0]

    # Gate by intensity where possible
    try:
        snd_full  = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
        intensity = snd_full.to_intensity(
            minimum_pitch=cfg.min_f0_hz,
            time_step=None,
            subtract_mean=False,
        )
        ints      = [intensity.get_value(t) for t in positions]
        positions = [
            t for t, iv in zip(positions, ints)
            if iv is not None and np.isfinite(iv) and iv > cfg.intensity_threshold_db
        ]
    except Exception:
        positions = list(positions)   # fall back to all positions

    if not positions:
        # Last resort: centre window
        n      = len(audio)
        centre = audio[n // 4 : 3 * n // 4]
        raw_f, _, _ = _extract_raw_formants(centre, sr, cfg, win_s)
        return group, speaker, _vtl_from_formants(raw_f)

    vtls      = []
    win_samps = int(round(win_s * sr))
    for t in positions:
        s0    = max(0, int((t - win_s / 2) * sr))
        s1    = min(len(audio), s0 + win_samps)
        frame = audio[s0:s1]
        if len(frame) < win_samps // 2:
            continue
        raw_f, _, _ = _extract_raw_formants(frame, sr, cfg, win_s)
        v = _vtl_from_formants(raw_f)
        if np.isfinite(v):
            vtls.append(v)

    if not vtls:
        return group, speaker, np.nan
    return group, speaker, float(np.median(vtls))


# ---------------------------------------------------------------------------
# Parallel extraction helper
# ---------------------------------------------------------------------------

def _resolve_workers(n_jobs: int) -> int:
    if n_jobs == -1:
        return os.cpu_count() or 1
    return max(1, n_jobs)


def _parallel_extract(
    samples: list[dict],
    cfg: LabellerConfig,
    smoother: VTLSmoother,
    pass_label: str,
    n_workers: int,
    verbose: bool,
    pass_name: str,
) -> list[list[dict]]:
    """Fan out _worker_extract across samples. Preserves order."""
    n    = len(samples)
    args = [(s, cfg, smoother, pass_label) for s in samples]

    if n_workers == 1:
        results = []
        for i, a in enumerate(args):
            if verbose and i % max(1, n // 10) == 0:
                print(f"  [{pass_name}] {i}/{n}", end="\r")
            results.append(_worker_extract(a))
        return results

    results_by_idx: dict[int, list[dict]] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker_extract, a): i for i, a in enumerate(args)}
        done = 0
        for fut in as_completed(futures):
            idx               = futures[fut]
            results_by_idx[idx] = fut.result()
            done             += 1
            if verbose and done % max(1, n // 10) == 0:
                print(f"  [{pass_name}] {done}/{n}", end="\r")

    return [results_by_idx[i] for i in range(n)]


# ---------------------------------------------------------------------------
# Two-pass dataset labeller
# ---------------------------------------------------------------------------

def label_dataset(
    samples: list[dict],
    cfg: Optional[LabellerConfig] = None,
    sr: int = 16000,
    verbose: bool = True,
) -> list[dict]:
    """
    Run the full two-pass labelling pipeline on a dataset.

    Parameters
    ----------
    samples : list of dicts from dataset.py / datasets_hf.py
    cfg     : LabellerConfig (uses defaults if None)
    sr      : sample rate (used if not present in sample dict)
    verbose : print progress

    Returns
    -------
    Flat list of per-window label dicts. Each window carries a
    ``_sample_idx`` field (index into ``samples``) for use by
    ``transpose_to_samples()``.

    Notes
    -----
    Wrap calls in ``if __name__ == "__main__":`` when running from a script
    to avoid worker process re-spawning on Windows/macOS (spawn start method).
    """
    if cfg is None:
        cfg = LabellerConfig()

    n_workers = _resolve_workers(cfg.n_jobs)

    for s in samples:
        s.setdefault("sr", sr)

    if verbose:
        print(f"[labeller] {len(samples)} samples, "
              f"{n_workers} worker{'s' if n_workers != 1 else ''}.")

    # -----------------------------------------------------------------------
    # Seed: robust multi-position VTL estimate per sample (parallel)
    # -----------------------------------------------------------------------
    if verbose:
        print("[labeller] Seeding VTL smoother...")

    seed_args = [(s, cfg) for s in samples]
    if n_workers == 1:
        seed_results = [_worker_seed_vtl(a) for a in seed_args]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            seed_results = list(pool.map(_worker_seed_vtl, seed_args))

    smoother_raw = VTLSmoother(
        prior_strength=cfg.vtl_prior_strength,
        sample_alpha=cfg.vtl_sample_alpha,
    )
    for group, speaker, vtl in seed_results:
        smoother_raw.update(group, speaker, vtl)

    # -----------------------------------------------------------------------
    # Pass 0: raw window extraction
    # -----------------------------------------------------------------------
    if verbose:
        print("[labeller] Pass 0 -- raw extraction...")

    pass0_by_sample = _parallel_extract(
        samples, cfg, smoother_raw, "raw", n_workers, verbose, "pass0"
    )

    # Aggregate pass-0 vtl_sample_mm (from assigned formants) for pass-1 smoother
    smoother_first = VTLSmoother(
        prior_strength=cfg.vtl_prior_strength,
        sample_alpha=cfg.vtl_sample_alpha,
    )
    for wins in pass0_by_sample:
        if not wins:
            continue
        vtls = [w["vtl_sample_mm"] for w in wins if np.isfinite(w["vtl_sample_mm"])]
        if vtls:
            smoother_first.update(wins[0]["group"], wins[0]["speaker"],
                                  float(np.mean(vtls)))

    # -----------------------------------------------------------------------
    # Pass 1: refined window extraction
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[labeller] Pass 1 -- refining VTL estimates...")

    pass1_by_sample = _parallel_extract(
        samples, cfg, smoother_first, "first", n_workers, verbose, "pass1"
    )

    # Aggregate pass-1 vtl_sample_mm for pass-2 smoother
    smoother_second = VTLSmoother(
        prior_strength=cfg.vtl_prior_strength,
        sample_alpha=cfg.vtl_sample_alpha,
    )
    for wins in pass1_by_sample:
        if not wins:
            continue
        vtls = [w["vtl_sample_mm"] for w in wins if np.isfinite(w["vtl_sample_mm"])]
        if vtls:
            smoother_second.update(wins[0]["group"], wins[0]["speaker"],
                                   float(np.mean(vtls)))

    # -----------------------------------------------------------------------
    # Pass 2: final extraction -- frozen per-speaker VTL, no sample blend
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[labeller] Pass 2 -- final formant assignment...")

    samples_pass2 = []
    for s in samples:
        speaker   = s.get("speaker", s.get("group", "unknown"))
        group     = s.get("group", speaker)
        final_vtl = smoother_second.smoothed_speaker_vtl(group, speaker)
        frozen    = VTLSmoother(prior_strength=1e9, sample_alpha=0.0)
        frozen._speaker_vtls[speaker] = [final_vtl]
        frozen._group_vtls[group]     = [final_vtl]
        samples_pass2.append((dict(s), cfg, frozen, "second"))

    if n_workers == 1:
        pass2_by_sample = []
        for i, a in enumerate(samples_pass2):
            if verbose and i % max(1, len(samples) // 10) == 0:
                print(f"  [pass2] {i}/{len(samples)}", end="\r")
            pass2_by_sample.append(_worker_extract(a))
    else:
        results_by_idx: dict[int, list[dict]] = {}
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            futures = {
                pool.submit(_worker_extract, a): i
                for i, a in enumerate(samples_pass2)
            }
            done = 0
            for fut in as_completed(futures):
                idx               = futures[fut]
                results_by_idx[idx] = fut.result()
                done             += 1
                if verbose and done % max(1, len(samples) // 10) == 0:
                    print(f"  [pass2] {done}/{len(samples)}", end="\r")
        pass2_by_sample = [results_by_idx[i] for i in range(len(samples))]

    # Flatten and tag with _sample_idx.
    # vtl_sample_mm in each window is already the unsmoothed per-window VTL
    # from pass-2 assigned formants -- no further overwriting needed.
    final_windows: list[dict] = []
    for i, wins in enumerate(pass2_by_sample):
        for w in wins:
            w["_sample_idx"] = i
        final_windows.extend(wins)

    if verbose:
        n_voiced = sum(1 for w in final_windows if np.isfinite(w["f0_hz"]))
        print(f"\n[labeller] Done. {len(final_windows)} windows "
              f"({n_voiced} voiced, "
              f"{len(final_windows) - n_voiced} unvoiced) "
              f"from {len(samples)} samples.")

    return final_windows


# ---------------------------------------------------------------------------
# Transposition: flat windows -> one dict per sample
# ---------------------------------------------------------------------------

def transpose_to_samples(
    flat_windows: list[dict],
    original_samples: list[dict],
) -> list[dict]:
    """
    Convert the flat per-window list from label_dataset() into one dict per
    original sample, with per-window acoustic features stored as numpy arrays.

    Parameters
    ----------
    flat_windows     : output of label_dataset()
    original_samples : the same samples list passed to label_dataset(),
                       used to attach the original audio arrays

    Returns
    -------
    List of sample dicts -- see module docstring for schema.
    The ``_sample_idx`` field is consumed here and not present in output.
    """
    # Group windows by sample index
    groups: dict[int, list[dict]] = defaultdict(list)
    for w in flat_windows:
        groups[w["_sample_idx"]].append(w)
    print(len(groups))

    scalar_fields = [
        "loudness_dbfs", "spectral_slope",
        "f0_hz", "periodicity", "f0_praat_hz", "f0_pyin_hz",
        "vtl_sample_mm", "vtl_raw_mm", "t_centre_s",
    ]
    formant_f_cols = [f"f{i+1}_hz" for i in range(N_FORMANTS)]
    formant_b_cols = [f"b{i+1}_hz" for i in range(N_FORMANTS)]

    transposed = []
    for idx, orig in enumerate(original_samples):
        wins = groups.get(idx, [])
        if not wins:
            continue

        # Stack scalar fields into 1-D arrays
        arrays = {
            f: np.array([w[f] for w in wins], dtype=np.float32)
            for f in scalar_fields
        }

        # Stack formants into (N, 7) arrays
        formant_hz    = np.stack(
            [np.array([w[c] for c in formant_f_cols], dtype=np.float32)
             for w in wins]
        )
        formant_bw_hz = np.stack(
            [np.array([w[c] for c in formant_b_cols], dtype=np.float32)
             for w in wins]
        )

        # Infer hop_ms from t_centre_s spacing
        t      = arrays["t_centre_s"]
        hop_ms = (float(np.diff(t).mean() * 1000.0)
                  if len(t) > 1
                  else wins[0]["window_length_ms"])
        sample = {
            "speaker":          wins[0]["speaker"],
            "group":            wins[0]["group"],
            "label":            wins[0]["label"],
            "source":           wins[0]["source"],
            "modality":         wins[0]["modality"],
            "dialect":          wins[0]["dialect"],
            "audio":            orig["audio"].astype(np.float32),
            "window_length_ms": wins[0]["window_length_ms"],
            "hop_ms":           hop_ms,
            "t_centre_s":       arrays["t_centre_s"],
            "loudness_dbfs":    arrays["loudness_dbfs"],
            "spectral_slope":   arrays["spectral_slope"],
            "f0_hz":            arrays["f0_hz"],
            "periodicity":      arrays["periodicity"],
            "f0_praat_hz":      arrays["f0_praat_hz"],
            "f0_pyin_hz":       arrays["f0_pyin_hz"],
            "formant_hz":       formant_hz,
            "formant_bw_hz":    formant_bw_hz,
            "vtl_sample_mm":    arrays["vtl_sample_mm"],
        }
        print(sample["speaker"])
        transposed.append(sample)

    return transposed


# ---------------------------------------------------------------------------
# Metadata: unsmoothed speaker / group VTL statistics
# ---------------------------------------------------------------------------

def build_metadata(
    transposed: list[dict],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Compute unsmoothed speaker and group VTL statistics from pass-2
    formant-derived VTLs (vtl_sample_mm arrays in the transposed dataset).

    No blending or smoothing is applied -- these are plain empirical
    means/stds over the raw per-window estimates, suitable for unit testing
    against the smoothed values used internally during labelling.

    Returns
    -------
    speaker_meta : {speaker_id: {group, vtl_mean_mm, vtl_std_mm,
                                  n_samples, n_windows}}
    group_meta   : {group: {vtl_mean_mm, vtl_std_mm, n_samples, n_windows,
                             vtl_literature_mm}}
    """
    spk_vtls:    dict[str, list[float]] = {}
    spk_group:   dict[str, str]         = {}
    spk_n:       dict[str, int]         = {}

    for s in transposed:
        spk  = s["speaker"]
        grp  = s["group"]
        vtls = s["vtl_sample_mm"]
        vtls = vtls[np.isfinite(vtls)]
        spk_vtls.setdefault(spk, []).extend(vtls.tolist())
        spk_group[spk] = grp
        spk_n[spk]     = spk_n.get(spk, 0) + 1

    speaker_meta: dict[str, dict] = {}
    for spk, vtls in spk_vtls.items():
        arr = np.array(vtls, dtype=np.float64)
        speaker_meta[spk] = {
            "group":       spk_group[spk],
            "vtl_mean_mm": float(np.mean(arr)) if len(arr) else float("nan"),
            "vtl_std_mm":  float(np.std(arr))  if len(arr) else float("nan"),
            "n_samples":   spk_n[spk],
            "n_windows":   len(arr),
        }

    grp_vtls: dict[str, list[float]] = {}
    grp_spks: dict[str, set]         = {}
    for spk, meta in speaker_meta.items():
        grp = meta["group"]
        grp_vtls.setdefault(grp, []).extend(spk_vtls.get(spk, []))
        grp_spks.setdefault(grp, set()).add(spk)

    group_meta: dict[str, dict] = {}
    for grp, vtls in grp_vtls.items():
        arr = np.array(vtls, dtype=np.float64)
        group_meta[grp] = {
            "vtl_mean_mm":       float(np.mean(arr)) if len(arr) else float("nan"),
            "vtl_std_mm":        float(np.std(arr))  if len(arr) else float("nan"),
            "n_samples":         len(grp_spks[grp]),
            "n_windows":         len(arr),
            "vtl_literature_mm": VTL_PRIOR_MM.get(grp, VTL_PRIOR_MM["_default"]),
        }

    return speaker_meta, group_meta


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_dataset(
    transposed: list[dict],
    speaker_meta: dict,
    group_meta: dict,
    path_prefix: str,
) -> None:
    """
    Save the full labelled dataset to disk.

    Creates:
        <path_prefix>_samples.pkl       -- transposed sample list (pickle)
        <path_prefix>_speaker_meta.json -- per-speaker VTL statistics
        <path_prefix>_group_meta.json   -- per-group VTL statistics

    Parameters
    ----------
    transposed   : output of transpose_to_samples()
    speaker_meta : output of build_metadata()[0]
    group_meta   : output of build_metadata()[1]
    path_prefix  : e.g. "output/my_dataset"
    """
    import pickle

    pkl_path = f"{path_prefix}_samples.pkl"
    spk_path = f"{path_prefix}_speaker_meta.json"
    grp_path = f"{path_prefix}_group_meta.json"

    with open(pkl_path, "wb") as f:
        pickle.dump(transposed, f, protocol=pickle.HIGHEST_PROTOCOL)

    with open(spk_path, "w", encoding="utf-8") as f:
        json.dump(speaker_meta, f, indent=2, ensure_ascii=False)

    with open(grp_path, "w", encoding="utf-8") as f:
        json.dump(group_meta, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(transposed)} samples  -> {pkl_path}")
    print(f"Speaker metadata ({len(speaker_meta)} speakers) -> {spk_path}")
    print(f"Group metadata   ({len(group_meta)} groups)   -> {grp_path}")


def load_dataset(
    path_prefix: str,
) -> tuple[list[dict], dict, dict]:
    """
    Load a dataset saved by save_dataset().

    Returns
    -------
    (transposed, speaker_meta, group_meta)
    """
    import pickle

    pkl_path = f"{path_prefix}_samples.pkl"
    spk_path = f"{path_prefix}_speaker_meta.json"
    grp_path = f"{path_prefix}_group_meta.json"

    with open(pkl_path, "rb") as f:
        transposed = pickle.load(f)

    with open(spk_path, "r", encoding="utf-8") as f:
        speaker_meta = json.load(f)

    with open(grp_path, "r", encoding="utf-8") as f:
        group_meta = json.load(f)

    print(f"Loaded {len(transposed)} samples from {pkl_path}")
    return transposed, speaker_meta, group_meta


# ---------------------------------------------------------------------------
# Legacy flat-list save / load  (kept for compatibility)
# ---------------------------------------------------------------------------

def save_windows(windows: list[dict], path: str) -> None:
    """Pickle a flat window list to disk."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(windows, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(windows)} windows to {path}")


def load_windows(path: str) -> list[dict]:
    """Load a pickled flat window list from disk."""
    import pickle
    with open(path, "rb") as f:
        windows = pickle.load(f)
    print(f"Loaded {len(windows)} windows from {path}")
    return windows


# ---------------------------------------------------------------------------
# Convenience: flat numpy array for model input (from flat window list)
# ---------------------------------------------------------------------------

FEATURE_COLS = (
    ["loudness_dbfs", "spectral_slope", "f0_hz", "periodicity"]
    + [f"f{i+1}_hz" for i in range(N_FORMANTS)]
    + [f"b{i+1}_hz" for i in range(N_FORMANTS)]
)


def to_arrays(
    windows: list[dict],
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    """
    Convert flat window dicts to a numpy feature matrix.

    Returns
    -------
    X          : (N, F) float32   -- acoustic features (NaN for unobservable)
    labels     : (N,)  object     -- IPA label strings
    feat_names : list[str]        -- column names for X
    """
    X = np.array(
        [[w.get(c, np.nan) for c in FEATURE_COLS] for w in windows],
        dtype=np.float32,
    )
    labels = np.array([w["label"] for w in windows], dtype=object)
    return X, labels, FEATURE_COLS
