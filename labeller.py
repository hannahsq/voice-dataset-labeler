# labeller.py
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

This module corrects that via a three-pass hierarchical VTL pipeline:

  Seed pass:
    For each sample, extract formants at up to 20 intensity-gated positions
    and take the median VTL.  Populate the global smoother before any full
    windowed pass begins.

  Pass 0 (raw):
    Extract all windows using the seed smoother.  vtl_sample_mm per window
    is derived from the *assigned* formants (not raw poles) so the inter-pass
    aggregation feeds on already-corrected estimates.

  Pass 1 (refinement):
    Re-estimate group/speaker VTL means from pass-0 vtl_sample_mm arrays.
    Re-extract all windows with the refined smoother.

  Pass 2 (final):
    Freeze each speaker's VTL at the pass-1 smoothed speaker mean (no
    per-sample blend in the final pass).  Re-extract all windows.
    vtl_sample_mm in the output is the unsmoothed per-window value from
    these final assigned formants — this is what you train on.

VTL blending
------------
    alpha = n / (n + n0)          (group and speaker level)
    vtl_sample = (1-alpha_s)*vtl_speaker + alpha_s*vtl_raw   (sample level)

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
    speaker          : str
    group            : str
    label            : str
    source           : str
    modality         : str
    dialect          : str
    audio            : (T,)    float32   original audio
    window_length_ms : float
    hop_ms           : float

    t_centre_s       : (N,)    float32
    loudness_dbfs    : (N,)    float32
    spectral_slope   : (N,)    float32   dB/oct; NaN if adaptive_preemphasis=False
    f0_hz            : (N,)    float32   NaN where unvoiced
    periodicity      : (N,)    float32
    f0_praat_hz      : (N,)    float32
    f0_pyin_hz       : (N,)    float32
    formant_hz       : (N, 7)  float32   F1-F7 frequencies
    formant_bw_hz    : (N, 7)  float32   F1-F7 bandwidths
    vtl_sample_mm    : (N,)    float32   unsmoothed per-window VTL
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

SPEED_OF_SOUND_MM_S = 350_000.0   # mm/s — warm, humid vocal tract interior

# Valid values for the sex and age dimensions of the group tuple.
# Exported so dataset loaders can validate against the same constants.
VALID_SEX: frozenset = frozenset({"male", "female", "unknown"})
VALID_AGE: frozenset = frozenset({"adult", "child",  "unknown"})

# Literature VTL priors (mm) keyed by (sex, age) tuples.
# Fitch & Giedd (1999) / Story (2005).
# Fallback hierarchy (most to least specific):
#   (sex, age) -> (sex, "unknown") -> ("unknown", age) -> ("unknown", "unknown")
VTL_PRIOR_MM: dict = {
    ("male",    "adult"):   174.0,
    ("female",  "adult"):   148.0,
    ("male",    "child"):   128.0,
    ("female",  "child"):   123.0,
    # collapsed dimensions
    ("male",    "unknown"): 151.0,   # mean of male adult + child
    ("female",  "unknown"): 135.5,   # mean of female adult + child
    ("unknown", "adult"):   161.0,   # mean of male + female adult
    ("unknown", "child"):   125.5,   # mean of male + female child
    ("unknown", "unknown"): 148.0,   # whole-dataset mean (literature)
}

FORMANT_MATCH_TOLERANCE = 0.45   # +/- fraction of delta-F for pole assignment

N_FORMANTS = 5


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LabellerConfig:
    win_ms:                 float = 10.0
    hop_ms:                 float = 5.0
    min_f0_hz:              float = 50.0
    max_f0_hz:              float = 600.0
    max_formant_hz:         float = 5500.0
    n_praat_formants:       int   = N_FORMANTS
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


# ---------------------------------------------------------------------------
# Acoustic helpers
# ---------------------------------------------------------------------------

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


def _pyin_f0(frame: np.ndarray, sr: int, cfg: LabellerConfig) -> float:
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            f0, voiced_flag, _ = librosa.pyin(frame.astype(np.float32),
                                               fmin=cfg.min_f0_hz,
                                               fmax=cfg.max_f0_hz,
                                               sr=sr)
        f0v = f0[voiced_flag.astype(bool)] if voiced_flag is not None else f0
        f0v = f0v[np.isfinite(f0v)]
        return float(np.median(f0v)) if len(f0v) else np.nan
    except Exception:
        return np.nan


def _blend_f0(f0_praat: float, f0_pyin: float,
              modality: str, cfg: LabellerConfig) -> float:
    if np.isfinite(f0_praat) and np.isfinite(f0_pyin):
        mean = 0.5 * (f0_praat + f0_pyin)
        if abs(f0_praat - f0_pyin) / mean <= cfg.f0_agreement_threshold:
            return mean
    if modality.startswith("sung"):
        return f0_pyin if np.isfinite(f0_pyin) else f0_praat
    return f0_praat if np.isfinite(f0_praat) else f0_pyin


# ---------------------------------------------------------------------------
# Spectral tilt / adaptive pre-emphasis
# ---------------------------------------------------------------------------

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

        x_log = np.log2(freqs[mask])
        A     = np.vstack([x_log, np.ones_like(x_log)]).T
        slope, _ = np.linalg.lstsq(A, 20.0 * np.log10(mag[mask]), rcond=None)[0]

        target       = -slope
        w12          = 2.0 * np.pi * np.array([500.0, 4000.0]) / sr
        desired_diff = target * np.log2(4000.0 / 500.0)
        best_alpha, best_err = 0.95, 1e9
        for alpha in np.linspace(0.70, 0.99, 300):
            H    = np.abs(1.0 - alpha * np.exp(-1j * w12))
            diff = 20.0 * np.log10(H[1] / H[0])
            if (err := abs(diff - desired_diff)) < best_err:
                best_err, best_alpha = err, alpha

        return float(slope), float(best_alpha)
    except Exception:
        return np.nan, np.nan


def _apply_preemphasis(x: np.ndarray, alpha: float) -> np.ndarray:
    y     = np.empty_like(x)
    y[0]  = x[0]
    y[1:] = x[1:] - alpha * x[:-1]
    return y


# ---------------------------------------------------------------------------
# Praat formant extraction
# ---------------------------------------------------------------------------

def _extract_raw_formants(
    frame: np.ndarray,
    sr: int,
    cfg: LabellerConfig,
    win_s: float,
    max_formant_hz: Optional[float] = None,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Extract raw Praat poles from a frame.  Applies adaptive pre-emphasis when
    cfg.adaptive_preemphasis is True, disabling Praat's own filter.

    max_formant_hz overrides cfg.max_formant_hz when provided — used by
    _extract_sample to pass in a speaker-adapted ceiling derived from the
    smoothed VTL estimate.

    Returns (freqs, bws, spectral_slope) — arrays are (n_praat_formants,).
    """
    ceiling        = max_formant_hz if max_formant_hz is not None else cfg.max_formant_hz
    spectral_slope = np.nan
    analysis       = frame.astype(np.float32)

    if cfg.adaptive_preemphasis:
        slope, alpha   = _estimate_spectral_tilt_alpha(frame, sr)
        spectral_slope = slope
        if np.isfinite(alpha):
            analysis = _apply_preemphasis(frame, alpha).astype(np.float32)
        praat_preemph = 20_000.0       # above Nyquist — disables Praat's filter
    else:
        praat_preemph = cfg.preemphasis_from_hz

    snd = parselmouth.Sound(analysis.astype(np.float64), sampling_frequency=sr)

    try:
        fp      = call(snd, "To FormantPath (burg)...",
                       win_s / 2.0, cfg.n_praat_formants,
                       ceiling, win_s,
                       praat_preemph, 0.05, 4)
        formant = call(fp, "Extract Formant")
    except Exception:
        try:
            formant = snd.to_formant_burg(
                time_step=win_s / 2.0,
                max_number_of_formants=cfg.n_praat_formants,
                maximum_formant=ceiling,
                window_length=win_s,
                pre_emphasis_from=praat_preemph,
            )
        except Exception:
            empty = np.full(cfg.n_praat_formants, np.nan, dtype=np.float32)
            return empty.copy(), empty.copy(), spectral_slope

    t_mid = snd.duration / 2.0
    freqs = np.full(cfg.n_praat_formants, np.nan, dtype=np.float32)
    bws   = np.full(cfg.n_praat_formants, np.nan, dtype=np.float32)
    for i in range(cfg.n_praat_formants):
        fv = formant.get_value_at_time(i + 1, t_mid)
        bv = formant.get_bandwidth_at_time(i + 1, t_mid)
        if fv is not None and np.isfinite(fv) and fv > 0:
            freqs[i] = fv
        if bv is not None and np.isfinite(bv) and bv > 0:
            bws[i] = bv

    return freqs, bws, spectral_slope


# ---------------------------------------------------------------------------
# VTL / formant utilities
# ---------------------------------------------------------------------------

def _vtl_from_formants(freqs: np.ndarray) -> float:
    """Median VTL estimate from assigned formant frequencies (mm)."""
    estimates = [
        (2 * (i + 1) - 1) * SPEED_OF_SOUND_MM_S / (4.0 * f)
        for i, f in enumerate(freqs)
        if np.isfinite(f) and f > 0
    ]
    return float(np.median(estimates)) if estimates else np.nan


def _assign_formant_indices(
    raw_freqs: np.ndarray,
    raw_bws: np.ndarray,
    vtl_mm: float,
    n_out: int = N_FORMANTS,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Greedily assign raw Praat poles to anatomically correct slots using
    expected positions from the uniform-tube model.  Unmatched slots -> NaN.
    """
    out_f = np.full(n_out, np.nan, dtype=np.float32)
    out_b = np.full(n_out, np.nan, dtype=np.float32)

    if not np.isfinite(vtl_mm) or vtl_mm <= 0:
        valid   = raw_freqs[np.isfinite(raw_freqs)][:n_out]
        valid_b = raw_bws[np.isfinite(raw_bws)][:n_out]
        out_f[:len(valid)]   = valid
        out_b[:len(valid_b)] = valid_b
        return out_f, out_b

    delta_f  = SPEED_OF_SOUND_MM_S / (2.0 * vtl_mm)
    expected = np.array([(2*n - 1) * SPEED_OF_SOUND_MM_S / (4.0 * vtl_mm)
                         for n in range(1, n_out + 1)], dtype=np.float64)
    tol      = FORMANT_MATCH_TOLERANCE * delta_f

    poles    = [(float(raw_freqs[i]),
                 float(raw_bws[i]) if np.isfinite(raw_bws[i]) else np.nan)
                for i in range(len(raw_freqs))
                if np.isfinite(raw_freqs[i]) and raw_freqs[i] > 0]

    used = set()
    for slot, exp_f in enumerate(expected):
        best_d, best_k = np.inf, -1
        for k, (pf, _) in enumerate(poles):
            if k in used:
                continue
            d = abs(pf - exp_f)
            if d < best_d and d <= tol:
                best_d, best_k = d, k
        if best_k >= 0:
            out_f[slot] = poles[best_k][0]
            out_b[slot] = poles[best_k][1]
            used.add(best_k)

    return out_f, out_b


# ---------------------------------------------------------------------------
# Hierarchical VTL smoother
# ---------------------------------------------------------------------------

class VTLSmoother:
    """
    Blends group/speaker VTL statistics with the literature prior using
    uncertainty-weighted alphas:  alpha = n / (n + n0).

    group is a (sex, age) tuple, e.g. ("female", "adult").
    Unknown dimensions are represented as "unknown":
        ("male",    "unknown")  -- sex known, age unknown
        ("unknown", "adult")    -- age known, sex unknown
        ("unknown", "unknown")  -- neither known

    Prior fallback hierarchy (most to least specific):
        (sex, age) -> (sex, "unknown") -> ("unknown", age) -> ("unknown", "unknown")
    """

    def __init__(self, prior_strength: float = 10.0, sample_alpha: float = 0.30):
        self.n0           = prior_strength
        self.sample_alpha = sample_alpha
        self._group_vtls:   dict[tuple, list[float]] = {}
        self._speaker_vtls: dict[str,   list[float]] = {}

    def update(self, group: tuple, speaker: str, vtl_mm: float) -> None:
        if not np.isfinite(vtl_mm):
            return
        self._group_vtls.setdefault(group, []).append(vtl_mm)
        self._speaker_vtls.setdefault(speaker, []).append(vtl_mm)

    def _prior(self, group: tuple) -> float:
        """Walk the fallback chain until a known prior is found."""
        sex, age = group
        for key in (
            (sex,       age),
            (sex,       "unknown"),
            ("unknown", age),
            ("unknown", "unknown"),
        ):
            if key in VTL_PRIOR_MM:
                return VTL_PRIOR_MM[key]
        return VTL_PRIOR_MM[("unknown", "unknown")]

    def smoothed_group_vtl(self, group: tuple) -> float:
        vals = self._group_vtls.get(group, [])
        n    = len(vals)
        if n == 0:
            return self._prior(group)
        alpha = n / (n + self.n0)
        return (1.0 - alpha) * self._prior(group) + alpha * float(np.mean(vals))

    def smoothed_speaker_vtl(self, group: tuple, speaker: str) -> float:
        vals      = self._speaker_vtls.get(speaker, [])
        n         = len(vals)
        group_vtl = self.smoothed_group_vtl(group)
        if n == 0:
            return group_vtl
        alpha = n / (n + self.n0)
        return (1.0 - alpha) * group_vtl + alpha * float(np.mean(vals))

    def smooth_vtl(self, group: tuple, speaker: str, raw_vtl: float) -> float:
        """literature -> group -> speaker -> sample blend."""
        spk_vtl = self.smoothed_speaker_vtl(group, speaker)
        if not np.isfinite(raw_vtl):
            return spk_vtl
        return (1.0 - self.sample_alpha) * spk_vtl + self.sample_alpha * raw_vtl


# ---------------------------------------------------------------------------
# Core per-sample extractor — returns transposed dict directly
# ---------------------------------------------------------------------------

def _extract_sample(
    sample: dict,
    cfg: LabellerConfig,
    vtl_smoother: VTLSmoother,
) -> dict:
    """
    Slice one sample into overlapping windows and extract acoustic features.

    vtl_smoother is read-only (frozen snapshot passed from the calling pass).
    Returns a single transposed dict matching the module output schema.
    Unvoiced windows are retained with NaN for F0/periodicity so the full
    timeline is available for downstream windowed std computation.
    """
    audio    = sample["audio"].astype(np.float32)
    sr       = sample.get("sr", 16000)
    speaker  = sample.get("speaker", sample.get("group", "unknown"))
    group    = sample.get("group", speaker)
    modality = sample.get("modality", "spoken")

    win_ms      = max(cfg.win_ms, 3000.0 / cfg.min_f0_hz)
    win_s       = win_ms / 1000.0
    win_samples = min(int(round(sr * win_s)), len(audio))
    hop_samples = max(1, int(round(sr * cfg.hop_ms / 1000.0)))

    # Derive a speaker-adapted formant ceiling from the smoothed speaker VTL.
    # Formula: N_FORMANTS * delta_F  where  delta_F = c / (2L)
    # Clamped to [4500, 8000] Hz so it stays physiologically meaningful.
    # Falls back to cfg.max_formant_hz when the smoother has no VTL data yet
    # (e.g. during the seed pass when called from _worker_seed_vtl).
    speaker_vtl = vtl_smoother.smoothed_speaker_vtl(group, speaker)
    if np.isfinite(speaker_vtl) and speaker_vtl > 0:
        # (N_FORMANTS + 0.5) * delta_F: half a spacing above the top formant
        # gives headroom without inviting spurious extra poles.
        dynamic_ceiling = float(np.clip(
            (N_FORMANTS + 0.5) * SPEED_OF_SOUND_MM_S / (2.0 * speaker_vtl),
            4500.0, 8000.0,
        ))
    else:
        dynamic_ceiling = cfg.max_formant_hz

    # Pre-allocate accumulators
    t_centres, loudness_arr, slope_arr = [], [], []
    f0_arr, period_arr, f0p_arr, f0y_arr = [], [], [], []
    formant_f_rows, formant_b_rows, vtl_sample_arr = [], [], []

    start = 0
    while start + win_samples <= len(audio):
        frame      = audio[start : start + win_samples]
        t_centre_s = (start + win_samples / 2.0) / sr

        loudness = _loudness_dbfs(frame)
        voiced   = loudness >= cfg.voicing_threshold_dbfs

        snd = parselmouth.Sound(frame.astype(np.float64), sampling_frequency=sr)
        if voiced:
            f0_praat = _praat_f0(snd, cfg)
            f0_pyin  = _pyin_f0(frame, sr, cfg)
            f0       = _blend_f0(f0_praat, f0_pyin, modality, cfg)
        else:
            f0_praat = f0_pyin = f0 = np.nan

        period = _periodicity(frame, sr, f0)

        raw_freqs, raw_bws, slope = _extract_raw_formants(frame, sr, cfg, win_s, dynamic_ceiling)
        vtl_raw      = _vtl_from_formants(raw_freqs)
        smoothed_vtl = vtl_smoother.smooth_vtl(group, speaker, vtl_raw)
        freqs_out, bws_out = _assign_formant_indices(raw_freqs, raw_bws, smoothed_vtl)
        vtl_sample   = _vtl_from_formants(freqs_out)

        t_centres.append(t_centre_s)
        loudness_arr.append(loudness)
        slope_arr.append(slope)
        f0_arr.append(f0)
        period_arr.append(period)
        f0p_arr.append(f0_praat)
        f0y_arr.append(f0_pyin)
        formant_f_rows.append(freqs_out)
        formant_b_rows.append(bws_out)
        vtl_sample_arr.append(vtl_sample)

        start += hop_samples

    def _f32(lst):
        return np.array(lst, dtype=np.float32)

    return {
        "speaker":          speaker,
        "group":            group,
        "label":            sample.get("label", ""),
        "source":           sample.get("source", ""),
        "modality":         modality,
        "dialect":          sample.get("dialect", "unknown"),
        "audio":            audio,
        "window_length_ms": win_ms,
        "hop_ms":           cfg.hop_ms,
        "t_centre_s":       _f32(t_centres),
        "loudness_dbfs":    _f32(loudness_arr),
        "spectral_slope":   _f32(slope_arr),
        "f0_hz":            _f32(f0_arr),
        "periodicity":      _f32(period_arr),
        "f0_praat_hz":      _f32(f0p_arr),
        "f0_pyin_hz":       _f32(f0y_arr),
        "formant_hz":       np.stack(formant_f_rows).astype(np.float32) if formant_f_rows
                            else np.empty((0, N_FORMANTS), dtype=np.float32),
        "formant_bw_hz":    np.stack(formant_b_rows).astype(np.float32) if formant_b_rows
                            else np.empty((0, N_FORMANTS), dtype=np.float32),
        "vtl_sample_mm":    _f32(vtl_sample_arr),
    }


# ---------------------------------------------------------------------------
# Module-level picklable workers
# ---------------------------------------------------------------------------

def _worker_extract(args: tuple[dict, LabellerConfig, VTLSmoother]) -> dict:
    """Extract one sample (parallel worker). Returns transposed sample dict."""
    sample, cfg, smoother = args
    return _extract_sample(sample, cfg, smoother)


def _worker_seed_vtl(
    args: tuple[dict, LabellerConfig],
) -> tuple[str, str, float]:
    """
    Compute a robust median VTL estimate from up to 20 intensity-gated
    positions across a sample.  Used to seed the smoother before pass 0.
    Returns (group, speaker, median_vtl_mm).
    """
    s, cfg   = args
    audio    = s["audio"].astype(np.float32)
    speaker  = s.get("speaker", s.get("group", "unknown"))
    group    = s.get("group", speaker)
    sr       = s.get("sr", 16000)
    win_s    = max(cfg.win_ms, 3000.0 / cfg.min_f0_hz) / 1000.0
    duration = len(audio) / sr

    margin    = win_s / 2.0
    positions = np.linspace(margin, duration - margin, 20)
    positions = positions[positions > 0]

    try:
        snd_full  = parselmouth.Sound(audio.astype(np.float64), sampling_frequency=sr)
        intensity = snd_full.to_intensity(minimum_pitch=cfg.min_f0_hz,
                                          time_step=None, subtract_mean=False)
        ivs       = [intensity.get_value(t) for t in positions]
        positions = [t for t, iv in zip(positions, ivs)
                     if iv is not None and np.isfinite(iv)
                     and iv > cfg.intensity_threshold_db]
    except Exception:
        positions = list(positions)

    if not positions:
        n      = len(audio)
        raw_f, _, _ = _extract_raw_formants(audio[n//4 : 3*n//4], sr, cfg, win_s)
        return group, speaker, _vtl_from_formants(raw_f)

    win_samps = int(round(win_s * sr))
    vtls = []
    for t in positions:
        s0    = max(0, int((t - win_s / 2) * sr))
        frame = audio[s0 : min(len(audio), s0 + win_samps)]
        if len(frame) < win_samps // 2:
            continue
        raw_f, _, _ = _extract_raw_formants(frame, sr, cfg, win_s)
        v = _vtl_from_formants(raw_f)
        if np.isfinite(v):
            vtls.append(v)

    return group, speaker, (float(np.median(vtls)) if vtls else np.nan)


# ---------------------------------------------------------------------------
# Parallel extraction helper
# ---------------------------------------------------------------------------

def _resolve_workers(n_jobs: int) -> int:
    return os.cpu_count() or 1 if n_jobs == -1 else max(1, n_jobs)


def _parallel_map(
    worker_args: list,
    n_workers: int,
    verbose: bool,
    pass_name: str,
) -> list:
    """Map _worker_extract over worker_args in parallel, preserving order."""
    n = len(worker_args)
    if n_workers == 1:
        results = []
        for i, a in enumerate(worker_args):
            if verbose and i % max(1, n // 10) == 0:
                print(f"  [{pass_name}] {i}/{n}", end="\r")
            results.append(_worker_extract(a))
        return results

    results_by_idx: dict[int, dict] = {}
    with ProcessPoolExecutor(max_workers=n_workers) as pool:
        futures = {pool.submit(_worker_extract, a): i
                   for i, a in enumerate(worker_args)}
        done = 0
        for fut in as_completed(futures):
            results_by_idx[futures[fut]] = fut.result()
            done += 1
            if verbose and done % max(1, n // 10) == 0:
                print(f"  [{pass_name}] {done}/{n}", end="\r")

    return [results_by_idx[i] for i in range(n)]


def _make_worker_args(
    samples: list[dict],
    cfg: LabellerConfig,
    smoother: VTLSmoother,
) -> list[tuple]:
    return [(s, cfg, smoother) for s in samples]


# ---------------------------------------------------------------------------
# Three-pass dataset labeller
# ---------------------------------------------------------------------------

def label_dataset(
    samples: list[dict],
    cfg: Optional[LabellerConfig] = None,
    sr: int = 16000,
    verbose: bool = True,
) -> list[dict]:
    """
    Run the full three-pass labelling pipeline on a dataset.

    Parameters
    ----------
    samples : list of dicts from dataset.py / datasets_hf.py
    cfg     : LabellerConfig (uses defaults if None)
    sr      : sample rate (used if not present in sample dict)
    verbose : print progress

    Returns
    -------
    List of transposed sample dicts — one per input sample, matching the
    output schema in the module docstring.

    Notes
    -----
    Guard top-level script calls with ``if __name__ == "__main__":`` to
    avoid worker re-spawning on Windows/macOS (spawn start method).
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
    # Seed: multi-position VTL estimate per sample (parallel)
    # -----------------------------------------------------------------------
    if verbose:
        print("[labeller] Seeding VTL smoother...")

    seed_args    = [(s, cfg) for s in samples]
    if n_workers == 1:
        seed_results = [_worker_seed_vtl(a) for a in seed_args]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            seed_results = list(pool.map(_worker_seed_vtl, seed_args))

    smoother_raw = VTLSmoother(cfg.vtl_prior_strength, cfg.vtl_sample_alpha)
    for group, speaker, vtl in seed_results:
        smoother_raw.update(group, speaker, vtl)

    # -----------------------------------------------------------------------
    # Pass 0
    # -----------------------------------------------------------------------
    if verbose:
        print("[labeller] Pass 0 -- raw extraction...")

    pass0 = _parallel_map(
        _make_worker_args(samples, cfg, smoother_raw),
        n_workers, verbose, "pass0",
    )

    smoother_first = VTLSmoother(cfg.vtl_prior_strength, cfg.vtl_sample_alpha)
    for s_out in pass0:
        vtls = s_out["vtl_sample_mm"]
        vtls = vtls[np.isfinite(vtls)]
        if len(vtls):
            smoother_first.update(s_out["group"], s_out["speaker"], float(np.mean(vtls)))

    # -----------------------------------------------------------------------
    # Pass 1
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[labeller] Pass 1 -- refining VTL estimates...")

    pass1 = _parallel_map(
        _make_worker_args(samples, cfg, smoother_first),
        n_workers, verbose, "pass1",
    )

    smoother_second = VTLSmoother(cfg.vtl_prior_strength, cfg.vtl_sample_alpha)
    for s_out in pass1:
        vtls = s_out["vtl_sample_mm"]
        vtls = vtls[np.isfinite(vtls)]
        if len(vtls):
            smoother_second.update(s_out["group"], s_out["speaker"], float(np.mean(vtls)))

    # -----------------------------------------------------------------------
    # Pass 2: frozen per-speaker VTL, no sample-level blend
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[labeller] Pass 2 -- final formant assignment...")

    frozen_args = []
    for s in samples:
        speaker   = s.get("speaker", s.get("group", "unknown"))
        group     = s.get("group", speaker)
        final_vtl = smoother_second.smoothed_speaker_vtl(group, speaker)
        frozen    = VTLSmoother(prior_strength=cfg.vtl_prior_strength, sample_alpha=cfg.vtl_sample_alpha)
        frozen._speaker_vtls[speaker] = [final_vtl]
        frozen._group_vtls[group]     = [final_vtl]
        frozen_args.append((dict(s), cfg, frozen))

    pass2 = _parallel_map(frozen_args, n_workers, verbose, "pass2")

    if verbose:
        n_voiced = sum(
            int(np.isfinite(s["f0_hz"]).any()) for s in pass2
        )
        print(f"\n[labeller] Done. {len(pass2)} samples "
              f"({n_voiced} with at least one voiced window).")

    return pass2



# ---------------------------------------------------------------------------
# Incremental labelling: add new samples using an existing labelled dataset
# ---------------------------------------------------------------------------

def _smoother_from_labelled(
    labelled: list[dict],
    cfg: LabellerConfig,
) -> VTLSmoother:
    """
    Build a VTLSmoother pre-populated from vtl_sample_mm arrays in an
    already-labelled dataset.  Uses per-speaker means so each speaker
    contributes one effective observation to the smoother, consistent with
    how inter-pass aggregation works in label_dataset().
    """
    smoother = VTLSmoother(cfg.vtl_prior_strength, cfg.vtl_sample_alpha)
    for s in labelled:
        vtls = s["vtl_sample_mm"][np.isfinite(s["vtl_sample_mm"])]
        if len(vtls):
            smoother.update(s["group"], s["speaker"], float(np.mean(vtls)))
    return smoother


def label_incremental(
    new_samples: list[dict],
    existing: list[dict],
    cfg: Optional[LabellerConfig] = None,
    sr: int = 16000,
    refine: bool = False,
    verbose: bool = True,
) -> list[dict]:
    """
    Label new samples using the pooled VTL statistics from an already-labelled
    dataset for smoothing, without re-running the full pipeline on everything.

    When ``refine=False`` (default):
        A seed pass is first run on ``new_samples`` to estimate each new
        speaker's VTL from their own audio, making the formant ceiling
        independent of the group prior.  A single pass-2-equivalent
        extraction then runs using a smoother combining the existing dataset
        statistics with the freshly seeded speaker VTLs.  Only the newly
        labelled samples are returned.

    When ``refine=True``:
        After the initial single-pass labelling of ``new_samples``, a full
        three-pass label_dataset() run is performed over the combined dataset
        (existing + new).  The full combined result is returned, so existing
        labels are updated as well.  Use this when the new samples represent
        a meaningfully different demographic and you want the smoothed VTL
        estimates to reflect the combined distribution.

    Parameters
    ----------
    new_samples : unlabelled sample dicts (same format as label_dataset input)
    existing    : already-labelled sample dicts (output of label_dataset)
    cfg         : LabellerConfig (uses defaults if None)
    sr          : fallback sample rate
    refine      : if True, re-run the full pipeline over the combined dataset
    verbose     : print progress

    Returns
    -------
    If refine=False: list of newly labelled sample dicts (len == len(new_samples))
    If refine=True:  combined list (existing re-labelled + new), len ==
                     len(existing) + len(new_samples)
    """
    if cfg is None:
        cfg = LabellerConfig()
    print(cfg)
    for s in new_samples:
        s.setdefault("sr", sr)

    n_workers = _resolve_workers(cfg.n_jobs)

    if verbose:
        print(f"[label_incremental] {len(new_samples)} new samples, "
              f"{len(existing)} existing — refine={refine}.")

    # Build smoother from existing labelled data
    smoother = _smoother_from_labelled(existing, cfg)

    if verbose:
        groups = {}
        for s in existing:
            groups[s["group"]] = groups.get(s["group"], 0) + 1
        breakdown = ", ".join(f"{g}={n}" for g, n in sorted(
            groups.items(), key=lambda x: str(x[0])
        ))
        print(f"[label_incremental] Smoother seeded from existing: {breakdown}")

    # Seed per-speaker VTL for new speakers from their own audio before the
    # main pass.  Without this the formant ceiling is derived purely from the
    # group prior, making the ceiling — and therefore the pole assignment —
    # sensitive to the group label even when vtl_sample_alpha is high.
    if verbose:
        print("[label_incremental] Seeding VTL for new speakers...")

    seed_args    = [(s, cfg) for s in new_samples]
    if n_workers == 1:
        seed_results = [_worker_seed_vtl(a) for a in seed_args]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            seed_results = list(pool.map(_worker_seed_vtl, seed_args))

    for group, speaker, vtl in seed_results:
        smoother.update(group, speaker, vtl)

    if verbose:
        new_speakers = {s.get("speaker", "unknown") for s in new_samples}
        for spk in sorted(new_speakers):
            grp = next((s["group"] for s in new_samples
                        if s.get("speaker") == spk), ("unknown","unknown"))
            vtl = smoother.smoothed_speaker_vtl(grp, spk)
            print(f"  seeded {spk} ({grp}): VTL={vtl:.1f}mm")

    # Single pass with frozen per-speaker VTL (pass-2 equivalent)
    if verbose:
        print("[label_incremental] Labelling new samples (single pass)...")

    frozen_args = []
    for s in new_samples:
        speaker   = s.get("speaker", s.get("group", "unknown"))
        group     = s.get("group", speaker)
        final_vtl = smoother.smoothed_speaker_vtl(group, speaker)
        frozen    = VTLSmoother(prior_strength=cfg.vtl_prior_strength, sample_alpha=cfg.vtl_sample_alpha)
        frozen._speaker_vtls[speaker] = [final_vtl]
        frozen._group_vtls[group]     = [final_vtl]
        frozen_args.append((dict(s), cfg, frozen))

    new_labelled = _parallel_map(frozen_args, n_workers, verbose, "incremental")

    if verbose:
        n_voiced = sum(int(np.isfinite(s["f0_hz"]).any()) for s in new_labelled)
        print(f"\n[label_incremental] {len(new_labelled)} new samples labelled "
              f"({n_voiced} with at least one voiced window).")

    if not refine:
        return new_labelled

    # Refinement: full pipeline over combined dataset
    if verbose:
        print("[label_incremental] Running refinement pass over combined dataset...")

    # Strip labeller output keys from existing so label_dataset receives
    # clean input dicts (it only needs the raw audio + metadata fields).
    _output_keys = {
        "t_centre_s", "loudness_dbfs", "spectral_slope", "f0_hz", "periodicity",
        "f0_praat_hz", "f0_pyin_hz", "formant_hz", "formant_bw_hz",
        "vtl_sample_mm", "window_length_ms", "hop_ms",
        "outlier_flags", "outlier_mask",
    }
    def _strip(s: dict) -> dict:
        return {k: v for k, v in s.items() if k not in _output_keys}

    combined_raw = [_strip(s) for s in existing] + list(new_samples)
    return label_dataset(combined_raw, cfg=cfg, sr=sr, verbose=verbose)


# ---------------------------------------------------------------------------
# Sanity-check probe: raw single pass, one sample per group
# ---------------------------------------------------------------------------

def probe_raw_formants(
    samples: list[dict],
    cfg: Optional[LabellerConfig] = None,
    sr: int = 16000,
    verbose: bool = True,
) -> list[dict]:
    """
    Run a single raw extraction pass on one sample per group per vowel,
    with no multi-pass smoothing.  The VTL prior is the bare literature
    value for each group — no data-driven blending is applied.

    Selection rules
    ---------------
    - Only vowels present in every group are included (intersection).
      Falls back to the union with a warning if the intersection is empty.
    - One sample is chosen per (group, vowel) pair.
    - No speaker appears more than once across the entire probe set.
    - Where a vowel cannot be filled for a group without reusing a speaker,
      it is skipped and a note is printed.

    Intended for sanity checking: compare returned formant_hz / vtl_sample_mm
    against your expectations before committing to a full label_dataset() run.

    Parameters
    ----------
    samples : same list passed to label_dataset()
    cfg     : LabellerConfig (uses defaults if None)
    sr      : fallback sample rate
    verbose : print selection summary

    Returns
    -------
    List of transposed sample dicts (same schema as label_dataset() output).
    No smoothing, no multi-pass VTL correction.
    """
    if cfg is None:
        cfg = LabellerConfig()

    for s in samples:
        s.setdefault("sr", sr)

    # --- Selection: no duplicate speakers or vowels; same vowel set per group ---

    # Pass 1: find which vowels are available in every group
    vowels_by_group: dict[str, set[str]] = defaultdict(set)
    for s in samples:
        g = s.get("group", s.get("speaker", "unknown"))
        vowels_by_group[g].add(s.get("label", ""))

    shared_vowels: set[str] = set.intersection(*vowels_by_group.values()) \
        if vowels_by_group else set()

    if not shared_vowels:
        # Fall back to the union if no vowel appears in every group
        shared_vowels = set.union(*vowels_by_group.values()) if vowels_by_group else set()
        if verbose:
            print("[probe_raw_formants] WARNING: no vowels common to all groups; "
                  "using union instead")

    # Pass 2: for each group pick one sample per shared vowel,
    # never reusing the same speaker
    group_selections: dict[str, dict[str, dict]] = defaultdict(dict)  # group -> vowel -> sample
    used_speakers: set[str] = set()

    for s in samples:
        g       = s.get("group", s.get("speaker", "unknown"))
        vowel   = s.get("label", "")
        speaker = s.get("speaker", s.get("group", "unknown"))

        if vowel not in shared_vowels:
            continue
        if vowel in group_selections[g]:          # vowel already filled for this group
            continue
        if speaker in used_speakers:              # speaker already used in any group
            continue

        group_selections[g][vowel] = s
        used_speakers.add(speaker)

    selected = [s for g_sels in group_selections.values() for s in g_sels.values()]

    if verbose:
        breakdown = ", ".join(
            f"{g}={len(sels)} vowels" for g, sels in sorted(group_selections.items())
        )
        missing = {
            g: shared_vowels - set(group_selections[g].keys())
            for g in vowels_by_group
            if shared_vowels - set(group_selections[g].keys())
        }
        print(f"[probe_raw_formants] {len(selected)} samples ({breakdown})")
        if missing:
            print(f"  NOTE: some vowels could not be filled without reusing a speaker: "
                  + ", ".join(f"{g}:{v}" for g, vs in missing.items() for v in vs))

    # Build a literature-only smoother — alpha=0 at every level means the
    # smooth_vtl() call returns the literature prior unchanged for any sample
    # VTL, giving us a clean baseline without any data influence.
    lit_smoother = VTLSmoother(prior_strength=0.0, sample_alpha=0.0)

    results = []
    for s in selected:
        speaker = s.get("speaker", s.get("group", "unknown"))
        group   = s.get("group", speaker)
        if verbose:
            print(f"  probing speaker={speaker!r} group={group!r} "
                  f"label={s.get('label', '?')!r}")
        result = _extract_sample(s, cfg, lit_smoother)
        results.append(result)

    return results


def probe_full(
    samples: list[dict],
    cfg: Optional[LabellerConfig] = None,
    sr: int = 16000,
    speakers_per_group: int = 2,
    verbose: bool = True,
) -> list[dict]:
    """
    Run the full three-pass label_dataset() pipeline on a small representative
    subset: up to ``speakers_per_group`` speakers per group, all vowels.

    Selection rules
    ---------------
    - Speakers are chosen greedily in dataset order.
    - No speaker appears in more than one group's quota (global uniqueness).
    - All samples belonging to the selected speakers are included, so the
      VTL smoother sees the same multi-vowel context as a real run.

    Parameters
    ----------
    samples            : same list passed to label_dataset()
    cfg                : LabellerConfig (uses defaults if None)
    sr                 : fallback sample rate
    speakers_per_group : how many speakers to include per group (default 2)
    verbose            : passed through to label_dataset()

    Returns
    -------
    List of transposed sample dicts (same schema as label_dataset() output),
    one per selected sample.
    """
    if cfg is None:
        cfg = LabellerConfig()

    # --- Select speakers ---
    group_speakers: dict[str, list[str]] = defaultdict(list)
    used_speakers:  set[str]             = set()

    for s in samples:
        group   = s.get("group", s.get("speaker", "unknown"))
        speaker = s.get("speaker", group)
        if speaker in used_speakers:
            continue
        if len(group_speakers[group]) < speakers_per_group:
            group_speakers[group].append(speaker)
            used_speakers.add(speaker)

    if verbose:
        breakdown = ", ".join(
            f"{g}={len(spks)} speakers ({', '.join(spks)})"
            for g, spks in sorted(group_speakers.items())
        )
        print(f"[probe_full] Selected: {breakdown}")

    # --- Collect all samples for selected speakers ---
    selected = [
        s for s in samples
        if s.get("speaker", s.get("group", "unknown")) in used_speakers
    ]

    if verbose:
        vowels_by_spk = defaultdict(set)
        for s in selected:
            vowels_by_spk[s.get("speaker", "?")].add(s.get("label", "?"))
        for spk, vowels in sorted(vowels_by_spk.items()):
            print(f"  {spk}: {len(vowels)} vowels ({', '.join(sorted(vowels))})")
        print(f"[probe_full] {len(selected)} samples total — running full pipeline...")

    return label_dataset(selected, cfg=cfg, sr=sr, verbose=verbose)


# ---------------------------------------------------------------------------
# Post-processing: per-window outlier flags
# ---------------------------------------------------------------------------

@dataclass
class OutlierConfig:
    # Per-group F1 bounds (Hz).  Groups not listed fall back to the
    # _default entry.  Ceilings are set generously to cover the open
    # vowels (/a/, /ɑ/) while still catching tracker misfires.
    f1_bounds_hz: dict = None

    def __post_init__(self):
        if self.f1_bounds_hz is None:
            self.f1_bounds_hz = {
                ("male",    "adult"):   (200.0,  950.0),
                ("female",  "adult"):   (200.0, 1100.0),
                ("male",    "child"):   (200.0, 1250.0),
                ("female",  "child"):   (200.0, 1250.0),
                # collapsed dimensions — use the more permissive ceiling
                ("male",    "unknown"): (200.0, 1250.0),
                ("female",  "unknown"): (200.0, 1250.0),
                ("unknown", "adult"):   (200.0, 1100.0),
                ("unknown", "child"):   (200.0, 1250.0),
                ("unknown", "unknown"): (200.0, 1250.0),
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


def flag_outliers(
    labelled: list[dict],
    cfg: Optional[OutlierConfig] = None,
) -> list[dict]:
    """
    Add per-window outlier flag arrays to each labelled sample dict.

    Modifies the dicts in-place and returns the same list for chaining.
    Each sample gains an ``outlier_flags`` dict of named boolean (N,) arrays
    (True = outlier window), and a combined ``outlier_mask`` (N,) bool array
    that is the logical OR of all active checks.

    Checks
    ------
    f1_bounds     : F1 below floor or above group-appropriate ceiling
    vtl_range     : VTL outside speaker_mean +/- vtl_std_threshold * speaker_std
    formant_gap   : adjacent assigned formants further apart than
                    gap_threshold * the uniform-tube expected spacing for that window
    unvoiced      : f0_hz is NaN  (only added when cfg.flag_unvoiced is True)

    Parameters
    ----------
    labelled : output of label_dataset(), probe_full(), or probe_raw_formants()
    cfg      : OutlierConfig (uses defaults if None)

    Returns
    -------
    The same list, with ``outlier_flags`` and ``outlier_mask`` added in-place.
    """
    if cfg is None:
        cfg = OutlierConfig()

    # Pre-compute per-speaker VTL mean and std across all their windows
    spk_vals: dict[str, list[float]] = {}
    for s in labelled:
        vtls = s["vtl_sample_mm"]
        spk_vals.setdefault(s["speaker"], []).extend(
            vtls[np.isfinite(vtls)].tolist()
        )
    spk_mean = {spk: float(np.mean(v)) for spk, v in spk_vals.items() if v}
    spk_std  = {spk: float(np.std(v))  for spk, v in spk_vals.items() if v}

    for s in labelled:
        N       = len(s["t_centre_s"])
        speaker = s["speaker"]
        group   = s["group"]
        flags: dict[str, np.ndarray] = {}

        # --- F1 bounds ---
        f1 = s["formant_hz"][:, 0]
        sex, age = group if isinstance(group, tuple) else ("unknown", "unknown")
        f1_floor, f1_ceil = next(
            (cfg.f1_bounds_hz[k] for k in (
                (sex, age),
                (sex, "unknown"),
                ("unknown", age),
                ("unknown", "unknown"),
            ) if k in cfg.f1_bounds_hz),
            (200.0, 1250.0),
        )
        flags["f1_bounds"] = (
            np.isfinite(f1) & ((f1 < f1_floor) | (f1 > f1_ceil))
        )

        # --- VTL range ---
        vtl   = s["vtl_sample_mm"]
        mu    = spk_mean.get(speaker, np.nan)
        sigma = spk_std.get(speaker, np.nan)
        if np.isfinite(mu) and np.isfinite(sigma) and sigma > 0:
            flags["vtl_range"] = (
                np.isfinite(vtl) &
                (np.abs(vtl - mu) > cfg.vtl_std_threshold * sigma)
            )
        else:
            flags["vtl_range"] = np.zeros(N, dtype=bool)

        # --- Formant gap ---
        fhz       = s["formant_hz"]   # (N, 7)
        gap_flags = np.zeros(N, dtype=bool)
        for w in range(N):
            vtl_w = float(vtl[w])
            if not np.isfinite(vtl_w) or vtl_w <= 0:
                continue
            expected_delta = SPEED_OF_SOUND_MM_S / (2.0 * vtl_w)
            row        = fhz[w]
            valid_idx  = np.where(np.isfinite(row))[0]
            for a, b in zip(valid_idx[:-1], valid_idx[1:]):
                if row[b] - row[a] > cfg.formant_gap_threshold * expected_delta:
                    gap_flags[w] = True
                    break
        flags["formant_gap"] = gap_flags

        # --- Unvoiced ---
        if cfg.flag_unvoiced:
            flags["unvoiced"] = ~np.isfinite(s["f0_hz"])

        s["outlier_flags"] = flags
        s["outlier_mask"]  = np.logical_or.reduce(list(flags.values()))

    return labelled



def build_metadata(
    labelled: list[dict],
) -> tuple[dict[str, dict], dict[str, dict]]:
    """
    Compute unsmoothed speaker and group VTL statistics from vtl_sample_mm
    arrays in the labelled dataset.

    Returns
    -------
    speaker_meta : {speaker: {group, vtl_mean_mm, vtl_std_mm,
                               n_samples, n_windows}}
    group_meta   : {group:   {vtl_mean_mm, vtl_std_mm, n_samples,
                               n_windows, vtl_literature_mm}}
    """
    spk_vtls: dict[str, list[float]] = {}
    spk_grp:  dict[str, str]         = {}
    spk_n:    dict[str, int]         = {}

    for s in labelled:
        spk  = s["speaker"]
        grp  = s["group"]
        vtls = s["vtl_sample_mm"][np.isfinite(s["vtl_sample_mm"])]
        spk_vtls.setdefault(spk, []).extend(vtls.tolist())
        spk_grp[spk] = grp
        spk_n[spk]   = spk_n.get(spk, 0) + 1

    speaker_meta: dict[str, dict] = {}
    for spk, vtls in spk_vtls.items():
        arr = np.array(vtls)
        speaker_meta[spk] = {
            "group":       spk_grp[spk],
            "vtl_mean_mm": float(np.mean(arr)) if len(arr) else float("nan"),
            "vtl_std_mm":  float(np.std(arr))  if len(arr) else float("nan"),
            "n_samples":   spk_n[spk],
            "n_windows":   len(arr),
        }

    grp_vtls: dict[str, list[float]] = {}
    grp_spks: dict[str, set]         = {}
    for spk, meta in speaker_meta.items():
        grp = meta["group"]
        grp_vtls.setdefault(grp, []).extend(spk_vtls[spk])
        grp_spks.setdefault(grp, set()).add(spk)

    group_meta: dict[str, dict] = {}
    for grp, vtls in grp_vtls.items():
        arr = np.array(vtls)
        group_meta[grp] = {
            "vtl_mean_mm":       float(np.mean(arr)) if len(arr) else float("nan"),
            "vtl_std_mm":        float(np.std(arr))  if len(arr) else float("nan"),
            "n_samples":         len(grp_spks[grp]),
            "n_windows":         len(arr),
            "vtl_literature_mm": VTL_PRIOR_MM.get(
                grp if isinstance(grp, tuple) else ("unknown", "unknown"),
                VTL_PRIOR_MM[("unknown", "unknown")],
            ),
        }

    return speaker_meta, group_meta


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_dataset(
    labelled: list[dict],
    speaker_meta: dict,
    group_meta: dict,
    path_prefix: str,
) -> None:
    """
    Save labelled dataset to disk.

    Files created:
        <path_prefix>_samples.pkl
        <path_prefix>_speaker_meta.json
        <path_prefix>_group_meta.json
    """
    import pickle
    pkl  = f"{path_prefix}_samples.pkl"
    spk  = f"{path_prefix}_speaker_meta.json"
    grp  = f"{path_prefix}_group_meta.json"

    with open(pkl, "wb") as f:
        pickle.dump(labelled, f, protocol=pickle.HIGHEST_PROTOCOL)
    with open(spk, "w", encoding="utf-8") as f:
        json.dump(speaker_meta, f, indent=2, ensure_ascii=False)
    with open(grp, "w", encoding="utf-8") as f:
        json.dump(group_meta, f, indent=2, ensure_ascii=False)

    print(f"Saved {len(labelled)} samples        -> {pkl}")
    print(f"Speaker metadata ({len(speaker_meta)} speakers) -> {spk}")
    print(f"Group metadata   ({len(group_meta)} groups)   -> {grp}")


def load_dataset(path_prefix: str) -> tuple[list[dict], dict, dict]:
    """Load a dataset saved by save_dataset(). Returns (labelled, speaker_meta, group_meta)."""
    import pickle
    with open(f"{path_prefix}_samples.pkl", "rb") as f:
        labelled = pickle.load(f)
    with open(f"{path_prefix}_speaker_meta.json", encoding="utf-8") as f:
        speaker_meta = json.load(f)
    with open(f"{path_prefix}_group_meta.json", encoding="utf-8") as f:
        group_meta = json.load(f)
    print(f"Loaded {len(labelled)} samples from {path_prefix}_samples.pkl")
    return labelled, speaker_meta, group_meta
