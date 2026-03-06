# labeller.py
"""
Acoustic feature labeller for vowel recordings.

Takes dataset dicts produced by dataset.py / datasets_hf.py and returns
per-window acoustic labels including loudness, F0, periodicity, and
formant frequencies + bandwidths with physiologically-corrected indices.

Formant index correction
------------------------
Parselmouth's Burg tracker returns however many poles it finds in a frame
without awareness of which anatomical resonance each pole corresponds to.
At high F0 or in noisy frames a resonance can be missed entirely, causing
all higher formant indices to shift down by one — a silent but serious error.

This module corrects for that via a two-pass hierarchical VTL estimation:

  Pass 0 (raw):
    Extract formants naïvely. Estimate per-sample VTL from raw ΔF.
    Blend: literature prior → group mean → speaker mean → sample VTL.
    Use smoothed sample VTL + uniform-tube model to assign formant indices
    via minimum-cost matching (greedy over 7 formants).  → firstpass_formants

  Pass 1 (refinement):
    Re-estimate group/speaker VTL means from firstpass_formants.
    Smooth again: literature prior → group mean → speaker mean.
    Re-assign formant indices using refined per-speaker prior.       → formants

VTL blending
------------
At group and speaker level, blending weight is:
    α = n / (n + n0)
where n is the number of samples in that group/speaker and n0 is a
tuneable prior-strength hyperparameter (default 10 — "trust the literature
as much as 10 samples").

At sample level a fixed configurable alpha (default 0.3) blends the
smoothed speaker VTL toward the raw sample VTL estimate:
    vtl_sample = (1 − α_sample) * vtl_speaker_smooth + α_sample * vtl_raw_sample

Voicing / unvoiced frames
--------------------------
A frame is considered unvoiced if its loudness is below `voicing_threshold_dbfs`
(default −40 dBFS).  F0 and periodicity are set to NaN for unvoiced frames.
Unvoiced formant frames are still labelled but contribute higher uncertainty.

F0 estimation
-------------
Both Praat's autocorrelation pitch tracker and librosa's PYIN are run.
If the two estimates agree within 10 % their mean is returned.
Otherwise the modality-based preference wins: Praat for "spoken*",
PYIN for "sung*".

Window length
-------------
Default 10 ms (configurable).  A minimum window length is enforced:
    win_ms ≥ 3000 / min_f0_hz    (covers 3 pitch periods at the lowest F0)
If the requested window is shorter than this it is silently extended.

Output schema (per window dict)
-------------------------------
    speaker          : str
    group            : str
    label            : str           IPA vowel label
    source           : str
    modality         : str
    dialect          : str
    window_length_ms : float         actual window duration in ms
    t_centre_s       : float         centre time within the source sample (s)

    loudness_dbfs    : float
    f0_hz            : float | NaN   (NaN if unvoiced)
    periodicity      : float | NaN   (normalised autocorrelation, NaN if unvoiced)
    f0_praat_hz      : float | NaN
    f0_pyin_hz       : float | NaN

    # 7 formant frequencies and bandwidths (NaN where index unassignable)
    f1_hz … f7_hz    : float | NaN
    b1_hz … b7_hz    : float | NaN

    # Intra-sample standard deviations (uncertainty for loss weighting)
    loudness_dbfs_std : float
    f0_hz_std         : float
    f1_hz_std … f7_hz_std : float

    vtl_mm           : float         final smoothed sample VTL in mm
    vtl_raw_mm       : float         raw sample VTL estimate in mm
"""

from __future__ import annotations

import os
import warnings
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import parselmouth
from parselmouth.praat import call
import librosa

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

SPEED_OF_SOUND_MM_S = 343_000.0   # mm/s at ~20 °C

# Literature VTL priors (mm) — from Fitch & Giedd (1999) / Story (2005)
# Used as the starting prior; blended toward data as samples accumulate.
VTL_PRIOR_MM: dict[str, float] = {
    "men":     174.0,
    "women":   148.0,
    "boys":    128.0,
    "girls":   123.0,
    # fallback for unknown / other group labels
    "_default": 148.0,
}

# Tolerance for formant-to-expected-position matching, as a fraction of ΔF.
# A Praat pole must be within this fraction of the expected spacing to be
# assigned to a formant slot; otherwise the slot is filled with NaN.
FORMANT_MATCH_TOLERANCE = 0.45   # ±45 % of ΔF

N_FORMANTS = 7


# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

@dataclass
class LabellerConfig:
    # Window parameters
    win_ms: float = 10.0          # target window length (ms)
    hop_ms: float = 5.0           # hop between windows (ms)
    min_f0_hz: float = 50.0       # lowest expected F0; sets minimum window length
    max_f0_hz: float = 600.0      # highest expected F0

    # Formant extraction
    max_formant_hz: float = 5500.0  # Praat ceiling (use 5000 for male, 5500 female)
    n_praat_formants: int = N_FORMANTS + 2   # ask Praat for extra poles as headroom

    # VTL blending
    vtl_prior_strength: float = 10.0   # n0: treat literature prior as this many samples
    vtl_sample_alpha: float = 0.30     # fixed alpha for sample-level VTL blend

    # Voicing
    voicing_threshold_dbfs: float = -40.0

    # F0 agreement threshold (fractional difference)
    f0_agreement_threshold: float = 0.10

    # Parallelism — number of worker processes.
    # -1 = use all logical CPUs (default), 1 = serial (useful for debugging).
    n_jobs: int = -1


# ---------------------------------------------------------------------------
# Low-level acoustic helpers
# ---------------------------------------------------------------------------

def _loudness_dbfs(frame: np.ndarray) -> float:
    """RMS loudness in dBFS.  Returns -inf for silent frames."""
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


def _blend_f0(f0_praat: float, f0_pyin: float, modality: str,
              cfg: LabellerConfig) -> float:
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
    # Disagreement or one missing — use modality preference
    if modality.startswith("sung"):
        return f0_pyin if np.isfinite(f0_pyin) else f0_praat
    else:
        return f0_praat if np.isfinite(f0_praat) else f0_pyin


# ---------------------------------------------------------------------------
# Praat formant + bandwidth extraction (raw poles, no relabelling yet)
# ---------------------------------------------------------------------------

def _extract_raw_formants(
    snd: parselmouth.Sound,
    cfg: LabellerConfig,
    win_s: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Run Praat's FormantPath tracker on a Sound object.

    Returns
    -------
    freqs : (n_praat_formants,) float32   — NaN where Praat has no estimate
    bws   : (n_praat_formants,) float32
    """
    try:
        # FormantPath tries several ceiling values and picks lowest-stress solution
        formant_path = call(
            snd,
            "To FormantPath (burg)...",
            0.0,                        # time step (0 = auto)
            cfg.n_praat_formants,       # max number of formants
            cfg.max_formant_hz,         # ceiling
            win_s,                      # window length
            50.0,                       # pre-emphasis from (Hz)
            0.05,                       # ceiling step fraction
            4,                          # number of steps on either side
        )
        formant = call(formant_path, "Extract Formant")
    except Exception:
        # FormantPath may not be available in older parselmouth — fall back
        try:
            formant = snd.to_formant_burg(
                time_step=0.0,
                max_number_of_formants=cfg.n_praat_formants,
                maximum_formant=cfg.max_formant_hz,
                window_length=win_s,
                pre_emphasis_from=50.0,
            )
        except Exception:
            return (
                np.full(cfg.n_praat_formants, np.nan, dtype=np.float32),
                np.full(cfg.n_praat_formants, np.nan, dtype=np.float32),
            )

    # Use the middle time point of the window
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

    return freqs, bws


# ---------------------------------------------------------------------------
# VTL estimation from raw formants
# ---------------------------------------------------------------------------

def _vtl_from_formants(freqs: np.ndarray) -> float:
    """
    Estimate vocal tract length from a set of formant frequencies using the
    uniform-tube formula:  Fₙ = (2n−1) * c / (4L)
    => L = (2n−1) * c / (4 * Fₙ)

    We take the median estimate across all valid formants to be robust to
    individual tracking errors.  Returns NaN if no valid formants.
    """
    estimates = []
    for i, f in enumerate(freqs):
        if np.isfinite(f) and f > 0:
            n = i + 1   # formant number (1-indexed)
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
    Given raw Praat poles and a VTL estimate, assign each pole to the
    anatomically correct formant slot using greedy minimum-distance matching.

    Expected formant positions from uniform-tube model:
        F_expected[n] = (2n−1) * c / (4L)   for n = 1, 2, …, n_out

    Each raw pole is matched to the nearest expected position, subject to:
      - Each slot assigned at most once
      - The pole must be within ±FORMANT_MATCH_TOLERANCE * ΔF of the expected
        position (ΔF ≈ c / (2L) is the uniform-tube spacing)

    Unmatched slots are filled with NaN.

    Returns
    -------
    freqs_out : (n_out,) float32
    bws_out   : (n_out,) float32
    """
    if not np.isfinite(vtl_mm) or vtl_mm <= 0:
        # No VTL estimate — pass through as many raw poles as we have
        out_f = np.full(n_out, np.nan, dtype=np.float32)
        out_b = np.full(n_out, np.nan, dtype=np.float32)
        valid = raw_freqs[np.isfinite(raw_freqs)][:n_out]
        valid_b = raw_bws[np.isfinite(raw_bws)][:n_out]
        out_f[:len(valid)] = valid
        out_b[:len(valid_b)] = valid_b
        return out_f, out_b

    delta_f = SPEED_OF_SOUND_MM_S / (2.0 * vtl_mm)
    expected = np.array(
        [(2 * n - 1) * SPEED_OF_SOUND_MM_S / (4.0 * vtl_mm) for n in range(1, n_out + 1)],
        dtype=np.float64,
    )
    tol = FORMANT_MATCH_TOLERANCE * delta_f

    # Collect valid (freq, bw, original_index) poles
    poles = [
        (float(raw_freqs[i]), float(raw_bws[i]) if np.isfinite(raw_bws[i]) else np.nan, i)
        for i in range(len(raw_freqs))
        if np.isfinite(raw_freqs[i]) and raw_freqs[i] > 0
    ]

    out_f = np.full(n_out, np.nan, dtype=np.float32)
    out_b = np.full(n_out, np.nan, dtype=np.float32)
    assigned_poles = set()

    for slot_idx, exp_f in enumerate(expected):
        best_dist = np.inf
        best_pole_idx = -1
        for k, (pf, pb, orig_i) in enumerate(poles):
            if k in assigned_poles:
                continue
            dist = abs(pf - exp_f)
            if dist < best_dist and dist <= tol:
                best_dist = dist
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

    Blending weight:  α = n / (n + n0)
    where n = number of samples seen for that group/speaker,
          n0 = prior_strength (default 10)

    Usage
    -----
    Call update() once per sample with the sample's raw VTL estimate.
    Call smooth_vtl() to get the smoothed VTL for a sample.
    """

    def __init__(self, prior_strength: float = 10.0, sample_alpha: float = 0.30):
        self.n0 = prior_strength
        self.sample_alpha = sample_alpha
        # {key: [vtl_values]}
        self._group_vtls: dict[str, list[float]] = {}
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
        n = len(vals)
        if n == 0:
            return self._group_prior(group)
        alpha = n / (n + self.n0)
        return (1.0 - alpha) * self._group_prior(group) + alpha * float(np.mean(vals))

    def smoothed_speaker_vtl(self, group: str, speaker: str) -> float:
        vals = self._speaker_vtls.get(speaker, [])
        n = len(vals)
        group_vtl = self.smoothed_group_vtl(group)
        if n == 0:
            return group_vtl
        alpha = n / (n + self.n0)
        return (1.0 - alpha) * group_vtl + alpha * float(np.mean(vals))

    def smooth_vtl(self, group: str, speaker: str, raw_sample_vtl: float) -> float:
        """
        Blend hierarchy: literature → group → speaker → sample.
        Sample-level blend uses fixed alpha.
        """
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

    Parameters
    ----------
    sample       : dataset dict (audio, label, speaker, group, …)
    cfg          : LabellerConfig
    vtl_smoother : pre-populated VTLSmoother (used for formant correction)
    pass_label   : "raw" | "first" | "second" — controls which VTL source
                   is used for formant assignment

    Returns
    -------
    List of window dicts.  Does NOT include intra-sample std fields —
    those are added by the caller after all windows for a sample are collected.
    """
    audio    = sample["audio"].astype(np.float32)
    sr       = sample.get("sr", 16000)
    speaker  = sample.get("speaker", sample.get("group", "unknown"))
    group    = sample.get("group", speaker)
    modality = sample.get("modality", "spoken")

    # Enforce minimum window length to cover 3 pitch periods
    min_win_ms = 3000.0 / cfg.min_f0_hz
    win_ms     = max(cfg.win_ms, min_win_ms)
    win_s      = win_ms / 1000.0

    win_samples = int(round(sr * win_s))
    hop_samples = int(round(sr * cfg.hop_ms / 1000.0))
    if win_samples > len(audio):
        win_samples = len(audio)
    if hop_samples < 1:
        hop_samples = 1

    windows = []
    start = 0
    while start + win_samples <= len(audio):
        frame = audio[start : start + win_samples]
        t_centre_s = (start + win_samples / 2) / sr

        # --- Loudness ---
        loudness = _loudness_dbfs(frame)
        voiced   = loudness >= cfg.voicing_threshold_dbfs

        # --- F0 ---
        snd = parselmouth.Sound(frame, sampling_frequency=sr)
        if voiced:
            f0_praat = _praat_f0(snd, cfg)
            f0_pyin  = _pyin_f0(frame, sr, cfg)
            f0       = _blend_f0(f0_praat, f0_pyin, modality, cfg)
        else:
            f0_praat = f0_pyin = f0 = np.nan

        # --- Periodicity ---
        period = _periodicity(frame, sr, f0)

        # --- Raw formant poles ---
        raw_freqs, raw_bws = _extract_raw_formants(snd, cfg, win_s)

        # --- VTL from raw poles ---
        raw_vtl = _vtl_from_formants(raw_freqs)

        # --- Smoothed VTL for formant assignment ---
        smoothed_vtl = vtl_smoother.smooth_vtl(group, speaker, raw_vtl)

        # --- Assign formant indices ---
        freqs_out, bws_out = _assign_formant_indices(
            raw_freqs, raw_bws, smoothed_vtl, n_out=N_FORMANTS
        )

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
            "f0_hz":            f0,
            "periodicity":      period,
            "f0_praat_hz":      f0_praat,
            "f0_pyin_hz":       f0_pyin,
            "vtl_mm":           smoothed_vtl,
            "vtl_raw_mm":       raw_vtl,
        }
        for i in range(N_FORMANTS):
            w[f"f{i+1}_hz"] = float(freqs_out[i]) if np.isfinite(freqs_out[i]) else np.nan
            w[f"b{i+1}_hz"] = float(bws_out[i])   if np.isfinite(bws_out[i])   else np.nan

        windows.append(w)
        start += hop_samples

    return windows


def _add_intra_sample_stds(windows: list[dict]) -> list[dict]:
    """
    Compute per-sample standard deviations over all windows from the same
    sample and add them as *_std fields to each window dict.

    Fields covered: loudness_dbfs, f0_hz, f1_hz … f7_hz
    """
    if not windows:
        return windows

    fields = ["loudness_dbfs", "f0_hz"] + [f"f{i+1}_hz" for i in range(N_FORMANTS)]

    stds = {}
    for f in fields:
        vals = np.array([w[f] for w in windows], dtype=np.float64)
        stds[f"{f}_std"] = float(np.nanstd(vals))

    for w in windows:
        w.update(stds)

    return windows


# ---------------------------------------------------------------------------
# Top-level picklable worker  (must be module-level for ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _worker_extract(
    args: tuple[dict, LabellerConfig, VTLSmoother, str],
) -> list[dict]:
    """
    Worker function: extract windows for one sample and add intra-sample stds.

    Receives a (sample, cfg, smoother, pass_label) tuple so it can be
    dispatched via ProcessPoolExecutor.map without lambda pickling issues.
    The smoother is passed by value (pickled read-only snapshot).
    """
    sample, cfg, smoother, pass_label = args
    wins = _extract_sample_windows(sample, cfg, smoother, pass_label)
    return _add_intra_sample_stds(wins)


def _worker_seed_vtl(
    args: tuple[dict, LabellerConfig],
) -> tuple[str, str, float]:
    """
    Worker function: compute a single centre-window VTL estimate for seeding.
    Returns (group, speaker, vtl_mm).
    """
    s, cfg = args
    audio   = s["audio"].astype(np.float32)
    speaker = s.get("speaker", s.get("group", "unknown"))
    group   = s.get("group", speaker)
    sr      = s.get("sr", 16000)
    n       = len(audio)
    centre  = audio[n // 4 : 3 * n // 4]
    snd     = parselmouth.Sound(centre, sampling_frequency=sr)
    win_s   = max(cfg.win_ms, 3000.0 / cfg.min_f0_hz) / 1000.0
    raw_f, _ = _extract_raw_formants(snd, cfg, win_s)
    return group, speaker, _vtl_from_formants(raw_f)


# ---------------------------------------------------------------------------
# Two-pass dataset labeller
# ---------------------------------------------------------------------------

def _resolve_workers(n_jobs: int) -> int:
    """Translate n_jobs=-1 to cpu_count(); clamp to at least 1."""
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
    """
    Fan out _worker_extract across all samples using ProcessPoolExecutor.
    Preserves sample order.  Falls back to serial on n_workers==1.
    """
    n = len(samples)
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
            idx = futures[fut]
            results_by_idx[idx] = fut.result()
            done += 1
            if verbose and done % max(1, n // 10) == 0:
                print(f"  [{pass_name}] {done}/{n}", end="\r")

    return [results_by_idx[i] for i in range(n)]


def label_dataset(
    samples: list[dict],
    cfg: Optional[LabellerConfig] = None,
    sr: int = 16000,
    verbose: bool = True,
) -> list[dict]:
    """
    Run the full two-pass labelling pipeline on a dataset.

    Parallelism is controlled by cfg.n_jobs:
        -1  use all logical CPUs  (default)
         1  serial — useful for debugging / profiling
         N  use exactly N worker processes

    The three extraction passes (pass 0, pass 1, pass 2) are each
    parallelised at the sample level.  The VTL smoother seeding and
    inter-pass aggregation steps are serial but fast (no Praat calls
    per window, just one centre-window scan per sample for seeding).

    Parameters
    ----------
    samples : list of dicts from dataset.py / datasets_hf.py
    cfg     : LabellerConfig (uses defaults if None)
    sr      : sample rate (used if not present in sample dict)
    verbose : print progress

    Returns
    -------
    List of per-window label dicts (see module docstring for schema).
    """
    if cfg is None:
        cfg = LabellerConfig()

    n_workers = _resolve_workers(cfg.n_jobs)

    # Attach sr to samples that don't have it
    for s in samples:
        s.setdefault("sr", sr)

    if verbose:
        print(f"[labeller] {len(samples)} samples, "
              f"{n_workers} worker{'s' if n_workers != 1 else ''}.")

    # -----------------------------------------------------------------------
    # Seed pass: one fast centre-window VTL estimate per sample (parallel)
    # -----------------------------------------------------------------------
    if verbose:
        print("[labeller] Seeding VTL smoother…")

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
    # Pass 0: full window extraction with seed smoother
    # -----------------------------------------------------------------------
    if verbose:
        print(f"[labeller] Pass 0 — raw extraction…")

    pass0_by_sample = _parallel_extract(
        samples, cfg, smoother_raw, "raw", n_workers, verbose, "pass0"
    )

    # -----------------------------------------------------------------------
    # Inter-pass aggregation: build pass-1 smoother from pass-0 VTLs (serial)
    # -----------------------------------------------------------------------
    smoother_first = VTLSmoother(
        prior_strength=cfg.vtl_prior_strength,
        sample_alpha=cfg.vtl_sample_alpha,
    )
    for wins in pass0_by_sample:
        if not wins:
            continue
        speaker = wins[0]["speaker"]
        group   = wins[0]["group"]
        vtls    = [w["vtl_mm"] for w in wins if np.isfinite(w["vtl_mm"])]
        if vtls:
            smoother_first.update(group, speaker, float(np.mean(vtls)))

    # -----------------------------------------------------------------------
    # Pass 1: re-extract with refined smoother
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[labeller] Pass 1 — refining VTL estimates…")

    pass1_by_sample = _parallel_extract(
        samples, cfg, smoother_first, "first", n_workers, verbose, "pass1"
    )

    # -----------------------------------------------------------------------
    # Inter-pass aggregation: build pass-2 smoother from pass-1 formants
    # -----------------------------------------------------------------------
    smoother_second = VTLSmoother(
        prior_strength=cfg.vtl_prior_strength,
        sample_alpha=cfg.vtl_sample_alpha,
    )
    for wins in pass1_by_sample:
        if not wins:
            continue
        speaker = wins[0]["speaker"]
        group   = wins[0]["group"]
        sample_vtls = []
        for w in wins:
            f_vec = np.array(
                [w.get(f"f{i+1}_hz", np.nan) for i in range(N_FORMANTS)],
                dtype=np.float32,
            )
            v = _vtl_from_formants(f_vec)
            if np.isfinite(v):
                sample_vtls.append(v)
        if sample_vtls:
            smoother_second.update(group, speaker, float(np.mean(sample_vtls)))

    # -----------------------------------------------------------------------
    # Pass 2: final extraction
    # For each sample we freeze the smoother at the per-speaker VTL
    # (no sample-level blend in the final pass) and store that value.
    # -----------------------------------------------------------------------
    if verbose:
        print(f"\n[labeller] Pass 2 — final formant assignment…")

    # Pre-compute frozen per-sample smoothers and inject final_vtl into each
    # sample dict so the worker can stamp it onto windows without needing
    # the live smoother object.
    samples_pass2 = []
    for s in samples:
        speaker   = s.get("speaker", s.get("group", "unknown"))
        group     = s.get("group", speaker)
        final_vtl = smoother_second.smoothed_speaker_vtl(group, speaker)
        frozen    = VTLSmoother(prior_strength=1e9, sample_alpha=0.0)
        frozen._speaker_vtls[speaker] = [final_vtl]
        frozen._group_vtls[group]     = [final_vtl]
        # Shallow copy to avoid mutating caller's dict
        s2 = dict(s, _final_vtl=final_vtl)
        samples_pass2.append((s2, cfg, frozen, "second"))

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
                idx = futures[fut]
                results_by_idx[idx] = fut.result()
                done += 1
                if verbose and done % max(1, len(samples) // 10) == 0:
                    print(f"  [pass2] {done}/{len(samples)}", end="\r")
        pass2_by_sample = [results_by_idx[i] for i in range(len(samples))]

    # Stamp final_vtl onto every window and flatten
    final_windows: list[dict] = []
    for s_orig, wins in zip(samples, pass2_by_sample):
        speaker   = s_orig.get("speaker", s_orig.get("group", "unknown"))
        group     = s_orig.get("group", speaker)
        final_vtl = smoother_second.smoothed_speaker_vtl(group, speaker)
        for w in wins:
            w["vtl_mm"] = final_vtl
        final_windows.extend(wins)

    if verbose:
        n_voiced = sum(1 for w in final_windows if np.isfinite(w["f0_hz"]))
        print(f"\n[labeller] Done. {len(final_windows)} windows "
              f"({n_voiced} voiced, "
              f"{len(final_windows) - n_voiced} unvoiced) "
              f"from {len(samples)} samples.")

    return final_windows


# ---------------------------------------------------------------------------
# Convenience: flat numpy arrays for model input
# ---------------------------------------------------------------------------

FEATURE_COLS = (
    ["loudness_dbfs", "f0_hz", "periodicity"]
    + [f"f{i+1}_hz" for i in range(N_FORMANTS)]
    + [f"b{i+1}_hz" for i in range(N_FORMANTS)]
)

UNCERTAINTY_COLS = (
    ["loudness_dbfs_std", "f0_hz_std"]
    + [f"f{i+1}_hz_std" for i in range(N_FORMANTS)]
)


# ---------------------------------------------------------------------------
# Persistence helpers
# ---------------------------------------------------------------------------

def save_windows(windows: list[dict], path: str) -> None:
    """Pickle a labelled window list to disk."""
    import pickle
    with open(path, "wb") as f:
        pickle.dump(windows, f, protocol=pickle.HIGHEST_PROTOCOL)
    print(f"Saved {len(windows)} windows to {path}")


def load_windows(path: str) -> list[dict]:
    """Load a pickled window list from disk."""
    import pickle
    with open(path, "rb") as f:
        windows = pickle.load(f)
    print(f"Loaded {len(windows)} windows from {path}")
    return windows


def to_arrays(
    windows: list[dict],
) -> tuple[np.ndarray, np.ndarray, np.ndarray, list[str]]:
    """
    Convert window dicts to numpy arrays suitable for model input.

    Returns
    -------
    X          : (N, F) float32   — acoustic features (NaN for unobservable)
    U          : (N, U) float32   — intra-sample std (uncertainty weights)
    labels     : (N,)  object     — IPA label strings
    feat_names : list[str]        — column names for X
    """
    X = np.array(
        [[w.get(c, np.nan) for c in FEATURE_COLS] for w in windows],
        dtype=np.float32,
    )
    U = np.array(
        [[w.get(c, np.nan) for c in UNCERTAINTY_COLS] for w in windows],
        dtype=np.float32,
    )
    labels = np.array([w["label"] for w in windows], dtype=object)
    return X, U, labels, FEATURE_COLS
