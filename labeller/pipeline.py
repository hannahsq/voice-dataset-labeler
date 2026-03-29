"""
Core extraction, parallel workers, and the convergence-based Bayesian
labelling pipeline.
"""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import parselmouth

from .config import LabellerConfig, N_FORMANTS, SPEED_OF_SOUND_MM_S, VTL_PRIOR_MM, _window_params
from .types import SpeakerMeta, Modality
from .acoustic import (
    _loudness_dbfs,
    _periodicity,
    _praat_f0,
    _praat_f0_all,
)
from .formants import (
    _vtl_from_formants,
    _assign_formant_indices,
)
from .bayesian_vtl import (
    BayesianVTLEstimator,
    NormalPosterior,
    fit_vtl_mixture,
    fit_vtl_single,
    _SampleObs,
    _iqr_var,
)
from extractors import FormantExtractor, PraatFormantExtractor


# ---------------------------------------------------------------------------
# Shared helpers: formant ceiling and pole count from a VTL estimate
# ---------------------------------------------------------------------------

def _formant_ceiling_hz(vtl_mm: float) -> float:
    """
    Praat max-formant ceiling for a speaker with the given VTL.
    (N_FORMANTS + 0.5) * delta_F gives half a spacing of headroom above
    the top expected formant, suppressing spurious high-frequency poles.
    Clamped to [4500, 8000] Hz.
    """
    delta_f = SPEED_OF_SOUND_MM_S / (2.0 * vtl_mm)
    return float(np.clip((N_FORMANTS + 0.5) * delta_f, 4500.0, 8000.0))


def _n_praat_formants(vtl_mm: float, nyquist_hz: float, cfg_max: int) -> int:
    """
    Number of Praat poles to request: floor(nyquist / delta_F), clipped to
    [1, cfg_max].  Prevents Praat from fitting spurious poles in a spectral
    region where no real formants can exist.
    """
    delta_f = SPEED_OF_SOUND_MM_S / (2.0 * vtl_mm)
    return max(1, min(cfg_max, int(np.floor(nyquist_hz / delta_f))))


# ---------------------------------------------------------------------------
# Core per-sample extractor — returns transposed dict directly
# ---------------------------------------------------------------------------

def _extract_sample(
    sample: dict,
    cfg: LabellerConfig,
    speaker_vtl_mm: float,
    extractor: FormantExtractor,
) -> dict:
    """
    Slice one sample into overlapping windows and extract acoustic features.

    speaker_vtl_mm is the current Bayesian posterior mean for this speaker's
    VTL.  It drives the dynamic formant ceiling and pole assignment.

    Returns a single transposed dict matching the module output schema.
    Includes a transient 'vtl_per_formant' field (W, N_FORMANTS) for
    downstream GMM fitting; callers should strip this before final output.
    """
    audio    = sample["audio"].astype(np.float32)
    sr       = sample.get("sr", 16000)
    speaker  = sample.get("speaker", "unknown")
    meta     = sample.get("speaker_meta")
    group    = (meta.vtl_class if isinstance(meta, SpeakerMeta)
                else sample.get("group", "unknown"))
    _mod     = sample.get("modality", "spoken")
    modality = _mod.modality if isinstance(_mod, Modality) else str(_mod)

    win_samples, hop_samples, win_ms = _window_params(
        len(audio), sr, cfg.win_ms, cfg.hop_ms, cfg.min_f0_hz)
    win_s = win_ms / 1000.0

    # Dynamic formant ceiling from Bayesian speaker VTL posterior
    if np.isfinite(speaker_vtl_mm) and speaker_vtl_mm > 0:
        dynamic_ceiling = _formant_ceiling_hz(speaker_vtl_mm)
    else:
        dynamic_ceiling = cfg.max_formant_hz

    # Build frame centres
    t_centres = []
    start = 0
    while start + win_samples <= len(audio):
        t_centres.append((start + win_samples / 2.0) / sr)
        start += hop_samples
    t_centres_arr = np.array(t_centres, dtype=np.float32)
    W = len(t_centres)

    # Nyquist-clipped formant count
    nyquist_hz = sr / 2.0
    if np.isfinite(speaker_vtl_mm) and speaker_vtl_mm > 0:
        n_praat = _n_praat_formants(speaker_vtl_mm, nyquist_hz, cfg.n_praat_formants)
    else:
        n_praat = cfg.n_praat_formants

    # --- Formant extraction: whole-sample when extractor supports it ---
    if hasattr(extractor, "extract_all"):
        raw_freqs_all, raw_bws_all, slope_arr_raw = extractor.extract_all(
            audio, sr, t_centres_arr, win_s,
            max_formant_hz=dynamic_ceiling, n_formants=n_praat,
        )
    else:
        # Per-frame fallback (TVLP or other extractors)
        raw_freqs_all = np.full((W, N_FORMANTS), np.nan, dtype=np.float32)
        raw_bws_all   = np.full((W, N_FORMANTS), np.nan, dtype=np.float32)
        slope_arr_raw = np.full(W, np.nan, dtype=np.float32)
        start = 0
        for w in range(W):
            frame = audio[start : start + win_samples]
            rf, rb, sl = extractor(
                frame, sr, win_s, max_formant_hz=dynamic_ceiling, n_formants=n_praat)
            raw_freqs_all[w, :len(rf)] = rf
            raw_bws_all[w, :len(rb)]   = rb
            slope_arr_raw[w]           = sl
            start += hop_samples

    # --- Whole-sample F0 extraction (single Praat call) ---
    f0_praat_all = _praat_f0_all(audio, sr, cfg, t_centres_arr)

    # --- Per-window: loudness, F0, periodicity, formant assignment ---
    loudness_arr = np.empty(W, dtype=np.float32)
    f0_arr       = np.empty(W, dtype=np.float32)
    period_arr   = np.empty(W, dtype=np.float32)
    formant_f_rows, formant_b_rows, vtl_sample_arr = [], [], []
    vtl_per_formant = np.full((W, N_FORMANTS), np.nan, dtype=np.float32)

    start = 0
    for w in range(W):
        frame    = audio[start : start + win_samples]
        loudness = _loudness_dbfs(frame)
        voiced   = loudness >= cfg.voicing_threshold_dbfs

        if voiced:
            f0 = float(f0_praat_all[w])
            if not np.isfinite(f0):
                f0 = np.nan
        else:
            f0 = np.nan
            f0_praat_all[w] = np.nan

        period = _periodicity(frame, sr, f0)

        raw_freqs = raw_freqs_all[w]
        raw_bws   = raw_bws_all[w]

        # Use speaker posterior mean for formant assignment (no per-window smoothing)
        freqs_out, bws_out = _assign_formant_indices(
            raw_freqs, raw_bws, speaker_vtl_mm, nyquist_hz=nyquist_hz)
        vtl_sample = _vtl_from_formants(freqs_out)

        # Per-formant VTL for GMM (from raw poles, not assigned)
        vtl_pf = _vtl_from_formants(raw_freqs, return_all=True)
        n_pf = min(len(vtl_pf), N_FORMANTS)
        vtl_per_formant[w, :n_pf] = vtl_pf[:n_pf].astype(np.float32)

        loudness_arr[w] = loudness
        f0_arr[w]       = f0
        period_arr[w]   = period
        formant_f_rows.append(freqs_out)
        formant_b_rows.append(bws_out)
        vtl_sample_arr.append(vtl_sample)

        start += hop_samples

    slope_arr = slope_arr_raw

    def _f32(lst):
        return np.array(lst, dtype=np.float32)

    return {
        "speaker":          speaker,
        "speaker_meta":     sample.get("speaker_meta"),
        "group":            group,
        "label":            sample.get("label", ""),
        "source":           sample.get("source", ""),
        "modality":         modality,
        "dialect":          sample.get("dialect", "unknown"),
        "audio":            audio,
        "sr":               sr,
        "window_length_ms": win_ms,
        "hop_ms":           cfg.hop_ms,
        "t_centre_s":       _f32(t_centres),
        "loudness_dbfs":    loudness_arr,
        "spectral_slope":   _f32(slope_arr),
        "f0_hz":            f0_arr,
        "periodicity":      period_arr,
        "f0_praat_hz":      f0_praat_all,
        "formant_hz":       np.stack(formant_f_rows).astype(np.float32) if formant_f_rows
                            else np.empty((0, N_FORMANTS), dtype=np.float32),
        "formant_bw_hz":    np.stack(formant_b_rows).astype(np.float32) if formant_b_rows
                            else np.empty((0, N_FORMANTS), dtype=np.float32),
        "vtl_sample_mm":    _f32(vtl_sample_arr),
        "vtl_per_formant":  vtl_per_formant,  # transient — stripped before final output
    }


# ---------------------------------------------------------------------------
# Module-level picklable workers
# ---------------------------------------------------------------------------

def _worker_extract(args: tuple[dict, LabellerConfig, float, FormantExtractor]) -> dict:
    """Extract one sample (parallel worker). Returns transposed sample dict."""
    sample, cfg, speaker_vtl_mm, extractor = args
    return _extract_sample(sample, cfg, speaker_vtl_mm, extractor)


def _worker_seed_vtl(
    args: tuple[dict, LabellerConfig, FormantExtractor],
) -> tuple[str, str, float, float]:
    """
    Compute a robust median VTL estimate from up to 20 intensity-gated
    positions across a sample.  Used to seed the estimator before the
    convergence loop.
    Returns (group, speaker, median_vtl_mm, var_vtl_mm).
    """
    s, cfg, extractor = args
    audio    = s["audio"].astype(np.float32)
    speaker  = s.get("speaker", "unknown")
    meta     = s.get("speaker_meta")
    group    = (meta.vtl_class if isinstance(meta, SpeakerMeta)
                else s.get("group", "unknown"))
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
        prior_vtl = VTL_PRIOR_MM.get(group, VTL_PRIOR_MM["unknown"])
        ceiling   = _formant_ceiling_hz(prior_vtl)
        n_seed    = _n_praat_formants(prior_vtl, sr / 2.0, cfg.n_praat_formants)
        n         = len(audio)
        raw_f, _, _ = extractor(audio[n//4 : 3*n//4], sr, win_s,
                                max_formant_hz=ceiling, n_formants=n_seed)
        v = _vtl_from_formants(raw_f)
        return group, speaker, v, 225.0  # no variance info, use wide default

    prior_vtl = VTL_PRIOR_MM.get(group, VTL_PRIOR_MM["unknown"])
    ceiling   = _formant_ceiling_hz(prior_vtl)
    n_seed    = _n_praat_formants(prior_vtl, sr / 2.0, cfg.n_praat_formants)

    win_samps = int(round(win_s * sr))
    vtls = []
    for t in positions:
        s0    = max(0, int((t - win_s / 2) * sr))
        frame = audio[s0 : min(len(audio), s0 + win_samps)]
        if len(frame) < win_samps // 2:
            continue
        raw_f, _, _ = extractor(frame, sr, win_s, max_formant_hz=ceiling, n_formants=n_seed)
        v = _vtl_from_formants(raw_f)
        if np.isfinite(v):
            vtls.append(v)

    if not vtls:
        return group, speaker, np.nan, 225.0
    median_vtl = float(np.median(vtls))
    var_vtl = _iqr_var(np.array(vtls))
    return group, speaker, median_vtl, var_vtl


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
    estimator: BayesianVTLEstimator,
    extractor: FormantExtractor,
    use_group_ceiling: bool = False,
) -> list[tuple]:
    """Build worker args using speaker VTL from the Bayesian estimator.

    If use_group_ceiling is True, use the group prior VTL for the ceiling
    instead of the per-speaker estimate.  This breaks the ceiling feedback
    loop on iteration 0 where speaker estimates may be unreliable.
    """
    args = []
    for s in samples:
        speaker = s.get("speaker", "unknown")
        if use_group_ceiling:
            meta = s.get("speaker_meta")
            group = (meta.vtl_class if isinstance(meta, SpeakerMeta)
                     else s.get("group", "unknown"))
            vtl_mm = estimator.group_vtl(group).mu
        else:
            vtl_mm = estimator.speaker_vtl(speaker)
        args.append((s, cfg, vtl_mm, extractor))
    return args


# ---------------------------------------------------------------------------
# GMM fitting: extract per-sample observations from extraction results
# ---------------------------------------------------------------------------

def _fit_gmms(
    results: list[dict],
    cfg: LabellerConfig,
) -> dict[str, dict[str, list[_SampleObs]]]:
    """Fit GMMs to vtl_per_formant arrays and structure observations
    for the BayesianVTLEstimator.

    Returns dict[group, dict[speaker, list[_SampleObs]]].
    """
    from scipy.stats import skew as _skew_fn
    observations: dict[str, dict[str, list[_SampleObs]]] = {}

    for s_out in results:
        group   = s_out["group"]
        speaker = s_out["speaker"]
        vtl_pf  = s_out.get("vtl_per_formant")

        if vtl_pf is None:
            continue

        # Pool all per-frame per-formant VTL observations for this sample
        vtl_flat = vtl_pf.ravel()
        vtl_flat = vtl_flat[np.isfinite(vtl_flat)]

        if len(vtl_flat) < 2:
            continue

        fitter = fit_vtl_mixture if cfg.vtl_use_gmm else fit_vtl_single
        narrow_mean, narrow_var, _ = fitter(
            vtl_flat, min_obs=cfg.vtl_gmm_min_obs)

        if not np.isfinite(narrow_mean) or not np.isfinite(narrow_var):
            continue

        sample_skew = float(_skew_fn(vtl_flat)) if len(vtl_flat) >= 3 else 0.0

        obs = _SampleObs(
            narrow_mean=narrow_mean,
            narrow_var=max(narrow_var, 1e-6),
            n_obs=1,
            skew=sample_skew,
        )

        observations.setdefault(group, {}).setdefault(speaker, []).append(obs)

    return observations


# ---------------------------------------------------------------------------
# Apply group priors to estimator
# ---------------------------------------------------------------------------

def _apply_group_priors(
    estimator: BayesianVTLEstimator,
    group_priors: Optional[dict] = None,
) -> None:
    """Register group-level VTL overrides if provided.

    Values may be plain floats (vtl_mm) or ``(vtl_mm, sigma_mm)`` tuples.
    """
    if group_priors:
        for group, val in group_priors.items():
            if isinstance(val, (list, tuple)):
                vtl_mm, sigma_mm = val
                estimator.register_group_override(group, vtl_mm, sigma_mm=sigma_mm)
            else:
                estimator.register_group_override(group, val)


# ---------------------------------------------------------------------------
# Convergence-based dataset labeller
# ---------------------------------------------------------------------------

def _dump_estimator(estimator: BayesianVTLEstimator, label: str) -> None:
    """Print a diagnostic snapshot of the estimator state."""
    print(f"\n--- estimator state [{label}] ---")
    print("  Groups:")
    from scipy.stats import skew as _skew
    for g, post in sorted(estimator._group_posteriors.items()):
        spread = estimator._group_spread_sq.get(g)
        spread_str = f"  spread_σ={np.sqrt(spread):.2f}mm" if spread is not None else ""
        # Skewness of speaker means within this group
        spk_in_g = [spk for spk, grp in estimator._speaker_to_group.items()
                     if grp == g and spk in estimator._speaker_posteriors
                     and spk not in estimator._speaker_override_mm]
        if len(spk_in_g) >= 3:
            mus = np.array([estimator._speaker_posteriors[s].mu for s in spk_in_g])
            skew_val = float(_skew(mus))
            skew_str = f"  skew={skew_val:+.2f}"
        else:
            skew_str = ""
        print(f"    {g}: μ={post.mu:.2f}mm  mean_σ={post.sigma:.2f}mm{spread_str}{skew_str}")
    print("  Speakers:")
    for spk in sorted(estimator._speaker_posteriors):
        post = estimator._speaker_posteriors[spk]
        grp = estimator._speaker_to_group.get(spk, "?")
        override = " [OVERRIDE]" if spk in estimator._speaker_override_mm else ""
        spk_obs_list = []
        for obs_group in getattr(estimator, '_last_observations', {}).values():
            if spk in obs_group:
                spk_obs_list.extend(obs_group[spk])
        obs_means = [o.narrow_mean for o in spk_obs_list if np.isfinite(o.narrow_mean)]
        if len(obs_means) >= 3:
            # Skew across multiple samples
            skew_str = f"  skew={float(_skew(obs_means)):+.2f}"
        elif len(spk_obs_list) == 1 and spk_obs_list[0].skew != 0.0:
            # Fall back to within-sample skew
            skew_str = f"  skew={spk_obs_list[0].skew:+.2f}*"
        else:
            skew_str = ""
        print(f"    {spk} ({grp}): μ={post.mu:.2f}mm  σ={post.sigma:.2f}mm{override}{skew_str}")
    print("---\n")


def label_dataset(
    samples: list[dict],
    cfg: Optional[LabellerConfig] = None,
    extractor: Optional[FormantExtractor] = None,
    sr: int = 16000,
    verbose: bool = True,
    group_priors: Optional[dict[str, float]] = None,
) -> list[dict]:
    """
    Run the Bayesian convergence-based labelling pipeline on a dataset.

    The pipeline iterates:
      1. Extract all samples using current speaker VTL posteriors
      2. Fit per-sample GMM to separate signal from noise
      3. Update Bayesian posteriors (group → speaker)
      4. Check convergence; repeat until group μ stabilises

    Parameters
    ----------
    samples   : list of dicts from dataset.py / datasets_hf.py
    cfg       : LabellerConfig (uses defaults if None)
    extractor : FormantExtractor backend (uses PraatFormantExtractor if None)
    sr        : sample rate (used if not present in sample dict)
    verbose   : print progress
    group_priors : optional dict of group → vtl_mm or (vtl_mm, sigma_mm)

    Returns
    -------
    List of transposed sample dicts — one per input sample.
    """
    if cfg is None:
        cfg = LabellerConfig()
    if extractor is None:
        extractor = PraatFormantExtractor(cfg)

    n_workers = _resolve_workers(cfg.n_jobs)
    for s in samples:
        s.setdefault("sr", sr)

    if verbose:
        print(f"[labeller] {len(samples)} samples, "
              f"{n_workers} worker{'s' if n_workers != 1 else ''}.")

    # -------------------------------------------------------------------
    # Seed: multi-position VTL estimate per sample (parallel)
    # -------------------------------------------------------------------
    if verbose:
        print("[labeller] Seeding VTL estimates...")

    seed_args = [(s, cfg, extractor) for s in samples]
    if n_workers == 1:
        seed_results = [_worker_seed_vtl(a) for a in seed_args]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            seed_results = list(pool.map(_worker_seed_vtl, seed_args))

    # Initialise Bayesian estimator
    estimator = BayesianVTLEstimator(prior_sigma_mm=cfg.vtl_prior_sigma_mm)
    _apply_group_priors(estimator, group_priors)

    # Register speaker overrides from SpeakerMeta
    for s in samples:
        meta = s.get("speaker_meta")
        if isinstance(meta, SpeakerMeta) and meta.vtl_prior_mm is not None:
            estimator.register_speaker_override(s["speaker"], meta.vtl_prior_mm)

    # Seed the estimator with rough VTL estimates
    for group, speaker, vtl, var in seed_results:
        if np.isfinite(vtl):
            estimator.seed_speaker(group, speaker, vtl, var)

    # Update group posteriors from seed data so iteration 0's group ceiling
    # reflects the actual data rather than literature priors
    estimator.update_groups_from_speakers()

    if verbose:
        _dump_estimator(estimator, "post-seed")

    # -------------------------------------------------------------------
    # Convergence loop
    # -------------------------------------------------------------------
    results = None
    for iteration in range(cfg.vtl_max_iterations):
        prev_mus = estimator.group_mus()

        if verbose:
            print(f"[labeller] Iteration {iteration} — extracting...")

        # Extract all samples — use group ceiling on iter 0 to avoid
        # ceiling feedback loop from potentially noisy seed VTL estimates
        worker_args = _make_worker_args(
            samples, cfg, estimator, extractor,
            use_group_ceiling=(iteration == 0))
        results = _parallel_map(worker_args, n_workers, verbose,
                                f"iter{iteration}")

        if verbose:
            print()

        # Fit GMMs and update posteriors
        observations = _fit_gmms(results, cfg)
        estimator.update_all(observations)

        # After first iteration, release speaker overrides so they can
        # be refined by data (the override served its purpose as a seed)
        if iteration == 0 and estimator._speaker_override_mm:
            if verbose:
                print(f"  [unfreeze] Releasing {len(estimator._speaker_override_mm)} "
                      f"speaker overrides: {sorted(estimator._speaker_override_mm)}")
            estimator._speaker_override_mm.clear()

        # Reseed speakers that have drifted outside n*σ of group spread
        if cfg.vtl_reseed_sigma > 0:
            reseeded = estimator.reseed_outliers(
                n_sigma=cfg.vtl_reseed_sigma, verbose=verbose)

        # Check convergence
        current_mus = estimator.group_mus()
        if verbose:
            for g, mu in sorted(current_mus.items()):
                prev = prev_mus.get(g, mu)
                sigma = estimator.group_vtl(g).sigma
                print(f"  group={g}: μ={mu:.1f}mm (Δ={mu-prev:+.2f}mm) σ={sigma:.1f}mm")

        if verbose:
            _dump_estimator(estimator, f"iter{iteration}")

        if iteration > 0 and estimator.has_converged(
                prev_mus, threshold=cfg.vtl_convergence_threshold_mm):
            if verbose:
                print(f"[labeller] Converged after {iteration + 1} iterations.")
            break
    else:
        if verbose:
            print(f"[labeller] Reached max iterations ({cfg.vtl_max_iterations}).")

    # -------------------------------------------------------------------
    # Final pass with converged VTL estimates
    # -------------------------------------------------------------------
    if verbose:
        print("[labeller] Final extraction with converged VTL estimates...")

    worker_args = _make_worker_args(samples, cfg, estimator, extractor)
    final = _parallel_map(worker_args, n_workers, verbose, "final")

    # Add posterior uncertainty and strip transient fields
    for s_out in final:
        speaker = s_out["speaker"]
        s_out["vtl_posterior_mu_mm"] = estimator.speaker_vtl(speaker)
        s_out["vtl_posterior_sigma_mm"] = estimator.speaker_uncertainty(speaker)
        s_out.pop("vtl_per_formant", None)

    if verbose:
        n_voiced = sum(
            int(np.isfinite(s["f0_hz"]).any()) for s in final
        )
        print(f"\n[labeller] Done. {len(final)} samples "
              f"({n_voiced} with at least one voiced window).")

    return final


# ---------------------------------------------------------------------------
# Incremental labelling: add new samples using an existing labelled dataset
# ---------------------------------------------------------------------------

def _estimator_from_labelled(
    labelled: list[dict],
    cfg: LabellerConfig,
) -> BayesianVTLEstimator:
    """
    Build a BayesianVTLEstimator from an already-labelled dataset.

    Each speaker's mean vtl_sample_mm becomes a tight-variance observation,
    reflecting the confidence from prior labelling.
    """
    estimator = BayesianVTLEstimator(prior_sigma_mm=cfg.vtl_prior_sigma_mm)

    # Collect per-speaker mean VTL
    speaker_vtls: dict[str, list[float]] = {}
    speaker_group: dict[str, str] = {}
    for s in labelled:
        vtls = s["vtl_sample_mm"][np.isfinite(s["vtl_sample_mm"])]
        if len(vtls):
            speaker = s["speaker"]
            group = s["group"]
            speaker_vtls.setdefault(speaker, []).append(float(np.mean(vtls)))
            speaker_group[speaker] = group

    # Build observations for the estimator
    observations: dict[str, dict[str, list[_SampleObs]]] = {}
    for speaker, vtls in speaker_vtls.items():
        group = speaker_group[speaker]
        mean_vtl = float(np.mean(vtls))
        var_vtl = max(float(np.var(vtls, ddof=1)) if len(vtls) > 1 else 25.0, 1.0)
        obs = _SampleObs(narrow_mean=mean_vtl, narrow_var=var_vtl, n_obs=len(vtls))
        observations.setdefault(group, {}).setdefault(speaker, []).append(obs)

    estimator.update_all(observations)
    return estimator


def label_incremental(
    new_samples: list[dict],
    existing: list[dict],
    cfg: Optional[LabellerConfig] = None,
    extractor: Optional[FormantExtractor] = None,
    sr: int = 16000,
    refine: bool = False,
    verbose: bool = True,
    group_priors: Optional[dict[str, float]] = None,
) -> list[dict]:
    """
    Label new samples using the pooled VTL statistics from an already-labelled
    dataset, without re-running the full pipeline on everything.

    When ``refine=False`` (default):
        Single-pass extraction using an estimator built from existing data +
        seed VTL from new samples.  Only the newly labelled samples are returned.

    When ``refine=True``:
        Full convergence-loop pipeline over the combined dataset
        (existing + new).  The full combined result is returned.
    """
    if cfg is None:
        cfg = LabellerConfig()
    if extractor is None:
        extractor = PraatFormantExtractor(cfg)

    for s in new_samples:
        s.setdefault("sr", sr)

    n_workers = _resolve_workers(cfg.n_jobs)

    if verbose:
        print(f"[label_incremental] {len(new_samples)} new samples, "
              f"{len(existing)} existing — refine={refine}.")

    # Build estimator from existing labelled data
    estimator = _estimator_from_labelled(existing, cfg)
    _apply_group_priors(estimator, group_priors)

    if verbose:
        groups = {}
        for s in existing:
            meta = s.get("speaker_meta")
            g    = (meta.vtl_class if isinstance(meta, SpeakerMeta)
                    else s.get("group", "unknown"))
            groups[g] = groups.get(g, 0) + 1
        breakdown = ", ".join(f"{g}={n}" for g, n in sorted(groups.items()))
        print(f"[label_incremental] Estimator seeded from existing: {breakdown}")

    # Seed per-speaker VTL for new speakers
    if verbose:
        print("[label_incremental] Seeding VTL for new speakers...")

    seed_args = [(s, cfg, extractor) for s in new_samples]
    if n_workers == 1:
        seed_results = [_worker_seed_vtl(a) for a in seed_args]
    else:
        with ProcessPoolExecutor(max_workers=n_workers) as pool:
            seed_results = list(pool.map(_worker_seed_vtl, seed_args))

    # Register speaker overrides from SpeakerMeta
    for s in new_samples:
        meta = s.get("speaker_meta")
        if isinstance(meta, SpeakerMeta) and meta.vtl_prior_mm is not None:
            estimator.register_speaker_override(s["speaker"], meta.vtl_prior_mm)

    for group, speaker, vtl, var in seed_results:
        if np.isfinite(vtl):
            estimator.seed_speaker(group, speaker, vtl, var)

    if verbose:
        new_speakers = {s.get("speaker", "unknown") for s in new_samples}
        for spk in sorted(new_speakers):
            vtl = estimator.speaker_vtl(spk)
            print(f"  seeded {spk}: VTL={vtl:.1f}mm")

    # Single pass with current estimator
    if verbose:
        print("[label_incremental] Labelling new samples (single pass)...")

    worker_args = _make_worker_args(new_samples, cfg, estimator, extractor)
    new_labelled = _parallel_map(worker_args, n_workers, verbose, "incremental")

    # Add posterior info and strip transient fields
    for s_out in new_labelled:
        speaker = s_out["speaker"]
        s_out["vtl_posterior_mu_mm"] = estimator.speaker_vtl(speaker)
        s_out["vtl_posterior_sigma_mm"] = estimator.speaker_uncertainty(speaker)
        s_out.pop("vtl_per_formant", None)

    if verbose:
        n_voiced = sum(int(np.isfinite(s["f0_hz"]).any()) for s in new_labelled)
        print(f"\n[label_incremental] {len(new_labelled)} new samples labelled "
              f"({n_voiced} with at least one voiced window).")

    if not refine:
        return new_labelled

    # Refinement: full pipeline over combined dataset
    if verbose:
        print("[label_incremental] Running refinement pass over combined dataset...")

    _output_keys = {
        "t_centre_s", "loudness_dbfs", "spectral_slope", "f0_hz", "periodicity",
        "f0_praat_hz", "formant_hz", "formant_bw_hz",
        "vtl_sample_mm", "vtl_per_formant", "vtl_posterior_mu_mm",
        "vtl_posterior_sigma_mm", "window_length_ms", "hop_ms",
        "outlier_flags", "outlier_mask",
    }
    def _strip(s: dict) -> dict:
        return {k: v for k, v in s.items() if k not in _output_keys}

    combined_raw = [_strip(s) for s in existing] + list(new_samples)
    return label_dataset(combined_raw, cfg=cfg, extractor=extractor, sr=sr,
                         verbose=verbose, group_priors=group_priors)


# ---------------------------------------------------------------------------
# Sanity-check probes
# ---------------------------------------------------------------------------

def probe_raw_formants(
    samples: list[dict],
    cfg: Optional[LabellerConfig] = None,
    extractor: Optional[FormantExtractor] = None,
    sr: int = 16000,
    verbose: bool = True,
) -> list[dict]:
    """
    Run a single raw extraction pass on one sample per group per vowel,
    with no multi-pass smoothing.  The VTL prior is the bare literature
    value for each group.
    """
    if cfg is None:
        cfg = LabellerConfig()
    if extractor is None:
        extractor = PraatFormantExtractor(cfg)

    for s in samples:
        s.setdefault("sr", sr)

    # --- Selection: no duplicate speakers or vowels; same vowel set per group ---
    vowels_by_group: dict[str, set[str]] = defaultdict(set)
    for s in samples:
        g = s.get("group", s.get("speaker", "unknown"))
        vowels_by_group[g].add(s.get("label", ""))

    shared_vowels: set[str] = set.intersection(*vowels_by_group.values()) \
        if vowels_by_group else set()

    if not shared_vowels:
        shared_vowels = set.union(*vowels_by_group.values()) if vowels_by_group else set()
        if verbose:
            print("[probe_raw_formants] WARNING: no vowels common to all groups; "
                  "using union instead")

    group_selections: dict[str, dict[str, dict]] = defaultdict(dict)
    used_speakers: set[str] = set()

    for s in samples:
        g       = s.get("group", s.get("speaker", "unknown"))
        vowel   = s.get("label", "")
        speaker = s.get("speaker", s.get("group", "unknown"))

        if vowel not in shared_vowels:
            continue
        if vowel in group_selections[g]:
            continue
        if speaker in used_speakers:
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

    # Use literature prior VTL for each sample's group
    results = []
    for s in selected:
        speaker = s.get("speaker", s.get("group", "unknown"))
        group   = s.get("group", speaker)
        lit_vtl = VTL_PRIOR_MM.get(group, VTL_PRIOR_MM["unknown"])
        if verbose:
            print(f"  probing speaker={speaker!r} group={group!r} "
                  f"label={s.get('label', '?')!r}")
        result = _extract_sample(s, cfg, lit_vtl, extractor)
        result.pop("vtl_per_formant", None)
        results.append(result)

    return results


def probe_full(
    samples: list[dict],
    cfg: Optional[LabellerConfig] = None,
    extractor: Optional[FormantExtractor] = None,
    sr: int = 16000,
    speakers_per_group: int = 2,
    verbose: bool = True,
) -> list[dict]:
    """
    Run the full convergence-based pipeline on a small representative subset.
    """
    if cfg is None:
        cfg = LabellerConfig()

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

    return label_dataset(selected, cfg=cfg, extractor=extractor, sr=sr, verbose=verbose)
