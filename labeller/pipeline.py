"""
Core extraction, parallel workers, and the multi-pass labelling pipeline.
"""

from __future__ import annotations

import os
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Optional

import numpy as np
import parselmouth

from .config import LabellerConfig, N_FORMANTS, SPEED_OF_SOUND_MM_S
from .types import SpeakerMeta, Modality
from .acoustic import (
    _loudness_dbfs,
    _periodicity,
    _praat_f0,
    _pyin_f0,
    _blend_f0,
)
from .formants import (
    _extract_raw_formants,
    _vtl_from_formants,
    _assign_formant_indices,
    VTLSmoother,
)
from .extractors import FormantExtractor, PraatFormantExtractor


# ---------------------------------------------------------------------------
# Core per-sample extractor — returns transposed dict directly
# ---------------------------------------------------------------------------

def _extract_sample(
    sample: dict,
    cfg: LabellerConfig,
    vtl_smoother: VTLSmoother,
    extractor: FormantExtractor,
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
    speaker  = sample.get("speaker", "unknown")
    # group is a vtl_class string; speaker_meta.vtl_class is the canonical source
    meta     = sample.get("speaker_meta")
    group    = (meta.vtl_class if isinstance(meta, SpeakerMeta)
                else sample.get("group", "unknown"))
    # modality: accept Modality object or plain string
    _mod     = sample.get("modality", "spoken")
    modality = _mod.modality if isinstance(_mod, Modality) else str(_mod)

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

        # Nyquist-clipped formant count: only ask Praat for poles that can
        # physically exist.  Uses smoothed speaker VTL so it's stable across
        # windows; falls back to cfg.n_praat_formants during the seed pass.
        nyquist_hz = sr / 2.0
        if np.isfinite(speaker_vtl) and speaker_vtl > 0:
            delta_f  = SPEED_OF_SOUND_MM_S / (2.0 * speaker_vtl)
            n_praat  = max(1, min(cfg.n_praat_formants,
                                  int(np.floor(nyquist_hz / delta_f))))
        else:
            n_praat  = cfg.n_praat_formants
        raw_freqs, raw_bws, slope = extractor(
            frame, sr, win_s, max_formant_hz=dynamic_ceiling, n_formants=n_praat)
        vtl_raw      = _vtl_from_formants(raw_freqs)
        smoothed_vtl = vtl_smoother.smooth_vtl(group, speaker, vtl_raw)
        freqs_out, bws_out = _assign_formant_indices(raw_freqs, raw_bws, smoothed_vtl, nyquist_hz=sr / 2.0)
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

def _worker_extract(args: tuple[dict, LabellerConfig, VTLSmoother, FormantExtractor]) -> dict:
    """Extract one sample (parallel worker). Returns transposed sample dict."""
    sample, cfg, smoother, extractor = args
    return _extract_sample(sample, cfg, smoother, extractor)


def _worker_seed_vtl(
    args: tuple[dict, LabellerConfig, FormantExtractor],
) -> tuple[str, str, float]:
    """
    Compute a robust median VTL estimate from up to 20 intensity-gated
    positions across a sample.  Used to seed the smoother before pass 0.
    Returns (group, speaker, median_vtl_mm).
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
        n      = len(audio)
        raw_f, _, _ = extractor(audio[n//4 : 3*n//4], sr, win_s)
        return group, speaker, _vtl_from_formants(raw_f)

    win_samps = int(round(win_s * sr))
    vtls = []
    for t in positions:
        s0    = max(0, int((t - win_s / 2) * sr))
        frame = audio[s0 : min(len(audio), s0 + win_samps)]
        if len(frame) < win_samps // 2:
            continue
        raw_f, _, _ = extractor(frame, sr, win_s)
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
    extractor: FormantExtractor,
) -> list[tuple]:
    return [(s, cfg, smoother, extractor) for s in samples]


# ---------------------------------------------------------------------------
# Three-pass dataset labeller
# ---------------------------------------------------------------------------

def label_dataset(
    samples: list[dict],
    cfg: Optional[LabellerConfig] = None,
    extractor: Optional[FormantExtractor] = None,
    sr: int = 16000,
    verbose: bool = True,
) -> list[dict]:
    """
    Run the full three-pass labelling pipeline on a dataset.

    Parameters
    ----------
    samples   : list of dicts from dataset.py / datasets_hf.py
    cfg       : LabellerConfig (uses defaults if None)
    extractor : FormantExtractor backend (uses PraatFormantExtractor if None)
    sr        : sample rate (used if not present in sample dict)
    verbose   : print progress

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
    if extractor is None:
        extractor = PraatFormantExtractor(cfg)

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

    seed_args    = [(s, cfg, extractor) for s in samples]
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
        _make_worker_args(samples, cfg, smoother_raw, extractor),
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
        _make_worker_args(samples, cfg, smoother_first, extractor),
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
        speaker   = s.get("speaker", "unknown")
        meta      = s.get("speaker_meta")
        group     = (meta.vtl_class if isinstance(meta, SpeakerMeta)
                     else s.get("group", "unknown"))
        final_vtl = smoother_second.smoothed_speaker_vtl(group, speaker)
        frozen    = VTLSmoother(prior_strength=1e9, sample_alpha=cfg.vtl_sample_alpha)
        frozen._speaker_vtls[speaker] = [final_vtl]
        frozen._group_vtls[group]     = [final_vtl]
        if meta and isinstance(meta, SpeakerMeta) and meta.vtl_prior_mm is not None:
            frozen.register_speaker_override(speaker, meta.vtl_prior_mm)
        frozen_args.append((dict(s), cfg, frozen, extractor))

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
    extractor: Optional[FormantExtractor] = None,
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
    if extractor is None:
        extractor = PraatFormantExtractor(cfg)

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
            meta = s.get("speaker_meta")
            g    = (meta.vtl_class if isinstance(meta, SpeakerMeta)
                    else s.get("group", "unknown"))
            groups[g] = groups.get(g, 0) + 1
        breakdown = ", ".join(f"{g}={n}" for g, n in sorted(groups.items()))
        print(f"[label_incremental] Smoother seeded from existing: {breakdown}")

    # Seed per-speaker VTL for new speakers from their own audio before the
    # main pass.  Without this the formant ceiling is derived purely from the
    # group prior, making the ceiling — and therefore the pole assignment —
    # sensitive to the group label even when vtl_sample_alpha is high.
    if verbose:
        print("[label_incremental] Seeding VTL for new speakers...")

    seed_args    = [(s, cfg, extractor) for s in new_samples]
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
            s_   = next((s for s in new_samples if s.get("speaker") == spk), {})
            meta = s_.get("speaker_meta")
            grp  = (meta.vtl_class if isinstance(meta, SpeakerMeta)
                    else s_.get("group", "unknown"))
            vtl  = smoother.smoothed_speaker_vtl(grp, spk)
            print(f"  seeded {spk} ({grp}): VTL={vtl:.1f}mm")

    # Single pass with frozen per-speaker VTL (pass-2 equivalent)
    if verbose:
        print("[label_incremental] Labelling new samples (single pass)...")

    frozen_args = []
    for s in new_samples:
        speaker   = s.get("speaker", "unknown")
        meta      = s.get("speaker_meta")
        group     = (meta.vtl_class if isinstance(meta, SpeakerMeta)
                     else s.get("group", "unknown"))
        final_vtl = smoother.smoothed_speaker_vtl(group, speaker)
        frozen    = VTLSmoother(prior_strength=1e9, sample_alpha=cfg.vtl_sample_alpha)
        frozen._speaker_vtls[speaker] = [final_vtl]
        frozen._group_vtls[group]     = [final_vtl]
        if meta and isinstance(meta, SpeakerMeta) and meta.vtl_prior_mm is not None:
            frozen.register_speaker_override(speaker, meta.vtl_prior_mm)
        frozen_args.append((dict(s), cfg, frozen, extractor))

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
    return label_dataset(combined_raw, cfg=cfg, extractor=extractor, sr=sr, verbose=verbose)


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
    if extractor is None:
        extractor = PraatFormantExtractor(cfg)

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
        result = _extract_sample(s, cfg, lit_smoother, extractor)
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

    return label_dataset(selected, cfg=cfg, extractor=extractor, sr=sr, verbose=verbose)
