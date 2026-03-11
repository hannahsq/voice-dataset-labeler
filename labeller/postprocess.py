"""
Post-processing: outlier flagging, metadata computation, and persistence.
"""

from __future__ import annotations

import json
import pickle
from typing import Optional

import numpy as np

from .config import SPEED_OF_SOUND_MM_S, VTL_PRIOR_MM, OutlierConfig


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
        f1       = s["formant_hz"][:, 0]
        vtl_cls  = (group if isinstance(group, str) and group in cfg.f1_bounds_hz
                    else "unknown")
        f1_floor, f1_ceil = cfg.f1_bounds_hz.get(vtl_cls, (200.0, 1250.0))
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
                grp if isinstance(grp, str) else "unknown",
                VTL_PRIOR_MM["unknown"],
            ),
        }

    return speaker_meta, group_meta


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
    with open(f"{path_prefix}_samples.pkl", "rb") as f:
        labelled = pickle.load(f)
    with open(f"{path_prefix}_speaker_meta.json", encoding="utf-8") as f:
        speaker_meta = json.load(f)
    with open(f"{path_prefix}_group_meta.json", encoding="utf-8") as f:
        group_meta = json.load(f)
    print(f"Loaded {len(labelled)} samples from {path_prefix}_samples.pkl")
    return labelled, speaker_meta, group_meta
