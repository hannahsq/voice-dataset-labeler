"""
test_synth_compat.py
====================
R11: synth_to_labelled produces a valid LabelledSample dict.

Tests that synthetic vowel output can be converted to the labeller's
per-window dict format and survives flag_outliers / build_metadata.
"""

from __future__ import annotations

from test_labeller_registry import (
    _check,
    _LABELLER_OK,
    _SYNTH_OK,
    N_FORMANTS,
)

import numpy as np


def test_r11_synth_to_labelled():
    """synth_to_labelled produces correct keys, shapes, and dtypes."""
    if not (_LABELLER_OK and _SYNTH_OK):
        _check("R11a-import", False, "labeller or synth import failed")
        return

    from synth import synth_to_labelled
    from labeller import LabellerConfig, flag_outliers, build_metadata
    from test_labeller_registry import _synth_neutral

    audio, meta = _synth_neutral(vtl_mm=155.0, f0_hz=120.0, sr=16000, seed=42)
    cfg = LabellerConfig()
    d = synth_to_labelled(audio, meta, cfg)

    # --- R11a: all expected keys present ---
    expected_keys = {
        "speaker", "speaker_meta", "group", "label", "source", "modality",
        "dialect", "audio", "sr", "window_length_ms", "hop_ms",
        "t_centre_s", "loudness_dbfs", "spectral_slope",
        "f0_hz", "periodicity", "f0_praat_hz", "f0_pyin_hz",
        "formant_hz", "formant_bw_hz", "vtl_sample_mm", "ground_truth",
    }
    missing = expected_keys - set(d.keys())
    _check("R11a-keys", len(missing) == 0, f"missing: {missing}")

    # --- R11b: per-window array shapes ---
    W = len(d["t_centre_s"])
    _check("R11b-W>0", W > 0, f"W={W}")

    for key in ("loudness_dbfs", "spectral_slope", "f0_hz", "periodicity",
                "f0_praat_hz", "f0_pyin_hz", "vtl_sample_mm"):
        arr = d[key]
        _check(f"R11b-{key}-shape", arr.shape == (W,),
               f"expected ({W},), got {arr.shape}")
        _check(f"R11b-{key}-dtype", arr.dtype == np.float32,
               f"expected float32, got {arr.dtype}")

    _check("R11b-formant_hz-shape", d["formant_hz"].shape == (W, N_FORMANTS),
           f"expected ({W}, {N_FORMANTS}), got {d['formant_hz'].shape}")
    _check("R11b-formant_bw_hz-shape", d["formant_bw_hz"].shape == (W, N_FORMANTS),
           f"expected ({W}, {N_FORMANTS}), got {d['formant_bw_hz'].shape}")

    # --- R11c: ground truth round-trip ---
    gt = d["ground_truth"]
    _check("R11c-gt-vtl", gt["vtl_mm"] == meta.vtl_mm, f"{gt['vtl_mm']} vs {meta.vtl_mm}")
    _check("R11c-gt-f0", gt["f0_hz"] == meta.f0_hz, f"{gt['f0_hz']} vs {meta.f0_hz}")
    _check("R11c-gt-formants", np.allclose(gt["formant_hz"], meta.formant_hz),
           "ground_truth formant_hz mismatch")

    # --- R11d: survives flag_outliers + build_metadata ---
    labelled = [d]
    try:
        flag_outliers(labelled)
        speaker_meta, group_meta = build_metadata(labelled)
        _check("R11d-postprocess", True, "flag_outliers + build_metadata OK")
    except Exception as e:
        _check("R11d-postprocess", False, f"raised {e!r}")


if __name__ == "__main__":
    test_r11_synth_to_labelled()
    from test_labeller_registry import _print_summary
    raise SystemExit(_print_summary())
