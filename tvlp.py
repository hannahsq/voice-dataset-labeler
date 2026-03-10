"""
tvlp.py
=======
Time-Varying Linear Prediction (TVLP) formant estimator.

Drop-in complement/replacement for Praat's Burg tracker in the labeller
pipeline.  The public interface mirrors _extract_raw_formants():

    freqs, bws, spectral_slope = extract_formants(
        frame, sr, n_formants, win_s, max_formant_hz, lambda_smooth
    )

Motivation
----------
Standard Burg / autocorrelation LP fits a *stationary* all-pole model to a
short window.  At high fundamental frequencies (singing, especially M2/M3
register) harmonics are widely spaced and the LP spectrum chases individual
partials rather than tracking the underlying vocal tract resonances.

TVLP addresses this by fitting a *slowly-varying* LP model across a longer
analysis window (~100–200 ms), adding a temporal smoothness penalty that
discourages rapid coefficient changes between adjacent sub-frames.  The LP
coefficients at the window centre are then used for pole extraction.

Method
------
The window is divided into n_sub equally-spaced sub-frames.  Let
a_k ∈ R^order be the LP coefficient vector for sub-frame k.  We minimise:

    J(a_1..a_K) = Σ_k ‖e_k(a_k)‖² + λ Σ_k ‖a_k - a_{k-1}‖²

where e_k is the LP prediction error for sub-frame k.  This is a block
tridiagonal least-squares problem, solved efficiently by stacking the normal
equations with off-diagonal coupling terms and calling np.linalg.solve.

This is the *time-varying LP* layer only — quasi-closed-phase (QCP) weighting
by glottal closure instants is a planned future extension.

Pole extraction
---------------
Roots of the LP polynomial A(z) = 1 - a_1 z^{-1} - ... - a_p z^{-p} are
found via np.roots().  Poles inside the unit circle with positive imaginary
part are converted to (frequency, bandwidth) pairs:

    freq = angle(pole) * sr / (2π)
    bw   = -log(|pole|) * sr / π

Poles are sorted by frequency and the n_formants lowest are returned.

Spectral slope
--------------
Estimated from the window centre sub-frame using the same log-magnitude linear
fit as labeller._estimate_spectral_tilt_alpha (300–4000 Hz).

Usage with labeller.py
----------------------
Replace the call to _extract_raw_formants() in _extract_sample() with:

    import tvlp
    raw_freqs, raw_bws, slope = tvlp.extract_formants(
        frame, sr, cfg, win_s,
        max_formant_hz=dynamic_ceiling,
        n_formants=n_praat,
    )

The rest of the pipeline (_assign_formant_indices, VTLSmoother, etc.) is
unchanged.
"""

from __future__ import annotations

import warnings
from typing import Optional

import numpy as np


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

_DEFAULT_WIN_S        = 0.15    # analysis window (s)
_DEFAULT_N_SUB        = 10      # sub-frames per window
_DEFAULT_ORDER        = 14      # LP order — sufficient for 7 formants
_DEFAULT_LAMBDA       = 2.0     # temporal smoothness strength
_DEFAULT_MAX_F        = 5500.0  # Hz — formant ceiling
_SLOPE_F_LOW          = 300.0   # Hz — spectral slope fit band
_SLOPE_F_HIGH         = 4000.0  # Hz


# ---------------------------------------------------------------------------
# Internal: per-sub-frame LP normal equations
# ---------------------------------------------------------------------------

def _lp_normal_equations(
    frame: np.ndarray,
    order: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Build the normal equations R a = r for autocorrelation LP.

    R is the (order × order) Toeplitz autocorrelation matrix,
    r is the (order,) right-hand side vector.

    Uses a biased autocorrelation estimate (divides by N, not N-lag).
    Returns (R, r).  On pathological input (zero energy), returns
    (identity, zeros) so the caller gets a zero coefficient vector.
    """
    x = frame.astype(np.float64)
    n = len(x)
    # Autocorrelation lags 0..order
    R_full = np.correlate(x, x, mode="full")
    mid    = len(R_full) // 2
    acf    = R_full[mid : mid + order + 1] / n  # r[0], r[1], ..., r[order]

    if not np.isfinite(acf[0]) or acf[0] < 1e-20:
        return np.eye(order), np.zeros(order)

    # Toeplitz matrix from lags
    row = acf[:order]
    col = acf[:order]
    R   = np.zeros((order, order))
    for i in range(order):
        for j in range(order):
            lag     = abs(i - j)
            R[i, j] = acf[lag] if lag <= order else 0.0

    r = acf[1 : order + 1]
    return R, r


# ---------------------------------------------------------------------------
# Core TVLP solver — exposed for testing (T08)
# ---------------------------------------------------------------------------

def fit_tvlp(
    audio: np.ndarray,
    sr: int,
    order: int = _DEFAULT_ORDER,
    n_sub: int = _DEFAULT_N_SUB,
    lambda_smooth: float = _DEFAULT_LAMBDA,
) -> np.ndarray:
    """
    Fit a time-varying LP model to `audio`.

    Divides the signal into `n_sub` overlapping sub-frames and solves the
    jointly-regularised least-squares system.

    Parameters
    ----------
    audio         : (T,) float32 or float64 audio frame
    sr            : sample rate (Hz) — unused in the solver but kept for API
                    consistency and future QCP extension
    order         : LP order
    n_sub         : number of sub-frames
    lambda_smooth : temporal smoothness regularisation weight.
                    0.0 = independent per-frame LP (no smoothing).
                    Higher values → smoother trajectories.

    Returns
    -------
    coeffs : (n_sub, order) float64 array of LP coefficients.
             Row k gives [a_1, ..., a_p] for sub-frame k such that
             A(z) = 1 - a_1·z⁻¹ - … - a_p·z⁻ᵖ.
    """
    audio = np.asarray(audio, dtype=np.float64)
    n     = len(audio)
    if n < order + 1:
        return np.zeros((n_sub, order))

    # Sub-frame boundaries — equal length, allow overlap at edges
    sub_len = max(order + 2, n // n_sub)
    starts  = np.linspace(0, max(0, n - sub_len), n_sub, dtype=int)

    # Build per-sub-frame normal equations
    Rs = []
    rs = []
    for k in range(n_sub):
        chunk = audio[starts[k] : starts[k] + sub_len]
        if len(chunk) < order + 1:
            chunk = np.pad(chunk, (0, order + 1 - len(chunk)))
        R_k, r_k = _lp_normal_equations(chunk, order)
        Rs.append(R_k)
        rs.append(r_k)

    # Assemble block system:
    # For each sub-frame k, the stationarity condition is:
    #   (R_k + 2λ I) a_k  - λ a_{k-1} - λ a_{k+1} = r_k
    # (boundary sub-frames have only one neighbour)
    # This gives a block-tridiagonal system of size (K*p × K*p).

    K   = n_sub
    p   = order
    dim = K * p

    A_mat = np.zeros((dim, dim))
    b_vec = np.zeros(dim)

    lam2 = 2.0 * lambda_smooth

    for k in range(K):
        i0 = k * p
        i1 = i0 + p

        # Diagonal block: R_k + (number of neighbours) * λ * I
        n_neighbours = (1 if k == 0 or k == K - 1 else 2) if K > 1 else 0
        A_mat[i0:i1, i0:i1] = Rs[k] + n_neighbours * lambda_smooth * np.eye(p)
        b_vec[i0:i1]         = rs[k]

        # Off-diagonal coupling blocks
        if k > 0:
            j0 = (k - 1) * p
            j1 = j0 + p
            A_mat[i0:i1, j0:j1] = -lambda_smooth * np.eye(p)
        if k < K - 1:
            j0 = (k + 1) * p
            j1 = j0 + p
            A_mat[i0:i1, j0:j1] = -lambda_smooth * np.eye(p)

    # Solve block system
    try:
        a_flat = np.linalg.solve(A_mat, b_vec)
    except np.linalg.LinAlgError:
        a_flat = np.zeros(dim)

    return a_flat.reshape(K, p)


# ---------------------------------------------------------------------------
# Pole extraction
# ---------------------------------------------------------------------------

def _poles_to_formants(
    coeffs: np.ndarray,
    sr: int,
    max_formant_hz: float,
    n_formants: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Extract (freq, bw) pairs from LP coefficients.

    Returns two (n_formants,) float32 arrays, NaN where no pole was found.
    """
    out_f = np.full(n_formants, np.nan, dtype=np.float32)
    out_b = np.full(n_formants, np.nan, dtype=np.float32)

    if not np.any(np.isfinite(coeffs)) or np.all(coeffs == 0):
        return out_f, out_b

    # LP polynomial: A(z) = 1 - a_1 z^{-1} - ... - a_p z^{-p}
    # numpy poly convention: [1, -a_1, -a_2, ..., -a_p]
    poly = np.concatenate([[1.0], -coeffs.astype(np.float64)])

    try:
        roots = np.roots(poly)
    except Exception:
        return out_f, out_b

    nyquist = sr / 2.0
    freqs   = []
    bws     = []

    for z in roots:
        # Keep only poles inside / on the unit circle with positive imaginary
        # part (conjugate pairs — positive freq only)
        r = abs(z)
        if r >= 1.0 or z.imag <= 0:
            continue
        freq = float(np.angle(z) * sr / (2.0 * np.pi))
        bw   = float(-np.log(r) * sr / np.pi)
        if freq <= 0 or freq > min(max_formant_hz, nyquist) or bw <= 0:
            continue
        freqs.append(freq)
        bws.append(bw)

    if not freqs:
        return out_f, out_b

    # Sort by frequency, keep lowest n_formants
    order = np.argsort(freqs)
    freqs = [freqs[i] for i in order]
    bws   = [bws[i]   for i in order]

    n_fill = min(n_formants, len(freqs))
    out_f[:n_fill] = np.array(freqs[:n_fill], dtype=np.float32)
    out_b[:n_fill] = np.array(bws[:n_fill],   dtype=np.float32)

    return out_f, out_b


# ---------------------------------------------------------------------------
# Spectral slope (mirrors labeller._estimate_spectral_tilt_alpha)
# ---------------------------------------------------------------------------

def _spectral_slope(frame: np.ndarray, sr: int) -> float:
    """
    Estimate spectral tilt (dB/octave) by log-log regression over 300–4000 Hz.
    Returns NaN on failure.
    """
    try:
        w     = np.hanning(len(frame))
        mag   = np.abs(np.fft.rfft(frame.astype(np.float64) * w)) + 1e-12
        freqs = np.fft.rfftfreq(len(frame), 1.0 / sr)
        mask  = (freqs >= _SLOPE_F_LOW) & (freqs <= _SLOPE_F_HIGH)
        if mask.sum() < 4:
            return np.nan
        x_log        = np.log2(freqs[mask])
        A            = np.vstack([x_log, np.ones_like(x_log)]).T
        slope, _     = np.linalg.lstsq(A, 20.0 * np.log10(mag[mask]), rcond=None)[0]
        return float(slope)
    except Exception:
        return np.nan


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def extract_formants(
    frame: np.ndarray,
    sr: int,
    n_formants: int = 7,
    win_s: Optional[float] = None,
    max_formant_hz: float = _DEFAULT_MAX_F,
    lambda_smooth: float = _DEFAULT_LAMBDA,
    order: Optional[int] = None,
    n_sub: int = _DEFAULT_N_SUB,
    preemphasis_alpha: float = 0.97,
) -> tuple[np.ndarray, np.ndarray, float]:
    """
    Estimate formant frequencies and bandwidths from an audio frame using
    time-varying LP.

    Parameters
    ----------
    frame             : (T,) float32 audio signal
    sr                : sample rate (Hz)
    n_formants        : number of formant slots to return (output array length)
    win_s             : analysis window length in seconds (unused — the full
                        frame is used; kept for API compatibility with
                        _extract_raw_formants)
    max_formant_hz    : reject poles above this frequency (Hz)
    lambda_smooth     : temporal smoothness regularisation weight.
                        Default 2.0 gives robust formant recovery at low F0;
                        set 0.0 for independent per-subframe LP (not recommended)
    order             : LP order (defaults to max(2*n_formants, 14))
    n_sub             : number of sub-frames for the time-varying fit
    preemphasis_alpha : first-order pre-emphasis coefficient applied before LP.
                        Flattens the glottal source slope so LP poles track
                        vocal tract resonances rather than harmonic structure.
                        Set 0.0 to disable.  Default 0.97 matches a ~3 dB/oct
                        boost that counteracts typical glottal roll-off.

    Returns
    -------
    freqs : (n_formants,) float32  — formant frequencies; NaN for missed slots
    bws   : (n_formants,) float32  — formant bandwidths;  NaN for missed slots
    slope : float                  — spectral slope (dB/oct); NaN on failure
    """
    out_f = np.full(n_formants, np.nan, dtype=np.float32)
    out_b = np.full(n_formants, np.nan, dtype=np.float32)

    frame = np.asarray(frame, dtype=np.float32)

    # Degenerate input guards
    if len(frame) < 4:
        return out_f, out_b, np.nan
    energy = float(np.mean(frame.astype(np.float64) ** 2))
    if not np.isfinite(energy) or energy < 1e-20:
        return out_f, out_b, np.nan
    if np.all(frame == frame[0]):
        # DC or constant — no AC energy, LP is ill-defined
        return out_f, out_b, np.nan

    # Spectral slope estimated before pre-emphasis (on the raw frame)
    slope = _spectral_slope(frame, sr)

    # Apply first-order pre-emphasis to flatten glottal source roll-off.
    # This is the same operation as labeller._apply_preemphasis and is
    # essential for LP to track resonances rather than harmonic peaks.
    if preemphasis_alpha > 0.0:
        analysis = np.empty_like(frame)
        analysis[0]  = frame[0]
        analysis[1:] = frame[1:] - preemphasis_alpha * frame[:-1]
    else:
        analysis = frame

    lp_order = order if order is not None else max(2 * n_formants, _DEFAULT_ORDER)

    # Fit time-varying LP and take centre sub-frame coefficients
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        coeffs_all = fit_tvlp(analysis, sr,
                               order=lp_order,
                               n_sub=n_sub,
                               lambda_smooth=lambda_smooth)

    centre_idx = n_sub // 2
    coeffs     = coeffs_all[centre_idx]

    freqs, bws = _poles_to_formants(coeffs, sr, max_formant_hz, n_formants)

    return freqs, bws, slope
