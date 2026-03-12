# synth/tests/test_synth_kl.py
"""
Kelly-Lochbaum waveguide unit tests (R11–R21).

All test setups and expected values derive from KL_SYNTH_SPEC.md §4.
Tolerance categories: EXACT (float64 precision), THEORY (DFT bin resolution),
EMPIRICAL (directional / sign only).
"""

from __future__ import annotations

import math
import sys
import traceback

import numpy as np

from test_kl_registry import _results, _check, _approx, _gt, _lt, _print_summary

# ---------------------------------------------------------------------------
# Imports from the synth package
# ---------------------------------------------------------------------------

_KL_ERR = ""
try:
    from synth.kl import (
        KLTract, reflection_coefficients, n_sections_from_vtl,
        _glottal_area_waveform, generate_vowel_kl,
    )
    from synth.kl_config import KLConfig, KLState, C_CM_S, RHO_G_CM3
    from synth.synth import SynthConfig
    _KL_OK = True
except Exception:
    _KL_OK = False
    _KL_ERR = traceback.format_exc()


# ---------------------------------------------------------------------------
# Test helper — eigenvalue reference frequencies (spec §4 Test Helper)
# ---------------------------------------------------------------------------

def _build_system_matrix(
    areas: np.ndarray,
    r_lip: float,
    alpha: float = 0.0,
) -> np.ndarray:
    """
    Build the 2N × 2N state-transition matrix M for the waveguide.

    State vector: x = [p⁺₀, ..., p⁺_{N-1}, p⁻₀, ..., p⁻_{N-1}]
    x[t+1] = M · x[t]

    Encodes junction scattering (KL-1), glottal total reflection (Ug=0),
    lip boundary (p⁻_{N-1} = −r_lip·p⁺_{N-1}), and loss.

    This is an entirely independent computation from the scattering loop.
    """
    N = len(areas)
    k = reflection_coefficients(areas)  # length N-1
    loss = 1.0 - alpha

    M = np.zeros((2 * N, 2 * N), dtype=np.float64)

    # Internal junction scattering via KL-2 matrix form:
    #   p⁺_{n+1} = (1+k)*p⁺_n - k*p⁻_{n+1}
    #   p⁻_n     =  k*p⁺_n + (1-k)*p⁻_{n+1}
    for n in range(N - 1):
        kn = k[n]
        row_fwd  = n + 1        # output: p⁺_{n+1}
        row_bwd  = N + n        # output: p⁻_n
        col_fwd  = n            # input:  p⁺_n
        col_bwd1 = N + n + 1   # input:  p⁻_{n+1}

        M[row_fwd, col_fwd]  = 1.0 + kn
        M[row_fwd, col_bwd1] = -kn
        M[row_bwd, col_fwd]  = kn
        M[row_bwd, col_bwd1] = 1.0 - kn

    # Glottal boundary: total reflection (Ug=0) → p⁺₀[t+1] = p⁻₀[t]
    M[0, N] = 1.0

    # Lip boundary: p⁻_{N-1}[t+1] = −r_lip · p⁺_{N-1}[t]
    M[2 * N - 1, N - 1] = -r_lip

    M *= loss
    return M


def _eigenvalue_resonances(
    areas: np.ndarray,
    sr: int,
    r_lip: float,
    alpha: float = 0.0,
    min_mag: float = 0.5,
) -> np.ndarray:
    """
    Compute resonance frequencies via matrix eigenvalues.

    Returns sorted array of frequencies (Hz) with |λ| > min_mag and f > 0.
    """
    M = _build_system_matrix(areas, r_lip, alpha)
    eigenvalues = np.linalg.eigvals(M)

    freqs = []
    for lam in eigenvalues:
        if abs(lam) > min_mag:
            angle = np.angle(lam)
            if angle > 0:
                f = angle * sr / (2.0 * math.pi)
                if 0 < f < sr / 2:
                    freqs.append(f)

    return np.sort(np.array(freqs))


# ---------------------------------------------------------------------------
# Other shared helpers
# ---------------------------------------------------------------------------

def _spectral_slope(signal: np.ndarray, sr: int, f0_hz: float,
                    n_harmonics: int = 10) -> float:
    """Spectral slope (dB/octave) via linear regression on harmonics H1..Hn."""
    N_fft = 2 ** int(np.ceil(np.log2(len(signal))))
    mag = np.abs(np.fft.rfft(signal, n=N_fft))
    bin_width = sr / N_fft

    amps_db, log_freqs = [], []
    for h in range(1, n_harmonics + 1):
        f_h = h * f0_hz
        idx = int(round(f_h / bin_width))
        if idx >= len(mag):
            break
        amp = mag[idx]
        if amp > 0:
            amps_db.append(20.0 * np.log10(amp))
            log_freqs.append(np.log2(f_h))

    if len(amps_db) < 3:
        return float('nan')

    x = np.array(log_freqs)
    y = np.array(amps_db)
    slope, _ = np.linalg.lstsq(np.vstack([x, np.ones(len(x))]).T, y, rcond=None)[0]
    return float(slope)


def _spectral_centroid(signal: np.ndarray, sr: int) -> float:
    """Spectral centroid (Hz)."""
    N_fft = 2 ** int(np.ceil(np.log2(len(signal))))
    mag = np.abs(np.fft.rfft(signal, n=N_fft))
    freqs = np.fft.rfftfreq(N_fft, 1.0 / sr)
    total = np.sum(mag)
    return float(np.sum(freqs * mag) / total) if total > 1e-30 else 0.0


def _impulse_ag(n_samples: int, A0: float, p_sub_cgs: float) -> np.ndarray:
    """
    Single-sample glottal area pulse producing p_fwd_0 = 1.0 at t=0
    in uncoupled mode with p_bwd_0_prev = 0.

    From GL-3 (p⁻₀=0): p⁺₀ = ρc·Ug/(2·A₀)  →  Ug = 2·A₀/ρc
    From GL-2:           Ug = ag·√(2·P_sub/ρ)  →  ag = 2·A₀/(ρc·√(2·P_sub/ρ))
    """
    rho_c = RHO_G_CM3 * C_CM_S
    ug_target = 2.0 * A0 / rho_c
    ag_val = ug_target / math.sqrt(2.0 * p_sub_cgs / RHO_G_CM3)
    ag = np.zeros(n_samples, dtype=np.float64)
    ag[0] = ag_val
    return ag


def _run_kl(
    areas: np.ndarray,
    sr: int,
    kl_cfg: KLConfig,
    ag: np.ndarray,
    p_sub_pa: float,
    record_state: bool = False,
) -> tuple:
    """Run the KL waveguide. Returns (p_rad, states[, p_fwd_hist, p_bwd_hist])."""
    tract = KLTract(areas, sr, kl_cfg)
    n = len(ag)
    output = np.zeros(n, dtype=np.float64)
    states = []
    p_fwd_hist, p_bwd_hist = [], []
    for t in range(n):
        st = tract.step(ag[t], p_sub_pa)
        output[t] = st.p_rad
        states.append(st)
        if record_state:
            p_fwd_hist.append(tract.p_fwd.copy())
            p_bwd_hist.append(tract.p_bwd.copy())
    if record_state:
        return output, states, p_fwd_hist, p_bwd_hist
    return output, states


# ===========================================================================
# Tests
# ===========================================================================

def test_r11_uniform_resonances():
    """
    R11: Uniform tube impulse → resonances match discrete eigenvalue method.
    Spec §4, Test 1.  Tolerance: THEORY ±1 FFT bin (≈1.95 Hz at N_fft=8192).
    """
    if not _KL_OK:
        _check("R11_import", False, _KL_ERR); return

    N = 16; sr = 16_000; r_lip = 0.9
    areas = np.full(N, 4.0)
    kl_cfg = KLConfig(loss_per_section=0.0, r_lip=r_lip, coupled=False)
    p_sub_pa = 800.0

    ag = _impulse_ag(8192, A0=4.0, p_sub_cgs=p_sub_pa * 10.0)
    output, _ = _run_kl(areas, sr, kl_cfg, ag, p_sub_pa)

    ref_freqs = _eigenvalue_resonances(areas, sr, r_lip)

    N_fft = 8192
    bin_width = sr / N_fft
    mag = np.abs(np.fft.rfft(output, n=N_fft))
    freqs = np.fft.rfftfreq(N_fft, 1.0 / sr)

    threshold = 0.05 * np.max(mag)
    peaks = [freqs[i] for i in range(1, len(mag)-1)
             if mag[i] > mag[i-1] and mag[i] > mag[i+1] and mag[i] > threshold]

    _check("R11_peaks_found",   len(peaks)     >= 3, f"found {len(peaks)} peaks")
    _check("R11_ref_found",     len(ref_freqs) >= 3, f"eigenvalue found {len(ref_freqs)} resonances")
    if not peaks or not len(ref_freqs): return

    for i, f_peak in enumerate(peaks[:3]):
        nearest = min(ref_freqs, key=lambda f: abs(f - f_peak))
        _approx(f"R11_resonance_{i+1}", f_peak, nearest, bin_width,
                 detail=f"peak={f_peak:.1f}Hz ref={nearest:.1f}Hz "
                        f"err={abs(f_peak-nearest):.1f}Hz tol=1bin={bin_width:.2f}Hz")


def test_r12_energy_conservation():
    """
    R12: Instantaneous power balance at every time step (lossless, r_lip=0.9).
    Spec §4, Test 2.  Tolerance: EXACT 1e-12.
    """
    if not _KL_OK:
        _check("R12_import", False, _KL_ERR); return

    N = 16; sr = 16_000; r_lip = 0.9
    areas = np.array([2.0 if i % 2 == 0 else 6.0 for i in range(N)], dtype=np.float64)
    kl_cfg = KLConfig(loss_per_section=0.0, r_lip=r_lip, coupled=False)

    n_inject = 100; n_total = n_inject + 200
    ag = np.zeros(n_total, dtype=np.float64)
    for t in range(n_inject):
        ag[t] = max(0.0, 0.05 * math.sin(2.0 * math.pi * 200.0 * t / sr))

    _, states, p_fwd_hist, p_bwd_hist = _run_kl(
        areas, sr, kl_cfg, ag, 800.0, record_state=True)

    # Power balance quantities — spec §4 Test 2
    def tube_energy(pf, pb):
        return float(np.sum(areas * (pf**2 + pb**2)))

    def power_in(pf, pb):
        return areas[0] * (pf[0]**2 - pb[0]**2)

    def power_out(pf, pb):
        return areas[N-1] * (pf[N-1]**2 - pb[N-1]**2)

    max_residual = 0.0
    e_prev = 0.0
    for t in range(n_total):
        e_curr = tube_energy(p_fwd_hist[t], p_bwd_hist[t])
        r = abs(e_curr - e_prev
                - power_in(p_fwd_hist[t], p_bwd_hist[t])
                + power_out(p_fwd_hist[t], p_bwd_hist[t]))
        max_residual = max(max_residual, r)
        e_prev = e_curr

    _lt("R12_power_balance", max_residual, 1e-12,
        f"max instantaneous power-balance residual = {max_residual:.2e}")
    _gt("R12_output_nonzero", float(np.max(np.abs([s.p_rad for s in states]))), 1e-15,
        "output should be nonzero")


def test_r13_uniform_delay():
    """
    R13: Uniform tube delay — forward wave arrives after N-1 samples.
    Spec §4, Test 3.  Tolerance: EXACT.
    """
    if not _KL_OK:
        _check("R13_import", False, _KL_ERR); return

    N = 16; sr = 16_000
    areas = np.full(N, 4.0)
    kl_cfg = KLConfig(loss_per_section=0.0, r_lip=0.0, coupled=False)
    p_sub_pa = 800.0

    ag = _impulse_ag(N + 15, A0=4.0, p_sub_cgs=p_sub_pa * 10.0)
    tract = KLTract(areas, sr, kl_cfg)
    u_lip = np.array([tract.step(ag[t], p_sub_pa).u_lip for t in range(len(ag))])

    first_nz = next((t for t in range(len(u_lip)) if abs(u_lip[t]) > 1e-15), -1)
    _check("R13_delay_lag", first_nz == N - 1,
           f"first nonzero u_lip at t={first_nz}, expected t={N-1}")

    # With r_lip=0: u_lip = A_{N-1}·(1+0)·p_fwd_{N-1}/(ρc) = A·1.0/ρc
    rho_c = RHO_G_CM3 * C_CM_S
    expected = areas[N-1] / rho_c
    if first_nz >= 0:
        _approx("R13_delay_amplitude", u_lip[first_nz], expected, 1e-12,
                 detail=f"u_lip[{first_nz}]={u_lip[first_nz]:.6e} expected={expected:.6e}")


def test_r14_two_tube_f1():
    """
    R14: Two-tube model F1 matches eigenvalue reference.
    Spec §4, Test 4.  Tolerance: THEORY ±1 FFT bin (≈1.95 Hz at N_fft=8192).
    """
    if not _KL_OK:
        _check("R14_import", False, _KL_ERR); return

    N = 16; sr = 16_000; r_lip = 0.9
    A1, A2 = 8.0, 2.0
    areas = np.array([A1]*8 + [A2]*8, dtype=np.float64)
    kl_cfg = KLConfig(loss_per_section=0.0, r_lip=r_lip, coupled=False)
    p_sub_pa = 800.0

    ag = _impulse_ag(8192, A0=A1, p_sub_cgs=p_sub_pa * 10.0)
    output, _ = _run_kl(areas, sr, kl_cfg, ag, p_sub_pa)

    ref_freqs = _eigenvalue_resonances(areas, sr, r_lip)

    N_fft = 8192
    bin_width = sr / N_fft
    mag = np.abs(np.fft.rfft(output, n=N_fft))
    freqs = np.fft.rfftfreq(N_fft, 1.0 / sr)

    threshold = 0.05 * np.max(mag)
    peaks = [freqs[i] for i in range(1, len(mag)-1)
             if mag[i] > mag[i-1] and mag[i] > mag[i+1] and mag[i] > threshold]

    _check("R14_peaks_found", len(peaks) >= 1, f"found {len(peaks)} peaks")
    _check("R14_ref_found",   len(ref_freqs) >= 1, f"eigenvalue found {len(ref_freqs)} resonances")
    if not peaks or not len(ref_freqs): return

    f1_fft = peaks[0]
    f1_ref  = min(ref_freqs, key=lambda f: abs(f - f1_fft))
    _approx("R14_f1_match", f1_fft, f1_ref, bin_width,
             detail=f"FFT F1={f1_fft:.1f}Hz ref={f1_ref:.1f}Hz "
                    f"err={abs(f1_fft-f1_ref):.1f}Hz tol=1bin={bin_width:.2f}Hz")

    # Sanity: two-tube F1 should differ from uniform
    uniform_refs = _eigenvalue_resonances(np.full(N, 4.0), sr, r_lip)
    if len(uniform_refs):
        _check("R14_f1_differs_from_uniform",
               abs(f1_fft - uniform_refs[0]) > bin_width,
               f"two-tube F1={f1_fft:.1f}Hz should differ from uniform F1={uniform_refs[0]:.1f}Hz")


def test_r15_coupling_modifies_slope():
    """
    R15: Coupling modifies spectral slope (|diff| > threshold, direction not asserted).
    Spec §4, Test 5.  Tolerance: EMPIRICAL nonzero difference.
    """
    if not _KL_OK:
        _check("R15_import", False, _KL_ERR); return

    sr = 16_000; f0_hz = 120.0
    areas = np.array([8.0]*8 + [2.0]*8, dtype=np.float64)
    n_samples = int(sr * 0.3)
    rng = np.random.default_rng(42)

    ag = _glottal_area_waveform(n_samples, sr, f0_hz, peak_area=0.10,
                                 oq=0.6, jitter_frac=0.0, shimmer_frac=0.0, rng=rng)

    out_c, _ = _run_kl(areas, sr, KLConfig(loss_per_section=0.001, r_lip=0.9, coupled=True),  ag, 800.0)
    out_u, _ = _run_kl(areas, sr, KLConfig(loss_per_section=0.001, r_lip=0.9, coupled=False), ag, 800.0)

    slope_c = _spectral_slope(out_c, sr, f0_hz)
    slope_u = _spectral_slope(out_u, sr, f0_hz)
    diff = abs(slope_c - slope_u)

    _gt("R15_coupling_changes_slope", diff, 0.05,
        f"|slope_coupled({slope_c:.3f}) - slope_uncoupled({slope_u:.3f})| = {diff:.3f} dB/oct")


def test_r16_coupling_scales_with_area():
    """
    R16: Coupling strength scales with glottal area.
    Spec §4, Test 6.  Tolerance: EMPIRICAL directional.
    """
    if not _KL_OK:
        _check("R16_import", False, _KL_ERR); return

    sr = 16_000; f0_hz = 120.0
    areas = np.array([8.0]*8 + [2.0]*8, dtype=np.float64)
    n_samples = int(sr * 0.3)

    def _slope_diff(peak_area: float) -> float:
        rng = np.random.default_rng(42)
        ag = _glottal_area_waveform(n_samples, sr, f0_hz, peak_area=peak_area,
                                     oq=0.6, jitter_frac=0.0, shimmer_frac=0.0, rng=rng)
        out_c, _ = _run_kl(areas, sr, KLConfig(loss_per_section=0.001, r_lip=0.9, coupled=True),  ag, 800.0)
        out_u, _ = _run_kl(areas, sr, KLConfig(loss_per_section=0.001, r_lip=0.9, coupled=False), ag, 800.0)
        return _spectral_slope(out_c, sr, f0_hz) - _spectral_slope(out_u, sr, f0_hz)

    diff_small = _slope_diff(0.05)
    diff_large = _slope_diff(0.20)

    _gt("R16_larger_area_stronger_coupling", abs(diff_large), abs(diff_small),
        f"|diff_large|={abs(diff_large):.3f} > |diff_small|={abs(diff_small):.3f}")


def test_r17_lip_radiation_highpass():
    """
    R17: Lip radiation is highpass (spectral centroid of p_rad > u_lip).
    Spec §4, Test 7.  Tolerance: EMPIRICAL directional.
    """
    if not _KL_OK:
        _check("R17_import", False, _KL_ERR); return

    sr = 16_000; f0_hz = 150.0
    areas = np.array([6.0]*8 + [3.0]*8, dtype=np.float64)
    n_samples = int(sr * 0.2)
    rng = np.random.default_rng(7)

    ag = _glottal_area_waveform(n_samples, sr, f0_hz, peak_area=0.10,
                                 oq=0.6, jitter_frac=0.0, shimmer_frac=0.0, rng=rng)
    _, states = _run_kl(areas, sr, KLConfig(loss_per_section=0.001, r_lip=0.9, coupled=False), ag, 800.0)

    u_lip_arr = np.array([s.u_lip for s in states])
    p_rad_arr = np.array([s.p_rad for s in states])

    c_lip = _spectral_centroid(u_lip_arr, sr)
    c_rad = _spectral_centroid(p_rad_arr, sr)
    _gt("R17_radiation_highpass", c_rad, c_lip,
        f"centroid_rad={c_rad:.1f}Hz > centroid_lip={c_lip:.1f}Hz")


def test_r18_reflection_coeff_bounds():
    """
    R18: Reflection coefficient bounds.
    Spec §4, Test 8.  Tolerance: EXACT.
    """
    if not _KL_OK:
        _check("R18_import", False, _KL_ERR); return

    _check("R18_equal_areas_zero",
           np.allclose(reflection_coefficients(np.array([4.0, 4.0, 4.0])), 0.0),
           "k should be all zero for equal areas")

    k_pos = reflection_coefficients(np.array([8.0, 2.0]))
    _check("R18_wider_to_narrower_positive", float(k_pos[0]) > 0,
           f"k={k_pos[0]:.4f} should be > 0 for 8→2")

    k_neg = reflection_coefficients(np.array([2.0, 8.0]))
    _check("R18_narrower_to_wider_negative", float(k_neg[0]) < 0,
           f"k={k_neg[0]:.4f} should be < 0 for 2→8")

    rng = np.random.default_rng(99)
    all_ok = all(
        np.all(np.abs(reflection_coefficients(rng.uniform(0.1, 50.0, size=10))) < 1.0)
        for _ in range(20)
    )
    _check("R18_bounded", all_ok, "|k| < 1 for 20 random area functions")

    _approx("R18_exact_value", float(reflection_coefficients(np.array([8.0, 2.0]))[0]),
             0.6, 1e-15, detail="k = (8-2)/(8+2) = 0.6")


def test_r19_uncoupled_flow_independence():
    """
    R19: Uncoupled glottal flow is independent of tract shape.
    Spec §4, Test 9.  Tolerance: EXACT 1e-12.
    """
    if not _KL_OK:
        _check("R19_import", False, _KL_ERR); return

    N = 16; sr = 16_000
    rng = np.random.default_rng(0)
    ag = _glottal_area_waveform(200, sr, 120.0, peak_area=0.10,
                                 oq=0.6, jitter_frac=0.0, shimmer_frac=0.0, rng=rng)

    areas_1 = np.full(N, 4.0)
    areas_2 = np.array([4.0]+[2.0 if i%2==0 else 8.0 for i in range(1,N)], dtype=np.float64)

    kl_cfg = KLConfig(loss_per_section=0.001, r_lip=0.9, coupled=False)
    _, s1 = _run_kl(areas_1, sr, kl_cfg, ag, 800.0)
    _, s2 = _run_kl(areas_2, sr, kl_cfg, ag, 800.0)

    max_diff = float(np.max(np.abs(np.array([s.ug for s in s1]) - np.array([s.ug for s in s2]))))
    _lt("R19_uncoupled_flow_identical", max_diff, 1e-12,
        f"max|Ug_1 - Ug_2| = {max_diff:.2e}")


def test_r20_coupled_flow_depends_on_tract():
    """
    R20: Coupled glottal flow depends on tract shape.
    Spec §4, Test 10.  Tolerance: EMPIRICAL directional.
    """
    if not _KL_OK:
        _check("R20_import", False, _KL_ERR); return

    N = 16; sr = 16_000
    rng = np.random.default_rng(0)
    ag = _glottal_area_waveform(400, sr, 120.0, peak_area=0.10,
                                 oq=0.6, jitter_frac=0.0, shimmer_frac=0.0, rng=rng)

    areas_1 = np.full(N, 4.0)
    areas_2 = np.array([4.0]+[2.0 if i%2==0 else 8.0 for i in range(1,N)], dtype=np.float64)

    kl_cfg = KLConfig(loss_per_section=0.001, r_lip=0.9, coupled=True)
    _, s1 = _run_kl(areas_1, sr, kl_cfg, ag, 800.0)
    _, s2 = _run_kl(areas_2, sr, kl_cfg, ag, 800.0)

    ug_1 = np.array([s.ug for s in s1])
    ug_2 = np.array([s.ug for s in s2])
    max_diff  = float(np.max(np.abs(ug_1 - ug_2)))
    peak_flow = max(float(np.max(np.abs(ug_1))), float(np.max(np.abs(ug_2))))
    threshold = 0.01 * peak_flow if peak_flow > 0 else 1e-6

    _gt("R20_coupled_flow_differs", max_diff, threshold,
        f"max|Ug_1 - Ug_2|={max_diff:.4f} threshold={threshold:.4f} (1% of {peak_flow:.4f})")


def test_r21_gl3_term_by_term():
    """
    R21: GL-3 boundary equation term-by-term verification.
    Spec §4, Test 11.  Tolerance: EXACT 1e-12.
    """
    if not _KL_OK:
        _check("R21_import", False, _KL_ERR); return

    N = 16; A0 = 4.0; sr = 16_000
    kl_cfg = KLConfig(loss_per_section=0.0, r_lip=0.9, coupled=False)
    p_sub_pa = 800.0; p_sub_cgs = p_sub_pa * 10.0
    rho_c = RHO_G_CM3 * C_CM_S

    rng = np.random.default_rng(0)
    ag = _glottal_area_waveform(200, sr, 120.0, peak_area=0.10,
                                 oq=0.6, jitter_frac=0.0, shimmer_frac=0.0, rng=rng)

    tract = KLTract(np.full(N, A0), sr, kl_cfg)
    max_gl2_err = max_gl3_err = 0.0

    for t in range(len(ag)):
        p_bwd_0_prev = tract.p_bwd[0]
        st = tract.step(ag[t], p_sub_pa)

        # GL-2 (uncoupled: ΔP = P_sub in CGS)
        if ag[t] > 0.0 and abs(p_sub_cgs) > 0.0:
            ug_exp = ag[t] * math.copysign(1.0, p_sub_cgs) * math.sqrt(
                2.0 * abs(p_sub_cgs) / RHO_G_CM3)
        else:
            ug_exp = 0.0
        max_gl2_err = max(max_gl2_err, abs(st.ug - ug_exp))

        # GL-3: p⁺₀ = (ρc·Ug)/(2·A₀) + p⁻₀[t-1]/2
        p_fwd_0_exp = (rho_c * st.ug) / (2.0 * A0) + p_bwd_0_prev / 2.0
        max_gl3_err = max(max_gl3_err, abs(st.p_fwd_0 - p_fwd_0_exp))

    _lt("R21_GL2_exact", max_gl2_err, 1e-12, f"max GL-2 error = {max_gl2_err:.2e}")
    _lt("R21_GL3_exact", max_gl3_err, 1e-12, f"max GL-3 error = {max_gl3_err:.2e}")


# ===========================================================================
# Runner
# ===========================================================================

if __name__ == "__main__":
    test_r11_uniform_resonances()
    test_r12_energy_conservation()
    test_r13_uniform_delay()
    test_r14_two_tube_f1()
    test_r15_coupling_modifies_slope()
    test_r16_coupling_scales_with_area()
    test_r17_lip_radiation_highpass()
    test_r18_reflection_coeff_bounds()
    test_r19_uncoupled_flow_independence()
    test_r20_coupled_flow_depends_on_tract()
    test_r21_gl3_term_by_term()

    n_fail = _print_summary()
    raise SystemExit(0 if n_fail == 0 else 1)
