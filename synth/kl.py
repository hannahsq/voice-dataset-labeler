# synth/kl.py
"""
Kelly-Lochbaum waveguide vocal-tract synthesis backend.

Implements a digital waveguide model that propagates forward (+) and backward (−)
pressure waves through a chain of cylindrical tube sections.  Source-filter
coupling arises because glottal flow depends on the pressure reflected back
from the tract.

See KL_SYNTH_SPEC.md for the full specification and mathematical derivations.
All equation references (GL-*, KL-*, LIP-*, EN-*) refer to that document.
"""

from __future__ import annotations

import math
import warnings
from typing import Optional

import numpy as np

from synth.kl_config import KLConfig, KLState, C_CM_S, RHO_G_CM3
from synth.synth import (
    SynthConfig, VowelMeta,
    SPEED_OF_SOUND_MM_S, VTL_MIN_MM, VTL_MAX_MM,
    F0_MIN_HZ, F0_MAX_HZ, _delta_f,
)


# ---------------------------------------------------------------------------
# Tube geometry helpers — spec §3.1
# ---------------------------------------------------------------------------

def n_sections_from_vtl(vtl_mm: float, sr: int) -> int:
    """
    Number of tube sections for a given VTL and sample rate.

    N = round(2 · L · sr / c)  where L = vtl_mm / 10 (mm → cm).
    This ensures a one-sample propagation delay per section (§3.1).
    """
    L_cm = vtl_mm / 10.0
    return round(2.0 * L_cm * sr / C_CM_S)


# ---------------------------------------------------------------------------
# Reflection coefficients — spec §3.3
# ---------------------------------------------------------------------------

def reflection_coefficients(areas: np.ndarray) -> np.ndarray:
    """
    Compute junction reflection coefficients from cross-sectional areas.

    k[n] = (A[n] − A[n+1]) / (A[n] + A[n+1])   for n = 0 .. N-2

    Returns array of length N-1 (float64).
    """
    areas = np.asarray(areas, dtype=np.float64)
    return (areas[:-1] - areas[1:]) / (areas[:-1] + areas[1:])


# ---------------------------------------------------------------------------
# KLTract — core waveguide engine
# ---------------------------------------------------------------------------

class KLTract:
    """
    Kelly-Lochbaum digital waveguide vocal-tract model.

    Exposes a per-sample ``step()`` method that returns internal state,
    enabling both batch synthesis and fine-grained test verification
    (spec §4, Test 11, option b).
    """

    def __init__(
        self,
        areas: np.ndarray,
        sr: int,
        kl_cfg: Optional[KLConfig] = None,
    ) -> None:
        if kl_cfg is None:
            kl_cfg = KLConfig()

        self._areas = np.asarray(areas, dtype=np.float64)
        self._N = len(self._areas)
        self._sr = sr
        self._cfg = kl_cfg

        # Precompute reflection coefficients (§3.3)
        self._k = reflection_coefficients(self._areas)

        # Physical constants
        self._rho_c = RHO_G_CM3 * C_CM_S       # ρc  (g / (cm² · s))

        # Loss factor (§3.8)
        self._one_minus_alpha = 1.0 - kl_cfg.loss_per_section

        # Allocate double buffers (§3.4)
        self._p_fwd     = np.zeros(self._N, dtype=np.float64)
        self._p_bwd     = np.zeros(self._N, dtype=np.float64)
        self._p_fwd_new = np.zeros(self._N, dtype=np.float64)
        self._p_bwd_new = np.zeros(self._N, dtype=np.float64)

        # Lip radiation state (§3.7, LIP-4)
        self._u_lip_prev = 0.0

    # -- public read-only access for tests (energy calculation, §3.6) ------

    @property
    def p_fwd(self) -> np.ndarray:
        return self._p_fwd

    @property
    def p_bwd(self) -> np.ndarray:
        return self._p_bwd

    @property
    def areas(self) -> np.ndarray:
        return self._areas

    @property
    def N(self) -> int:
        return self._N

    def reset(self) -> None:
        """Zero all internal state."""
        self._p_fwd[:]     = 0.0
        self._p_bwd[:]     = 0.0
        self._p_fwd_new[:] = 0.0
        self._p_bwd_new[:] = 0.0
        self._u_lip_prev   = 0.0

    # -- core time-step (spec §3.4 update order) --------------------------

    def step(self, ag: float, p_sub: float) -> KLState:
        """
        Advance the waveguide by one sample.

        Parameters
        ----------
        ag    : glottal area (cm²) at this time step
        p_sub : subglottal pressure (dyn/cm² in CGS, but we treat the
                value as consistent with the constants — see note below)

        Returns
        -------
        KLState with radiated pressure, lip velocity, and internal values.

        Note: p_sub from KLConfig.subglottal_p_pa is in Pa. Since we work
        in CGS, 1 Pa = 10 dyn/cm². The conversion is applied here.
        """
        p_sub_cgs = p_sub * 10.0   # Pa → dyn/cm²

        N   = self._N
        k   = self._k
        p_f = self._p_fwd
        p_b = self._p_bwd
        pf  = self._p_fwd_new
        pb  = self._p_bwd_new

        # 1. Glottal boundary (§3.5, GL-1 / GL-2 / GL-3)
        p_bwd_0_prev = p_b[0]   # p⁻₀[t-1]

        if self._cfg.coupled:
            delta_p = p_sub_cgs - p_bwd_0_prev          # GL-1
        else:
            delta_p = p_sub_cgs                          # uncoupled

        # GL-2: Bernoulli glottal flow
        if ag > 0.0 and abs(delta_p) > 0.0:
            ug = ag * math.copysign(1.0, delta_p) * math.sqrt(
                2.0 * abs(delta_p) / RHO_G_CM3
            )
        else:
            ug = 0.0

        # GL-3: forward wave at glottal end
        A0 = self._areas[0]
        pf[0] = (self._rho_c * ug) / (2.0 * A0) + p_bwd_0_prev / 2.0

        # 2. Junction scattering — double-buffer (§3.3, KL-1a / KL-1b)
        for n in range(N - 1):
            s = k[n] * (p_f[n] - p_b[n + 1])
            pf[n + 1] = p_f[n] + s        # KL-1a
            pb[n]     = p_b[n + 1] + s     # KL-1b

        # 3. Lip boundary (§3.7, LIP-1)
        r_lip = self._cfg.r_lip
        pb[N - 1] = -r_lip * pf[N - 1]

        # 4. Lip volume velocity (§3.7, LIP-3) and radiation (LIP-4)
        u_lip = (self._areas[N - 1] * (1.0 + r_lip) * pf[N - 1]) / self._rho_c
        p_rad = u_lip - self._u_lip_prev
        self._u_lip_prev = u_lip

        # 5. Per-section losses (§3.8)
        if self._one_minus_alpha != 1.0:
            pf *= self._one_minus_alpha
            pb *= self._one_minus_alpha

        # 6. Buffer swap (reference swap, not copy)
        self._p_fwd, self._p_fwd_new = pf, p_f
        self._p_bwd, self._p_bwd_new = pb, p_b

        # 7. Return state snapshot
        return KLState(
            p_rad   = p_rad,
            u_lip   = u_lip,
            p_fwd_0 = pf[0],      # after swap, pf is the "current" buffer
            p_bwd_0 = p_bwd_0_prev,
            ug      = ug,
        )


# ---------------------------------------------------------------------------
# Glottal area waveform
# ---------------------------------------------------------------------------

def _glottal_area_waveform(
    n_samples:    int,
    sr:           int,
    f0_hz:        float,
    peak_area:    float = 0.10,     # cm²
    oq:           float = 0.6,      # open quotient
    jitter_frac:  float = 0.005,
    shimmer_frac: float = 0.03,
    rng:          Optional[np.random.Generator] = None,
) -> np.ndarray:
    """
    Generate a per-sample glottal area waveform (half-rectified sine).

    Each period of duration T = sr/f0:
      - Open phase:  ag = peak_area · sin(π · t / T_open)  for t in [0, T_open]
      - Closed phase: ag = 0  for t in [T_open, T]
    where T_open = oq · T.

    Jitter and shimmer are applied per period (same convention as Klatt backend).
    """
    if rng is None:
        rng = np.random.default_rng()

    ag = np.zeros(n_samples, dtype=np.float64)
    period = sr / f0_hz
    t = 0.0
    amp = peak_area

    while t < n_samples:
        T = period
        # Jitter
        T += rng.normal(0.0, jitter_frac * period)
        T = max(T, 2.0)  # safety floor

        T_open = oq * T
        n_open = int(round(T_open))
        pos = int(round(t))

        if pos >= n_samples:
            break

        # Write half-sine open phase
        for i in range(n_open):
            idx = pos + i
            if idx >= n_samples:
                break
            phase = math.pi * i / T_open
            ag[idx] = amp * math.sin(phase)

        # Advance to next period
        t += T

        # Shimmer
        amp = peak_area * (1.0 + rng.normal(0.0, shimmer_frac))
        amp = max(amp * 0.2, min(amp, peak_area * 2.0))

    return ag


# ---------------------------------------------------------------------------
# Public synthesis wrapper
# ---------------------------------------------------------------------------

def generate_vowel_kl(
    cfg: SynthConfig,
    rng: Optional[np.random.Generator] = None,
) -> tuple[np.ndarray, VowelMeta]:
    """
    Generate one synthetic vowel clip using the Kelly-Lochbaum backend.

    Currently produces a uniform-tube (neutral vowel) synthesis.
    Area-from-formants is deferred to a follow-up spec (§3.2).
    """
    if rng is None:
        rng = np.random.default_rng(cfg.rng_seed)

    kl_cfg = cfg.backend_cfg if isinstance(cfg.backend_cfg, KLConfig) else KLConfig()

    # --- Sample F0 and VTL ---
    f0_hz  = cfg.f0_hz  if cfg.f0_hz  is not None else float(rng.uniform(F0_MIN_HZ, F0_MAX_HZ))
    vtl_mm = cfg.vtl_mm if cfg.vtl_mm is not None else float(rng.uniform(VTL_MIN_MM, VTL_MAX_MM))
    df     = _delta_f(vtl_mm)

    # --- Tube geometry (§3.1) ---
    N = kl_cfg.n_sections if kl_cfg.n_sections > 0 else n_sections_from_vtl(vtl_mm, cfg.sr)
    areas = np.full(N, 4.0, dtype=np.float64)   # uniform tube, 4 cm²

    # --- Glottal area waveform ---
    n_samples = int(round(cfg.sr * cfg.duration_s))
    oq = kl_cfg.oq if kl_cfg.oq is not None else 0.6
    ag = _glottal_area_waveform(
        n_samples, cfg.sr, f0_hz,
        peak_area=0.10, oq=oq,
        jitter_frac=cfg.jitter_frac,
        shimmer_frac=cfg.shimmer_frac,
        rng=rng,
    )

    # --- Run waveguide ---
    tract = KLTract(areas, cfg.sr, kl_cfg)
    p_sub = kl_cfg.subglottal_p_pa

    output = np.zeros(n_samples, dtype=np.float64)
    for t in range(n_samples):
        state = tract.step(ag[t], p_sub)
        output[t] = state.p_rad

    # --- Normalise amplitude ---
    peak = np.max(np.abs(output))
    if peak > 1e-9:
        output = (output / peak * cfg.amplitude).astype(np.float32)
    else:
        warnings.warn("KL synthesised signal has near-zero amplitude; check parameters.")
        output = output.astype(np.float32)

    # --- Ground-truth metadata ---
    # Uniform tube formants: f_n = (2n-1) · c/(4L)  (spec §3.1)
    L_cm = vtl_mm / 10.0
    formant_hz = np.array([
        (2 * i - 1) * C_CM_S / (4.0 * L_cm)
        for i in range(1, cfg.n_formants + 1)
    ], dtype=np.float32)

    fn_norm = (formant_hz / df).astype(np.float32)

    # Bandwidths: approximate from loss factor (§3.8): BW ≈ α · sr / π
    bw_approx = kl_cfg.loss_per_section * cfg.sr / math.pi
    if cfg.bw_hz is not None:
        bw_hz = np.asarray(cfg.bw_hz, dtype=np.float32)
    else:
        bw_hz = np.full(cfg.n_formants, bw_approx, dtype=np.float32)

    bn_norm = (bw_hz / df).astype(np.float32)

    meta = VowelMeta(
        sr             = cfg.sr,
        f0_hz          = f0_hz,
        vtl_mm         = vtl_mm,
        delta_f_hz     = df,
        formant_hz     = formant_hz,
        formant_bw_hz  = bw_hz,
        fn_norm        = fn_norm,
        bn_norm        = bn_norm,
        duration_s     = n_samples / cfg.sr,
        register       = kl_cfg.register,
        effort         = kl_cfg.effort,
        phonation_type = kl_cfg.phonation_type,
        glottal_gap_frac = kl_cfg.glottal_gap_frac,
    )

    return output, meta
