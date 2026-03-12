# synth/kl_config.py
"""
Kelly-Lochbaum backend configuration, per-sample state, and physical constants.

See KL_SYNTH_SPEC.md §2 (KLConfig) and §3.1 (constants).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple, Optional


# ---------------------------------------------------------------------------
# Physical constants — CGS, per spec §3.1
# ---------------------------------------------------------------------------

C_CM_S: float = 34_300.0        # speed of sound in warm vocal tract (cm/s)
RHO_G_CM3: float = 1.14e-3     # air density at ~35 °C (g/cm³)


# ---------------------------------------------------------------------------
# Per-sample state returned by KLTract.step()
# ---------------------------------------------------------------------------

class KLState(NamedTuple):
    """Internal state snapshot from one time step of the KL waveguide."""
    p_rad: float      # radiated pressure sample (LIP-4)
    u_lip: float      # lip volume velocity before differentiation (LIP-3)
    p_fwd_0: float    # forward wave at glottal end after this step
    p_bwd_0: float    # backward wave at glottal end (from previous step)
    ug: float         # glottal volume velocity (GL-2)


# ---------------------------------------------------------------------------
# Backend configuration — spec §2
# ---------------------------------------------------------------------------

@dataclass
class KLConfig:
    """Configuration for the Kelly-Lochbaum waveguide backend."""
    n_sections:       int   = 0       # 0 = auto from VTL and sr (§3.1)
    register:         Optional[str]   = None   # "M1"|"M2"|"M3"; None = random
    effort:           Optional[str]   = None   # "soft"|"moderate"|"loud"; None = random
    phonation_type:   Optional[str]   = None   # "modal"|"breathy"|"pressed"; None = random
    glottal_gap_frac: Optional[float] = None   # 0–1; None = derived from phonation_type
    subglottal_p_pa:  float = 800.0
    loss_per_section: float = 0.002
    r_lip:            float = 0.9              # lip reflection magnitude
    coupled:          bool  = True             # False = fixed flow, no feedback
    oq:  Optional[float] = None               # open quotient
    sq:  Optional[float] = None               # speed quotient
    rd:  Optional[float] = None               # Rd glottal shape parameter
