# Kelly-Lochbaum Synthesis Backend — Specification

**Status:** Draft
**Scope:** Core scattering loop, boundary conditions, and unit tests.
LF parameterisation and register/effort/phonation sampling are deferred to a follow-up spec.

---

## 1. Overview

This document specifies a Kelly-Lochbaum (KL) waveguide vocal-tract model with
nonlinear source-filter coupling at the glottal boundary. The implementation
lives behind the existing `generate_vowel` / `generate_dataset` public API
(see §2) and is selected via `SynthConfig.backend = "kelly_lochbaum"`.

The model propagates forward (+) and backward (−) *pressure* waves through a
chain of cylindrical tube sections. Coupling arises because glottal flow depends
on the pressure reflected back from the tract.

---

## 2. Public Interface (unchanged)

```python
@dataclass
class SynthConfig:
    sr:             int   = 16_000
    n_formants:     int   = 5
    duration_s:     float = 0.5
    f0_hz:          Optional[float] = None
    vtl_mm:         Optional[float] = None
    fn_norm:        Optional[np.ndarray] = None
    bw_hz:          Optional[np.ndarray] = None
    jitter_frac:    float = JITTER_FRAC
    shimmer_frac:   float = SHIMMER_FRAC
    amplitude:      float = 0.5
    rng_seed:       Optional[int] = None
    backend:        str = "klatt"            # "klatt" | "kelly_lochbaum"
    backend_cfg:    Optional[Any] = None     # KLConfig | KlattConfig | None
```

```python
@dataclass
class VowelMeta:
    sr:              int
    f0_hz:           float
    vtl_mm:          float
    delta_f_hz:      float
    formant_hz:      np.ndarray        # (n_formants,) float32
    formant_bw_hz:   np.ndarray        # (n_formants,) float32
    fn_norm:         np.ndarray
    bn_norm:         np.ndarray
    duration_s:      float
    # Optional KL-specific metadata
    register:        Optional[str]   = None
    effort:          Optional[str]   = None
    phonation_type:  Optional[str]   = None
    glottal_gap_frac: Optional[float] = None
    spectral_slope:  Optional[float] = None
    loudness_dbfs:   Optional[float] = None
```

```python
@dataclass
class KLConfig:
    n_sections:       int   = 0       # 0 = auto from VTL and sr (see §3.1)
    register:         Optional[str]   = None   # "M1"|"M2"|"M3"; None = random
    effort:           Optional[str]   = None   # "soft"|"moderate"|"loud"; None = random
    phonation_type:   Optional[str]   = None   # "modal"|"breathy"|"pressed"; None = random
    glottal_gap_frac: Optional[float] = None   # 0–1; None = derived from phonation_type
    subglottal_p_pa:  float = 800.0
    loss_per_section: float = 0.002
    r_lip:            float = 0.9              # lip reflection magnitude
    coupled:          bool  = True             # False = inject fixed flow (no feedback)
    oq:  Optional[float] = None
    sq:  Optional[float] = None
    rd:  Optional[float] = None
```

`generate_vowel` and `generate_dataset` signatures are unchanged.

---

## 3. Vocal Tract Model

### 3.1 Tube Geometry

The vocal tract of physical length `L` (derived from `vtl_mm`) is divided into
`N` equal-length cylindrical sections. Each section has length:

    Δx = c / (2 · sr)

where `c = 34300 cm/s` (speed of sound in air at body temperature, ~35°C) and
`sr` is the sample rate. This choice ensures a one-sample propagation delay per
section. The number of sections is:

    N = round(2 · L · sr / c)

For example: L = 17.5 cm, sr = 16000 → N = round(2 · 17.5 · 16000 / 34300) = round(16.33) = 16.

Each section n ∈ {0, 1, ..., N−1} has a cross-sectional area Aₙ (cm²).
Section 0 is the glottal (source) end; section N−1 is the lip (radiation) end.

**Constants:**

| Symbol | Value | Unit | Description |
|--------|-------|------|-------------|
| c      | 34300 | cm/s | Speed of sound at ~35°C |
| ρ      | 1.14e-3 | g/cm³ | Air density at ~35°C |

### 3.2 Area Function from Formants

*Deferred to follow-up spec.* For the core loop tests, area functions are
specified directly (uniform tube, two-tube, etc.).

### 3.3 Junction Scattering — One-Multiply Kelly-Lochbaum

At each junction between section n and section n+1, define the reflection
coefficient:

    kₙ = (Aₙ − Aₙ₊₁) / (Aₙ + Aₙ₊₁)

The scattering update (one-multiply form) at junction n is:

    s  = kₙ · (p⁺ₙ − p⁻ₙ₊₁)
    p⁺ₙ₊₁ ← p⁺ₙ  + s                               ... (KL-1a)
    p⁻ₙ   ← p⁻ₙ₊₁ + s                               ... (KL-1b)

where p⁺ₙ is the forward-travelling pressure wave leaving section n (propagating
toward the lips) and p⁻ₙ₊₁ is the backward-travelling wave leaving section n+1
(propagating toward the glottis). Both values are from the *previous* time step
(one-sample delay per section is implicit in the update order).

**Equivalent matrix form** (for reference / verification):

    ⎡ p⁺ₙ₊₁ ⎤     ⎡ 1+kₙ   kₙ  ⎤ ⎡ p⁺ₙ   ⎤
    ⎢        ⎥  =  ⎢             ⎥ ⎢        ⎥      ... (KL-2)
    ⎣ p⁻ₙ   ⎦     ⎣  kₙ   1-kₙ ⎤ ⎣ p⁻ₙ₊₁ ⎦

Note: This matrix is NOT unitary in the Euclidean sense (‖M x‖₂ ≠ ‖x‖₂ in
general). Energy conservation must be verified in the *acoustic power* sense
(see §3.6).

### 3.4 Propagation and Update Order

At each time step t, the update proceeds:

1. **Glottal boundary** (§3.5): compute p⁺₀[t] from subglottal pressure,
   glottal area, and p⁻₀[t−1].
2. **Forward pass** through junctions 0 .. N−2:
   apply (KL-1) using p⁺ₙ[t−1] and p⁻ₙ₊₁[t−1] for each junction.
   After the forward pass, all p⁺ₙ and p⁻ₙ are updated for time t.
3. **Lip boundary** (§3.7): compute p⁻_{N-1}[t] from p⁺_{N-1}[t] and
   compute the radiated output.
4. **Apply losses** (§3.8): multiply all waves by (1 − α).

**Implementation note:** The forward and backward waves are stored in two
arrays of length N. The update must be done with care to avoid overwriting
values needed later in the same time step. The standard approach is to update
junctions in order from glottis to lips for the forward wave and lips to glottis
for the backward wave, using *two separate passes*, or equivalently to read from
one buffer and write to another (double-buffering).

Specifically, the safe update order is:

```
# Double-buffer: read from (p_fwd, p_bwd), write to (p_fwd_new, p_bwd_new)
for n in range(N - 1):         # junctions 0 .. N-2
    s = k[n] * (p_fwd[n] - p_bwd[n + 1])
    p_fwd_new[n + 1] = p_fwd[n] + s
    p_bwd_new[n]     = p_bwd[n + 1] + s
```

Or equivalently, propagate in-place from the *lip end* backward for p_fwd
and from the *glottal end* forward for p_bwd, exploiting the one-sample delay
structure. Either approach is acceptable; double-buffering is less error-prone.

### 3.5 Glottal Boundary (Source-Filter Coupling)

Define:
- P_sub : subglottal pressure (Pa), constant or slowly varying
- ag[t] : time-varying glottal area (cm²), from the parametric source model
- p⁻₀[t−1] : backward wave arriving at the glottis from the tract

The transglottal pressure drop is:

    ΔP[t] = P_sub − p⁻₀[t−1]                        ... (GL-1)

The glottal volume velocity (Bernoulli flow, with sign handling):

    Ug[t] = ag[t] · sign(ΔP[t]) · √(2 · |ΔP[t]| / ρ)   ... (GL-2)

The forward wave injected into section 0:

    p⁺₀[t] = (ρ · c · Ug[t]) / (2 · A₀) + p⁻₀[t−1] / 2   ... (GL-3)

**Derivation of (GL-3):** The pressure at the entrance to section 0 is
p₀ = p⁺₀ + p⁻₀. The volume velocity entering the tract is
Ug = A₀ · (p⁺₀ − p⁻₀) / (ρc). Solving for p⁺₀:

    p⁺₀ = ρc · Ug / (2 A₀) + p⁻₀ / 2

This uses p⁻₀ from the *previous* time step (t−1), making the equation
explicit (no implicit solve needed). The one-sample delay is physically
correct since the backward wave takes one sample to reach the glottis.

**Uncoupled mode** (`KLConfig.coupled = False`):
Set ΔP[t] = P_sub (ignore p⁻₀), so the flow is determined solely by
subglottal pressure and glottal area. This is useful for A/B testing
the effect of coupling.

### 3.6 Energy Conservation (Lossless Case)

For a lossless system (α = 0), acoustic power must be conserved at each
junction. The acoustic power flux in section n is:

    Wₙ = (Aₙ / (ρc)) · [(p⁺ₙ)² − (p⁻ₙ)²]          ... (EN-1)

**Claim:** The scattering equations (KL-1) conserve power across each junction,
i.e., the power entering junction n from the left equals the power leaving to
the right plus the power reflected back to the left.

**Proof:** Substituting (KL-1) into (EN-1) for sections n and n+1:

Power entering from section n:

    W_in = (Aₙ / (ρc)) · [(p⁺ₙ)² − (p⁻ₙ)²]

Power leaving into section n+1:

    W_out = (Aₙ₊₁ / (ρc)) · [(p⁺ₙ₊₁)² − (p⁻ₙ₊₁)²]

Substituting p⁺ₙ₊₁ = p⁺ₙ + s and p⁻ₙ = p⁻ₙ₊₁ + s where s = kₙ(p⁺ₙ − p⁻ₙ₊₁):

    Aₙ₊₁ · [(p⁺ₙ + s)² − (p⁻ₙ₊₁)²]
      = Aₙ₊₁ · [(p⁺ₙ)² + 2·p⁺ₙ·s + s² − (p⁻ₙ₊₁)²]

    Aₙ · [(p⁺ₙ)² − (p⁻ₙ₊₁ + s)²]
      = Aₙ · [(p⁺ₙ)² − (p⁻ₙ₊₁)² − 2·p⁻ₙ₊₁·s − s²]

Setting W_in = W_out and letting d = p⁺ₙ − p⁻ₙ₊₁, s = kₙ · d:

    Aₙ · [(p⁺ₙ)² − (p⁻ₙ₊₁)² − 2·p⁻ₙ₊₁·kₙ·d − kₙ²·d²]
    = Aₙ₊₁ · [(p⁺ₙ)² + 2·p⁺ₙ·kₙ·d + kₙ²·d² − (p⁻ₙ₊₁)²]

Noting (p⁺ₙ)² − (p⁻ₙ₊₁)² = d · (p⁺ₙ + p⁻ₙ₊₁):

    Aₙ · [d·(p⁺ₙ + p⁻ₙ₊₁) − 2·p⁻ₙ₊₁·kₙ·d − kₙ²·d²]
    = Aₙ₊₁ · [d·(p⁺ₙ + p⁻ₙ₊₁) + 2·p⁺ₙ·kₙ·d + kₙ²·d²]

Dividing through by d (nonzero in the general case):

    Aₙ · [(p⁺ₙ + p⁻ₙ₊₁) − 2·kₙ·p⁻ₙ₊₁ − kₙ²·d]
    = Aₙ₊₁ · [(p⁺ₙ + p⁻ₙ₊₁) + 2·kₙ·p⁺ₙ + kₙ²·d]

Substituting kₙ = (Aₙ − Aₙ₊₁)/(Aₙ + Aₙ₊₁) and simplifying... this
algebra is involved. The clean way to verify is numerically in the test suite
(Test 2). The analytical proof follows from the fact that the one-multiply
KL form is derived from impedance-matched boundary conditions at each junction,
which conserve power by construction.

**Test strategy:** Rather than completing the algebraic proof here, we verify
energy conservation numerically to EXACT (float64) tolerance in Test 2. This
is stronger than an analytical proof because it also catches implementation
bugs.

### 3.7 Lip Radiation Boundary

At the lip end (section N−1), the vocal tract terminates in a radiation
impedance. The simplest model is a frequency-independent real reflection:

    p⁻_{N-1}[t] = −r_lip · p⁺_{N-1}[t]              ... (LIP-1)

where r_lip ∈ (0, 1) is the lip reflection coefficient magnitude. The negative
sign models the open-end (pressure-release) boundary condition.

The radiated output pressure is proportional to the net outgoing wave:

    p_out[t] = (1 + r_lip) · p⁺_{N-1}[t]             ... (LIP-2)

Wait — let me be careful. At an open end with reflection coefficient −r_lip:

    p_total = p⁺ + p⁻ = p⁺ − r_lip · p⁺ = (1 − r_lip) · p⁺

The radiated pressure is proportional to the *volume velocity* at the lips,
not the pressure. Volume velocity is:

    U_lip = A_{N-1} · (p⁺ − p⁻) / (ρc)
          = A_{N-1} · (1 + r_lip) · p⁺_{N-1} / (ρc)  ... (LIP-3)

The far-field radiated pressure is proportional to dU_lip/dt, which in the
discrete case is approximated by a first difference:

    p_rad[t] = U_lip[t] − U_lip[t−1]                  ... (LIP-4)

This first-difference acts as the +6 dB/octave radiation characteristic.

For synthesis training data, the output signal is p_rad (after appropriate
scaling to the target amplitude).

### 3.8 Per-Section Losses

After each full time-step update, all travelling waves are scaled:

    p⁺ₙ[t] ← (1 − α) · p⁺ₙ[t]     ∀ n
    p⁻ₙ[t] ← (1 − α) · p⁻ₙ[t]     ∀ n

where α = `KLConfig.loss_per_section`. This models wall losses and viscous
damping. The resulting formant bandwidths are approximately:

    BW ≈ α · sr / π    (Hz)

For α = 0.002, sr = 16000: BW ≈ 10.2 Hz. Real formant bandwidths are
50–200+ Hz, so additional damping may be needed. Options:
- Increase α (but this applies uniformly to all formants)
- Add frequency-dependent losses (more complex, deferred)
- Accept narrow bandwidths and rely on the `bw_hz` override in SynthConfig
  to inform the ground-truth labels rather than requiring the tract model
  to produce exact bandwidths

**Decision:** For the initial implementation, use a single α for simplicity.
The formant *frequencies* from the tract model should be accurate; the
*bandwidths* in VowelMeta are set from the target specification, not
measured from the output. Bandwidth matching can be improved in a later
iteration by adding per-section frequency-dependent losses or by
post-filtering.

---

## 4. Unit Test Specifications

Tests are organised by tolerance category following the project convention:

- **EXACT**: Deterministic properties that must hold to float64 machine
  precision (or a small multiple thereof, e.g., 1e-12).
- **THEORY**: Results derivable from closed-form expressions. Tolerance is
  set by the DFT frequency resolution or known numerical approximation
  error, documented per test.
- **EMPIRICAL**: Directional or approximate properties verified by sign
  or by measured bounds with ~2.5× margin.

All tests use the registry pattern (`_check`/`_approx`/`_gt`/`_lt`) and
print a summary table rather than raising on first failure.

### Test Helper — Eigenvalue Reference Frequencies

Tests 1 and 4 require reference resonance frequencies that are independent
of the sample-by-sample scattering implementation. This helper computes
them via the state-transition matrix eigenvalue method.

**Algorithm:**

1. Construct the 2N × 2N state-transition matrix M that maps the full
   waveguide state vector at time t to time t+1:

       x[t] = [p⁺₀[t], p⁺₁[t], ..., p⁺_{N-1}[t],
               p⁻₀[t], p⁻₁[t], ..., p⁻_{N-1}[t]]

       x[t+1] = M · x[t]

   M encodes:
   - Junction scattering (KL-1) at each internal junction
   - Glottal boundary: for this helper, use a *fully closed* glottal end
     (total reflection, p⁺₀ = p⁻₀) since we want the natural resonances
     of the tract, not the driven response. This is equivalent to setting
     Ug = 0 and letting the tract ring freely.
   - Lip boundary: p⁻_{N-1} = −r_lip · p⁺_{N-1}
   - Loss: multiply all elements by (1 − α) if α > 0
   - Propagation delays (one sample per section, implicit in the indexing)

2. Compute eigenvalues λₖ of M.

3. Convert to frequencies: fₖ = |atan2(Im(λₖ), Re(λₖ))| · sr / (2π)

4. Filter: keep only eigenvalues with |λₖ| > 0.5 (resonant modes, not
   heavily damped) and fₖ > 0 and fₖ < sr/2.

5. Sort by frequency and return.

**Implementation note:** Building M requires carefully encoding the same
boundary conditions and scattering equations as the loop implementation,
but in matrix form. This is a different computational path (matrix
eigenvalues vs iterative sample-by-sample propagation), so agreement
between the two validates both. The matrix should be constructed directly
from the area array and boundary parameters, NOT by calling the
scattering loop code.

### Test 1 — Uniform Tube: Impulse → Resonances from Discrete Transfer Function

**Setup:**
- N = 16 sections, all areas A = 4.0 cm²
- r_lip = 0.9, α = 0 (lossless)
- Inject a unit impulse at glottal end at t=0 (p⁺₀[0] = 1.0, no coupling)
- Record p_rad for 8192 samples

**Expected resonance frequencies — derived from the discrete system, not
from c/(4L):**

The continuous-domain formula fₘ = (2m−1)·c/(4L) does not exactly predict
the resonances of the discrete waveguide because boundary conditions may
introduce additional sample delays depending on implementation. The correct
reference frequencies must come from the discrete z-domain transfer function.

For a uniform tube (all kₙ = 0), the scattering junctions are transparent
and the system reduces to a pure delay line with reflections at the two
ends. The transfer function from glottal input to lip output depends on:
- The number of delay elements in the forward path
- The glottal boundary reflection (how p⁺₀ is computed from the input
  and p⁻₀)
- The lip boundary reflection (p⁻_{N-1} = −r_lip · p⁺_{N-1})

**Test helper — reference pole computation:**

Rather than deriving a closed-form H(z) that may embed the same boundary
assumptions as the implementation, compute the reference frequencies
empirically from a *second, independent method*:

1. Build the system matrix M (2N × 2N) that maps the full state vector
   [p⁺₀, ..., p⁺_{N-1}, p⁻₀, ..., p⁻_{N-1}] at time t to time t+1,
   including both boundary conditions and all junction scattering.
2. Compute eigenvalues of M.
3. Convert eigenvalues to frequencies: fₖ = |angle(λₖ)| · sr / (2π).
4. Select the eigenvalues with |λₖ| close to 1 (resonant modes, not
   overdamped); their frequencies are the expected resonances.

This is a completely independent computation from the sample-by-sample
scattering loop — it tests the loop against a matrix eigenvalue method.

**Verification:**
- Compute FFT magnitude of p_rad (N_fft = 8192, bin width ≈ 1.95 Hz)
- Find spectral peaks
- Compare each peak against the nearest reference frequency from the
  eigenvalue method
- Each peak should be within ±1 FFT bin of the reference

**Tolerance:** THEORY — ±1 FFT bin. With N_fft = 8192, sr = 16000,
bin width = 1.95 Hz. The eigenvalue method gives exact discrete
frequencies; the only error is the DFT bin quantisation.

### Test 2 — Lossless Energy Conservation (Instantaneous Power Balance)

**Setup:**
- N = 16, non-uniform area function (e.g., alternating 2.0 and 6.0 cm²)
- α = 0 (lossless), r_lip = 0.9 (partial reflection — energy both
  reflects and radiates, exercising the full system)
- Inject a known signal at glottal end (e.g., 100 samples of a sine burst
  at 200 Hz), then zero input for 200 more samples
- No coupling (fixed flow injection)
- The implementation must expose per-sample internal state: all p⁺ₙ[t]
  and p⁻ₙ[t]

**Instantaneous energy and power definitions:**

Total energy stored in the tube at time t (proportional to acoustic
energy; constant factors cancel in the balance equation):

    E_tube[t] = Σₙ Aₙ · [(p⁺ₙ[t])² + (p⁻ₙ[t])²]

Net power flux into the tube at the glottal end (positive = into tube):

    P_in[t] = A₀ · [(p⁺₀[t])² − (p⁻₀[t])²]

Net power flux out of the tube at the lip end (positive = out of tube):

    P_out[t] = A_{N-1} · [(p⁺_{N-1}[t])² − (p⁻_{N-1}[t])²]

**Expected:**
At every time step t, the power balance holds:

    E_tube[t] − E_tube[t−1] = P_in[t] − P_out[t]

Equivalently, the residual:

    R[t] = E_tube[t] − E_tube[t−1] − P_in[t] + P_out[t]

should be zero at every time step.

**Verification:**
- Compute R[t] for every time step t = 1 .. n_total
- Assert: max|R[t]| < 1e-12

This formulation is superior to the drain-and-check approach because:
(a) it does not require choosing a drain time (energy may take many
    round trips to exit a tube with strong internal reflections),
(b) it checks conservation at *every* time step, catching transient
    bugs that a final-state-only check would miss,
(c) it works with any r_lip value, not just r_lip = 0.

**Tolerance:** EXACT — 1e-12 absolute residual in float64.

**Implementation note:** This test requires access to the full internal
state at each time step. Use the `step()` interface (see Test 11 notes)
or a state-recording callback.

### Test 3 — Uniform Tube Delay

**Setup:**
- N sections, all areas equal, α = 0
- r_lip such that p⁻_{N-1} = −r_lip · p⁺_{N-1}
- Inject arbitrary input sequence at glottal end (uncoupled mode)

**Expected:**
- The forward wave arrives at the lip end after N−1 sample delays
- For a uniform tube with all k = 0, no scattering occurs; the forward
  wave propagates unchanged
- Output (before radiation differentiation) at time t should be
  proportional to input at time t − (N−1)

**Verification:**
- Cross-correlate input and output; peak should be at lag = N−1
- Amplitude ratio should be (1 + r_lip) (for volume velocity output)
  or (1 − r_lip) (for pressure output) depending on LIP formulation

**Tolerance:** EXACT for lag. EXACT for amplitude ratio (float64).

### Test 4 — Two-Tube Model: Known F1

**Setup:**
- N = 16 (8 back sections + 8 front sections)
- Back cavity area A₁ = 8.0 cm², front cavity area A₂ = 2.0 cm²
- Single junction with k = (8 − 2)/(8 + 2) = 0.6
- α = 0, r_lip = 0.9
- Impulse excitation (uncoupled)
- Record p_rad for 8192 samples

**Expected:**
- F1 should differ from the uniform-tube case (Test 1) because the area
  discontinuity shifts resonance frequencies.
- The reference F1 is computed using the same eigenvalue method as Test 1:
  build the 2N × 2N system matrix for the two-tube configuration and
  extract pole frequencies. This is the same test helper, just called
  with different area arrays.

**Verification:**
- Compute FFT of impulse response (N_fft = 8192)
- Measure F1 from the first spectral peak
- Compare against F1 from eigenvalue computation
- They should agree to within ±1 FFT bin

**Tolerance:** THEORY — ±1 FFT bin (bin width ≈ 1.95 Hz with N_fft = 8192).

**Additional sanity check:**
- Assert: two-tube F1 ≠ uniform-tube F1 (the area discontinuity must
  shift the resonance). This is a weak directional check that catches
  the degenerate case where the junction scattering has no effect.

**Note:** The eigenvalue test helper is shared with Test 1. It should be
implemented as a standalone function that takes an area array, r_lip,
and boundary conditions, and returns resonance frequencies — independent
of the scattering loop implementation.

### Test 5 — Coupling Modifies Spectral Slope

**Setup:**
- Non-uniform area function (e.g., vowel /ɑ/ approximation)
- Parametric glottal source: half-rectified sine, f0 = 120 Hz
- Run twice: once with coupling (KLConfig.coupled=True), once without
  (KLConfig.coupled=False)
- Same source parameters in both cases

**Expected:**
- The coupled and uncoupled outputs should have *different* spectral
  slopes, because the vocal tract impedance modulates the glottal flow
  waveform shape when coupling is active.

**Note on direction:** The original spec predicted the coupled slope would
be steeper (more negative). In practice, the direction depends on whether
the vocal tract loading is predominantly inertive or compliant at the
dominant source harmonics, which varies with the specific area function
and F0. Rather than asserting a direction that may be wrong for the
chosen test configuration, this test checks only that the slopes differ.
Once the implementation is validated (especially via Test 11), the
measured direction can be documented and the test tightened if desired.

**Verification:**
- Compute harmonic amplitudes for both outputs (H1..H10)
- Compute spectral slope via linear regression on log-frequency
- Assert: |slope_coupled − slope_uncoupled| > threshold

The threshold should be set empirically after initial implementation:
measure the actual difference, then set threshold at ~40% of measured
(2.5× margin), documenting the measured value.

**Tolerance:** EMPIRICAL — nonzero difference with margin.

### Test 6 — Coupling Strength Scales with Glottal Area

**Setup:**
- Same area function and f0 as Test 5
- Two runs with coupling, differing only in peak glottal area:
  ag_small = 0.05 cm², ag_large = 0.20 cm²
- For each, compute spectral slope difference (coupled − uncoupled)

**Expected:**
- |slope_diff_large| > |slope_diff_small|
- Larger glottal opening means lower glottal impedance, so the tract
  loading has a stronger effect on the flow

**Verification:**
- Four synthesis runs: {small, large} × {coupled, uncoupled}
- Assert: |slope_coupled_large − slope_uncoupled_large|
        > |slope_coupled_small − slope_uncoupled_small|

**Tolerance:** EMPIRICAL directional.

### Test 7 — Lip Radiation is Highpass

**Setup:**
- Any non-uniform tract, any source
- Capture two signals: (a) volume velocity at the lip end U_lip,
  (b) radiated output p_rad (which is dU_lip/dt)

**Expected:**
- p_rad has higher spectral centroid than U_lip
- The difference operator boosts high frequencies by +6 dB/octave

**Verification:**
- Compute spectral centroid of both signals
- Assert: centroid_rad > centroid_lip

**Tolerance:** EMPIRICAL directional.

### Test 8 — Reflection Coefficient Bounds

**Setup:** Pure unit test on the reflection coefficient computation.

**Verification:**
- For any positive areas A₁, A₂: |k| < 1
- k = 0 when A₁ = A₂
- k > 0 when A₁ > A₂ (wider to narrower)
- k < 0 when A₁ < A₂ (narrower to wider)
- k = (A−B)/(A+B) computed for a range of area ratios

**Tolerance:** EXACT (float64).

### Test 9 — Uncoupled Glottal Flow Independence

**Setup:**
- Same tract and source, coupled=False
- Run with two different area functions that share the same section 0 area

**Expected:**
- The glottal flow waveform Ug[t] should be identical in both runs
  (since it depends only on P_sub, ag[t], and ρ — not on the tract)
- The output will differ (different tract filtering), but the source is
  the same

**Verification:**
- Record Ug[t] from both runs
- Assert: max |Ug_1 − Ug_2| < 1e-12

**Tolerance:** EXACT.

### Test 10 — Coupled Glottal Flow Depends on Tract

**Setup:**
- Same as Test 9 but with coupled=True

**Expected:**
- The glottal flow *should differ* between the two tract configurations,
  because p⁻₀ feeds back into the flow equation

**Verification:**
- Record Ug[t] from both runs
- Assert: max |Ug_1 − Ug_2| > threshold (e.g., 1% of peak flow)

**Tolerance:** EMPIRICAL — directional (nonzero difference). The 1%
threshold may need adjustment after initial implementation; document
the measured value and set threshold at ~40% of measured (2.5× margin).

### Test 11 — Glottal Boundary Equation (GL-3) Term-by-Term Verification

**Motivation:** Tests 1–10 catch most errors in the scattering loop and
boundary conditions, but a subtle scaling error in GL-3 could produce
physically plausible (energy-conserving, directionally correct) output
with the wrong coupling strength. This test validates GL-3 directly by
checking the computed p⁺₀ against the predicted value term by term.

**Setup:**
- Uniform tube, N = 16, all areas A₀ = 4.0 cm²
- α = 0 (lossless), r_lip = 0.9
- Uncoupled mode (coupled=False)
- Drive with a known glottal flow sequence Ug[t] (e.g., a half-rectified
  sine at 120 Hz for 200 samples)
- The implementation must expose (or allow recording of) the internal
  values: p⁺₀[t], p⁻₀[t], and Ug[t] at each time step

**Verification:**
At each time step t, verify:

    p⁺₀[t] == (ρ · c · Ug[t]) / (2 · A₀) + p⁻₀[t−1] / 2

by comparing the internally computed p⁺₀[t] against the right-hand side
calculated from the recorded Ug[t] and p⁻₀[t−1].

Additionally, in uncoupled mode, verify that Ug[t] matches the expected
flow from GL-2 with ΔP = P_sub (no tract feedback):

    Ug[t] == ag[t] · sign(P_sub) · √(2 · |P_sub| / ρ)

This validates both GL-2 and GL-3 independently.

**Tolerance:** EXACT — max absolute error < 1e-12 (float64). There is no
approximation involved; the equation is evaluated in closed form at each
sample. Any coefficient, sign, or scaling error will produce a nonzero
residual.

**Implementation note:** This test requires access to internal state
(p⁺₀, p⁻₀, Ug) that is not part of the public API. The KL backend
should either:
(a) accept an optional callback/recorder that captures per-sample state, or
(b) expose a lower-level `step()` function that returns internal state,
    with the public `generate_vowel` being a wrapper that calls `step()`
    in a loop.

Option (b) is preferred as it also enables future real-time streaming use.

---

## 5. Implementation Notes

### 5.1 Numerical Precision

All internal computations should use float64 to avoid accumulation errors
in the scattering loop (thousands of multiply-add operations per output
sample). The output audio array is cast to float32 at the end.

### 5.2 Performance

For sr = 16000 and duration = 0.5 s, that's 8000 samples. Each sample
requires ~N junction updates (~16–40 multiplies). Total: ~128K–320K
multiplies per clip. This is trivially fast in pure NumPy with the inner
loop in Python; if it becomes a bottleneck for large dataset generation,
the scattering loop can be moved to Cython or Numba, but this is not
needed initially.

Actually — the inner loop is inherently sequential (each sample depends
on the previous), so it *cannot* be vectorised along the time axis.
It can be vectorised along the section axis (all junctions updated in
parallel per time step) if using the double-buffer approach, but the
time loop must remain explicit. Consider using `@numba.njit` for the
core loop if pure Python is too slow for generating 5000+ clips.

### 5.3 File Organisation

```
synth/                — Package root
  __init__.py
  config.py           — SynthConfig, VowelMeta, KLConfig, KlattConfig
  api.py              — Public API (generate_vowel, generate_dataset)
  kl.py               — Kelly-Lochbaum backend implementation
                        (scattering loop, boundaries, area functions)
  klatt.py            — Existing Klatt backend (moved from current synth.py)
  glottal.py          — Glottal area waveform generation (§7)
  tests/
    test_kl.py        — KL scattering + boundary tests (§4)
    test_glottal.py   — Glottal source tests (§8)
```

---

## 6. Glottal Area Waveform

### 6.1 Design Rationale

The KL boundary (GL-2) takes a time-varying glottal area ag[t] as input and
computes glottal flow via the Bernoulli equation. The area waveform represents
the biomechanics of the vocal folds (opening and closing); the aerodynamics
(flow through the constriction, modulated by tract loading) are handled by
GL-2 and GL-3.

This separation means that in uncoupled mode (ΔP = P_sub = const), flow is
proportional to area: Ug ∝ ag · √(P_sub). With coupling, the tract loading
skews the flow pulse relative to the area pulse — this is the Level 1
source-filter interaction.

We parameterise the area waveform using OQ (open quotient) and SQ (speed
quotient) directly, rather than going through the LF Rd parameter. This is
more transparent and avoids embedding approximate Rd→OQ/SQ mappings. An
optional Rd convenience constructor can be added later.

### 6.2 Waveform Definition

One pitch period T₀ = 1/f₀ is divided into three phases:

    ┌──────────┬───────────┬──────────┐
    │ Opening  │  Closing  │  Closed  │
    │ 0 → t_op │ t_op→t_cl │ t_cl→T₀  │
    └──────────┴───────────┴──────────┘

where:

    t_cl  = OQ · T₀                     (end of open phase)
    t_op  = SQ / (1 + SQ) · t_cl        (peak area, transition from opening to closing)

**Phase definitions (half-cosine segments):**

Opening (0 ≤ t < t_op):

    ag(t) = ag_min + (ag_max − ag_min) · ½ · (1 − cos(π · t / t_op))

Closing (t_op ≤ t < t_cl):

    ag(t) = ag_min + (ag_max − ag_min) · ½ · (1 + cos(π · (t − t_op) / (t_cl − t_op)))

Closed (t_cl ≤ t < T₀):

    ag(t) = ag_min

**Parameters:**

| Parameter | Symbol  | Unit | Description |
|-----------|---------|------|-------------|
| f0        | f₀      | Hz   | Fundamental frequency |
| OQ        | OQ      | –    | Open quotient: fraction of T₀ the glottis is open |
| SQ        | SQ      | –    | Speed quotient: opening_time / closing_time |
| ag_max    | ag_max  | cm²  | Peak glottal area |
| ag_min    | ag_min  | cm²  | Minimum glottal area (0 = complete closure) |

**Continuity properties:**
- ag(t) is continuous everywhere
- dag/dt is continuous at t = 0 (zero slope into opening) and at t_op
  (smooth peak)
- dag/dt is discontinuous at t_cl when ag_min = 0: the closing phase
  ends with nonzero negative slope, then the derivative jumps to zero.
  This discontinuity is physically realistic (abrupt vocal fold contact)
  and is the primary source of high-frequency excitation.
- When ag_min > 0 (glottal gap), the discontinuity in dag/dt at t_cl is
  smaller but still present.

### 6.3 Derived Quantities

**Glottal gap fraction:**

    gap_frac = ag_min / ag_max

- gap_frac = 0: complete closure (modal male voice, pressed voice)
- gap_frac ≈ 0.05–0.2: typical incomplete closure (many female voices, breathy)
- gap_frac > 0.3: very breathy, approaching whisper

**Spectral tilt relationship (uncoupled, qualitative):**

Higher OQ → more sinusoidal pulse → steeper spectral slope (less HF energy)
Higher SQ → more symmetric pulse → steeper spectral slope
Higher gap_frac → DC offset in flow → boosted H1 relative to other harmonics
Lower OQ + lower SQ → sharper closure → more HF energy (pressed voice)

### 6.4 Multi-Period Generation

For a clip of duration `duration_s` at frequency f₀:

    N_periods = ceil(duration_s · f₀)
    N_samples = round(duration_s · sr)

Generate the area waveform sample-by-sample:

```python
def generate_glottal_area(
    f0_hz:    float,
    oq:       float,
    sq:       float,
    ag_max:   float,
    ag_min:   float,
    sr:       int,
    n_samples: int,
    jitter_frac:  float = 0.0,
    shimmer_frac: float = 0.0,
    rng: Optional[np.random.Generator] = None,
) -> np.ndarray:
    """Returns ag[t] array of shape (n_samples,), float64."""
```

**Jitter:** Each period's T₀ is perturbed by a random factor:

    T₀_perturbed = T₀ · (1 + jitter_frac · U(-1, 1))

where U(-1,1) is uniform random on [-1, 1]. This models cycle-to-cycle
pitch perturbation.

**Shimmer:** Each period's ag_max is perturbed:

    ag_max_perturbed = ag_max · (1 + shimmer_frac · U(-1, 1))

This models cycle-to-cycle amplitude perturbation.

### 6.5 Parameter Ranges by Register / Effort / Phonation

These ranges define the sampling bounds for synthetic training data
generation. Bounds are derived as min-of-min / max-of-max across
literature sources. Overlap between adjacent categories is intentional.

#### 6.5.1 Register

| Parameter | M1 (modal)  | M2 (falsetto) | M3 (whistle)  |
|-----------|-------------|---------------|---------------|
| f0 (Hz)   | 80 – 500    | 200 – 1000    | 600 – 1600    |
| OQ        | 0.35 – 0.65 | 0.55 – 0.85   | 0.80 – 0.95   |
| SQ        | 1.5 – 3.5   | 1.0 – 2.0     | 0.8 – 1.2     |
| ag_max (cm²)| 0.05 – 0.25| 0.02 – 0.12  | 0.01 – 0.05   |

#### 6.5.2 Effort (applied as modifiers to register ranges)

| Parameter delta | Soft     | Moderate | Loud      |
|-----------------|----------|----------|-----------|
| OQ              | +0.05    | 0        | −0.10     |
| SQ              | −0.3     | 0        | +0.5      |
| ag_max scale    | ×0.6     | ×1.0     | ×1.5      |
| P_sub (Pa)      | 400–600  | 600–1000 | 1000–2000 |

Effort modifiers are added to / multiplied with the register base values,
then clamped to physically valid ranges (OQ ∈ (0, 1), SQ > 0, ag_max > 0).

#### 6.5.3 Phonation Type

| Parameter     | Modal       | Breathy     | Pressed     |
|---------------|-------------|-------------|-------------|
| OQ modifier   | 0           | +0.10       | −0.10       |
| gap_frac      | 0.0 – 0.05  | 0.05 – 0.30 | 0.0         |
| noise_level   | −40 dB      | −25 dB      | −45 dB      |

`noise_level` is the amplitude of additive aspiration noise relative to
the glottal flow signal, applied during the open phase only. This models
turbulent airflow through the glottis.

#### 6.5.4 Parameter Sampling Procedure

For each synthetic clip:

1. Sample register ∈ {M1, M2, M3} (uniform or weighted)
2. Sample effort ∈ {soft, moderate, loud} (uniform or weighted)
3. Sample phonation_type ∈ {modal, breathy, pressed} (uniform or weighted)
4. Sample f0 from the register's f0 range (uniform in log-frequency)
5. Sample OQ, SQ, ag_max from register range
6. Apply effort modifiers; clamp
7. Apply phonation type modifiers; clamp
8. Sample gap_frac from phonation type range
9. Compute ag_min = gap_frac · ag_max

All continuous parameters are sampled uniformly within their (possibly
modified) ranges unless otherwise noted. f0 is sampled in log-space to
give equal weight per octave.

---

## 7. Glottal Source Tests

### Test G1 — Waveform Bounds

**Setup:** Generate glottal area for a range of parameter combinations.

**Verification:**
- ag(t) ≥ ag_min for all t
- ag(t) ≤ ag_max for all t
- During closed phase: ag(t) == ag_min exactly

**Tolerance:** EXACT (float64).

### Test G2 — Peak Location

**Setup:** Generate area waveform with known OQ, SQ, f0.

**Expected:**
- Peak (ag = ag_max) occurs at sample closest to t_op = SQ/(1+SQ) · OQ · T₀
- Peak is unique within each period (no secondary maxima)

**Verification:**
- Find argmax within each period
- Compare to expected t_op (±1 sample for discretisation)
- Verify no other samples equal ag_max within the period

**Tolerance:** THEORY — ±1 sample.

### Test G3 — Open Quotient

**Setup:** Generate area waveform with known OQ, ag_min = 0 (complete closure).

**Expected:**
- Fraction of samples where ag > 0 should equal OQ (±1/N_period_samples
  for discretisation)

**Verification:**
- Count samples with ag > 0 per period
- Compare ratio to OQ

**Tolerance:** THEORY — ±1 sample per period.

### Test G4 — Speed Quotient

**Setup:** Generate area waveform with known SQ.

**Expected:**
- Number of samples in opening phase / number in closing phase ≈ SQ
- Opening phase = samples from ag first exceeding ag_min to argmax
- Closing phase = samples from argmax to ag returning to ag_min

**Verification:**
- Measure opening and closing durations in samples
- Compute ratio, compare to SQ

**Tolerance:** THEORY — ±2 samples total (discretisation at both boundaries).

### Test G5 — Continuity at Phase Boundaries

**Setup:** Generate area waveform, examine transitions at t=0, t_op, t_cl.

**Verification:**
- |ag(t_op − 1) − ag(t_op)| < expected_max_step
  where expected_max_step is the maximum sample-to-sample difference
  for a half-cosine of the given frequency and amplitude
- |ag(t_cl − 1) − ag(t_cl)| may be large (closing discontinuity) —
  do NOT test continuity here
- dag/dt at t = 0 should be ≈ 0 (zero slope into opening):
  |ag(1) − ag(0)| < |ag(2) − ag(1)| (first step smaller than second)

**Tolerance:** EMPIRICAL directional for the derivative check.
THEORY for the max step size (derivable from the half-cosine formula).

### Test G6 — Complete Closure Produces Silence in Closed Phase

**Setup:** ag_min = 0, generate area waveform.

**Verification:**
- During closed phase (ag = 0), if this area is fed to GL-2, the flow
  Ug = 0 regardless of ΔP (since ag = 0 multiplies everything)
- This is a property of GL-2 not the area generator, but it's worth
  verifying end-to-end: run the KL model with ag_min = 0 and verify
  that during closed phase samples, the glottal flow is exactly zero

**Tolerance:** EXACT.

### Test G7 — Jitter Perturbs Period Duration

**Setup:** Generate two waveforms with identical parameters except
jitter_frac = 0 vs jitter_frac = 0.02. Same RNG seed is NOT used
(we want different realisations).

**Verification:**
- In the jitter=0 case, all periods have identical duration
  (measured by peak-to-peak interval)
- In the jitter>0 case, peak-to-peak intervals vary
- Standard deviation of period durations should be approximately
  jitter_frac · T₀ / √3 (std of uniform distribution)

**Tolerance:**
- Identical periods in jitter=0 case: EXACT
- Std deviation of jittered periods: EMPIRICAL — within 2× of the
  theoretical value (allow margin for small number of periods).

### Test G8 — Shimmer Perturbs Amplitude

**Setup:** Same structure as Test G7 but for shimmer.

**Verification:**
- In shimmer=0 case, all periods have identical peak amplitude
- In shimmer>0 case, peak amplitudes vary
- Standard deviation of peak amplitudes ≈ shimmer_frac · ag_max / √3

**Tolerance:** Same as Test G7.

### Test G9 — Glottal Gap Increases DC Component

**Setup:** Two waveforms with identical parameters except ag_min = 0 vs
ag_min = 0.1 · ag_max.

**Verification:**
- Mean of the ag_min > 0 waveform is strictly greater than mean of
  the ag_min = 0 waveform
- In the uncoupled KL model, the DC component of the flow should
  also be higher (since flow ∝ area when ΔP is constant)

**Tolerance:** EMPIRICAL directional.

### Test G10 — SQ Symmetry

**Setup:** Generate area waveform with SQ = 1.0 (symmetric pulse).

**Verification:**
- Within each open phase, the waveform should be symmetric about t_op
- Specifically: ag(t_op − k) ≈ ag(t_op + k) for all valid k

**Tolerance:** THEORY — the half-cosine segments with equal duration
produce an exactly symmetric pulse. Allow ±1e-12 for float64 arithmetic.

### Test G11 — Extreme OQ Values

**Setup:** Test boundary conditions:
- OQ = 0.01 (nearly always closed): only ~1% of period is open
- OQ = 0.99 (nearly always open): only ~1% of period is closed

**Verification:**
- OQ = 0.01: waveform is mostly ag_min, with a brief narrow pulse
  each period. Count of open samples ≈ 0.01 · period_samples ± 1
- OQ = 0.99: waveform is mostly open, with a brief dip to ag_min.
  Count of open samples ≈ 0.99 · period_samples ± 1
- No NaN, no negative values, no values > ag_max in either case

**Tolerance:** Bounds checks: EXACT. Sample counts: THEORY ±1.

---

## 8. Deferred Items

- Area function from formant frequencies (§3.2)
- Rd convenience constructor (Rd → OQ, SQ mapping)
- Frequency-dependent tract losses for bandwidth control
- Subglottal coupling
- Aspiration noise generation (referenced in §6.5.3 but not yet specified)
- Vowel-specific area function library (Story area functions or equivalent)
