"""
Bayesian VTL estimation with Normal-Normal conjugate updates and GMM
noise separation.

Replaces the ad-hoc VTLSmoother with proper hierarchical Bayesian
inference at group → speaker → sample levels.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np

from .config import VTL_PRIOR_MM


# ---------------------------------------------------------------------------
# Normal posterior — the atomic unit of Bayesian VTL estimation
# ---------------------------------------------------------------------------

@dataclass
class NormalPosterior:
    """N(mu, sigma_sq) posterior with conjugate update."""

    mu: float
    sigma_sq: float

    def conjugate_update(
        self, x_bar: float, sigma_obs_sq: float, n: int,
    ) -> NormalPosterior:
        """Return a new posterior after observing n samples with mean x_bar
        and observation variance sigma_obs_sq."""
        if n <= 0 or not np.isfinite(x_bar) or sigma_obs_sq <= 0:
            return NormalPosterior(self.mu, self.sigma_sq)
        prec_prior = 1.0 / self.sigma_sq
        prec_obs = n / sigma_obs_sq
        sigma_post_sq = 1.0 / (prec_prior + prec_obs)
        mu_post = sigma_post_sq * (prec_prior * self.mu + prec_obs * x_bar)
        return NormalPosterior(mu_post, sigma_post_sq)

    @property
    def sigma(self) -> float:
        return float(np.sqrt(self.sigma_sq))


# ---------------------------------------------------------------------------
# Robust variance estimate
# ---------------------------------------------------------------------------

def _iqr_var(x: np.ndarray, floor: float = 1.0) -> float:
    """IQR-based robust variance estimate.

    For a normal distribution IQR ≈ 1.35σ, so σ² ≈ (IQR/1.35)².
    Falls back to np.var(ddof=1) when fewer than 4 observations
    (IQR is unreliable with very few points).
    """
    if len(x) < 4:
        return max(float(np.var(x, ddof=1)) if len(x) > 1 else 225.0, floor)
    q75, q25 = np.percentile(x, [75, 25])
    iqr = q75 - q25
    if iqr <= 0:
        return max(float(np.var(x, ddof=1)), floor)
    return max((iqr / 1.35) ** 2, floor)


# ---------------------------------------------------------------------------
# 1D two-component Gaussian mixture model (EM from scratch)
# ---------------------------------------------------------------------------

@dataclass
class _MixtureResult:
    """Result of a 2-component GMM fit."""
    narrow_mean: float
    narrow_var: float
    narrow_weight: float
    broad_mean: float
    broad_var: float
    broad_weight: float


def fit_vtl_mixture(
    vtl_obs: np.ndarray,
    min_obs: int = 6,
    max_iter: int = 50,
    tol: float = 1e-4,
) -> tuple[float, float, float]:
    """Fit a 2-component GMM to 1D VTL observations.

    Returns (narrow_mean, narrow_var, narrow_weight) where the narrow
    component (smaller variance) represents the signal.

    Falls back to (mean, var, 1.0) if fewer than min_obs valid
    observations.
    """
    x = np.asarray(vtl_obs, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        m = float(np.mean(x)) if len(x) else np.nan
        return m, 225.0, 1.0  # 15mm² default variance

    if len(x) < min_obs:
        return float(np.mean(x)), _iqr_var(x), 1.0

    # EM initialisation: split at the median
    med = float(np.median(x))
    lo = x[x <= med]
    hi = x[x > med]

    mu = np.array([
        float(np.mean(lo)) if len(lo) else med - 5.0,
        float(np.mean(hi)) if len(hi) else med + 5.0,
    ])
    var = np.array([
        max(float(np.var(lo, ddof=0)), 1.0) if len(lo) else 25.0,
        max(float(np.var(hi, ddof=0)), 1.0) if len(hi) else 25.0,
    ])
    pi = np.array([0.5, 0.5])

    n = len(x)
    prev_ll = -np.inf

    for _ in range(max_iter):
        # E-step: responsibilities
        # log p(x | k) = -0.5*log(2π*var_k) - (x-mu_k)²/(2*var_k)
        log_p = np.zeros((2, n))
        for k in range(2):
            log_p[k] = (-0.5 * np.log(2.0 * np.pi * var[k])
                        - (x - mu[k]) ** 2 / (2.0 * var[k])
                        + np.log(pi[k] + 1e-30))

        # Log-sum-exp for numerical stability
        max_lp = np.maximum(log_p[0], log_p[1])
        log_sum = max_lp + np.log(np.exp(log_p[0] - max_lp)
                                  + np.exp(log_p[1] - max_lp))
        ll = float(np.sum(log_sum))

        resp = np.exp(log_p - log_sum[None, :])  # (2, n)

        # Check convergence
        if abs(ll - prev_ll) < tol:
            break
        prev_ll = ll

        # M-step
        nk = resp.sum(axis=1) + 1e-10  # (2,)
        for k in range(2):
            mu[k] = float(np.sum(resp[k] * x) / nk[k])
            var[k] = max(float(np.sum(resp[k] * (x - mu[k]) ** 2) / nk[k]),
                         1e-6)
        pi = nk / n

    # Identify narrow component (smaller variance)
    narrow = 0 if var[0] <= var[1] else 1
    return float(mu[narrow]), float(var[narrow]), float(pi[narrow])


def fit_vtl_single(
    vtl_obs: np.ndarray,
    min_obs: int = 6,       # unused, kept for signature compat
    max_iter: int = 50,     # unused
    tol: float = 1e-4,      # unused
) -> tuple[float, float, float]:
    """Fit a single Gaussian to VTL observations (no mixture separation).

    Same return signature as fit_vtl_mixture: (mean, var, weight=1.0).
    Use this as a baseline to test whether the GMM helps.
    """
    x = np.asarray(vtl_obs, dtype=np.float64)
    x = x[np.isfinite(x)]

    if len(x) < 2:
        m = float(np.mean(x)) if len(x) else np.nan
        return m, 225.0, 1.0

    return float(np.median(x)), _iqr_var(x), 1.0


# ---------------------------------------------------------------------------
# Hierarchical Bayesian VTL estimator
# ---------------------------------------------------------------------------

@dataclass
class _SampleObs:
    """Observation summary from one sample's GMM fit."""
    narrow_mean: float
    narrow_var: float
    n_obs: int
    skew: float = 0.0


class BayesianVTLEstimator:
    """Hierarchical Bayesian VTL estimator with Normal-Normal conjugate
    updates.

    Hierarchy: literature prior → group posterior → speaker posterior.

    Parameters
    ----------
    prior_sigma_mm : float
        Standard deviation of the group-level literature prior (mm).
    """

    def __init__(self, prior_sigma_mm: float = 15.0) -> None:
        self.prior_sigma_mm = prior_sigma_mm
        self._prior_sigma_sq = prior_sigma_mm ** 2

        # Group-level posteriors (initialised lazily from literature priors)
        self._group_posteriors: dict[str, NormalPosterior] = {}

        # Between-speaker spread per group (empirical variance of speaker
        # means within the group).  This does NOT shrink with more speakers —
        # it's a property of the population, not uncertainty about the mean.
        self._group_spread_sq: dict[str, float] = {}

        # Speaker-level posteriors
        self._speaker_posteriors: dict[str, NormalPosterior] = {}
        self._speaker_to_group: dict[str, str] = {}

        # Overrides
        self._group_prior_override: dict[str, float] = {}
        self._group_sigma_override: dict[str, float] = {}
        self._speaker_override_mm: dict[str, float] = {}

    # -- Prior helpers ------------------------------------------------------

    def _group_prior(self, group: str) -> NormalPosterior:
        """Literature prior for a group."""
        mu = self._group_prior_override.get(
            group, VTL_PRIOR_MM.get(group, VTL_PRIOR_MM["unknown"]))
        sigma_sq = self._group_sigma_override.get(
            group, self._prior_sigma_sq)
        return NormalPosterior(mu, sigma_sq)

    def _ensure_group(self, group: str) -> None:
        if group not in self._group_posteriors:
            self._group_posteriors[group] = self._group_prior(group)

    def _ensure_speaker(self, group: str, speaker: str) -> None:
        self._ensure_group(group)
        self._speaker_to_group[speaker] = group
        if speaker not in self._speaker_posteriors:
            if speaker in self._speaker_override_mm:
                # Override: very tight prior around manual measurement
                self._speaker_posteriors[speaker] = NormalPosterior(
                    self._speaker_override_mm[speaker], 1.0)
            else:
                # Inherit group mean + between-speaker spread
                spread_sq = self._group_spread_sq.get(
                    group, self._prior_sigma_sq)
                self._speaker_posteriors[speaker] = NormalPosterior(
                    self._group_posteriors[group].mu,
                    spread_sq,
                )

    # -- Public interface: overrides ----------------------------------------

    def register_group_override(
        self, group: str, vtl_mm: float,
        sigma_mm: Optional[float] = None,
    ) -> None:
        """Set an empirical group prior (replaces literature value)."""
        self._group_prior_override[group] = vtl_mm
        if sigma_mm is not None:
            self._group_sigma_override[group] = sigma_mm ** 2
        # Reset group posterior to the new prior
        self._group_posteriors[group] = self._group_prior(group)

    def register_speaker_override(self, speaker: str, vtl_mm: float) -> None:
        """Pin a speaker's VTL to a manual measurement (very tight prior)."""
        self._speaker_override_mm[speaker] = vtl_mm
        self._speaker_posteriors[speaker] = NormalPosterior(vtl_mm, 1.0)

    # -- Public interface: seeding ------------------------------------------

    def seed_speaker(
        self, group: str, speaker: str, vtl_mm: float,
        var_mm: float = 225.0,
    ) -> None:
        """Register a rough initial VTL estimate for a speaker (from seed
        pass).  Uses the data-derived variance from the seed positions if
        available, otherwise falls back to a wide default (225 mm²)."""
        self._ensure_group(group)
        self._speaker_to_group[speaker] = group
        if speaker in self._speaker_override_mm:
            return  # don't overwrite manual overrides
        # Use data-derived variance, clamped to a reasonable floor
        sigma_sq = max(float(var_mm), 1.0)
        self._speaker_posteriors[speaker] = NormalPosterior(vtl_mm, sigma_sq)

    def update_groups_from_speakers(self) -> None:
        """Update group posteriors and spread from current speaker posteriors.

        Call after seeding to ensure group estimates reflect the seed data
        rather than just the literature priors.
        """
        # Collect speakers per group
        group_speakers: dict[str, list[str]] = {}
        for spk, grp in self._speaker_to_group.items():
            if (spk in self._speaker_posteriors
                    and spk not in self._speaker_override_mm):
                group_speakers.setdefault(grp, []).append(spk)

        for group, speakers in group_speakers.items():
            if not speakers:
                continue

            # Update group mean posterior
            prior = self._group_prior(group)
            post = prior
            for spk in speakers:
                sp = self._speaker_posteriors[spk]
                post = post.conjugate_update(sp.mu, sp.sigma_sq, 1)
            self._group_posteriors[group] = post

            # Update between-speaker spread
            if len(speakers) >= 2:
                spk_mus = np.array([self._speaker_posteriors[spk].mu
                                    for spk in speakers])
                self._group_spread_sq[group] = _iqr_var(spk_mus)

    # -- Public interface: update from observations -------------------------

    def update_all(
        self,
        observations: dict[str, dict[str, list[_SampleObs]]],
    ) -> None:
        """Update all posteriors from GMM observations.

        Parameters
        ----------
        observations : dict[group, dict[speaker, list[_SampleObs]]]
            Nested dict of per-sample GMM results.
        """
        self._last_observations = observations
        # 1. Update speaker posteriors from their sample observations
        for group, speakers in observations.items():
            self._ensure_group(group)
            for speaker, sample_obs_list in speakers.items():
                self._speaker_to_group[speaker] = group

                if speaker in self._speaker_override_mm:
                    # Override speakers: allow tiny data-driven adjustment
                    prior = NormalPosterior(
                        self._speaker_override_mm[speaker], 1.0)
                elif speaker in self._speaker_posteriors:
                    # Existing speaker: keep previous posterior as prior
                    prior = self._speaker_posteriors[speaker]
                else:
                    # New speaker: inherit group mean + between-speaker spread
                    spread_sq = self._group_spread_sq.get(
                        group, self._prior_sigma_sq)
                    prior = NormalPosterior(
                        self._group_posteriors[group].mu,
                        spread_sq,
                    )

                # Pool all sample observations for this speaker
                post = prior
                for obs in sample_obs_list:
                    if (np.isfinite(obs.narrow_mean)
                            and np.isfinite(obs.narrow_var)
                            and obs.narrow_var > 0
                            and obs.n_obs > 0):
                        post = post.conjugate_update(
                            obs.narrow_mean, obs.narrow_var, obs.n_obs)

                self._speaker_posteriors[speaker] = post

        # 2. Update group posteriors from speaker posteriors
        for group in observations:
            speakers_in_group = [
                spk for spk, g in self._speaker_to_group.items()
                if g == group and spk in self._speaker_posteriors
                and spk not in self._speaker_override_mm
            ]
            if not speakers_in_group:
                continue

            # 2a. Update group mean posterior (uncertainty about the mean)
            prior = self._group_prior(group)
            post = prior
            for spk in speakers_in_group:
                sp = self._speaker_posteriors[spk]
                post = post.conjugate_update(sp.mu, sp.sigma_sq, 1)
            self._group_posteriors[group] = post

            # 2b. Compute between-speaker spread (population variance).
            # This does NOT shrink with more speakers — it reflects the
            # natural VTL diversity within the group.
            # Use IQR-based estimate for robustness against outlier speakers.
            if len(speakers_in_group) >= 2:
                spk_mus = np.array([self._speaker_posteriors[spk].mu
                                    for spk in speakers_in_group])
                self._group_spread_sq[group] = _iqr_var(spk_mus)

    # -- Public interface: queries ------------------------------------------

    def reseed_outliers(self, n_sigma: float = 3.0, verbose: bool = False) -> list[str]:
        """Reset speakers whose posterior mean is > n_sigma from their
        group mean (using between-speaker spread).  Returns list of
        reseeded speaker IDs."""
        reseeded = []
        for group, post in self._group_posteriors.items():
            spread_sq = self._group_spread_sq.get(
                group, self._prior_sigma_sq)
            spread = np.sqrt(spread_sq)
            lo = post.mu - n_sigma * spread
            hi = post.mu + n_sigma * spread

            for spk, grp in list(self._speaker_to_group.items()):
                if grp != group:
                    continue
                if spk in self._speaker_override_mm:
                    continue
                if spk not in self._speaker_posteriors:
                    continue
                sp = self._speaker_posteriors[spk]
                if sp.mu < lo or sp.mu > hi:
                    if verbose:
                        print(f"  [reseed] {spk}: μ={sp.mu:.1f}mm "
                              f"outside [{lo:.1f}, {hi:.1f}]mm "
                              f"→ reset to group μ={post.mu:.1f}mm")
                    self._speaker_posteriors[spk] = NormalPosterior(
                        post.mu, spread_sq)
                    reseeded.append(spk)
        return reseeded

    def speaker_vtl(self, speaker: str) -> float:
        """Return posterior mean VTL for a speaker (mm)."""
        if speaker in self._speaker_posteriors:
            return self._speaker_posteriors[speaker].mu
        return VTL_PRIOR_MM.get("unknown", 148.0)

    def speaker_uncertainty(self, speaker: str) -> float:
        """Return posterior σ for a speaker (mm)."""
        if speaker in self._speaker_posteriors:
            return self._speaker_posteriors[speaker].sigma
        return self.prior_sigma_mm

    def group_vtl(self, group: str) -> NormalPosterior:
        """Return group-level posterior."""
        self._ensure_group(group)
        return self._group_posteriors[group]

    def group_mus(self) -> dict[str, float]:
        """Return current group posterior means (for convergence check)."""
        return {g: p.mu for g, p in self._group_posteriors.items()}

    def has_converged(
        self,
        prev_mus: dict[str, float],
        threshold: float = 0.5,
    ) -> bool:
        """Check if all group posteriors have stabilised."""
        for group, post in self._group_posteriors.items():
            if group in prev_mus:
                if abs(post.mu - prev_mus[group]) > threshold:
                    return False
        return True
