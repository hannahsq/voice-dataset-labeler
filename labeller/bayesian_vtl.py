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
        return float(np.mean(x)), float(np.var(x, ddof=1)), 1.0

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


# ---------------------------------------------------------------------------
# Hierarchical Bayesian VTL estimator
# ---------------------------------------------------------------------------

@dataclass
class _SampleObs:
    """Observation summary from one sample's GMM fit."""
    narrow_mean: float
    narrow_var: float
    n_obs: int


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
                # Inherit group posterior
                self._speaker_posteriors[speaker] = NormalPosterior(
                    self._group_posteriors[group].mu,
                    self._group_posteriors[group].sigma_sq,
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
    ) -> None:
        """Register a rough initial VTL estimate for a speaker (from seed
        pass).  This sets the speaker posterior to a moderately confident
        estimate centered on vtl_mm."""
        self._ensure_group(group)
        self._speaker_to_group[speaker] = group
        if speaker in self._speaker_override_mm:
            return  # don't overwrite manual overrides
        # Seed: moderately confident (σ² = prior_σ²)
        self._speaker_posteriors[speaker] = NormalPosterior(
            vtl_mm, self._prior_sigma_sq)

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
        # 1. Update speaker posteriors from their sample observations
        for group, speakers in observations.items():
            self._ensure_group(group)
            for speaker, sample_obs_list in speakers.items():
                self._speaker_to_group[speaker] = group

                if speaker in self._speaker_override_mm:
                    # Override speakers: allow tiny data-driven adjustment
                    prior = NormalPosterior(
                        self._speaker_override_mm[speaker], 1.0)
                else:
                    # Normal speakers: use current group posterior as prior
                    prior = NormalPosterior(
                        self._group_posteriors[group].mu,
                        self._group_posteriors[group].sigma_sq,
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

            prior = self._group_prior(group)
            post = prior
            for spk in speakers_in_group:
                sp = self._speaker_posteriors[spk]
                # Each speaker posterior contributes as an observation
                # with variance = speaker posterior variance
                post = post.conjugate_update(sp.mu, sp.sigma_sq, 1)

            self._group_posteriors[group] = post

    # -- Public interface: queries ------------------------------------------

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
