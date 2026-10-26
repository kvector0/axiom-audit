"""Borda-Conformal Hybrid: aggregator, split-conformal calibration, per-pair certificate and gap-marginalized bound."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations
from math import ceil
from typing import Sequence

import numpy as np

from aggregation.profiles import Profile, SocialOrder


def per_voter_borda(profile: Profile) -> np.ndarray:
    """Per-(judge, criterion) Borda counts B~_{j,c}(x) = |{y: s_{j,c}(x) > s_{j,c}(y)}|, shape (n, k, m)."""
    pm = profile.pairwise_matrix()
    counts = (pm == 1).sum(axis=2)
    return counts.reshape(profile.n_judges, profile.n_criteria, profile.n_alternatives).astype(float)


def _weights(profile: Profile, criterion_weights: Sequence[float] | None) -> np.ndarray:
    k = profile.n_criteria
    if criterion_weights is None:
        return np.full(k, 1.0 / k)
    w = np.asarray(criterion_weights, dtype=float)
    if w.shape != (k,):
        raise ValueError(f"criterion_weights must have length k={k}")
    if np.any(w < 0) or not np.isclose(w.sum(), 1.0):
        raise ValueError("criterion_weights must lie in the simplex")
    return w


def bch_scores(profile: Profile, criterion_weights: Sequence[float] | None = None) -> np.ndarray:
    """B^_BCH(x) = sum_c w_c * median_j B~_{j,c}(x)."""
    w = _weights(profile, criterion_weights)
    med = np.median(per_voter_borda(profile), axis=0)
    return w @ med


def f_bch(profile: Profile, criterion_weights: Sequence[float] | None = None) -> SocialOrder:
    """F_BCH: the weak order induced by B^_BCH."""
    return SocialOrder.from_values(profile.alternatives, bch_scores(profile, criterion_weights))


def pair_gaps(profile: Profile, criterion_weights: Sequence[float] | None = None) -> dict[tuple[str, str], float]:
    """Ranking gaps Delta_{x,y}(s) = |B^_BCH(x) - B^_BCH(y)| for every unordered pair."""
    b = bch_scores(profile, criterion_weights)
    idx = {x: i for i, x in enumerate(profile.alternatives)}
    return {(x, y): float(abs(b[idx[x]] - b[idx[y]])) for x, y in combinations(profile.alternatives, 2)}


def conformal_quantile(residuals: Sequence[float], alpha: float) -> float:
    """Split-conformal quantile: the ceil((n+1)(1-alpha))-th smallest residual."""
    r = np.sort(np.asarray(residuals, dtype=float))
    n = len(r)
    if n == 0:
        raise ValueError("no calibration residuals")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must be in (0, 1)")
    k = ceil((n + 1) * (1.0 - alpha))
    if k > n:
        return float("inf")
    return float(r[k - 1])


def gap_marginalized_bound(alpha: float, small_gap_fraction: float) -> float:
    """4 alpha + p, with p the small-gap fraction."""
    return 4.0 * alpha + small_gap_fraction


@dataclass
class CalibrationResult:
    alpha: float
    n_cal: int
    q_hat: float
    residuals: list[float]
    residual_alternatives: list[str]
    gaps: list[float]
    small_gap_fraction: float
    bound: float
    gap_threshold: float
    empirical_coverage: float
    gap_histogram: dict[str, int] = field(default_factory=dict)

    def as_dict(self) -> dict:
        return {
            "alpha": self.alpha,
            "n_cal": self.n_cal,
            "q_hat": self.q_hat,
            "gap_threshold_4q": self.gap_threshold,
            "small_gap_fraction_p": self.small_gap_fraction,
            "bound_4alpha_plus_p": self.bound,
            "empirical_coverage": self.empirical_coverage,
            "gap_histogram": self.gap_histogram,
            "n_gaps": len(self.gaps),
            "residuals": self.residuals,
            "residual_alternatives": self.residual_alternatives,
        }


def calibrate(
    cal_profiles: Sequence[Profile],
    mu_agg: Sequence[dict[str, float]],
    alpha: float = 0.05,
    criterion_weights: Sequence[float] | None = None,
    rng: np.random.Generator | int | None = 0,
    residual_alternatives: Sequence[str] | None = None,
) -> CalibrationResult:
    """Split-conformal calibration of BCH: one aggregated residual per calibration profile at one alternative."""
    if len(cal_profiles) != len(mu_agg):
        raise ValueError("one mu_agg mapping per calibration profile is required")
    if len(cal_profiles) == 0:
        raise ValueError("empty calibration set")
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    residuals: list[float] = []
    chosen: list[str] = []
    gaps: list[float] = []
    for i, (prof, mu) in enumerate(zip(cal_profiles, mu_agg)):
        b = bch_scores(prof, criterion_weights)
        if residual_alternatives is not None:
            x = residual_alternatives[i]
        else:
            x = prof.alternatives[int(gen.integers(prof.n_alternatives))]
        residuals.append(float(abs(b[prof.alt_index(x)] - mu[x])))
        chosen.append(x)
        gaps.extend(pair_gaps(prof, criterion_weights).values())
    q_hat = conformal_quantile(residuals, alpha)
    thr = 4.0 * q_hat
    gaps_arr = np.asarray(gaps)
    p = float(np.mean(gaps_arr <= thr + 1e-12)) if len(gaps_arr) else float("nan")
    hist: dict[str, int] = {}
    for g in np.round(gaps_arr, 6):
        key = f"{g:g}"
        hist[key] = hist.get(key, 0) + 1
    coverage = float(np.mean(np.asarray(residuals) <= q_hat + 1e-12))
    return CalibrationResult(
        alpha=alpha,
        n_cal=len(residuals),
        q_hat=q_hat,
        residuals=residuals,
        residual_alternatives=chosen,
        gaps=[float(g) for g in gaps_arr],
        small_gap_fraction=p,
        bound=gap_marginalized_bound(alpha, p),
        gap_threshold=thr,
        empirical_coverage=coverage,
        gap_histogram=dict(sorted(hist.items(), key=lambda kv: float(kv[0]))),
    )


def certify_pair(profile: Profile, x: str, y: str, calibration: CalibrationResult,
                 criterion_weights: Sequence[float] | None = None) -> dict:
    """Per-pair coverage certificate for the pair (x, y)."""
    gaps = pair_gaps(profile, criterion_weights)
    gap = gaps[(x, y)] if (x, y) in gaps else gaps[(y, x)]
    large = gap > calibration.gap_threshold
    return {
        "pair": [x, y],
        "gap": gap,
        "threshold_4q": calibration.gap_threshold,
        "large_gap": large,
        "swap_probability_bound": 4.0 * calibration.alpha if large else 1.0,
    }


def mu_agg_from_rankings(rankings: Sequence[Sequence[str]]) -> dict[str, float]:
    """Reference aggregated Borda score mu_agg(x): mean over annotators of the per-annotator Borda count."""
    alts = list(rankings[0])
    m = len(alts)
    acc = {x: 0.0 for x in alts}
    for r in rankings:
        if set(r) != set(alts) or len(r) != m:
            raise ValueError("every annotator ranking must be a permutation of the same alternatives")
        for pos, x in enumerate(r):
            acc[x] += float(m - 1 - pos)
    return {x: acc[x] / len(rankings) for x in alts}
