"""Rates, standard errors, Clopper-Pearson intervals, McNemar tests, bootstrap, Krippendorff's alpha."""
from __future__ import annotations

from dataclasses import dataclass
from math import sqrt
from typing import Sequence

import numpy as np
from scipy import stats as sps


def rate_se(p_hat: float, n: int) -> float:
    """Normal-approximation standard error sqrt(p(1-p)/n)."""
    if n <= 0:
        return float("nan")
    return sqrt(p_hat * (1.0 - p_hat) / n)


def clopper_pearson(k: int, n: int, level: float = 0.95) -> tuple[float, float]:
    """Exact (Clopper-Pearson) binomial confidence interval for k successes in n."""
    if n <= 0:
        return (float("nan"), float("nan"))
    a = 1.0 - level
    lo = 0.0 if k == 0 else float(sps.beta.ppf(a / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(sps.beta.ppf(1 - a / 2, k + 1, n - k))
    return (lo, hi)


@dataclass(frozen=True)
class McNemarResult:
    b: int
    c: int
    n_concordant: int
    chi2: float
    p_chi2: float
    p_exact: float
    n_both: int = 0
    n_neither: int = 0

    def as_dict(self) -> dict:
        return {"b": self.b, "c": self.c, "n_concordant": self.n_concordant,
                "contingency": {"both": self.n_both, "first_only": self.b, "second_only": self.c, "neither": self.n_neither},
                "chi2": self.chi2, "p_chi2": self.p_chi2, "p_exact": self.p_exact}


def mcnemar(first: Sequence[bool], second: Sequence[bool]) -> McNemarResult:
    """Paired McNemar test on prompt-level violation indicators."""
    f = np.asarray(first, dtype=bool)
    s = np.asarray(second, dtype=bool)
    if f.shape != s.shape:
        raise ValueError("paired indicator vectors must have equal length")
    b = int(np.sum(f & ~s))
    c = int(np.sum(~f & s))
    n_conc = int(len(f) - b - c)
    both, neither = int(np.sum(f & s)), int(np.sum(~f & ~s))
    if b + c == 0:
        return McNemarResult(b, c, n_conc, 0.0, 1.0, 1.0, both, neither)
    chi2 = (b - c) ** 2 / (b + c)
    p_chi2 = float(sps.chi2.sf(chi2, df=1))
    p_exact = float(min(1.0, 2.0 * sps.binom.cdf(min(b, c), b + c, 0.5)))
    return McNemarResult(b, c, n_conc, float(chi2), p_chi2, p_exact, both, neither)


def bonferroni_threshold(alpha: float = 0.05, n_comparisons: int = 7) -> float:
    """Bonferroni-corrected significance threshold alpha / n_comparisons."""
    return alpha / n_comparisons


def bootstrap_ci(values: Sequence[float], n_resamples: int = 1000, level: float = 0.95,
                 clusters: Sequence[int] | None = None, rng: np.random.Generator | int | None = 0,
                 statistic=np.mean) -> tuple[float, float]:
    """Percentile bootstrap CI of ``statistic`` over ``values``."""
    gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
    v = np.asarray(values, dtype=float)
    if len(v) == 0:
        return (float("nan"), float("nan"))
    stats_out = np.empty(n_resamples)
    if clusters is None:
        for i in range(n_resamples):
            idx = gen.integers(0, len(v), len(v))
            stats_out[i] = statistic(v[idx])
    else:
        cl = np.asarray(clusters)
        uniq = np.unique(cl)
        members = {u: np.flatnonzero(cl == u) for u in uniq}
        for i in range(n_resamples):
            picked = gen.choice(uniq, size=len(uniq), replace=True)
            idx = np.concatenate([members[u] for u in picked])
            stats_out[i] = statistic(v[idx])
    a = (1.0 - level) / 2
    return (float(np.quantile(stats_out, a)), float(np.quantile(stats_out, 1 - a)))


def krippendorff_alpha(data: np.ndarray) -> float:
    """Krippendorff's alpha (ordinal metric) for a (coders, units) matrix; NaN marks missing values."""
    d = np.asarray(data, dtype=float)
    if d.ndim != 2:
        raise ValueError("data must be (coders, units)")
    vals = np.unique(d[~np.isnan(d)])
    if len(vals) < 2:
        return 1.0
    v_index = {v: i for i, v in enumerate(vals)}
    V = len(vals)
    o = np.zeros((V, V))
    for u in range(d.shape[1]):
        col = d[:, u]
        col = col[~np.isnan(col)]
        m_u = len(col)
        if m_u < 2:
            continue
        for i, a in enumerate(col):
            for j, b in enumerate(col):
                if i != j:
                    o[v_index[a], v_index[b]] += 1.0 / (m_u - 1)
    n_c = o.sum(axis=1)
    n = n_c.sum()
    if n <= 1:
        return 1.0
    cum = np.cumsum(n_c)
    delta = np.zeros((V, V))
    for c in range(V):
        for k in range(V):
            if c == k:
                continue
            lo, hi = min(c, k), max(c, k)
            between = cum[hi] - (cum[lo - 1] if lo > 0 else 0.0)
            delta[c, k] = (between - (n_c[lo] + n_c[hi]) / 2.0) ** 2
    do = np.sum(o * delta)
    de = np.sum(np.outer(n_c, n_c) * delta) / (n - 1)
    if de == 0:
        return 1.0
    return float(1.0 - do / de)


def rate_summary(k: int, n: int, level: float = 0.95) -> dict:
    p = k / n if n else float("nan")
    lo, hi = clopper_pearson(k, n, level)
    return {"count": k, "n": n, "rate": p, "se": rate_se(p, n) if n else float("nan"),
            "ci_low": lo, "ci_high": hi}
