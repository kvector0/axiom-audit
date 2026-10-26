"""Audited aggregators and frontier witnesses, with a name-indexed registry."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, Sequence

import numpy as np

from aggregation.profiles import Profile, SocialOrder
from bch.bch import f_bch

Aggregator = Callable[..., SocialOrder]


def borda_scores(profile: Profile) -> np.ndarray:
    """B(x) = sum_v |{y: x >_v y}|."""
    pm = profile.pairwise_matrix()
    return (pm == 1).sum(axis=(0, 2)).astype(float)


def borda(profile: Profile) -> SocialOrder:
    return SocialOrder.from_values(profile.alternatives, borda_scores(profile))


def score_mean(profile: Profile) -> SocialOrder:
    """M(x) = (1/N) sum_v s_v(x)."""
    return SocialOrder.from_values(profile.alternatives, profile.voter_scores().mean(axis=0))


def mtbench_avg(profile: Profile) -> SocialOrder:
    """Within-judge cross-criterion mean, then cross-judge mean."""
    per_judge = profile.scores.mean(axis=1)
    return SocialOrder.from_values(profile.alternatives, per_judge.mean(axis=0))


def single_judge_mean(profile: Profile, judge: str) -> SocialOrder:
    """Single-judge cross-criterion mean (the audited Best-of-Judge rule)."""
    ji = profile.judge_index(judge)
    return SocialOrder.from_values(profile.alternatives, profile.scores[ji].mean(axis=0))


def dictatorship(profile: Profile, judge: str, criterion: str) -> SocialOrder:
    """Single-(judge, criterion) dictatorship."""
    ji, ci = profile.judge_index(judge), profile.criteria.index(criterion)
    return SocialOrder.from_values(profile.alternatives, profile.scores[ji, ci])


def weighted_utilitarian(profile: Profile, criterion_weights: Sequence[float] | None = None) -> SocialOrder:
    """Criterion-weighted utilitarian rule argsort(sum_c omega_c sum_j s_{j,c}(.)), omega_c > 0."""
    k = profile.n_criteria
    w = np.ones(k) if criterion_weights is None else np.asarray(criterion_weights, dtype=float)
    if w.shape != (k,) or np.any(w <= 0):
        raise ValueError("criterion weights must be a length-k vector with omega_c > 0")
    per_c = profile.scores.sum(axis=0)
    return SocialOrder.from_values(profile.alternatives, w @ per_c)


def median_score_mean(profile: Profile, criterion_weights: Sequence[float] | None = None) -> SocialOrder:
    """Median-over-judges variant of Score-Mean."""
    k = profile.n_criteria
    w = np.full(k, 1.0 / k) if criterion_weights is None else np.asarray(criterion_weights, dtype=float)
    med = np.median(profile.scores, axis=0)
    return SocialOrder.from_values(profile.alternatives, w @ med)


def majority_vote(profile: Profile) -> SocialOrder:
    """Pairwise majority over all N = nk virtual voters (intransitive on Condorcet-cyclic profiles)."""
    pm = profile.pairwise_matrix()
    m = profile.n_alternatives
    rel = np.zeros((m, m), dtype=np.int8)
    for i in range(m):
        for j in range(m):
            if i == j:
                continue
            x_weak = int(np.sum(pm[:, i, j] >= 0))
            y_weak = int(np.sum(pm[:, j, i] >= 0))
            if x_weak > y_weak:
                rel[i, j] = 1
            elif y_weak > x_weak:
                rel[i, j] = -1
    return SocialOrder(profile.alternatives, rel)


def constant_rule(profile: Profile, order: Sequence[str] | SocialOrder) -> SocialOrder:
    """F(s) == a fixed weak order."""
    if isinstance(order, SocialOrder):
        return order.restrict(profile.alternatives)
    present = [x for x in order if x in profile.alternatives]
    if set(present) != set(profile.alternatives):
        raise ValueError("constant order must cover every alternative")
    return SocialOrder.from_ranking(present, profile.alternatives)


def pairwise_outcomes(profile: Profile) -> np.ndarray:
    """(m, m) matrix of wins of x over y summed over virtual voters; ties count 1/2 each."""
    pm = profile.pairwise_matrix()
    wins = (pm == 1).sum(axis=0).astype(float)
    ties = (pm == 0).sum(axis=0).astype(float)
    np.fill_diagonal(ties, 0.0)
    return wins + 0.5 * ties


def bradley_terry_strengths(wins: np.ndarray, prior: float = 0.01, max_iter: int = 10_000, tol: float = 1e-10) -> np.ndarray:
    """Bradley-Terry maximum-likelihood strengths via the MM algorithm."""
    m = wins.shape[0]
    w = np.asarray(wins, dtype=float) + prior
    np.fill_diagonal(w, 0.0)
    n_pair = w + w.T
    total_wins = w.sum(axis=1)
    pi = np.ones(m)
    for _ in range(max_iter):
        denom = np.zeros(m)
        for i in range(m):
            for j in range(m):
                if i != j and n_pair[i, j] > 0:
                    denom[i] += n_pair[i, j] / (pi[i] + pi[j])
        new = total_wins / denom
        new = new / np.exp(np.mean(np.log(new)))
        if np.max(np.abs(new - pi)) < tol:
            pi = new
            break
        pi = new
    return pi


def arena_bt(profile: Profile, prior: float = 0.01, strength_tol: float = 1e-6) -> SocialOrder:
    """Arena-style Bradley-Terry on score-derived pairwise outcomes."""
    strengths = bradley_terry_strengths(pairwise_outcomes(profile), prior=prior)
    return SocialOrder.from_values(profile.alternatives, np.log(strengths), tol=strength_tol)


def win_rate(profile: Profile) -> SocialOrder:
    """Mean pairwise win rate of x over all opponents and virtual voters."""
    m = profile.n_alternatives
    wins = pairwise_outcomes(profile)
    rate = wins.sum(axis=1) / (profile.n_voters * (m - 1))
    return SocialOrder.from_values(profile.alternatives, rate)


def lc_win_rate(profile: Profile, lengths: Sequence[float], length_coef: float, length_scale: float = 1.0) -> SocialOrder:
    """Length-controlled win rate."""
    m = profile.n_alternatives
    L = np.asarray(lengths, dtype=float)
    if L.shape != (m,):
        raise ValueError("lengths must have one entry per alternative")
    o = pairwise_outcomes(profile) / profile.n_voters
    np.fill_diagonal(o, 0.0)
    corr = np.tanh((L[:, None] - L[None, :]) / length_scale)
    adj = o - length_coef * corr
    np.fill_diagonal(adj, 0.0)
    return SocialOrder.from_values(profile.alternatives, adj.sum(axis=1) / (m - 1))


def fit_length_coefficient(profiles: Sequence[Profile], lengths: Sequence[Sequence[float]], length_scale: float = 1.0) -> float:
    """Least-squares fit of the global length coefficient across prompts."""
    xs: list[float] = []
    ys: list[float] = []
    for prof, L in zip(profiles, lengths):
        L = np.asarray(L, dtype=float)
        o = pairwise_outcomes(prof) / prof.n_voters
        m = prof.n_alternatives
        for i in range(m):
            for j in range(m):
                if i != j:
                    xs.append(float(np.tanh((L[i] - L[j]) / length_scale)))
                    ys.append(float(o[i, j] - 0.5))
    x = np.asarray(xs)
    y = np.asarray(ys)
    denom = float(x @ x)
    return float(x @ y / denom) if denom > 0 else 0.0


@dataclass(frozen=True)
class AggregatorSpec:
    name: str
    fn: Aggregator
    pipeline: str
    anonymity_analytic: bool
    iia_analytic: bool
    ud_structural: bool
    params: tuple[str, ...] = ()
    pi_cell: str = "composite"


REGISTRY: dict[str, AggregatorSpec] = {
    "majority_vote": AggregatorSpec("majority_vote", majority_vote, "sym", True, True, False),
    "score_mean": AggregatorSpec("score_mean", score_mean, "sym", True, False, True),
    "borda": AggregatorSpec("borda", borda, "fwd", True, False, True, pi_cell="judge_level"),
    "arena_bt": AggregatorSpec("arena_bt", arena_bt, "sym", True, False, True),
    "mtbench_avg": AggregatorSpec("mtbench_avg", mtbench_avg, "fwd", True, False, True),
    "best_of_judge": AggregatorSpec("best_of_judge", single_judge_mean, "sym", False, False, True, ("judge",)),
    "perm_consensus": AggregatorSpec("perm_consensus", borda, "sym", True, False, True),
    "bch": AggregatorSpec("bch", f_bch, "sym", True, False, True, ("criterion_weights",)),
    "bch_no_sym": AggregatorSpec("bch_no_sym", f_bch, "fwd", True, False, True, ("criterion_weights",), pi_cell="judge_level"),
    "median_score_mean": AggregatorSpec("median_score_mean", median_score_mean, "sym", True, False, True, ("criterion_weights",)),
    "win_rate": AggregatorSpec("win_rate", win_rate, "sym", True, False, True),
    "lc_win_rate": AggregatorSpec("lc_win_rate", lc_win_rate, "sym", True, False, True, ("lengths", "length_coef", "length_scale")),
    "dictatorship": AggregatorSpec("dictatorship", dictatorship, "sym", False, True, True, ("judge", "criterion")),
    "weighted_utilitarian": AggregatorSpec("weighted_utilitarian", weighted_utilitarian, "sym", True, False, True, ("criterion_weights",)),
    "constant": AggregatorSpec("constant", constant_rule, "sym", True, True, True, ("order",)),
}


def get(name: str) -> AggregatorSpec:
    return REGISTRY[name]


AUDIT_ORDER = ["majority_vote", "score_mean", "borda", "arena_bt", "mtbench_avg", "best_of_judge", "perm_consensus", "bch"]
