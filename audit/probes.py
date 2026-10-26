"""Axiom probes: universal domain, weak Pareto, ordinal IIA, judge anonymity, criterion monotonicity, position invariance."""
from __future__ import annotations

from dataclasses import dataclass, field
from itertools import combinations, permutations
from typing import Callable, Sequence

import numpy as np

from aggregation.aggregators import majority_vote
from aggregation.profiles import Profile, SocialOrder

AggFn = Callable[..., SocialOrder]


@dataclass
class IIAProbeResult:
    subset: str
    compare_alternatives: list[str]
    eligible_pairs: list[tuple[str, str]]
    changed_pairs: list[tuple[str, str]]
    violated: bool
    full_order: dict = field(default_factory=dict)
    subset_order: dict = field(default_factory=dict)
    n_tie_involved: int = 0
    n_strict_flip: int = 0

    def as_dict(self) -> dict:
        return {"subset": self.subset, "compare_alternatives": self.compare_alternatives,
                "eligible_pairs": [list(p) for p in self.eligible_pairs],
                "changed_pairs": [list(p) for p in self.changed_pairs], "violated": self.violated,
                "full_order": self.full_order, "subset_order": self.subset_order,
                "n_tie_involved": self.n_tie_involved, "n_strict_flip": self.n_strict_flip}


def iia_probe(agg: AggFn, full: Profile, probe: Profile, compare_alternatives: Sequence[str],
              params: dict | None = None, subset: str = "") -> IIAProbeResult:
    """Ordinal IIA: pairs whose virtual-voter restriction is unchanged must keep their social relation."""
    params = params or {}
    f_full = agg(full, **params)
    f_probe = agg(probe, **params)
    comp = list(compare_alternatives)
    eligible: list[tuple[str, str]] = []
    changed: list[tuple[str, str]] = []
    for x, y in combinations(comp, 2):
        if not np.array_equal(full.pairwise_restriction(x, y), probe.pairwise_restriction(x, y)):
            continue
        eligible.append((x, y))
        if f_full.relation(x, y) != f_probe.relation(x, y):
            changed.append((x, y))
    n_tie = n_strict = 0
    for x, y in changed:
        if f_full.relation(x, y) == 0 or f_probe.relation(x, y) == 0:
            n_tie += 1
        else:
            n_strict += 1
    return IIAProbeResult(subset, comp, eligible, changed, bool(changed),
                          f_full.restrict(comp).as_dict(), f_probe.restrict(comp).as_dict(),
                          n_tie, n_strict)


def prompt_level_iia(results: Sequence[IIAProbeResult]) -> bool:
    """A prompt counts as violated if any of its probes violates."""
    return any(r.violated for r in results)


def pi_composite(agg: AggFn, fwd: Profile, rev: Profile, params: dict | None = None) -> bool:
    """Composite PI violation: the social rankings of the forward- and reverse-order profiles differ."""
    params = params or {}
    return not agg(fwd, **params).agrees_with(agg(rev, **params))


def verdict_flip(verdicts: dict) -> bool | None:
    """True if the judge's preferred candidate differs between (A,B) and (B,A); None if either failed to parse."""
    ab, ba = verdicts.get("AB"), verdicts.get("BA")
    if ab is None or ba is None:
        return None
    return ab != ba


@dataclass
class CMResult:
    n_perturbations: int
    n_demotions: int
    demotions: list[tuple[str, str, str]]

    @property
    def violated(self) -> bool:
        return self.n_demotions > 0


def cm_probe(agg: AggFn, profile: Profile, delta: float = 1.0, params: dict | None = None) -> CMResult:
    """Criterion-Monotonicity: raising one s_{j,c}(x) by delta must not weaken x's relation to any y."""
    params = params or {}
    base = agg(profile, **params)
    n = 0
    dem: list[tuple[str, str, str]] = []
    for j in profile.judges:
        for c in profile.criteria:
            for x in profile.alternatives:
                n += 1
                pert = agg(profile.with_score(j, c, x, profile.score(j, c, x) + delta), **params)
                for y in profile.alternatives:
                    if y != x and pert.relation(x, y) < base.relation(x, y):
                        dem.append((j, c, x))
                        break
    return CMResult(n, len(dem), dem)


@dataclass
class WPResult:
    n_unanimous_pairs: int
    n_violations: int
    violations: list[tuple[str, str]]

    @property
    def violated(self) -> bool:
        return self.n_violations > 0


def wp_probe(agg: AggFn, profile: Profile, params: dict | None = None) -> WPResult:
    """Weak Pareto: unanimous s_v(x) > s_v(y) yet y weakly above x."""
    order = agg(profile, **(params or {}))
    pairs = profile.unanimous_pairs()
    viol = [(x, y) for x, y in pairs if order.relation(x, y) <= 0]
    return WPResult(len(pairs), len(viol), viol)


def anonymity_probe(agg: AggFn, profile: Profile, params: dict | None = None, max_perms: int = 120,
                    rng: np.random.Generator | int | None = 0) -> bool:
    """Judge-Anonymity: does any judge permutation change F(s)?"""
    params = params or {}
    base = agg(profile, **params)
    idx = list(range(profile.n_judges))
    perms = list(permutations(idx))
    if len(perms) > max_perms:
        gen = rng if isinstance(rng, np.random.Generator) else np.random.default_rng(rng)
        perms = [tuple(gen.permutation(idx)) for _ in range(max_perms)]
    for p in perms:
        if list(p) == idx:
            continue
        if not agg(profile.permute_judges(list(p)), **params).agrees_with(base):
            return True
    return False


def condorcet_cycle(profile: Profile) -> list[str] | None:
    return majority_vote(profile).strict_cycle()


def ud_probe(agg: AggFn, profile: Profile, params: dict | None = None) -> bool:
    """True if F fails to return a complete transitive weak order on this profile."""
    return not agg(profile, **(params or {})).is_transitive()
