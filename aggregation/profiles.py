"""Score profiles over the virtual electorate, social orders, and order-reversal symmetrization."""
from __future__ import annotations

from dataclasses import dataclass
from itertools import combinations
from typing import Iterable, Mapping, Sequence

import numpy as np

TIE_TOL = 1e-9


def _sign(a: float, b: float, tol: float = TIE_TOL) -> int:
    if a > b + tol:
        return 1
    if b > a + tol:
        return -1
    return 0


@dataclass(frozen=True)
class Profile:
    """A score profile s_{j,c}(x), tensor shape (n, k, m)."""

    scores: np.ndarray
    judges: tuple[str, ...]
    criteria: tuple[str, ...]
    alternatives: tuple[str, ...]

    def __post_init__(self) -> None:
        scores = np.asarray(self.scores, dtype=float)
        object.__setattr__(self, "scores", scores)
        object.__setattr__(self, "judges", tuple(self.judges))
        object.__setattr__(self, "criteria", tuple(self.criteria))
        object.__setattr__(self, "alternatives", tuple(self.alternatives))
        n, k, m = len(self.judges), len(self.criteria), len(self.alternatives)
        if scores.shape != (n, k, m):
            raise ValueError(f"scores shape {scores.shape} != (n_judges={n}, n_criteria={k}, n_alts={m})")
        if len(set(self.alternatives)) != m:
            raise ValueError("alternative names must be unique")

    @property
    def n_judges(self) -> int:
        return len(self.judges)

    @property
    def n_criteria(self) -> int:
        return len(self.criteria)

    @property
    def n_alternatives(self) -> int:
        return len(self.alternatives)

    @property
    def n_voters(self) -> int:
        return self.n_judges * self.n_criteria

    def alt_index(self, x: str) -> int:
        return self.alternatives.index(x)

    def judge_index(self, j: str) -> int:
        return self.judges.index(j)

    def score(self, j: str, c: str, x: str) -> float:
        return float(self.scores[self.judge_index(j), self.criteria.index(c), self.alt_index(x)])

    def voter_scores(self) -> np.ndarray:
        """(N, m) matrix of virtual-voter score vectors s_v(.)."""
        return self.scores.reshape(self.n_voters, self.n_alternatives)

    def pairwise_matrix(self) -> np.ndarray:
        """(N, m, m) matrix of sign(s_v(x) - s_v(y)) for every virtual voter."""
        vs = self.voter_scores()
        diff = vs[:, :, None] - vs[:, None, :]
        out = np.zeros_like(diff, dtype=np.int8)
        out[diff > TIE_TOL] = 1
        out[diff < -TIE_TOL] = -1
        return out

    def pairwise_restriction(self, x: str, y: str) -> np.ndarray:
        """Vector over v of the relation of x vs y: +1 (x >_v y), 0, -1."""
        xi, yi = self.alt_index(x), self.alt_index(y)
        return self.pairwise_matrix()[:, xi, yi]

    def restrict(self, keep: Sequence[str]) -> "Profile":
        idx = [self.alt_index(x) for x in keep]
        return Profile(self.scores[:, :, idx], self.judges, self.criteria, tuple(keep))

    def permute_judges(self, sigma: Mapping[str, str] | Sequence[int]) -> "Profile":
        if isinstance(sigma, Mapping):
            idx = [self.judge_index(sigma[j]) for j in self.judges]
        else:
            idx = list(sigma)
        return Profile(self.scores[idx], self.judges, self.criteria, self.alternatives)

    def with_score(self, j: str, c: str, x: str, value: float) -> "Profile":
        s = self.scores.copy()
        s[self.judge_index(j), self.criteria.index(c), self.alt_index(x)] = value
        return Profile(s, self.judges, self.criteria, self.alternatives)

    def unanimous_pairs(self) -> list[tuple[str, str]]:
        """Ordered pairs (x, y) with s_v(x) > s_v(y) for every v."""
        pm = self.pairwise_matrix()
        out = []
        for xi, x in enumerate(self.alternatives):
            for yi, y in enumerate(self.alternatives):
                if xi != yi and np.all(pm[:, xi, yi] == 1):
                    out.append((x, y))
        return out


def symmetrize(fwd: Profile, rev: Profile) -> Profile:
    """Order-symmetrizing pipeline: s = 1/2 (s^fwd + s^rev)."""
    if fwd.judges != rev.judges or fwd.criteria != rev.criteria:
        raise ValueError("fwd/rev profiles must share judges and criteria")
    if set(fwd.alternatives) != set(rev.alternatives):
        raise ValueError("fwd/rev profiles must share the alternative set")
    rev_aligned = rev.restrict(fwd.alternatives)
    return Profile(0.5 * (fwd.scores + rev_aligned.scores), fwd.judges, fwd.criteria, fwd.alternatives)


class SocialOrder:
    """A complete pairwise relation on X returned by an aggregator."""

    __slots__ = ("alternatives", "rel")

    def __init__(self, alternatives: Sequence[str], rel: np.ndarray):
        self.alternatives = tuple(alternatives)
        rel = np.asarray(rel, dtype=np.int8)
        m = len(self.alternatives)
        if rel.shape != (m, m):
            raise ValueError("relation matrix must be (m, m)")
        if not np.array_equal(rel, -rel.T):
            raise ValueError("relation must be antisymmetric with zero diagonal")
        self.rel = rel

    @classmethod
    def from_values(cls, alternatives: Sequence[str], values: Sequence[float], tol: float = TIE_TOL) -> "SocialOrder":
        """Weak order induced by 'higher value is better'."""
        vals = np.asarray(values, dtype=float)
        m = len(alternatives)
        rel = np.zeros((m, m), dtype=np.int8)
        for i in range(m):
            for j in range(m):
                if i != j:
                    rel[i, j] = _sign(vals[i], vals[j], tol)
        return cls(alternatives, rel)

    @classmethod
    def from_ranking(cls, ranking: Sequence[str], alternatives: Sequence[str] | None = None) -> "SocialOrder":
        """Strict linear order from a best-to-worst list."""
        alts = tuple(alternatives) if alternatives is not None else tuple(ranking)
        pos = {x: i for i, x in enumerate(ranking)}
        return cls.from_values(alts, [-pos[x] for x in alts])

    def relation(self, x: str, y: str) -> int:
        return int(self.rel[self.alternatives.index(x), self.alternatives.index(y)])

    def strict_cycle(self) -> list[str] | None:
        """A directed cycle of strict preferences, if one exists."""
        m = len(self.alternatives)
        adj = [[j for j in range(m) if self.rel[i, j] == 1] for i in range(m)]
        color = [0] * m
        stack: list[int] = []

        def dfs(u: int) -> list[int] | None:
            color[u] = 1
            stack.append(u)
            for v in adj[u]:
                if color[v] == 1:
                    return stack[stack.index(v):] + [v]
                if color[v] == 0:
                    found = dfs(v)
                    if found:
                        return found
            stack.pop()
            color[u] = 2
            return None

        for s in range(m):
            if color[s] == 0:
                cyc = dfs(s)
                if cyc:
                    return [self.alternatives[i] for i in cyc]
        return None

    def is_transitive(self) -> bool:
        m = len(self.alternatives)
        r = self.rel
        for i in range(m):
            for j in range(m):
                for k in range(m):
                    if r[i, j] >= 0 and r[j, k] >= 0 and r[i, k] < 0:
                        return False
                    if r[i, j] == 1 and r[j, k] >= 0 and r[i, k] != 1:
                        return False
                    if r[i, j] >= 0 and r[j, k] == 1 and r[i, k] != 1:
                        return False
        return True

    def tiers(self) -> tuple[frozenset[str], ...]:
        if not self.is_transitive():
            raise ValueError("relation is intransitive; no weak order exists")
        m = len(self.alternatives)
        n_above = [int(np.sum(self.rel[:, i] == 1)) for i in range(m)]
        levels = sorted(set(n_above))
        return tuple(frozenset(self.alternatives[i] for i in range(m) if n_above[i] == lv) for lv in levels)

    def restrict(self, keep: Sequence[str]) -> "SocialOrder":
        idx = [self.alternatives.index(x) for x in keep]
        return SocialOrder(tuple(keep), self.rel[np.ix_(idx, idx)])

    def agrees_with(self, other: "SocialOrder", pairs: Iterable[tuple[str, str]] | None = None) -> bool:
        """True iff both relations coincide on every listed pair (default: all common pairs)."""
        if pairs is None:
            common = [x for x in self.alternatives if x in other.alternatives]
            pairs = combinations(common, 2)
        return all(self.relation(x, y) == other.relation(x, y) for x, y in pairs)

    def __repr__(self) -> str:
        if self.is_transitive():
            return "SocialOrder(" + " > ".join("{" + ", ".join(sorted(t)) + "}" for t in self.tiers()) + ")"
        return f"SocialOrder(intransitive, cycle={self.strict_cycle()})"

    def as_dict(self) -> dict:
        return {
            "alternatives": list(self.alternatives),
            "transitive": self.is_transitive(),
            "tiers": [sorted(t) for t in self.tiers()] if self.is_transitive() else None,
        }
