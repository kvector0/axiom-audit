"""Calibration set construction from human rankings and the human-jury IIA comparator."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Sequence

import numpy as np

from aggregation.aggregators import borda_scores
from aggregation.profiles import Profile, SocialOrder
from audit.stats import bootstrap_ci, krippendorff_alpha, rate_summary
from bch.bch import mu_agg_from_rankings


@dataclass
class Annotation:
    prompt_id: str
    benchmark: str
    candidates: list[str]
    rankings: list[list[str]]

    @classmethod
    def from_dict(cls, d: dict) -> "Annotation":
        return cls(d["prompt_id"], d["benchmark"], list(d["candidates"]), [list(r) for r in d["rankings"]])

    def to_dict(self) -> dict:
        return {"prompt_id": self.prompt_id, "benchmark": self.benchmark, "candidates": self.candidates, "rankings": self.rankings}

    def rank_matrix(self) -> np.ndarray:
        """(annotators, candidates) matrix of rank positions (1 = best)."""
        return np.array([[r.index(x) + 1 for x in self.candidates] for r in self.rankings], dtype=float)

    def borda_profile(self) -> Profile:
        """Profile with annotators as judges, one criterion, score = Borda count (m-1-position)."""
        m = len(self.candidates)
        arr = np.array([[[m - 1 - r.index(x) for x in self.candidates]] for r in self.rankings], dtype=float)
        return Profile(arr, tuple(f"annotator_{i}" for i in range(len(self.rankings))), ("ranking",), tuple(self.candidates))


def load_annotations(path) -> list[Annotation]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            if line.strip():
                out.append(Annotation.from_dict(json.loads(line)))
    return out


def save_annotations(path, annotations: Sequence[Annotation]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for a in annotations:
            f.write(json.dumps(a.to_dict()) + "\n")


def per_prompt_agreement(ann: Annotation) -> float:
    """Per-prompt Krippendorff's alpha (ordinal metric) on the annotators' rankings."""
    return krippendorff_alpha(ann.rank_matrix())


def corpus_krippendorff(annotations: Sequence[Annotation]) -> float:
    """Krippendorff's alpha (ordinal) over all (prompt, candidate) units."""
    mats = [a.rank_matrix() for a in annotations]
    n_coders = max(m.shape[0] for m in mats)
    padded = []
    for m in mats:
        pad = np.full((n_coders - m.shape[0], m.shape[1]), np.nan)
        padded.append(np.concatenate([m, pad], axis=0))
    return krippendorff_alpha(np.concatenate(padded, axis=1))


@dataclass
class CalibrationSet:
    retained: list[Annotation]
    dropped: list[Annotation]
    threshold: float
    agreements: dict[str, float] = field(default_factory=dict)

    @property
    def n_cal(self) -> int:
        return len(self.retained)

    def mu_agg(self) -> list[dict[str, float]]:
        return [mu_agg_from_rankings(a.rankings) for a in self.retained]

    def summary(self) -> dict:
        vals = [self.agreements[a.prompt_id] for a in self.retained]
        return {
            "threshold": self.threshold,
            "n_annotated": len(self.retained) + len(self.dropped), "n_cal": self.n_cal,
            "mean_agreement_retained": float(np.mean(vals)) if vals else float("nan"),
            "min_agreement_retained": float(np.min(vals)) if vals else float("nan"),
            "max_agreement_retained": float(np.max(vals)) if vals else float("nan"),
            "by_benchmark": {b: sum(1 for a in self.retained if a.benchmark == b) for b in sorted({a.benchmark for a in self.retained})},
        }


def build_calibration_set(annotations: Sequence[Annotation], threshold: float = 0.7) -> CalibrationSet:
    """Retain prompts with per-prompt agreement strictly above ``threshold``."""
    agreements = {a.prompt_id: per_prompt_agreement(a) for a in annotations}
    retained = [a for a in annotations if agreements[a.prompt_id] > threshold]
    dropped = [a for a in annotations if agreements[a.prompt_id] <= threshold]
    return CalibrationSet(retained, dropped, threshold, agreements)


def threshold_sensitivity(annotations: Sequence[Annotation], thresholds=(0.5, 0.6, 0.7, 0.8)) -> dict[str, int]:
    """Calibration-set size as a function of the agreement threshold."""
    return {str(t): build_calibration_set(annotations, t).n_cal for t in thresholds}


def aggregated_borda_order(ann: Annotation, keep: Sequence[str] | None = None) -> SocialOrder:
    """Sum of per-annotator Borda counts, re-derived on the restricted candidate set."""
    prof = ann.borda_profile()
    if keep is not None:
        prof = prof.restrict(list(keep))
    return SocialOrder.from_values(prof.alternatives, borda_scores(prof))


def jury_probes(ann: Annotation) -> list[dict]:
    """The remove-one probes of one jury prompt on the annotators' fixed rankings."""
    full = aggregated_borda_order(ann)
    out = []
    for z in ann.candidates:
        keep = [x for x in ann.candidates if x != z]
        sub = aggregated_borda_order(ann, keep)
        changed = [(x, y) for x, y in combinations(keep, 2) if full.relation(x, y) != sub.relation(x, y)]
        out.append({"removed": z, "violated": bool(changed), "changed_pairs": [list(p) for p in changed]})
    return out


def jury_iia(annotations: Sequence[Annotation], n_bootstrap: int = 1000, seed: int = 0) -> dict:
    """Probe-level human IIA rate with prompt-clustered bootstrap CI, prompt-level rate and its possible range."""
    flags: list[int] = []
    clusters: list[int] = []
    per_prompt: list[dict] = []
    for i, ann in enumerate(annotations):
        probes = jury_probes(ann)
        per_prompt.append({"prompt_id": ann.prompt_id, "benchmark": ann.benchmark, "probes": probes,
                           "prompt_violated": any(p["violated"] for p in probes)})
        for p in probes:
            flags.append(int(p["violated"]))
            clusters.append(i)
    k, n = int(sum(flags)), len(flags)
    lo, hi = bootstrap_ci(flags, n_resamples=n_bootstrap, clusters=clusters, rng=seed)
    n_prompts = len(annotations)
    m = len(annotations[0].candidates) if annotations else 4
    prompt_level_min = int(np.ceil(k / m)) / n_prompts if n_prompts else float("nan")
    prompt_level_max = min(k, n_prompts) / n_prompts if n_prompts else float("nan")
    prompt_level_obs = sum(1 for pp in per_prompt if pp["prompt_violated"]) / n_prompts if n_prompts else float("nan")
    alpha_k = corpus_krippendorff(annotations) if annotations else float("nan")
    alpha_ci = [float("nan"), float("nan")]
    if len(annotations) > 1:
        gen = np.random.default_rng(seed + 1)
        vals = []
        for _ in range(n_bootstrap):
            picked = gen.integers(0, len(annotations), size=len(annotations))
            vals.append(corpus_krippendorff([annotations[i] for i in picked]))
        alpha_ci = [float(np.quantile(vals, 0.025)), float(np.quantile(vals, 0.975))]
    return {
        "n_prompts": n_prompts, "n_probes": n,
        "probe_level": {**rate_summary(k, n), "bootstrap_ci_clustered": [lo, hi]},
        "prompt_level": {"observed": prompt_level_obs, "min_possible": prompt_level_min, "max_possible": prompt_level_max},
        "krippendorff_alpha_ordinal": alpha_k, "krippendorff_alpha_ci95_bootstrap": alpha_ci,
        "by_benchmark": {b: sum(1 for a in annotations if a.benchmark == b) for b in sorted({a.benchmark for a in annotations})},
        "per_prompt": per_prompt,
    }
