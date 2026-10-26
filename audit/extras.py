"""Self-enhancement audit."""
from __future__ import annotations

from typing import Sequence

import numpy as np
from scipy import stats as sps

from judging.judges import FULL, PromptRecord, ScoreStore


def self_enhancement(store: ScoreStore, prompts: Sequence[PromptRecord], judges: Sequence[str], criteria: Sequence[str],
                     judge: str, candidate: str, n_matched_pairs: int | None = None, order: str = "sym") -> dict:
    """Self-judging uplift of ``judge`` on ``candidate``: prompt-level matched pairs, length control, paired t-test."""
    js = list(judges)
    seen: set[str] = set()
    rows = []
    for p in prompts:
        if p.prompt_id in seen:
            continue
        seen.add(p.prompt_id)
        prof = store.profile(p.prompt_id, js, list(criteria), p.candidate_ids, FULL, order)
        ji = prof.judge_index(judge)
        for x in prof.alternatives:
            xi = prof.alt_index(x)
            per_c = prof.scores[ji, :, xi]
            rows.append((p.prompt_id, x, float(per_c.mean()), float(np.log(max(1.0, p.length(x)))), per_c))
    if not rows:
        raise ValueError("no prompts")
    L = np.array([r[3] for r in rows])
    S = np.array([r[2] for r in rows])
    A = np.vstack([np.ones_like(L), L]).T
    coef, *_ = np.linalg.lstsq(A, S, rcond=None)
    resid = S - A @ coef
    by_prompt: dict[str, dict] = {}
    for (pid, x, s, ll, per_c), r in zip(rows, resid):
        by_prompt.setdefault(pid, {})[x] = {"score": s, "resid": float(r), "loglen": ll, "per_c": per_c}
    pairs = []
    for pid, d in by_prompt.items():
        if candidate not in d:
            continue
        others = [x for x in d if x != candidate]
        if not others:
            continue
        own = d[candidate]
        oth_score = float(np.mean([d[x]["score"] for x in others]))
        oth_resid = float(np.mean([d[x]["resid"] for x in others]))
        oth_loglen = float(np.mean([d[x]["loglen"] for x in others]))
        per_c_diff = own["per_c"] - np.mean([d[x]["per_c"] for x in others], axis=0)
        pairs.append({"prompt_id": pid, "own": own["score"], "other": oth_score, "own_resid": own["resid"],
                      "other_resid": oth_resid, "length_gap": abs(own["loglen"] - oth_loglen), "per_c_diff": per_c_diff})
    if n_matched_pairs is not None and n_matched_pairs < len(pairs):
        pairs = sorted(pairs, key=lambda q: q["length_gap"])[:n_matched_pairs]
    n = len(pairs)
    own = np.array([q["own"] for q in pairs])
    oth = np.array([q["other"] for q in pairs])
    own_r = np.array([q["own_resid"] for q in pairs])
    oth_r = np.array([q["other_resid"] for q in pairs])

    def ttest(a, b):
        d = np.asarray(a) - np.asarray(b)
        if len(d) < 2:
            return {"t": float("nan"), "p": float("nan")}
        if np.allclose(d, d[0]):
            return {"t": float("nan"), "p": 1.0 if np.isclose(d[0], 0.0) else 0.0}
        res = sps.ttest_rel(a, b)
        return {"t": float(res.statistic), "p": float(res.pvalue)}

    per_c = np.mean([q["per_c_diff"] for q in pairs], axis=0) if pairs else np.full(len(criteria), np.nan)
    cand_ids = list(dict.fromkeys(c for p in prompts for c in p.candidate_ids))
    mean_scores = {j: {c: [] for c in cand_ids} for j in js}
    for p in prompts:
        prof = store.profile(p.prompt_id, js, list(criteria), p.candidate_ids, FULL, order)
        for j in js:
            for x in prof.alternatives:
                mean_scores[j][x].append(float(prof.scores[prof.judge_index(j), :, prof.alt_index(x)].mean()))
    matrix = {}
    for j in js:
        if j in {c for p in prompts for c in p.candidate_ids}:
            diffs = []
            for p in prompts:
                prof = store.profile(p.prompt_id, js, list(criteria), p.candidate_ids, FULL, order)
                ji = prof.judge_index(j)
                own_s = prof.scores[ji, :, prof.alt_index(j)].mean()
                oth_s = np.mean([prof.scores[ji, :, prof.alt_index(x)].mean() for x in prof.alternatives if x != j])
                diffs.append(own_s - oth_s)
            matrix[j] = {"uplift": float(np.mean(diffs)), "n": len(diffs)}
    return {
        "judge": judge, "candidate": candidate, "n_pairs": n, "length_matched": n_matched_pairs is not None,
        "uplift_raw": float(np.mean(own - oth)) if n else float("nan"), "ttest_raw": ttest(own, oth),
        "uplift_length_adjusted": float(np.mean(own_r - oth_r)) if n else float("nan"), "ttest_length_adjusted": ttest(own_r, oth_r),
        "length_coefficient": float(coef[1]),
        "per_criterion_uplift": {c: float(v) for c, v in zip(criteria, per_c)},
        "self_judging_uplift_by_judge": matrix,
        "judge_candidate_mean_scores": {j: {c: float(np.mean(v)) if v else float("nan") for c, v in d.items()} for j, d in mean_scores.items()},
    }
