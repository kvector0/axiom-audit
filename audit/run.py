"""Audit orchestration: aggregator binding, six-cell axiom profiles, statistics, calibration and jury runs, report writer."""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np
import yaml

from aggregation import aggregators as agg_mod
from aggregation.aggregators import AggregatorSpec, fit_length_coefficient
from aggregation.profiles import Profile, symmetrize
from audit import probes
from audit.extras import self_enhancement
from audit.human import build_calibration_set, jury_iia, load_annotations, threshold_sensitivity
from audit.stats import bonferroni_threshold, bootstrap_ci, mcnemar, rate_se, rate_summary
from bch.bch import calibrate, certify_pair
from judging.judges import (FULL, JudgeSpec, PromptRecord, ScoreStore, SyntheticWorld, judge_fingerprint, load_prompts, make_judge,
                            subset_key_add, subset_key_remove, subset_key_replace)


def load_config(path: str | Path) -> dict:
    cfg = yaml.safe_load(Path(path).read_text(encoding="utf-8"))
    cfg["_config_path"] = str(path)
    cfg.setdefault("criteria", ["correctness", "helpfulness", "safety", "brevity"])
    cfg.setdefault("criterion_weights", [1.0 / len(cfg["criteria"])] * len(cfg["criteria"]))
    cfg.setdefault("scoring", {})
    cfg["scoring"].setdefault("orders", ["fwd", "rev"])
    cfg.setdefault("probes", {})
    cfg["probes"].setdefault("iia", {"variants": ["remove_one"]})
    cfg["probes"].setdefault("pi", {"per_criterion": True, "include_overall": False})
    cfg["probes"].setdefault("cm", {"delta": 1.0})
    cfg["probes"].setdefault("anonymity", {"empirical": False})
    cfg.setdefault("pipeline_overrides", {})
    cfg.setdefault("pi_cell_overrides", {})
    cfg.setdefault("aggregators", list(agg_mod.AUDIT_ORDER))
    cfg.setdefault("bch", {"alpha": 0.05})
    cfg.setdefault("stats", {})
    cfg["stats"].setdefault("bootstrap_resamples", 1000)
    cfg["stats"].setdefault("mcnemar_reference", "bch")
    cfg["stats"].setdefault("bonferroni_alpha", 0.05)
    cfg.setdefault("seed", 0)
    return cfg


def judge_specs(cfg: dict) -> list[JudgeSpec]:
    out = []
    for j in cfg["judges"]:
        j = dict(j)
        out.append(JudgeSpec(name=j.pop("name"), client=j.pop("client"), model=j.pop("model", ""),
                             temperature=float(j.pop("temperature", 0.0)), max_tokens=int(j.pop("max_tokens", 64)), extra=j))
    return out


def candidate_specs(cfg: dict, key: str = "candidates") -> list[JudgeSpec]:
    out = []
    for c in cfg.get(key, []):
        c = dict(c)
        out.append(JudgeSpec(name=c.pop("name"), client=c.pop("client"), model=c.pop("model", ""),
                             temperature=float(c.pop("temperature", 0.0)), max_tokens=int(c.pop("max_tokens", 512)), extra=c))
    return out


def synthetic_world(cfg: dict) -> SyntheticWorld | None:
    syn = cfg.get("synthetic", {})
    if not syn.get("enabled", False):
        return None
    return SyntheticWorld(seed=int(cfg["seed"]), criteria=cfg["criteria"],
                          candidate_means=syn.get("candidate_means", {}), sd=float(syn.get("quality_sd", 1.0)))


def build_judges(cfg: dict):
    world = synthetic_world(cfg)
    return [make_judge(s, cfg["scoring"], world) for s in judge_specs(cfg)]


def open_store(cfg: dict, path: str | Path, prompts: Sequence[PromptRecord]) -> ScoreStore:
    """Score store, refused if its entries were produced under other judge settings or candidate responses."""
    store = ScoreStore(path)
    try:
        world = synthetic_world(cfg)
        store.check_inputs({s.name: judge_fingerprint(s, cfg["scoring"], world if s.client == "synthetic" else None)
                            for s in judge_specs(cfg)}, prompts)
    except ValueError as e:
        raise SystemExit(str(e))
    return store


@dataclass
class BoundAggregator:
    name: str
    spec: AggregatorSpec
    pipeline: str
    params: dict = field(default_factory=dict)
    per_prompt: Callable[[PromptRecord, Sequence[str]], dict] | None = None
    pi_cell: str = "composite"

    def fn(self, prompt: PromptRecord | None = None):
        """Aggregator callable f(profile) with all parameters bound."""
        base = self.spec.fn
        params = dict(self.params)
        pp = self.per_prompt

        def f(profile: Profile, **overrides):
            p = dict(params)
            if pp is not None and prompt is not None:
                p.update(pp(prompt, profile.alternatives))
            p.update(overrides)
            return base(profile, **p)

        return f


def bind_aggregator(name: str, cfg: dict, prompts: Sequence[PromptRecord], judges: Sequence[str],
                    lc_state: dict | None = None) -> BoundAggregator:
    spec = agg_mod.get(name)
    pipeline = cfg.get("pipeline_overrides", {}).get(name, spec.pipeline)
    if pipeline not in ("fwd", "sym"):
        raise ValueError(f"pipeline_overrides[{name!r}] must be fwd or sym")
    pi_cell = cfg.get("pi_cell_overrides", {}).get(name, spec.pi_cell)
    if pi_cell not in ("judge_level", "composite"):
        raise ValueError(f"pi_cell_overrides[{name!r}] must be judge_level or composite")
    params: dict = {}
    per_prompt = None
    if name == "best_of_judge":
        params["judge"] = cfg.get("best_of_judge", {}).get("judge", judges[0])
    elif name in ("bch", "bch_no_sym", "median_score_mean", "weighted_utilitarian"):
        w = cfg.get("criterion_weights")
        params["criterion_weights"] = [float(v) for v in w] if w else None
    elif name == "arena_bt":
        bt = cfg.get("arena_bt", {})
        params["prior"] = float(bt.get("prior", 0.01))
        params["strength_tol"] = float(bt.get("strength_tol", 1e-6))
    elif name == "dictatorship":
        d = cfg.get("dictatorship", {})
        params["judge"] = d.get("judge", judges[0])
        params["criterion"] = d.get("criterion", cfg["criteria"][0])
    elif name == "constant":
        default = [c["name"] for c in cfg["candidates"]] + [e["name"] for e in cfg.get("extra_candidates", [])]
        params["order"] = cfg.get("constant", {}).get("order", default)
    elif name == "lc_win_rate":
        if lc_state is None:
            raise ValueError("lc_win_rate needs the fitted length state")
        coef, scale = lc_state["length_coef"], lc_state["length_scale"]
        by_id = {p.prompt_id: p for p in prompts}

        def lengths(prompt: PromptRecord, alternatives: Sequence[str]) -> dict:
            rec = by_id[prompt.prompt_id]
            return {"lengths": [rec.length(x) for x in alternatives], "length_coef": coef, "length_scale": scale}

        per_prompt = lengths
    return BoundAggregator(name, spec, pipeline, params, per_prompt, pi_cell)


def fit_lc_state(cfg: dict, prompts: Sequence[PromptRecord], store: ScoreStore, judges: Sequence[str]) -> dict:
    """Global length-coefficient fit for the length-controlled win rate."""
    profs, lens = [], []
    for p in prompts:
        prof = store.profile(p.prompt_id, judges, cfg["criteria"], p.candidate_ids, FULL, "sym")
        profs.append(prof)
        lens.append([p.length(x) for x in prof.alternatives])
    scale_cfg = cfg.get("lc_win_rate", {}).get("length_scale", "auto")
    if scale_cfg == "auto":
        all_len = np.array([l for ls in lens for l in ls], dtype=float)
        scale = float(np.std(all_len)) if len(all_len) > 1 and np.std(all_len) > 0 else 1.0
    else:
        scale = float(scale_cfg)
    coef = fit_length_coefficient(profs, lens, length_scale=scale)
    return {"length_coef": coef, "length_scale": scale}


def _iia_probe_profiles(store: ScoreStore, prompt: PromptRecord, judges: Sequence[str], criteria: Sequence[str],
                        pipeline: str, variant: str) -> list[tuple[str, Profile, list[str]]]:
    order = "sym" if pipeline == "sym" else "fwd"
    cands = prompt.candidate_ids
    extras = [e["id"] for e in prompt.extra_candidates]
    out = []
    if variant == "remove_one":
        for z in cands:
            keep = [c for c in cands if c != z]
            key = subset_key_remove(z)
            out.append((key, store.profile(prompt.prompt_id, judges, criteria, cands, key, order), keep))
    elif variant == "add_one" and extras:
        key = subset_key_add(extras[0])
        out.append((key, store.profile(prompt.prompt_id, judges, criteria, cands, key, order), list(cands)))
    elif variant == "replace_one" and extras:
        for z in cands:
            keep = [c for c in cands if c != z]
            key = subset_key_replace(z, extras[0])
            out.append((key, store.profile(prompt.prompt_id, judges, criteria, cands, key, order), keep))
    return out


def audit_aggregator(bound: BoundAggregator, cfg: dict, store: ScoreStore, prompts: Sequence[PromptRecord],
                     judges: Sequence[str], criteria: Sequence[str] | None = None, variants: Sequence[str] | None = None,
                     judge_subset: Sequence[str] | None = None) -> dict:
    """Six-cell axiom profile of one aggregator over the audit prompts."""
    criteria = list(criteria or cfg["criteria"])
    variants = list(variants or cfg["probes"]["iia"].get("variants", ["remove_one"]))
    delta = float(cfg["probes"]["cm"].get("delta", 1.0))
    js = list(judge_subset or judges)
    order = "sym" if bound.pipeline == "sym" else "fwd"
    n_boot = int(cfg["stats"]["bootstrap_resamples"])
    seed = int(cfg["seed"])

    per_prompt: list[dict] = []
    iia_prompt_flags: dict[str, list[bool]] = {v: [] for v in variants}
    iia_probe_flags: dict[str, list[bool]] = {v: [] for v in variants}
    iia_eligible_total: dict[str, int] = {v: 0 for v in variants}
    iia_tie_involved: dict[str, int] = {v: 0 for v in variants}
    iia_strict_flip: dict[str, int] = {v: 0 for v in variants}
    iia_n_with_probe: dict[str, int] = {v: 0 for v in variants}
    strata: dict[str, dict[str, int]] = {}
    wp_pairs = wp_viol = wp_prompt_viol = 0
    cm_n = cm_dem = cm_prompt_viol = 0
    ud_fail = 0
    ud_cycles = 0
    pi_composite_viol = 0
    anon_viol = 0
    anon_checked = 0

    for prompt in prompts:
        f = bound.fn(prompt)
        cands = prompt.candidate_ids
        full = store.profile(prompt.prompt_id, js, criteria, cands, FULL, order)
        rec: dict[str, Any] = {"prompt_id": prompt.prompt_id, "benchmark": prompt.benchmark}
        st = strata.setdefault(prompt.benchmark, {"n": 0, "iia": 0, "wp": 0, "cm": 0, "ud": 0, "pi": 0})
        st["n"] += 1

        ud_bad = probes.ud_probe(f, full)
        ud_fail += ud_bad
        st["ud"] += ud_bad
        rec["ud_failure"] = bool(ud_bad)
        if bound.name == "majority_vote":
            cyc = probes.condorcet_cycle(full)
            ud_cycles += cyc is not None
            rec["condorcet_cycle"] = cyc

        wp = probes.wp_probe(f, full)
        wp_pairs += wp.n_unanimous_pairs
        wp_viol += wp.n_violations
        wp_prompt_viol += wp.violated
        st["wp"] += wp.violated
        rec["wp"] = {"unanimous_pairs": wp.n_unanimous_pairs, "violations": wp.n_violations}

        rec["iia"] = {}
        for v in variants:
            res = [probes.iia_probe(f, full, probe_prof, comp, None, key)
                   for key, probe_prof, comp in _iia_probe_profiles(store, prompt, js, criteria, bound.pipeline, v)]
            if not res:
                continue
            iia_n_with_probe[v] += 1
            violated = probes.prompt_level_iia(res)
            iia_prompt_flags[v].append(violated)
            for r in res:
                iia_probe_flags[v].append(r.violated)
                iia_eligible_total[v] += len(r.eligible_pairs)
                iia_tie_involved[v] += r.n_tie_involved
                iia_strict_flip[v] += r.n_strict_flip
            rec["iia"][v] = {"prompt_violated": violated, "probes": [r.as_dict() for r in res]}
            if v == "remove_one":
                st["iia"] += violated
            else:
                st[f"iia_{v}"] = st.get(f"iia_{v}", 0) + violated

        cm = probes.cm_probe(f, full, delta)
        cm_n += cm.n_perturbations
        cm_dem += cm.n_demotions
        cm_prompt_viol += cm.violated
        st["cm"] += cm.violated
        rec["cm"] = {"perturbations": cm.n_perturbations, "demotions": cm.n_demotions}

        if cfg["probes"]["anonymity"].get("empirical", False):
            av = probes.anonymity_probe(f, full)
            anon_viol += av
            anon_checked += 1
            rec["anonymity_violation"] = bool(av)

        fwd_prof = store.profile(prompt.prompt_id, js, criteria, cands, FULL, "fwd")
        rev_prof = store.profile(prompt.prompt_id, js, criteria, cands, FULL, "rev")
        if bound.pipeline == "sym":
            changed = probes.pi_composite(f, symmetrize(fwd_prof, rev_prof), symmetrize(rev_prof, fwd_prof))
        else:
            changed = probes.pi_composite(f, fwd_prof, rev_prof)
        pi_composite_viol += changed
        st["pi"] += changed
        rec["pi_composite_changed"] = bool(changed)
        per_prompt.append(rec)

    n = len(prompts)
    out: dict[str, Any] = {
        "name": bound.name, "pipeline": bound.pipeline, "params": bound.params, "criteria": criteria,
        "n_prompts": n, "judges": js,
        "ud": {"structural": bound.spec.ud_structural, "intransitive_outputs": ud_fail, "condorcet_cycles": ud_cycles,
               "rate": ud_fail / n if n else float("nan")},
        "wp": {"unanimous_pairs": wp_pairs, "pair_violations": wp_viol, **rate_summary(wp_prompt_viol, n)},
        "anonymity": {"analytic": bound.spec.anonymity_analytic,
                      "empirical_violation_rate": anon_viol / anon_checked if anon_checked else None},
        "cm": {"perturbations": cm_n, "demotions": cm_dem, **rate_summary(cm_prompt_viol, n)},
        "pi": {"secured_by_pi_star": bound.pipeline == "sym", "cell_type": bound.pi_cell, **rate_summary(pi_composite_viol, n)},
        "iia": {"analytic": bound.spec.iia_analytic},
        "strata": strata,
        "per_prompt": per_prompt,
    }
    for v in variants:
        flags = iia_prompt_flags[v]
        pf = iia_probe_flags[v]
        summary = rate_summary(int(sum(flags)), len(flags))
        if flags:
            lo, hi = bootstrap_ci([float(x) for x in flags], n_resamples=n_boot, rng=seed)
            summary["bootstrap_ci"] = [lo, hi]
        out["iia"][v] = {"prompt_level": summary, "probe_level": rate_summary(int(sum(pf)), len(pf)),
                         "eligible_pairs_total": iia_eligible_total[v],
                         "changed_tie_involved": iia_tie_involved[v], "changed_strict_flip": iia_strict_flip[v],
                         "n_prompts_with_probe": iia_n_with_probe[v],
                         "prompt_flags": [bool(x) for x in flags]}
    return out


def pi_verdict_rates(store: ScoreStore, prompts: Sequence[PromptRecord], judges: Sequence[str], criteria: Sequence[str]) -> dict:
    """Judge-level verdict-flip rates over (prompt, pair) probes, by (judge, criterion) and by (judge, benchmark)."""
    by_jc: dict[str, dict[str, dict]] = {}
    by_jb: dict[str, dict[str, dict]] = {}
    has_overall = any("overall" in store.pairwise.get(p.prompt_id, {}).get(j, {}) for p in prompts for j in judges)
    columns = list(criteria) + (["overall"] if has_overall else [])
    for j in judges:
        by_jc[j] = {}
        by_jb[j] = {}
        for c in columns:
            num = den = 0
            per_b: dict[str, list[int]] = {}
            for p in prompts:
                pairs = store.pairwise.get(p.prompt_id, {}).get(j, {}).get(c, {})
                flips = [probes.verdict_flip(v) for v in pairs.values()]
                valid = [x for x in flips if x is not None]
                if not valid:
                    continue
                num += int(sum(valid))
                den += len(valid)
                per_b.setdefault(p.benchmark, []).extend(int(x) for x in valid)
            by_jc[j][c] = rate_summary(num, den) if den else {"rate": None, "n": 0}
            if c == "overall":
                continue
            for b, vals in per_b.items():
                cell = by_jb[j].setdefault(b, {"num": 0, "den": 0})
                cell["num"] += int(sum(vals))
                cell["den"] += len(vals)
        for b, cell in by_jb[j].items():
            by_jb[j][b] = rate_summary(cell["num"], cell["den"])
    cells = [by_jc[j][c]["rate"] for j in judges for c in criteria if by_jc[j][c].get("n")]
    return {"by_judge_criterion": by_jc, "by_judge_benchmark": by_jb,
            "mean_over_cells": float(np.mean(cells)) if cells else None,
            "min_cell": float(np.min(cells)) if cells else None, "max_cell": float(np.max(cells)) if cells else None}


def run_audit(cfg: dict, log=print) -> dict:
    paths = cfg["paths"]
    prompts = load_prompts(paths["prompts"])
    store = open_store(cfg, paths["scores"], prompts)
    judges = [j["name"] for j in cfg["judges"]]
    criteria = list(cfg["criteria"])
    n = len(prompts)
    log(f"audit: {n} prompts, {len(judges)} judges, {len(criteria)} criteria, aggregators={cfg['aggregators']}")

    extra_groups = [list(cfg.get(k, [])) for k in ("ablation_aggregators", "leaderboard_aggregators", "witness_aggregators")]
    all_names = list(dict.fromkeys(list(cfg["aggregators"]) + [a for g in extra_groups for a in g]))
    lc_state = fit_lc_state(cfg, prompts, store, judges) if "lc_win_rate" in all_names else None

    results: dict[str, Any] = {
        "config": cfg.get("name"), "config_path": cfg.get("_config_path"), "seed": cfg["seed"], "n_prompts": n,
        "strata": {b: sum(1 for p in prompts if p.benchmark == b) for b in sorted({p.benchmark for p in prompts})},
        "judges": judges, "criteria": criteria,
        "iia_variants": list(cfg["probes"]["iia"].get("variants", ["remove_one"])),
        "judge_stats": store.meta.get("judge_stats", {}),
        "aggregators": {}, "ablation": {}, "leaderboard_map": {}, "witnesses": list(cfg.get("witness_aggregators", [])),
    }

    bound_all = {name: bind_aggregator(name, cfg, prompts, judges, lc_state) for name in all_names}
    variants = list(cfg["probes"]["iia"].get("variants", ["remove_one"]))
    for name in all_names:
        log(f"  auditing {name} (pipeline={bound_all[name].pipeline})")
        results["aggregators"][name] = audit_aggregator(bound_all[name], cfg, store, prompts, judges)
        for v in variants:
            iia_v = results["aggregators"][name]["iia"].get(v, {})
            nwp = iia_v.get("n_prompts_with_probe")
            if nwp is not None and nwp < n:
                log(f"    warning: {name}/{v}: {n - nwp}/{n} prompts excluded (no held-out candidate)")

    pi = pi_verdict_rates(store, prompts, judges, criteria)
    results["pi_verdicts"] = pi
    for name, res in results["aggregators"].items():
        res["pi"]["judge_level_mean_flip_rate"] = pi["mean_over_cells"]
        res["pi"]["judge_level_mean_flip_se"] = rate_se(pi["mean_over_cells"], n) if pi["mean_over_cells"] is not None else None

    ref = cfg["stats"]["mcnemar_reference"]
    comparisons = [a for a in cfg["aggregators"] if a != ref]
    results["bonferroni"] = {"alpha": cfg["stats"]["bonferroni_alpha"], "n_comparisons": len(comparisons),
                             "threshold": bonferroni_threshold(cfg["stats"]["bonferroni_alpha"], max(1, len(comparisons)))}
    results["mcnemar"] = {}
    if ref in results["aggregators"]:
        ref_flags = results["aggregators"][ref]["iia"].get("remove_one", {}).get("prompt_flags")
        for other in comparisons:
            oth_flags = results["aggregators"][other]["iia"].get("remove_one", {}).get("prompt_flags")
            if ref_flags is None or oth_flags is None:
                continue
            mc = mcnemar(oth_flags, ref_flags)
            d = mc.as_dict()
            d["effect_size_points"] = 100.0 * (float(np.mean(oth_flags)) - float(np.mean(ref_flags)))
            d["bonferroni_significant_chi2"] = mc.p_chi2 < results["bonferroni"]["threshold"]
            d["bonferroni_significant_exact"] = mc.p_exact < results["bonferroni"]["threshold"]
            results["mcnemar"][f"{other}_vs_{ref}"] = d

    for name in cfg.get("ablation_aggregators", []):
        r = results["aggregators"][name]
        results["ablation"][name] = {"pi": r["pi"]["rate"] if r["pipeline"] != "sym" else 0.0,
                                     "pi_judge_level": r["pi"].get("judge_level_mean_flip_rate"),
                                     "pi_cell_type": r["pi"].get("cell_type", "composite"),
                                     "iia": r["iia"].get("remove_one", {}).get("prompt_level", {}).get("rate")}
    lb_names = {"mtbench_avg": "MT-Bench (cross-criterion mean)", "arena_bt": "Arena-Hard (Bradley-Terry)",
                "lc_win_rate": "AlpacaEval 2 (LC win rate)", "win_rate": "AlpacaEval 2 (win rate, no LC)"}
    for name in cfg.get("leaderboard_aggregators", []):
        r = results["aggregators"][name]
        results["leaderboard_map"][lb_names.get(name, name)] = r["iia"].get("remove_one", {}).get("prompt_level", {}).get("rate")
    if lc_state:
        results["lc_state"] = lc_state

    if cfg.get("per_judge_counts", True):
        results["per_judge_counts"] = per_judge_counts(cfg, store, prompts, judges, bound_all, all_names, log)

    variants = cfg["probes"]["iia"].get("variants", ["remove_one"])
    results["multi_probe"] = {name: {v: results["aggregators"][name]["iia"].get(v, {}).get("prompt_level", {}).get("rate")
                                     for v in variants} for name in cfg["aggregators"]}
    return results


def per_judge_counts(cfg: dict, store: ScoreStore, prompts: Sequence[PromptRecord], judges: Sequence[str],
                     bound_all: dict[str, BoundAggregator], names: Sequence[str], log=print) -> dict:
    """Integer counts of every probe per (aggregator, judge, benchmark), each judge audited on its own."""
    out: dict = {}
    for name in names:
        out[name] = {}
        for j in judges:
            try:
                res = audit_aggregator(bound_all[name], cfg, store, prompts, judges, judge_subset=[j])
            except (ValueError, KeyError) as e:
                out[name][j] = {"skipped": str(e)}
                continue
            out[name][j] = {b: {"iia_count": st["iia"], **{k: v for k, v in st.items() if k != "iia"}} for b, st in res["strata"].items()}
            total = {"iia_count": res["iia"]["remove_one"]["prompt_level"]["count"], "n": res["n_prompts"]}
            for v, r in res["iia"].items():
                if v != "remove_one" and isinstance(r, dict):
                    total[f"iia_{v}_count"] = r["prompt_level"]["count"]
            for k in ("wp", "cm", "pi"):
                total[f"{k}_count"] = res[k]["count"]
            total["ud_count"] = res["ud"]["intransitive_outputs"]
            out[name][j]["total"] = total
        if log:
            log(f"  per-judge counts: {name}")
    return out


def audit_pair_certificates(cfg: dict, calibration, judges: Sequence[str]) -> dict | None:
    """Per-pair coverage certificates of the audit prompts under the fitted calibration."""
    paths = cfg["paths"]
    if not (Path(paths["prompts"]).exists() and Path(paths["scores"]).exists()):
        return None
    prompts = load_prompts(paths["prompts"])
    store = open_store(cfg, paths["scores"], prompts)
    n_pairs = n_large = 0
    per_pair: list[dict] = []
    for p in prompts:
        try:
            prof = store.profile(p.prompt_id, judges, cfg["criteria"], p.candidate_ids, FULL, "sym")
        except KeyError:
            return None
        for x, y in combinations(prof.alternatives, 2):
            cert = certify_pair(prof, x, y, calibration, cfg.get("criterion_weights"))
            n_pairs += 1
            n_large += cert["large_gap"]
            per_pair.append({"prompt_id": p.prompt_id, **cert})
    return {"n_pairs": n_pairs, "n_large_gap": n_large, "large_gap_fraction": n_large / n_pairs if n_pairs else float("nan"),
            "swap_bound_large_gap": 4.0 * calibration.alpha, "swap_bound_small_gap": 1.0, "per_pair": per_pair}


def run_calibration(cfg: dict, log=print) -> dict:
    paths = cfg["paths"]
    ann = load_annotations(paths["calibration_annotations"])
    cal_cfg = cfg.get("calibration", {})
    thr = float(cal_cfg.get("iaa_threshold", 0.7))
    cal = build_calibration_set(ann, thr)
    prompts = {p.prompt_id: p for p in load_prompts(paths["calibration_prompts"])}
    store = open_store(cfg, paths["calibration_scores"], list(prompts.values()))
    judges = [j["name"] for j in cfg["judges"]]
    profiles = []
    mu = []
    for a, m in zip(cal.retained, cal.mu_agg()):
        p = prompts[a.prompt_id]
        profiles.append(store.profile(p.prompt_id, judges, cfg["criteria"], p.candidate_ids, FULL, "sym"))
        mu.append(m)
    alpha = float(cfg["bch"].get("alpha", 0.05))
    res = calibrate(profiles, mu, alpha=alpha, criterion_weights=cfg.get("criterion_weights"), rng=int(cfg["seed"]))
    log(f"calibration: n_cal={res.n_cal} q_hat={res.q_hat:.4g} p={res.small_gap_fraction:.4f} bound={res.bound:.4f}")
    return {
        "calibration_set": cal.summary(),
        "threshold_sensitivity": threshold_sensitivity(ann, tuple(cal_cfg.get("thresholds", (0.5, 0.6, 0.7, 0.8)))),
        "conformal": res.as_dict(),
        "audit_pair_certificates": audit_pair_certificates(cfg, res, judges),
    }


def run_extras(cfg: dict, log=print) -> dict:
    """Self-enhancement audit, when configured."""
    ex = cfg.get("extras", {})
    if not ex:
        return {}
    prompts = load_prompts(cfg["paths"]["prompts"])
    store = open_store(cfg, cfg["paths"]["scores"], prompts)
    judges = [j["name"] for j in cfg["judges"]]
    out: dict = {}
    se = ex.get("self_enhancement")
    if se:
        out["self_enhancement"] = self_enhancement(store, prompts, judges, cfg["criteria"], se["judge"], se["candidate"],
                                                   se.get("n_matched_pairs"))
        r = out["self_enhancement"]
        log(f"self-enhancement: {se['judge']} on {se['candidate']}: +{r['uplift_raw']:.2f} raw, +{r['uplift_length_adjusted']:.2f} "
            f"length-adjusted (n={r['n_pairs']}, p_raw={r['ttest_raw']['p']:.3g})")
    return out


def run_jury(cfg: dict, log=print) -> dict:
    ann = load_annotations(cfg["paths"]["jury_annotations"])
    res = jury_iia(ann, n_bootstrap=int(cfg["stats"]["bootstrap_resamples"]), seed=int(cfg["seed"]))
    log(f"jury: {res['probe_level']['count']}/{res['probe_level']['n']} probes violated "
        f"({100 * res['probe_level']['rate']:.1f}%), alpha_K={res['krippendorff_alpha_ordinal']:.3f}")
    return res


ROW_LABELS = {"majority_vote": "Majority Vote", "score_mean": "Score-Mean", "borda": "Borda", "arena_bt": "Arena-BT",
              "mtbench_avg": "MT-Bench Avg", "best_of_judge": "Best-of-Judge", "perm_consensus": "Perm-Consensus",
              "bch": "BCH", "bch_no_sym": "F_BCH without order symmetrization", "median_score_mean": "Score-Mean, median-over-judges",
              "win_rate": "Win rate", "lc_win_rate": "LC win rate", "constant": "Constant rule",
              "dictatorship": "Dictatorship", "weighted_utilitarian": "Weighted utilitarian"}


def _pct(x: float | None, se: float | None = None) -> str:
    if x is None or x != x:
        return "n/a"
    return f"{100 * x:.1f}" + (f" ± {100 * se:.1f}" if se is not None and se == se else "")


def axiom_matrix_markdown(results: dict, names: Sequence[str] | None = None) -> str:
    names = list(names or [n for n in agg_mod.AUDIT_ORDER if n in results["aggregators"]])
    lines = ["| Agg. | UD | WP | IIA (%) | A | CM | PI (%) |", "|---|---|---|---|---|---|---|"]
    for name in names:
        r = results["aggregators"][name]
        if r["ud"]["structural"]:
            ud = "✓" if r["ud"]["intransitive_outputs"] == 0 else f"× ({r['ud']['intransitive_outputs']} intransitive)"
        else:
            ud = f"×(†) {r['ud']['condorcet_cycles']} cycles, {r['ud']['intransitive_outputs']} intransitive"
        wp = "✓" if r["wp"]["count"] == 0 else _pct(r["wp"]["rate"], r["wp"]["se"])
        iia_r = r["iia"].get("remove_one", {}).get("prompt_level")
        iia = "✓ (analytic)" if r["iia"]["analytic"] else (_pct(iia_r["rate"], iia_r["se"]) if iia_r else "n/a")
        if r["iia"]["analytic"] and iia_r and iia_r["count"]:
            iia += f" [probe: {_pct(iia_r['rate'])}]"
        n_judges = len(r.get("judges", results.get("judges", [])))
        a = "✓" if r["anonymity"]["analytic"] or n_judges <= 1 else "100.0 (by construction)"
        if r["anonymity"].get("empirical_violation_rate") not in (None, 0.0):
            a += f" [empirical {_pct(r['anonymity']['empirical_violation_rate'])}]"
        cm = "✓" if r["cm"]["count"] == 0 else _pct(r["cm"]["rate"], r["cm"]["se"])
        if r["pi"]["secured_by_pi_star"]:
            pi = "✓*"
        elif r["pi"].get("cell_type") == "judge_level":
            pi = _pct(r["pi"].get("judge_level_mean_flip_rate"), r["pi"].get("judge_level_mean_flip_se")) + " (judge-level)"
        else:
            pi = _pct(r["pi"]["rate"], r["pi"]["se"]) + " (composite)"
        lines.append(f"| {ROW_LABELS.get(name, name)} | {ud} | {wp} | {iia} | {a} | {cm} | {pi} |")
    return "\n".join(lines)


def report_markdown(results: dict, calibration: dict | None = None, jury: dict | None = None, extras: dict | None = None) -> str:
    out = [f"# Axiom audit report ({results.get('config')})", "",
           f"Prompts: {results['n_prompts']} ({results['strata']}); judges: {results['judges']}; criteria: {results['criteria']}",
           f"IIA probe variants: {results['iia_variants']}", "",
           "## Aggregator-by-axiom matrix", "", axiom_matrix_markdown(results), "",
           "✓ = deterministic satisfaction (0 violations observed on the audited profiles, or analytic); ✓* = PI secured by order symmetrization; "
           "numeric = empirical violation rate ± SE (SE = sqrt(p(1-p)/n)). "
           "PI cells: (judge-level) = mean of per-(judge, criterion) pairwise verdict-flip rates; "
           "(composite) = ranking-change rate Pr[G(p,π) ≠ G(p,π')]; the two are not directly comparable.", ""]
    elig_parts = []
    for name in (n for n in agg_mod.AUDIT_ORDER if n in results["aggregators"]):
        iia_ro = results["aggregators"][name]["iia"].get("remove_one", {})
        ep = iia_ro.get("eligible_pairs_total")
        ti = iia_ro.get("changed_tie_involved", 0)
        sf = iia_ro.get("changed_strict_flip", 0)
        if ep is not None:
            elig_parts.append(f"{ROW_LABELS.get(name, name)} {ep} ({ti} tie-involved, {sf} strict-flip changes)")
    if elig_parts:
        out += [f"IIA eligible pairs (remove-one): {'; '.join(elig_parts)}.", ""]
    if results.get("witnesses"):
        out += ["## Frontier witnesses", "", axiom_matrix_markdown(results, results["witnesses"]), ""]
    if results.get("mcnemar"):
        out += ["## McNemar tests vs BCH (prompt-level pairing)", "",
                f"Bonferroni threshold: p < {results['bonferroni']['threshold']:.5f} ({results['bonferroni']['n_comparisons']} comparisons)", "",
                "| comparison | both | b (other only) | c (BCH only) | neither | χ² | p (χ²) | p (exact) | effect (pts) | sig. (χ²) |",
                "|---|---|---|---|---|---|---|---|---|---|"]
        for k, d in results["mcnemar"].items():
            t = d["contingency"]
            out.append(f"| {k} | {t['both']} | {d['b']} | {d['c']} | {t['neither']} | {d['chi2']:.3f} | {d['p_chi2']:.4f} | {d['p_exact']:.4f} | "
                       f"{d['effect_size_points']:+.1f} | {d['bonferroni_significant_chi2']} |")
        out.append("")
    if results.get("ablation"):
        out += ["## BCH ablation", "", "| Variant | PI (%) | IIA (%) |", "|---|---|---|"]
        for name, d in results["ablation"].items():
            pi_v = d["pi_judge_level"] if d.get("pi_cell_type") == "judge_level" else d["pi"]
            out.append(f"| {ROW_LABELS.get(name, name)} | {_pct(pi_v)} | {_pct(d['iia'])} |")
        out.append("")
    if results.get("leaderboard_map"):
        out += ["## Leaderboard aggregators", "", "| Leaderboard (aggregator) | IIA (%) |", "|---|---|"]
        for k, v in results["leaderboard_map"].items():
            out.append(f"| {k} | {_pct(v)} |")
        out.append("")
    pv = results.get("pi_verdicts", {})
    if pv.get("by_judge_criterion"):
        crits = results["criteria"]
        out += ["## PI verdict-flip rate (%) by judge and criterion", "", "| Judge | " + " | ".join(crits) + " |", "|---|" + "---|" * len(crits)]
        for j, row in pv["by_judge_criterion"].items():
            out.append(f"| {j} | " + " | ".join(_pct(row[c].get("rate")) for c in crits) + " |")
        out += ["", f"Mean over cells: {_pct(pv.get('mean_over_cells'))}", ""]
    mp = results.get("multi_probe", {})
    if mp and any(len(v) > 1 for v in mp.values()):
        variants = list(next(iter(mp.values())).keys())
        out += ["## Multi-probe IIA (%)", "", "| Aggregator | " + " | ".join(variants) + " |", "|---|" + "---|" * len(variants)]
        for name, d in mp.items():
            out.append(f"| {ROW_LABELS.get(name, name)} | " + " | ".join(_pct(d[v]) for v in variants) + " |")
        out.append("")
    if calibration:
        c = calibration["conformal"]
        out += ["## BCH conformal certificate", "",
                f"n_cal = {c['n_cal']} (threshold sensitivity: {calibration['threshold_sensitivity']}); α = {c['alpha']}; "
                f"q̂_α = {c['q_hat']:.4g}; 4q̂ = {c['gap_threshold_4q']:.4g}; small-gap fraction p = {c['small_gap_fraction_p']:.4f}; "
                f"bound 4α + p = {c['bound_4alpha_plus_p']:.4f}; empirical coverage = {c['empirical_coverage']:.4f}", ""]
        cert = calibration.get("audit_pair_certificates")
        if cert:
            out += [f"Audit pairs with gap > 4q̂ (swap bound 4α): {cert['n_large_gap']}/{cert['n_pairs']} = {_pct(cert['large_gap_fraction'])}%", ""]
    if jury:
        pl = jury["probe_level"]
        out += ["## Human-jury comparator", "",
                f"{pl['count']}/{pl['n']} probes violated = {_pct(pl['rate'])} (bootstrap CI clustered by prompt: "
                f"[{_pct(pl['bootstrap_ci_clustered'][0])}, {_pct(pl['bootstrap_ci_clustered'][1])}]); prompt-level observed "
                f"{_pct(jury['prompt_level']['observed'])} (possible range [{_pct(jury['prompt_level']['min_possible'])}, "
                f"{_pct(jury['prompt_level']['max_possible'])}]); Krippendorff α = {jury['krippendorff_alpha_ordinal']:.3f} "
                f"[{jury['krippendorff_alpha_ci95_bootstrap'][0]:.2f}, {jury['krippendorff_alpha_ci95_bootstrap'][1]:.2f}]", ""]
    if extras:
        se = extras.get("self_enhancement")
        if se:
            out += ["## Self-enhancement audit", "",
                    f"{se['judge']} judging {se['candidate']}: uplift {se['uplift_raw']:+.2f} points raw "
                    f"(paired t, p = {se['ttest_raw']['p']:.3g}), {se['uplift_length_adjusted']:+.2f} length-adjusted "
                    f"(p = {se['ttest_length_adjusted']['p']:.3g}); n = {se['n_pairs']} matched pairs; per-criterion: "
                    + ", ".join(f"{c} {v:+.2f}" for c, v in se["per_criterion_uplift"].items()) + "; self-judging uplift by judge: "
                    + (", ".join(f"{j} {d['uplift']:+.2f}" for j, d in se["self_judging_uplift_by_judge"].items()) or "n/a"), ""]
    return "\n".join(out)


def strip_per_prompt(results: dict) -> dict:
    """Copy without per-prompt detail (written to a separate file)."""
    slim = json.loads(json.dumps(results, default=float))
    for r in slim.get("aggregators", {}).values():
        r.pop("per_prompt", None)
    return slim


def write_results(results_dir: str | Path, results: dict, calibration: dict | None = None, jury: dict | None = None,
                  extras: dict | None = None) -> dict[str, Path]:
    d = Path(results_dir)
    d.mkdir(parents=True, exist_ok=True)
    files = {}
    files["results"] = d / "results.json"
    files["results"].write_text(json.dumps(strip_per_prompt(results), indent=1, default=float), encoding="utf-8")
    files["per_prompt"] = d / "per_prompt.json"
    files["per_prompt"].write_text(json.dumps({n: r["per_prompt"] for n, r in results["aggregators"].items()}, default=float), encoding="utf-8")
    counts = {"strata": {n: r["strata"] for n, r in results["aggregators"].items()},
              "per_judge": results.get("per_judge_counts", {}), "mcnemar": results.get("mcnemar", {}),
              "bonferroni": results.get("bonferroni", {})}
    files["counts"] = d / "counts.json"
    files["counts"].write_text(json.dumps(counts, indent=1, default=float), encoding="utf-8")
    for name, res in (("calibration", calibration), ("jury", jury), ("extras", extras or None)):
        path = d / f"{name}.json"
        if res is None:
            path.unlink(missing_ok=True)
            continue
        files[name] = path
        path.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    files["report"] = d / "report.md"
    files["report"].write_text(report_markdown(results, calibration, jury, extras), encoding="utf-8")
    return files
