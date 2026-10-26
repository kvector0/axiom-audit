"""Command-line entry point: data preparation, judge scoring, calibration, jury comparator, extras and the axiom audit."""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from audit import run as audit_mod
from audit.human import save_annotations
from audit.run import (build_judges, candidate_specs, load_config, run_audit, run_calibration, run_extras, run_jury,
                       synthetic_world, write_results)
from judging.benchmarks import (download_benchmarks, generate_candidates, load_pool, partition_prompts, synthetic_annotations,
                                synthetic_prompts)
from judging.judges import PromptRecord, ScoreStore, load_prompts, run_pairwise, run_scoring, save_prompts


def _log(msg: str) -> None:
    print(msg, flush=True)


def _allocation(cfg: dict) -> dict:
    return {b: {k: int(v) for k, v in d.items()} for b, d in cfg["benchmarks"].items()}


def cmd_prepare_data(cfg: dict, args) -> None:
    raw = cfg["paths"].get("raw_dir", "data/raw")
    download_benchmarks(raw, list(cfg["benchmarks"]), log=_log)
    pools = {b: load_pool(b, raw) for b in cfg["benchmarks"]}
    for b, pool in pools.items():
        _log(f"{b}: pool size {len(pool)}")
    parts = partition_prompts(pools, _allocation(cfg), int(cfg["seed"]))
    splits = [("audit", "prompts"), ("calibration", "calibration_prompts"), ("jury", "jury_prompts")]
    for split, key in splits:
        path = Path(cfg["paths"][key])
        if path.exists():
            kept = {r.prompt_id for r in load_prompts(path)}
            expected = {p["prompt_id"] for p in parts[split]}
            if kept != expected:
                raise SystemExit(f"preserved {split} split does not match the partition this config produces; "
                                 f"delete the existing prompt files to repartition")
    for split, key in splits:
        path = Path(cfg["paths"][key])
        if path.exists():
            _log(f"kept existing {split} prompts -> {path}")
            continue
        recs = [PromptRecord(p["prompt_id"], p["benchmark"], p["text"], []) for p in parts[split]]
        save_prompts(path, recs)
        _log(f"wrote {len(recs)} {split} prompts ({len({r.prompt_id for r in recs})} distinct; no candidates yet) -> {path}")


def cmd_generate_candidates(cfg: dict, args) -> None:
    key = {"audit": "prompts", "calibration": "calibration_prompts", "jury": "jury_prompts"}[args.split]
    path = Path(cfg["paths"][key])
    prompts = [p.to_dict() for p in load_prompts(path)]
    world = synthetic_world(cfg)
    gens = candidate_specs(cfg, "candidates")
    extras = candidate_specs(cfg, "extra_candidates") if args.split == "audit" else []
    recs = generate_candidates(prompts, gens, world, extras, cfg.get("synthetic", {}).get("verbosity"), log=_log)
    save_prompts(path, recs)
    _log(f"wrote {len(recs)} prompts with {len(gens)} candidates (+{len(extras)} extra) -> {path}")


def cmd_synth(cfg: dict, args) -> None:
    syn = cfg.get("synthetic", {})
    if not syn.get("enabled"):
        raise SystemExit("synth requires synthetic.enabled: true in the config")
    world = synthetic_world(cfg)
    alloc = _allocation(cfg)
    pools = synthetic_prompts({b: sum(alloc[b].values()) + 5 for b in alloc}, int(cfg["seed"]))
    parts = partition_prompts(pools, alloc, int(cfg["seed"]))
    gens = candidate_specs(cfg, "candidates")
    extras = candidate_specs(cfg, "extra_candidates")
    n_ann = int(syn.get("n_annotators", 2))
    for split, key in (("audit", "prompts"), ("calibration", "calibration_prompts"), ("jury", "jury_prompts")):
        recs = generate_candidates(parts[split], gens, world, extras if split == "audit" else [], syn.get("verbosity"), log=None)
        save_prompts(cfg["paths"][key], recs)
        _log(f"synth: {len(recs)} {split} prompts -> {cfg['paths'][key]}")
        if split == "calibration":
            ann = synthetic_annotations(recs, world, n_ann, float(syn.get("annotator_noise_sd", 0.6)), tag="cal")
            save_annotations(cfg["paths"]["calibration_annotations"], ann)
        if split == "jury":
            ann = synthetic_annotations(recs, world, n_ann, float(syn.get("jury_noise_sd", 0.5)), tag="jury")
            save_annotations(cfg["paths"]["jury_annotations"], ann)


def cmd_score(cfg: dict, args) -> None:
    try:
        _score(cfg, args)
    except ValueError as e:
        raise SystemExit(str(e))


def _score(cfg: dict, args) -> None:
    judges = build_judges(cfg)
    if args.split == "audit":
        prompts = load_prompts(cfg["paths"]["prompts"])
        store = ScoreStore(cfg["paths"]["scores"])
        variants = cfg["probes"]["iia"].get("variants", ["remove_one"])
        n = run_scoring(store, judges, prompts, cfg["criteria"], variants, cfg["scoring"]["orders"], log=_log)
        _log(f"scoring: {n} new queries")
        pi = cfg["probes"]["pi"]
        crits = list(cfg["criteria"]) if pi.get("per_criterion", True) else []
        if pi.get("include_overall", False) or not crits:
            crits.append(None)
        n = run_pairwise(store, judges, prompts, crits, log=_log)
        _log(f"pairwise: {n} new pairs")
    elif args.split == "calibration":
        prompts = load_prompts(cfg["paths"]["calibration_prompts"])
        store = ScoreStore(cfg["paths"]["calibration_scores"])
        n = run_scoring(store, judges, prompts, cfg["criteria"], [], cfg["scoring"]["orders"], log=_log)
        _log(f"calibration scoring: {n} new queries")
    else:
        raise SystemExit("--split must be audit or calibration")
    store.record_judge_stats(judges)
    store.save()
    _log("judge queries: " + "; ".join(f"{j.name} {j.stats()}" for j in judges))


def _optional(fn, cfg, name):
    try:
        return fn(cfg, log=_log)
    except FileNotFoundError as e:
        _log(f"{name}: skipped ({e.filename} not found)")
        return None
    except KeyError as e:
        _log(f"{name}: skipped (no stored entry for {e}; check that the score step and the ranking files cover the same prompts)")
        return None


def _required(fn, cfg, name):
    try:
        return fn(cfg, log=_log)
    except FileNotFoundError as e:
        raise SystemExit(f"{name}: {e.filename} not found")
    except KeyError as e:
        raise SystemExit(f"{name}: no stored entry for {e}; check that the score step and the ranking files cover the same prompts")


def _write_json(cfg: dict, name: str, res: dict) -> None:
    out = Path(cfg["paths"]["results_dir"]) / name
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, indent=1, default=float), encoding="utf-8")
    _log(f"wrote {out}")


def cmd_calibrate(cfg: dict, args) -> None:
    _write_json(cfg, "calibration.json", _required(run_calibration, cfg, "calibrate"))


def cmd_jury(cfg: dict, args) -> None:
    _write_json(cfg, "jury.json", _required(run_jury, cfg, "jury"))


def cmd_extras(cfg: dict, args) -> None:
    _write_json(cfg, "extras.json", run_extras(cfg, log=_log))


def cmd_audit(cfg: dict, args) -> None:
    results = run_audit(cfg, log=_log)
    cal = _optional(run_calibration, cfg, "calibration")
    jury = _optional(run_jury, cfg, "jury")
    extras = run_extras(cfg, log=_log) if cfg.get("extras") else None
    files = write_results(cfg["paths"]["results_dir"], results, cal, jury, extras)
    _log("wrote " + ", ".join(str(p) for p in files.values()))
    print(f"\naggregator-by-axiom matrix ({cfg.get('name', 'audit')}, {results['n_prompts']} prompts)\n")
    print(audit_mod.axiom_matrix_markdown(results))


def cmd_all(cfg: dict, args) -> None:
    cmd_synth(cfg, args)
    cmd_score(cfg, argparse.Namespace(split="audit"))
    cmd_score(cfg, argparse.Namespace(split="calibration"))
    cmd_audit(cfg, argparse.Namespace())


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(prog="main.py", description=__doc__)
    sub = ap.add_subparsers(dest="cmd", required=True)
    for name in ("prepare-data", "synth", "calibrate", "jury", "extras", "all"):
        p = sub.add_parser(name)
        p.add_argument("--config", required=True)
    p = sub.add_parser("generate-candidates")
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="audit", choices=["audit", "calibration", "jury"])
    p = sub.add_parser("score")
    p.add_argument("--config", required=True)
    p.add_argument("--split", default="audit", choices=["audit", "calibration"])
    p = sub.add_parser("audit")
    p.add_argument("--config", required=True)
    args = ap.parse_args(argv)
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    cfg = load_config(args.config)
    {"prepare-data": cmd_prepare_data, "generate-candidates": cmd_generate_candidates, "synth": cmd_synth, "score": cmd_score,
     "calibrate": cmd_calibrate, "jury": cmd_jury, "extras": cmd_extras, "audit": cmd_audit, "all": cmd_all}[args.cmd](cfg, args)
    return 0


if __name__ == "__main__":
    sys.exit(main())
