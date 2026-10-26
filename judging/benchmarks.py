"""Benchmark download, seeded prompt partition, candidate generation and the synthetic world for the smoke configuration."""
from __future__ import annotations

import hashlib
import json
import os
import zlib
from pathlib import Path
from typing import Sequence

import numpy as np

from audit.human import Annotation
from judging.judges import Judge, JudgeSpec, PromptRecord, SyntheticWorld, make_judge

BENCHMARK_SOURCES = {
    "mt_bench": {
        "url": "https://raw.githubusercontent.com/lm-sys/FastChat/b494d0c6b4e7935f1764f8439e75da3e66beccc7/fastchat/llm_judge/data/mt_bench/question.jsonl",
        "file": "mt_bench_question.jsonl",
        "n_expected": 80,
        "sha256": "119565adbab82227089cefdb44c8d7e2cf04dc0a0ec233634c82e7d4e2a944f7",
    },
    "alpaca_eval_2": {
        "url": "https://huggingface.co/datasets/tatsu-lab/alpaca_eval/resolve/2edc6fad8be6b14ea7230aabfd08188da6b8b814/alpaca_eval.json",
        "file": "alpaca_eval.json",
        "n_expected": 805,
        "sha256": "d92b92c51e8f1962a21193abe74e6f727c2bc8286035f4041505ff38a7c3ae51",
    },
    "arena_hard": {
        "url": "https://raw.githubusercontent.com/lmarena/arena-hard-auto/900183d0e8783b180ddb34e3fa952463f12e134f/data/arena-hard-v0.1/question.jsonl",
        "file": "arena_hard_question.jsonl",
        "n_expected": 500,
        "sha256": "2650242d2883f22b2abd498f273b6a7e02253cce533f5c91a858e1d44aa3002f",
    },
}


def download_benchmarks(raw_dir: str | os.PathLike, benchmarks: Sequence[str] = tuple(BENCHMARK_SOURCES), log=print) -> dict[str, Path]:
    import requests

    raw_dir = Path(raw_dir)
    raw_dir.mkdir(parents=True, exist_ok=True)
    out = {}
    for b in benchmarks:
        src = BENCHMARK_SOURCES[b]
        dest = raw_dir / src["file"]
        if not dest.exists():
            log(f"downloading {b} from {src['url']}")
            r = requests.get(src["url"], timeout=120)
            r.raise_for_status()
            dest.write_bytes(r.content)
            log(f"  sha256: {hashlib.sha256(r.content).hexdigest()}")
        expected = src.get("sha256")
        if expected is not None:
            actual = hashlib.sha256(dest.read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f"{b}: sha256 mismatch (expected {expected}, got {actual})")
        out[b] = dest
    return out


def load_pool(benchmark: str, raw_dir: str | os.PathLike) -> list[tuple[str, str]]:
    """[(prompt_id, prompt_text)] for a benchmark; MT-Bench uses the first turn."""
    src = BENCHMARK_SOURCES[benchmark]
    path = Path(raw_dir) / src["file"]
    items: list[tuple[str, str]] = []
    if benchmark == "mt_bench":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                items.append((f"mt_bench-{d['question_id']}", d["turns"][0]))
    elif benchmark == "alpaca_eval_2":
        for i, d in enumerate(json.loads(path.read_text(encoding="utf-8"))):
            items.append((f"alpaca_eval_2-{i}", d["instruction"]))
    elif benchmark == "arena_hard":
        for line in path.read_text(encoding="utf-8").splitlines():
            if line.strip():
                d = json.loads(line)
                items.append((f"arena_hard-{d['uid']}", d["prompt"]))
    else:
        raise ValueError(benchmark)
    n_exp = src.get("n_expected")
    if n_exp is not None and len(items) != n_exp:
        raise ValueError(f"{benchmark}: expected {n_exp} prompts, got {len(items)}")
    return items


def partition_prompts(pools: dict[str, list[tuple[str, str]]], allocation: dict[str, dict[str, int]], seed: int) -> dict[str, list[dict]]:
    """Seeded partition per benchmark into disjoint audit / calibration splits and a jury split drawn from the rest, then from the audit split."""
    out: dict[str, list[dict]] = {"audit": [], "calibration": [], "jury": []}
    for b, pool in pools.items():
        rng = np.random.default_rng([seed, zlib.crc32(b.encode())])
        alloc = allocation.get(b, {})
        order = rng.permutation(len(pool))
        n_a, n_c, n_j = int(alloc.get("audit", 0)), int(alloc.get("calibration", 0)), int(alloc.get("jury", 0))
        if n_a + n_c > len(pool):
            raise ValueError(f"{b}: audit + calibration ({n_a + n_c}) exceeds pool size {len(pool)}")
        audit_idx = order[:n_a]
        cal_idx = order[n_a:n_a + n_c]
        rest = order[n_a + n_c:]
        jury_idx = list(rest[:n_j])
        if len(jury_idx) < n_j:
            jury_idx += list(audit_idx[: n_j - len(jury_idx)])
        for split, idx in (("audit", audit_idx), ("calibration", cal_idx), ("jury", jury_idx)):
            for i in idx:
                pid, text = pool[int(i)]
                out[split].append({"prompt_id": pid, "benchmark": b, "text": text})
    return out


class SyntheticResponder:
    """Fabricates candidate texts whose length follows a latent verbosity (smoke configuration)."""

    def __init__(self, world: SyntheticWorld, verbosity: dict[str, float] | None = None):
        self.world = world
        self.verbosity = verbosity or {}

    def respond(self, model: str, prompt_id: str, prompt: str) -> str:
        h = hashlib.sha256(f"{self.world.seed}|resp|{model}|{prompt_id}".encode()).digest()
        rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
        n_sent = int(np.clip(round(rng.normal(4 + 2 * self.verbosity.get(model, 0.0), 1.5)), 1, 12))
        q = self.world.overall(prompt_id, model)
        sents = [f"[{model} | q={q:+.2f}] Synthetic sentence {i + 1} answering: {prompt[:60]!r}." for i in range(n_sent)]
        return " ".join(sents)


def generate_candidates(prompts: Sequence[dict], generators: Sequence[JudgeSpec], world: SyntheticWorld | None = None,
                        extra_generators: Sequence[JudgeSpec] = (), verbosity: dict[str, float] | None = None,
                        log=print) -> list[PromptRecord]:
    """One response per generator per prompt."""
    clients: dict[str, Judge | SyntheticResponder] = {}
    responder = SyntheticResponder(world, verbosity) if world is not None else None
    for spec in list(generators) + list(extra_generators):
        if spec.client == "synthetic" and responder is None:
            raise ValueError("synthetic generators need a SyntheticWorld")

    def client(spec: JudgeSpec) -> Judge | SyntheticResponder:
        if spec.name not in clients:
            clients[spec.name] = responder if spec.client == "synthetic" else make_judge(spec)
        return clients[spec.name]

    records = []
    for i, p in enumerate(prompts):
        def make(spec: JudgeSpec) -> dict:
            cl = client(spec)
            text = cl.respond(spec.name, p["prompt_id"], p["text"]) if isinstance(cl, SyntheticResponder) else cl.complete(p["text"])
            return {"id": spec.name, "text": text, "length": len(text)}

        rec = PromptRecord(p["prompt_id"], p["benchmark"], p["text"], [make(s) for s in generators], [make(s) for s in extra_generators])
        records.append(rec)
        if log and (i + 1) % 50 == 0:
            log(f"generated candidates for {i + 1}/{len(prompts)} prompts")
    return records


def synthetic_annotations(prompts: Sequence[PromptRecord], world: SyntheticWorld, n_annotators: int = 2,
                          noise_sd: float = 0.6, tag: str = "cal") -> list[Annotation]:
    """Simulated annotators rank candidates by latent overall quality plus noise."""
    out = []
    for p in prompts:
        cands = p.candidate_ids
        rankings = []
        for a in range(n_annotators):
            h = hashlib.sha256(f"{world.seed}|{tag}|annot{a}|{p.prompt_id}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
            util = {c: world.overall(p.prompt_id, c) + rng.normal(0, noise_sd) for c in cands}
            rankings.append(sorted(cands, key=lambda c: -util[c]))
        out.append(Annotation(p.prompt_id, p.benchmark, list(cands), rankings))
    return out


def synthetic_prompts(n_per_benchmark: dict[str, int], seed: int) -> dict[str, list[tuple[str, str]]]:
    """Offline stand-in prompt pools."""
    rng = np.random.default_rng(seed)
    pools = {}
    for b, n in n_per_benchmark.items():
        pools[b] = [(f"{b}-synth{i}", f"Synthetic {b} prompt #{i}: explain topic {int(rng.integers(1000))} clearly.") for i in range(n)]
    return pools
