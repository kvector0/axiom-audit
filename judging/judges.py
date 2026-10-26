"""Judge clients, prompt templates, score cache and the scoring and pairwise runners."""
from __future__ import annotations

import hashlib
import json
import os
import re
import time
import unicodedata
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from itertools import combinations
from pathlib import Path
from typing import Iterable, Sequence

import numpy as np

from aggregation.profiles import Profile, symmetrize

CRITERIA_DEFAULT = ("correctness", "helpfulness", "safety", "brevity")
ORDERS = ("fwd", "rev")

SCORING_TEMPLATE = (
    "You are evaluating responses on the criterion of {criterion}, anchored on a {lo}–{hi} scale "
    "where {lo} = unacceptable and {hi} = excellent. Given the prompt below and the {n} responses, "
    "return a single integer score between {lo} and {hi} for each response, one per line in the order "
    "presented, formatted as \"Response i: <score>\". Do not include any text other than the scores.\n\n"
    "Prompt: {prompt}\n"
    "{responses}\n"
    "{criterion} scores:"
)

PAIRWISE_TEMPLATE = (
    "Compare two responses to the prompt below on {focus}. Return either \"A\" or \"B\" indicating "
    "which is better. Do not include any text other than the letter.\n\n"
    "Prompt: {prompt}\n"
    "Response A: {response_a}\n"
    "Response B: {response_b}\n"
    "Choice:"
)


def scoring_prompt(prompt: str, responses: Sequence[str], criterion: str, lo: int = 1, hi: int = 10) -> str:
    block = "\n".join(f"Response {i + 1}: {r}" for i, r in enumerate(responses))
    return SCORING_TEMPLATE.format(criterion=criterion, prompt=prompt, responses=block, n=len(responses), lo=lo, hi=hi)


def pairwise_prompt(prompt: str, response_a: str, response_b: str, criterion: str | None = None) -> str:
    focus = "overall quality" if criterion is None else criterion
    return PAIRWISE_TEMPLATE.format(focus=focus, prompt=prompt, response_a=response_a, response_b=response_b)


LABEL = re.compile(r"response ?#? ?\d", re.I | re.A)
SCORE_LINE = re.compile(
    r"response ?#? ?(\d{1,4})(?: of (\d{1,4}))?(?: \((?:1 ?[-–] ?(\d{1,4})|1 to (\d{1,4})|out of (\d{1,4}))\))?(?: (?:score|rating))?"
    r"(?: ?[:=：] ?| [-–—] )(?:(?:score|rating) ?[:=：]? ?)?(?:<(-?\d{1,4})(?:\.0+)?>|\[(-?\d{1,4})(?:\.0+)?\]|\((-?\d{1,4})(?:\.0+)?\)|"
    r"\"(-?\d{1,4})(?:\.0+)?\"|(-?\d{1,4})(?:\.0+)?)(?: ?/ ?(\d{1,4})| out of (\d{1,4}))?\.?",
    re.I | re.A)
NUMBER = re.compile(r"-?\d{1,4}(?:\.0+)?", re.A)
VERDICT_LINE = re.compile(
    r"(?:(?:my|the|final|best|better|preferred|overall) ){0,2}(?:choice|answer|verdict) ?[:：-]? ?[\"'(\[]?([ab])[\"')\]]?[.!]?",
    re.I | re.A)
FENCE = re.compile(r"```[A-Za-z]*")


def _clean(text: str) -> str:
    return re.sub(r"[\u200b-\u200f\u2060\ufeff]", "", unicodedata.normalize("NFKC", text))


def _norm(line: str) -> str:
    line = " ".join(re.sub(r"[*`]", "", line).replace("_", " ").split())
    return re.sub(r"^(?:[-–•>#]+ )+", "", line)


def _lines(text: str) -> list[str]:
    return [ln for ln in (_norm(x) for x in text.splitlines() if not FENCE.fullmatch(x.strip())) if ln]


TYPOGRAPHY = frozenset("\u2013\u2014\u2018\u2019\u201c\u201d\u2022\uff1a\u2026\u00a0\u200b\u200c\u200d\u2060\ufeff")


def _odd(text: str) -> bool:
    """True if the reply contains anything but printable ASCII, tab, newlines and a few typographic marks."""
    return any(not (" " <= ch <= "~" or ch in "\t\n\r" or ch in TYPOGRAPHY) for ch in text)


def _hidden(text: str) -> bool:
    """True if Markdown would hide or change part of the reply: HTML comments or entities, text on a code-fence line."""
    return bool(re.search(r"<!--|&#?\w+;|\r(?!\n)", text)) or any("```" in x and not FENCE.fullmatch(x.strip())
                                                                   for x in text.splitlines())


def parse_scores(text: str, n: int, lo: int = 1, hi: int = 10, truncated: bool = False, criterion: str | None = None) -> list[int | None]:
    """Scores of a reply whose every line is one "Response i: s" line (after an optional "<criterion> scores:" header), or that is n bare numbers."""
    out: list[int | None] = [None] * n
    if truncated or _odd(text) or _hidden(text) or "`" in "".join(x for x in text.splitlines() if not FENCE.fullmatch(x.strip())):
        return out
    text = _clean(text)
    if re.search(r"\d[*_`]+\d|[-–—]_|_[-–—]", text):
        return out
    lines = _lines(text)
    if not lines or re.search(r"rank", text, flags=re.I):
        return out
    name = re.escape(criterion.replace("_", " ")) + " " if criterion else ""
    header = re.compile(r"(?:" + name + r")?scores?:?", re.I)
    if any(LABEL.search(ln) for ln in lines):
        body = lines[1:] if header.fullmatch(lines[0]) else lines
        vals: dict[int, int] = {}
        for line in body:
            m = SCORE_LINE.fullmatch(line) if len(line) <= 120 else None
            if m is None:
                return out
            idx, of, top, den = int(m.group(1)), m.group(2), m.group(3) or m.group(4) or m.group(5), m.group(11) or m.group(12)
            v = int(m.group(6) or m.group(7) or m.group(8) or m.group(9) or m.group(10))
            if not 1 <= idx <= n or not lo <= v <= hi or any(x is not None and int(x) != want for x, want in ((of, n), (top, hi), (den, hi))) \
                    or vals.get(idx, v) != v:
                return out
            vals[idx] = v
        return [vals.get(k) for k in range(1, n + 1)]
    toks = re.split(r", | ", lines[0]) if len(lines) == 1 else lines
    if not re.fullmatch(r"[0-9\s,.-]+", text) or len(toks) != n or not all(NUMBER.fullmatch(t) for t in toks):
        return out
    vals_ = [int(float(t)) for t in toks]
    return vals_ if all(lo <= v <= hi for v in vals_) else out


def parse_pairwise(text: str, truncated: bool = False) -> str | None:
    """The verdict of a reply that is one line: the bare letter, or "Choice: X" / "Answer: X" / "Verdict: X"."""
    if truncated or _odd(text):
        return None
    text = _clean(text)
    if _hidden(text) or "```" in text or re.search(r"\]\s*:", text):
        return None
    lines = _lines(text)
    if lines and re.fullmatch(r"(?:choice|answer|verdict) ?[:\uff1a]", lines[0], flags=re.I):
        lines = lines[1:]
    if len(lines) != 1:
        return None
    bare = re.fullmatch(r"[\"'\u201c\u2018(\[]?([ABab])[\"'\u201d\u2019)\]]?\.?", lines[0])
    if bare:
        return bare.group(1).upper()
    m = VERDICT_LINE.fullmatch(lines[0])
    return m.group(1).upper() if m else None


@dataclass
class JudgeSpec:
    name: str
    client: str
    model: str = ""
    temperature: float = 0.0
    max_tokens: int = 64
    extra: dict = field(default_factory=dict)


class Judge(ABC):
    """A judge answers free-text prompts; the scoring and pairwise methods build on complete()."""

    def __init__(self, spec: JudgeSpec, scale: tuple[int, int] = (1, 10), invalid_retries: int = 1, fallback_score: int = 5):
        self.spec = spec
        self.name = spec.name
        self.scale = scale
        self.invalid_retries = invalid_retries
        self.fallback_score = fallback_score
        self.n_calls = 0
        self.n_requeries = 0
        self.n_fallback_slots = 0
        self.n_pairwise_failures = 0
        self.last_truncated = False

    @abstractmethod
    def complete(self, prompt: str) -> str:
        """Reply text; sets ``last_truncated`` when the reply stopped at the max_tokens limit."""

    def stats(self) -> dict:
        return {"calls": self.n_calls, "requeries": self.n_requeries, "fallback_slots": self.n_fallback_slots,
                "pairwise_failures": self.n_pairwise_failures}

    def fingerprint(self) -> dict:
        return judge_fingerprint(self.spec, {"scale": self.scale, "invalid_retries": self.invalid_retries,
                                             "fallback_score": self.fallback_score})

    def score(self, prompt_id: str, prompt: str, criterion: str, cand_ids: Sequence[str], responses: Sequence[str]) -> list[int]:
        lo, hi = self.scale
        n = len(responses)
        best: list[int | None] = [None] * n
        for attempt in range(self.invalid_retries + 1):
            self.n_calls += 1
            self.n_requeries += attempt > 0
            text = self.complete(scoring_prompt(prompt, responses, criterion, lo, hi))
            parsed = parse_scores(text, n, lo, hi, self.last_truncated, criterion)
            best = [b if b is not None else p for p, b in zip(parsed, best)]
            if all(v is not None for v in best):
                break
        self.n_fallback_slots += sum(v is None for v in best)
        return [v if v is not None else self.fallback_score for v in best]

    def pairwise(self, prompt_id: str, prompt: str, criterion: str | None, a_id: str, a_text: str, b_id: str, b_text: str) -> str | None:
        """Id of the preferred candidate, or None when no verdict parses."""
        for attempt in range(self.invalid_retries + 1):
            self.n_calls += 1
            self.n_requeries += attempt > 0
            text = self.complete(pairwise_prompt(prompt, a_text, b_text, criterion))
            v = parse_pairwise(text, self.last_truncated)
            if v is not None:
                return a_id if v == "A" else b_id
        self.n_pairwise_failures += 1
        return None


def _retry(fn, tries: int = 5, base: float = 1.0):
    last: Exception | None = None
    for i in range(tries):
        try:
            return fn()
        except Exception as e:
            last = e
            time.sleep(base * (2 ** i))
    raise RuntimeError(f"judge call failed after {tries} attempts: {last}")


class OpenAIJudge(Judge):
    """OpenAI chat completions client."""

    def __init__(self, spec: JudgeSpec, **kw):
        super().__init__(spec, **kw)
        from openai import OpenAI

        self._client = OpenAI(**spec.extra.get("client_kwargs", {}))

    def complete(self, prompt: str) -> str:
        def call():
            r = self._client.chat.completions.create(
                model=self.spec.model, temperature=self.spec.temperature, max_tokens=self.spec.max_tokens,
                messages=[{"role": "user", "content": prompt}], seed=self.spec.extra.get("seed", 0),
            )
            self.last_truncated = r.choices[0].finish_reason != "stop"
            return r.choices[0].message.content or ""

        return _retry(call)


class AnthropicJudge(Judge):
    """Anthropic messages client."""

    def __init__(self, spec: JudgeSpec, **kw):
        super().__init__(spec, **kw)
        from anthropic import Anthropic

        self._client = Anthropic(**spec.extra.get("client_kwargs", {}))

    def complete(self, prompt: str) -> str:
        def call():
            r = self._client.messages.create(
                model=self.spec.model, temperature=self.spec.temperature, max_tokens=self.spec.max_tokens,
                messages=[{"role": "user", "content": prompt}],
            )
            self.last_truncated = r.stop_reason not in ("end_turn", "stop_sequence")
            return "".join(getattr(b, "text", "") for b in r.content)

        return _retry(call)


class HFLocalJudge(Judge):
    """Local causal language model loaded through transformers."""

    def __init__(self, spec: JudgeSpec, **kw):
        super().__init__(spec, **kw)
        import torch
        from transformers import AutoModelForCausalLM, AutoTokenizer

        self._torch = torch
        self._tok = AutoTokenizer.from_pretrained(spec.model)
        self._model = AutoModelForCausalLM.from_pretrained(
            spec.model, device_map=spec.extra.get("device_map", "auto"),
            torch_dtype=getattr(torch, spec.extra.get("dtype", "float16")),
        )
        self._model.eval()

    def complete(self, prompt: str) -> str:
        tok = self._tok
        if getattr(tok, "chat_template", None):
            ids = tok.apply_chat_template([{"role": "user", "content": prompt}], add_generation_prompt=True, return_tensors="pt")
        else:
            ids = tok(prompt, return_tensors="pt").input_ids
        ids = ids.to(self._model.device)
        with self._torch.no_grad():
            out = self._model.generate(ids, max_new_tokens=self.spec.max_tokens, do_sample=False)
        gen = out[0, ids.shape[1]:]
        eos = self._model.generation_config.eos_token_id
        eos = set(eos) if isinstance(eos, (list, tuple)) else {eos}
        self.last_truncated = len(gen) > 0 and len(gen) >= self.spec.max_tokens and int(gen[-1]) not in eos
        return tok.decode(gen, skip_special_tokens=True)


class SyntheticJudge(Judge):
    """Deterministic simulated judge for the smoke configuration."""

    def __init__(self, spec: JudgeSpec, latent, **kw):
        super().__init__(spec, **kw)
        self.latent = latent
        x = spec.extra
        self.bias = float(x.get("bias", 0.0))
        self.noise_sd = float(x.get("noise_sd", 0.8))
        self.position_effect = float(x.get("position_effect", 0.6))
        self.context_effect = float(x.get("context_effect", 0.5))
        self.pairwise_position_bias = float(x.get("pairwise_position_bias", 0.4))
        self.pairwise_noise_sd = float(x.get("pairwise_noise_sd", 0.7))
        self.context_noise_share = float(x.get("context_noise_share", 0.5))

    def complete(self, prompt: str) -> str:
        raise NotImplementedError("SyntheticJudge produces structured scores directly")

    def fingerprint(self) -> dict:
        return judge_fingerprint(self.spec, {"scale": self.scale, "invalid_retries": self.invalid_retries,
                                             "fallback_score": self.fallback_score}, self.latent)

    def _rng(self, *key) -> np.random.Generator:
        h = hashlib.sha256("|".join(map(str, key)).encode()).digest()
        return np.random.default_rng(int.from_bytes(h[:8], "little"))

    def _score(self, prompt_id: str, criterion: str, cand_ids: Sequence[str], position: int) -> int:
        lo, hi = self.scale
        cid = cand_ids[position]
        qs = [self.latent.quality(prompt_id, c, criterion) for c in cand_ids]
        q = qs[position]
        n = len(cand_ids)
        rank = sum(1 for v in qs if v > q)
        contrast = self.context_effect * ((n - 1) - 2 * rank) / max(1, n - 1)
        rho = self.context_noise_share
        r_persist = self._rng(self.name, prompt_id, criterion, cid, "persistent")
        r_context = self._rng(self.name, prompt_id, criterion, cid, tuple(cand_ids), position)
        noise = self.noise_sd * (np.sqrt(1 - rho) * r_persist.normal() + np.sqrt(rho) * r_context.normal())
        raw = 5.5 + q + self.bias + contrast + noise
        raw += self.position_effect * (0.5 - position / max(1, n - 1))
        return int(np.clip(round(raw), lo, hi))

    def score(self, prompt_id, prompt, criterion, cand_ids, responses) -> list[int]:
        self.n_calls += 1
        return [self._score(prompt_id, criterion, list(cand_ids), i) for i in range(len(cand_ids))]

    def pairwise(self, prompt_id, prompt, criterion, a_id, a_text, b_id, b_text) -> str | None:
        self.n_calls += 1
        crit = criterion or "overall"
        qa = self.latent.overall(prompt_id, a_id) if criterion is None else self.latent.quality(prompt_id, a_id, criterion)
        qb = self.latent.overall(prompt_id, b_id) if criterion is None else self.latent.quality(prompt_id, b_id, criterion)
        rng = self._rng(self.name, prompt_id, crit, "pair", a_id, b_id)
        margin = (qa - qb) + self.pairwise_position_bias + rng.normal(0.0, self.pairwise_noise_sd)
        return a_id if margin >= 0 else b_id


class SyntheticWorld:
    """Latent qualities shared by synthetic judges, candidates and annotators (seeded)."""

    def __init__(self, seed: int, criteria: Sequence[str] = CRITERIA_DEFAULT, candidate_means: dict[str, float] | None = None, sd: float = 1.0):
        self.seed = int(seed)
        self.criteria = tuple(criteria)
        self.candidate_means = candidate_means or {}
        self.sd = sd
        self._cache: dict[tuple[str, str, str], float] = {}

    def quality(self, prompt_id: str, cand_id: str, criterion: str) -> float:
        key = (prompt_id, cand_id, criterion)
        if key not in self._cache:
            h = hashlib.sha256(f"{self.seed}|{prompt_id}|{cand_id}|{criterion}".encode()).digest()
            rng = np.random.default_rng(int.from_bytes(h[:8], "little"))
            mu = self.candidate_means.get(cand_id, 0.0)
            self._cache[key] = float(mu + rng.normal(0.0, self.sd))
        return self._cache[key]

    def overall(self, prompt_id: str, cand_id: str) -> float:
        return float(np.mean([self.quality(prompt_id, cand_id, c) for c in self.criteria]))


def _scoring_kwargs(scoring_cfg: dict | None) -> dict:
    cfg = scoring_cfg or {}
    return dict(scale=tuple(cfg.get("scale", (1, 10))), invalid_retries=int(cfg.get("invalid_retries", 1)),
                fallback_score=int(cfg.get("fallback_score", 5)))


def judge_fingerprint(spec: JudgeSpec, scoring_cfg: dict | None = None, world: "SyntheticWorld | None" = None) -> dict:
    """Every judge setting and template that changes a cached score or verdict (client credentials excluded)."""
    kw = _scoring_kwargs(scoring_cfg)
    fp = {"client": spec.client, "model": spec.model, "temperature": spec.temperature, "max_tokens": spec.max_tokens,
          "extra": {k: v for k, v in sorted(spec.extra.items()) if k != "client_kwargs"}, "scale": list(kw["scale"]),
          "invalid_retries": kw["invalid_retries"], "fallback_score": kw["fallback_score"],
          "templates_sha256": hashlib.sha256((SCORING_TEMPLATE + PAIRWISE_TEMPLATE).encode()).hexdigest()}
    if world is not None:
        fp["world"] = {"seed": world.seed, "criteria": list(world.criteria), "candidate_means": world.candidate_means, "sd": world.sd}
    return json.loads(json.dumps(fp))


def make_judge(spec: JudgeSpec, scoring_cfg: dict | None = None, world: SyntheticWorld | None = None) -> Judge:
    kw = _scoring_kwargs(scoring_cfg)
    if spec.client == "openai":
        return OpenAIJudge(spec, **kw)
    if spec.client == "anthropic":
        return AnthropicJudge(spec, **kw)
    if spec.client == "hf":
        return HFLocalJudge(spec, **kw)
    if spec.client == "synthetic":
        if world is None:
            raise ValueError("synthetic judges need a SyntheticWorld")
        return SyntheticJudge(spec, world, **kw)
    raise ValueError(f"unknown judge client {spec.client!r}")


FULL = "full"


def subset_key_remove(cand: str) -> str:
    return f"minus:{cand}"


def subset_key_add(extra: str) -> str:
    return f"plus:{extra}"


def subset_key_replace(cand: str, extra: str) -> str:
    return f"swap:{cand}:{extra}"


def subset_candidates(key: str, candidates: Sequence[str]) -> list[str]:
    """Canonical (forward) candidate order for a subset key."""
    cands = list(candidates)
    if key == FULL:
        return cands
    kind, *rest = key.split(":")
    if kind == "minus":
        return [c for c in cands if c != rest[0]]
    if kind == "plus":
        return cands + [rest[0]]
    if kind == "swap":
        old, new = rest
        return [new if c == old else c for c in cands]
    raise ValueError(f"bad subset key {key!r}")


class ScoreStore:
    """JSON-backed cache of every judge query."""

    def __init__(self, path: str | os.PathLike | None = None):
        self.path = Path(path) if path else None
        self.scores: dict = {}
        self.pairwise: dict = {}
        self.meta: dict = {}
        self._recorded: dict = {}
        if self.path and self.path.exists():
            self.load()

    def load(self) -> None:
        d = json.loads(Path(self.path).read_text(encoding="utf-8"))
        self.scores, self.pairwise, self.meta = d.get("scores", {}), d.get("pairwise", {}), d.get("meta", {})

    def save(self) -> None:
        if self.path is None:
            return
        self.path.parent.mkdir(parents=True, exist_ok=True)
        tmp = self.path.with_suffix(".tmp")
        tmp.write_text(json.dumps({"meta": self.meta, "scores": self.scores, "pairwise": self.pairwise}), encoding="utf-8")
        os.replace(tmp, self.path)

    def record_judge_stats(self, judges: Sequence[Judge]) -> None:
        """Add the query counts made since the last call to the counts of the runs that filled the store."""
        acc = self.meta.setdefault("judge_stats", {})
        for j in judges:
            cur = acc.setdefault(j.name, {})
            done = self._recorded.setdefault(j.name, {})
            for k, v in j.stats().items():
                cur[k] = int(cur.get(k, 0)) + int(v) - int(done.get(k, 0))
                done[k] = int(v)

    def check_inputs(self, fingerprints: dict[str, dict], prompts: Sequence["PromptRecord"]) -> None:
        """Refuse cached entries produced under other judge settings, templates or candidate responses."""
        known = self.meta.setdefault("judges", {})
        for name, fp in fingerprints.items():
            if known.setdefault(name, fp) != fp:
                raise ValueError(f"{self.path}: cached scores of judge {name!r} were produced with other judge settings or "
                                 f"templates; use a new scores path or delete the file")
        digests = self.meta.setdefault("prompts", {})
        for p in prompts:
            d = p.digest()
            if digests.setdefault(p.prompt_id, d) != d:
                raise ValueError(f"{self.path}: cached scores of prompt {p.prompt_id!r} were produced for other candidate "
                                 f"responses; use a new scores path or delete the file")

    def has_scores(self, prompt: str, judge: str, criterion: str, subset: str, order: str) -> bool:
        return order in self.scores.get(prompt, {}).get(judge, {}).get(criterion, {}).get(subset, {})

    def get_scores(self, prompt: str, judge: str, criterion: str, subset: str, order: str) -> dict[str, float]:
        return self.scores[prompt][judge][criterion][subset][order]

    def put_scores(self, prompt: str, judge: str, criterion: str, subset: str, order: str, values: dict[str, float]) -> None:
        self.scores.setdefault(prompt, {}).setdefault(judge, {}).setdefault(criterion, {}).setdefault(subset, {})[order] = dict(values)

    @staticmethod
    def pair_key(a: str, b: str) -> str:
        return "|".join(sorted((a, b)))

    def has_pairwise(self, prompt: str, judge: str, criterion: str, a: str, b: str) -> bool:
        return self.pair_key(a, b) in self.pairwise.get(prompt, {}).get(judge, {}).get(criterion, {})

    def put_pairwise(self, prompt: str, judge: str, criterion: str, a: str, b: str, verdicts: dict) -> None:
        self.pairwise.setdefault(prompt, {}).setdefault(judge, {}).setdefault(criterion, {})[self.pair_key(a, b)] = dict(verdicts)

    def profile(self, prompt: str, judges: Sequence[str], criteria: Sequence[str], candidates: Sequence[str],
                subset: str = FULL, order: str = "sym") -> Profile:
        """Profile for (prompt, subset) under a presentation: 'fwd', 'rev' or 'sym' (order-symmetrized)."""
        alts = subset_candidates(subset, candidates)
        if order == "sym":
            return symmetrize(self.profile(prompt, judges, criteria, candidates, subset, "fwd"),
                              self.profile(prompt, judges, criteria, candidates, subset, "rev"))
        arr = np.empty((len(judges), len(criteria), len(alts)))
        for ji, j in enumerate(judges):
            for ci, c in enumerate(criteria):
                sc = self.get_scores(prompt, j, c, subset, order)
                arr[ji, ci] = [sc[x] for x in alts]
        return Profile(arr, tuple(judges), tuple(criteria), tuple(alts))


@dataclass
class PromptRecord:
    prompt_id: str
    benchmark: str
    text: str
    candidates: list[dict]
    extra_candidates: list[dict] = field(default_factory=list)

    def digest(self) -> str:
        """SHA-256 of the prompt text and every candidate response the judges see."""
        content = [self.text, [[c["id"], c["text"]] for c in self.candidates], [[c["id"], c["text"]] for c in self.extra_candidates]]
        return hashlib.sha256(json.dumps(content).encode()).hexdigest()

    @property
    def candidate_ids(self) -> list[str]:
        return [c["id"] for c in self.candidates]

    def response(self, cand_id: str) -> str:
        for c in self.candidates + self.extra_candidates:
            if c["id"] == cand_id:
                return c["text"]
        raise KeyError(cand_id)

    def length(self, cand_id: str) -> float:
        for c in self.candidates + self.extra_candidates:
            if c["id"] == cand_id:
                return float(c.get("length", len(c["text"])))
        raise KeyError(cand_id)

    @classmethod
    def from_dict(cls, d: dict) -> "PromptRecord":
        return cls(d["prompt_id"], d["benchmark"], d["text"], list(d["candidates"]), list(d.get("extra_candidates", [])))

    def to_dict(self) -> dict:
        return {"prompt_id": self.prompt_id, "benchmark": self.benchmark, "text": self.text,
                "candidates": self.candidates, "extra_candidates": self.extra_candidates}


def load_prompts(path: str | os.PathLike) -> list[PromptRecord]:
    out = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                out.append(PromptRecord.from_dict(json.loads(line)))
    return out


def save_prompts(path: str | os.PathLike, prompts: Iterable[PromptRecord]) -> None:
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for p in prompts:
            f.write(json.dumps(p.to_dict()) + "\n")


def subset_keys_for(prompt: PromptRecord, variants: Sequence[str]) -> list[str]:
    """Subset keys needed for the configured IIA probe variants."""
    keys = [FULL]
    cands = prompt.candidate_ids
    extras = [e["id"] for e in prompt.extra_candidates]
    if "remove_one" in variants:
        keys += [subset_key_remove(c) for c in cands]
    if "add_one" in variants and extras:
        keys.append(subset_key_add(extras[0]))
    if "replace_one" in variants and extras:
        keys += [subset_key_replace(c, extras[0]) for c in cands]
    return keys


def run_scoring(store: ScoreStore, judges: Sequence[Judge], prompts: Sequence[PromptRecord], criteria: Sequence[str],
                subset_variants: Sequence[str] = ("remove_one",), orders: Sequence[str] = ORDERS,
                save_every: int = 50, log=print) -> int:
    """Populate the store with scores for every (prompt, judge, criterion, subset, order)."""
    store.check_inputs({j.name: j.fingerprint() for j in judges}, prompts)
    n_new = 0
    since_save = 0
    for pi, prompt in enumerate(prompts):
        for key in subset_keys_for(prompt, subset_variants):
            fwd_cands = subset_candidates(key, prompt.candidate_ids)
            for judge in judges:
                for c in criteria:
                    for order in orders:
                        if store.has_scores(prompt.prompt_id, judge.name, c, key, order):
                            continue
                        cands = fwd_cands if order == "fwd" else list(reversed(fwd_cands))
                        vals = judge.score(prompt.prompt_id, prompt.text, c, cands, [prompt.response(x) for x in cands])
                        store.put_scores(prompt.prompt_id, judge.name, c, key, order, dict(zip(cands, vals)))
                        n_new += 1
                        since_save += 1
            if since_save >= save_every:
                store.record_judge_stats(judges)
                store.save()
                since_save = 0
        if log and (pi + 1) % 25 == 0:
            log(f"scored {pi + 1}/{len(prompts)} prompts ({n_new} new queries)")
    store.record_judge_stats(judges)
    store.save()
    return n_new


def run_pairwise(store: ScoreStore, judges: Sequence[Judge], prompts: Sequence[PromptRecord], criteria: Sequence[str | None],
                 save_every: int = 50, log=print) -> int:
    """Pairwise queries for the PI probe: every candidate pair, both presentation orders, per criterion."""
    store.check_inputs({j.name: j.fingerprint() for j in judges}, prompts)
    n_new = 0
    since_save = 0
    for pi, prompt in enumerate(prompts):
        cands = prompt.candidate_ids
        for judge in judges:
            for c in criteria:
                ckey = c if c is not None else "overall"
                for a, b in combinations(cands, 2):
                    if store.has_pairwise(prompt.prompt_id, judge.name, ckey, a, b):
                        continue
                    ab = judge.pairwise(prompt.prompt_id, prompt.text, c, a, prompt.response(a), b, prompt.response(b))
                    ba = judge.pairwise(prompt.prompt_id, prompt.text, c, b, prompt.response(b), a, prompt.response(a))
                    store.put_pairwise(prompt.prompt_id, judge.name, ckey, a, b, {"AB": ab, "BA": ba, "first": a, "second": b})
                    n_new += 1
                    since_save += 1
            if since_save >= save_every:
                store.record_judge_stats(judges)
                store.save()
                since_save = 0
        if log and (pi + 1) % 25 == 0:
            log(f"pairwise {pi + 1}/{len(prompts)} prompts ({n_new} new pairs)")
    store.record_judge_stats(judges)
    store.save()
    return n_new
