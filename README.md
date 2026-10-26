# axiom-audit

Audits LLM-as-judge aggregators for violations of Universal Domain, Weak Pareto, Independence of Irrelevant Alternatives, Judge-Anonymity, Position-Invariance and Criterion-Monotonicity, and implements the Borda-Conformal Hybrid aggregator with its split-conformal calibration.

## Install

Python 3.11 or later.

```
pip install -r requirements.txt
```

```
pip install -r requirements-judges.txt
```

## Run

```
python main.py all --config configs/smoke.yaml
```

```
export OPENAI_API_KEY=... ANTHROPIC_API_KEY=...
export HF_TOKEN=...   # meta-llama/Llama-3.1-8B-Instruct is gated on the Hugging Face Hub

CFG="--config configs/audit_200.yaml"
python main.py prepare-data        $CFG
python main.py generate-candidates $CFG --split audit
python main.py generate-candidates $CFG --split calibration
python main.py generate-candidates $CFG --split jury
python main.py score               $CFG --split audit
python main.py score               $CFG --split calibration
python main.py audit               $CFG
```

## Layout

```
main.py         entry point
configs/        audit_200.yaml, smoke.yaml
aggregation/    score profiles and aggregators
bch/            Borda-Conformal Hybrid
judging/        judge clients, prompt templates, benchmarks
audit/          axiom probes, statistics, human comparator, orchestration
```

| File | Contents |
|---|---|
| `aggregation/profiles.py` | score profiles, social orders, order-reversal symmetrization |
| `aggregation/aggregators.py` | audited aggregators and frontier witnesses |
| `bch/bch.py` | BCH scores, split-conformal calibration, per-pair certificate, gap-marginalized bound |
| `judging/judges.py` | judge clients, scoring and pairwise templates, score store |
| `judging/benchmarks.py` | benchmark download, seeded partition, candidate generation |
| `audit/probes.py` | probes for the six axioms |
| `audit/stats.py` | standard errors, Clopper-Pearson, McNemar, Bonferroni, bootstrap, Krippendorff's alpha |
| `audit/human.py` | calibration set, agreement filter, human-jury IIA comparator |
| `audit/extras.py` | self-enhancement audit |
| `audit/run.py` | aggregator binding, axiom profiles, report writer |

## Configuration

| Key | Values |
|---|---|
| `seed` | integer (partition, calibration-alternative and bootstrap seed) |
| `benchmarks.<name>` | prompts per split (`audit`, `calibration`, `jury`) |
| `candidates`, `extra_candidates`, `judges` | client, model identifier and decoding settings of each generator and judge |
| `criteria` | list of criterion names (`[correctness, helpfulness, safety, brevity]`) |
| `criterion_weights` | simplex weights over criteria (`[0.25, 0.25, 0.25, 0.25]`) |
| `scoring.invalid_retries` | `1` (re-queries after an unparseable reply) |
| `scoring.fallback_score` | `5` (score of a response slot that still does not parse) |
| `scoring.orders` | `[fwd, rev]` (candidate listing orders averaged by the order-symmetrizing pipeline) |
| `probes.iia.variants` | `remove_one`, `add_one`, `replace_one` |
| `probes.cm.delta` | `1.0` (size of the single-score increase) |
| `pipeline_overrides` | per-aggregator pipeline: `fwd` (single presentation) or `sym` (order-symmetrized) |
| `pi_cell_overrides` | per-aggregator PI cell: `judge_level` (mean verdict-flip rate) or `composite` (ranking-change rate) |
| `best_of_judge.judge` | oracle judge of Best-of-Judge |
| `arena_bt.prior` | `0.01` (Bradley-Terry pseudo-count) |
| `arena_bt.strength_tol` | `1e-6` (log-strength tie band) |
| `lc_win_rate.length_scale` | `auto` (standard deviation of candidate lengths) or a number |
| `bch.alpha` | `0.05` (conformal miscoverage level) |
| `calibration.iaa_threshold` | `0.7` (per-prompt Krippendorff's alpha, ordinal, retained above this value) |
| `calibration.thresholds` | `[0.5, 0.6, 0.7, 0.8]` (agreement thresholds of the sensitivity check) |
| `stats.bootstrap_resamples` | `1000` |
| `stats.mcnemar_reference` | `bch` (aggregator every other aggregator is paired against) |
| `stats.bonferroni_alpha` | `0.05` (family-wise significance level) |
| `extras.self_enhancement.n_matched_pairs` | `100` (length-matched pairs) |
