# FrontierRank

Reranking that spends compute in proportion to **ranking regret** rather than
candidate count. A small local typed-decision model (Laya, 421M, Apache-2.0)
scores every candidate; a hosted large-slate teacher (Jev) is called only on the
uncertain top-k frontier; every teacher call becomes training data.

Companion to the design doc. Everything here runs on CPU with numpy; torch is
only needed for `losses.py`.

## Install

```bash
pip install -e .            # numpy only
pip install -e '.[train]'   # + torch, transformers
pip install -e '.[eval]'    # + scipy, lightgbm
```

## The one-minute version

```bash
python scripts/smoke_test.py          # end-to-end, synthetic scorer, ~10s
python scripts/anchor_budget_sweep.py # anchors vs slate capacity
python scripts/cost_model.py          # $/1k queries, all inputs cited in-file
```

## What the smoke test shows

Two findings, and the second one corrected the project's original claim.

| | naive | anchored | what it means |
|---|---|---|---|
| **nDCG@10**, slates blocked by first-stage rank | 0.679 | **0.902** | anchoring rescues the obvious implementation |
| **nDCG@10**, slates stratified round-robin | 0.895 | 0.897 | **stratification is a free fix — anchors add nothing here** |
| **Cross-query correlation** with true utility | 0.918 | **0.961** | only anchoring gives an absolute scale |

Stratify for ranking. Anchor for scale. They fix different problems, and it is
the *scale* that the frontier, the thresholds and the multi-round re-scoring
need.

## Modules

| Module | What it does |
|---|---|
| `slates.py` | slate construction under a hard option-slot budget (M=10 for Laya) |
| `calibrate.py` | anchor estimators — shift, affine, ridge — recovering the per-slate offset |
| `score.py` | `AnchoredScorer`: slates → one globally comparable utility per candidate |
| `frontier.py` | top-k regret frontier, swap probabilities, expected nDCG regret |
| `controller.py` | EVI action choice: stop / read_more / widen / teacher |
| `teacher.py` | teacher interface + a Jev batched-rubric implementation |
| `losses.py` | λ-RankNet, ListNet, teacher-distribution KL, CORAL, RPS |
| `evaluate.py` | nDCG (TREC convention), ECE, paired bootstrap |

## Plugging in a real model

`ScorerProtocol` is the whole contract:

```python
class MyScorer:
    def choice_logprobs(self, instructions, options, state) -> np.ndarray:
        """log-probabilities over `options`, normalised across them."""
```

The Laya SDK's `Router` / `RLAgent.system_one` fits this directly, as does a
Jev `Choice` call. `TeacherProtocol` is similarly thin — swap Jev for a
Qwen3-Reranker or an LLM judge in one line. Nothing above `teacher.py` imports
a vendor name, deliberately: Jev is closed-weights, early access, and cannot be
fine-tuned, so all vendor risk sits on that one interface.

## Constraints encoded here

- **M = 10 option slots.** Laya ships `temperature_by_options` with
  `choice:11+ → T = 0.1006`, a 10× sharpening that only fits if the raw logits
  are near-uniform past ~10 options. The cliff is at 11, not 20.
- **State is not shared across questions.** `build_sequence` runs once per
  question; measured T4 marginal cost ~14 ms/question. Five factorised
  questions per candidate = five encoder passes per candidate. Do not fan out.
- **Jev caps at 255 options**, hard-rejects at 256, and its latency is flat in
  option count — so send the whole frontier, not a subset.
- **Grade 1 in TREC DL is NOT relevant.** `recall_at_k`/`mrr_at_k` default to
  `threshold=2`, matching `trec_eval -l 2`.

## Not included

Retrieval, the evidence slicer, and the BEIR/TREC harness are deliberately
out of scope — they depend on your index. `AnchoredScorer.score()` takes
signatures and a first-stage order; produce those however you like.
