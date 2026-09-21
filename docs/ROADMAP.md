# Roadmap

The tracker. One row per task, tied to the phase gate it serves, the command
that runs it, and the finding it produces. `plan.md` Part X defines the gates;
this file records progress against them.

Status: `[ ]` todo · `[~]` in progress · `[x]` done · `[!]` blocked · `[-]` dropped

Findings live in [FINDINGS.md](FINDINGS.md) (F-numbers below link there).
Dated narrative is in [LAB_NOTEBOOK.md](LAB_NOTEBOOK.md).

---

## Phase 0 — the mechanism  ·  **gate PASSED, two items open**

> Gate: `c_S` behaves as a per-slate shift, and anchor regression residuals are
> on the order of the measurement floor. If not, stop — the design does not survive.

| | Task | Command | Gate / output |
|---|---|---|---|
| [x] | Read the vendor source; pin it in `docs/vendor/` | — | F1, F2 |
| [x] | Token-budget arithmetic, verified against the real tokenizer | `make budget` | F1 — 16/16 exact |
| [x] | Faithful Laya reimplementation, `strict=True` load | `make test` | F10 — noul 0.9237 vs Jev 0.95 |
| [x] | Teacher layer: Jev/OpenRouter, in-session human, cache | `pytest tests/integration -m network` | F5 |
| [x] | BEIR loaders, BM25, evidence slicer | — | — |
| [x] | **T2 — additive fit `log p = l_i − c_S`** | `make phase0` | **F6 — PASS, 0.84× floor** |
| [x] | T1 — order-noise floor (replaces "marker noise") | `make phase0` | F3, F4 |
| [x] | Signal vs starvation: OPTIONS vs STATE layout | `make phase0-signal` | F7 — STATE wins by +0.27 nDCG |
| [x] | **Confirm F7 at full scale with bootstrap CIs** | `make phase0-signal-full` | F7 — +0.232 [+0.151,+0.312] vs BM25, n=50 |
| [!] | T4 — does anchoring buy the cross-slate scale? | `make phase0-anchors` | F8 — underpowered (+0.010 ± 0.013) |
| [x] | Templated pivots + probe: are they actually pivots? | `make anchor-probe` | F11 — only weakly ordered (0.33) |
| [x] | Teacher-graded pivots: real text, continuous utilities | — | F11 — now the default |
| [ ] | Re-run T4 with teacher pivots | `make phase0-anchors` | unblocks F8 |
| [ ] | T5 — does anchor *choice* swing results >1 nDCG point? | — | plan Part VIII test 5 |

**Why T4 is blocked, not failed:** trec-covid has three grades, so the anchor
regression has three distinct x-values against a 0.38-nat noise floor. Jev's
Score returns a *fractional* expected grade, which gives pivots real leverage.
That is also what production would use.

---

## Phase 1 — anchored scoring  ·  **next**

> Gate: anchored beats naive by a large margin, and beats a pointwise rubric at
> equal or lower pass count. If anchoring only matches the rubric, use the rubric.

| | Task | Command | Gate / output |
|---|---|---|---|
| [x] | Full reranking pipeline on a realistic top-100 pool | `make train` | via `make_eval_fn`; pools frozen |
| [x] | Validate BM25 against published BEIR numbers | `make check-first-stage` | F12 — 2/3 within 0.011 |
| [x] | Ordinal CORAL head (`P(y > k)` on the 0–3 scale) | `pytest tests/unit` | monotone by construction |
| [ ] | Retarget the act head to "will a teacher call change top-k?" | — | plan Part IX, head 3 |
| [ ] | Close the trec-covid BM25 gap (−0.077) | `make check-first-stage` | F12 |
| [x] | Slate dataset: anchors mirror inference, order randomised | — | difficulty mix hard/mixed/easy |
| [ ] | Hard-negative mining across BM25 + dense + late-interaction | — | plan Part IX — "worth more than the loss" |
| [~] | Trainer on MPS (grad checkpointing, accumulation) | `make train` | first full run in flight |
| [ ] | Curriculum: warm start → anchor-aware → domain | — | plan Part IX |
| [ ] | **Re-run the whole Phase 0 battery on the tuned checkpoint** | `make phase0-all` | this is the real T4 test |

---

## Phase 2 — the frontier

> Gate: matches Qwen3-Reranker-4B quality at under 10% escalation.

| | Task | Command |
|---|---|---|
| [ ] | Temperature refit per (type, option-count) bucket, on our data | `make calibrate` |
| [ ] | Split-conformal intervals; report coverage, not just ECE | — |
| [ ] | Regret estimation wired to the tuned σ | — |
| [ ] | Fixed-threshold escalation to Jev over the frontier | — |
| [ ] | Distillation buffer: every teacher call becomes a label | — |

---

## Phase 3 — the controller

> Gate: the controller beats the best fixed threshold at matched cost. If not,
> report that and ship the threshold.

| | Task |
|---|---|
| [ ] | Offline counterfactual logging of every action |
| [ ] | Fit `V(s,a)` with LightGBM; act greedily |
| [ ] | `read_more` as a first-class action (the least prior art) |
| [ ] | λ_C sweep → the Pareto curve |

---

## Phase 4 — annealing

> Gate: quality rises at fixed escalation rate. Note the cost curve saturates
> below ~5% escalation, so the claim to test is *quality* annealing.

| | Task |
|---|---|
| [ ] | Periodic retrain on the buffer |
| [ ] | Track escalation rate and quality over rounds |

---

## Cross-cutting

| | Task | Note |
|---|---|---|
| [ ] | TREC DL19/20 loader | the only clean 4-level qrels; needs MS MARCO |
| [ ] | Dense retriever + RRF hybrid first stage | plan Part X holds this fixed across all rows |
| [ ] | Baselines: Qwen3-Reranker-0.6B/4B, bge-reranker, MiniLM | the honest competition |
| [ ] | Ablation ladder (plan Part IX) | the two that matter: no anchors, fixed threshold |
| [ ] | Pareto scatter: $/1M searches vs nDCG@10 | **the deliverable** — plan Part X |
| [ ] | Paired bootstrap + Holm–Bonferroni across the grid | 10k resamples |

---

## Decisions taken

| Date | Decision | Why |
|---|---|---|
| 2026-09-21 | STATE layout as default, not OPTIONS | F7: +0.27 nDCG@10 on the base checkpoint |
| 2026-09-21 | trec-covid as the Phase 0 corpus | only small BEIR set with explicit grade-0 judgements |
| 2026-09-21 | Jev via OpenRouter `/api/v1/systemone` | F5: works, reports real per-call cost |
| 2026-09-21 | Train on M4/MPS | user's constraint; caps the "strong specialist" scale in plan Part IX |
| 2026-09-21 | Teacher-graded pivots as default | F11: templates only weakly ordered; Jev gives continuous utilities on real text |
| 2026-09-21 | Train on nfcorpus, evaluate on trec-covid | 323 training queries vs 50; tests transfer, not fit |
| 2026-09-21 | Temperature OFF for Phase 0 | shipped map is fitted on the vendor's mix; rescaling changes what the anchor regression estimates |
