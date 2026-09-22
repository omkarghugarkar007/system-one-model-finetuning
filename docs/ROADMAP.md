# Roadmap

The repo has been reframed as a **recipe for fine-tuning System One models**
([RECIPE.md](../RECIPE.md)); the reranking work below is now the worked example
that produced it. Phases 0–1 are done and published; 2–4 are built but not
measured.

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
| [x] | **T4 — does anchoring buy the cross-slate scale?** | `make phase0-anchors` | **F8 RESOLVED — yes, but only when composition varies** |
| [x] | Templated pivots + probe: are they actually pivots? | `make anchor-probe` | F11 — only weakly ordered (0.33) |
| [x] | Teacher-graded pivots: real text, continuous utilities | — | F11 — now the default |
| [x] | Re-run T4 with teacher pivots (continuous utilities) | `make phase0-anchors` | F8 — leverage problem fixed, SE 3× tighter |
| [ ] | T5 — does anchor *choice* swing results >1 nDCG point? | — | plan Part VIII test 5 |
| [ ] | Confirm F8 on a second corpus before generalising | — | n=14, one corpus |

**Why T4 is blocked, not failed:** trec-covid has three grades, so the anchor
regression has three distinct x-values against a 0.38-nat noise floor. Jev's
Score returns a *fractional* expected grade, which gives pivots real leverage.
That is also what production would use.

---

## Phase 1 — anchored scoring  ·  **next**

> Gate (as written in plan.md Part X): anchored beats naive by a large margin,
> and beats a pointwise rubric at equal or lower pass count.
>
> **Corrected.** That gate contradicts plan.md's own Part VIII, which measures
> stratified A=4 at naive 0.895 vs anchored 0.897 and concludes anchors buy the
> *cross-query scale*, not within-query ranking. Measured here: +0.0004 — a tie,
> matching the simulation's +0.002. The operative gate is falsification test 4.

| | Task | Command | Gate / output |
|---|---|---|---|
| [x] | Full reranking pipeline on a realistic top-100 pool | `make train` | via `make_eval_fn`; pools frozen |
| [x] | Validate BM25 against published BEIR numbers | `make check-first-stage` | F12 — 2/3 within 0.011 |
| [x] | Ordinal CORAL head (`P(y > k)` on the 0–3 scale) | `pytest tests/unit` | monotone by construction |
| [ ] | Retarget the act head to "will a teacher call change top-k?" | — | plan Part IX, head 3 |
| [ ] | Close the trec-covid BM25 gap (−0.077) | `make check-first-stage` | F12 |
| [x] | Slate dataset: anchors mirror inference, order randomised | — | difficulty mix hard/mixed/easy |
| [ ] | Hard-negative mining across BM25 + dense + late-interaction | — | plan Part IX — "worth more than the loss" |
| [x] | Trainer on MPS (grad checkpointing, accumulation) | `make train` | F13 — +0.152, beats BM25 by +0.079 |
| [x] | Memory levers: layer freezing, periodic saves | — | F14 — 6.76 GB all-trainable swaps a 16 GB M4 |
| [x] | **Falsification test 4 on the TUNED checkpoint** | `make phase0-anchors` | F8 — +0.181 ± 0.026 when c_S varies; null when it does not |
| [x] | Anchor probe on the tuned checkpoint | `make anchor-probe` | F14 — rank corr 0.33 → 0.795, transferred |
| [ ] | Curriculum: warm start → anchor-aware → domain | — | plan Part IX |
| [ ] | **Re-run the whole Phase 0 battery on the tuned checkpoint** | `make phase0-all` | this is the real T4 test |

---

## Recipe (the published artifact)

| | Task | Where |
|---|---|---|
| [x] | Generic typed-decision API (choice/score/noul) | `systemone.data.typed` |
| [x] | Generic objective with ordinal handling | `systemone.train.losses.TypedDecisionLoss` |
| [x] | Generic trainer with preflight budget check | `systemone.train.trainer` |
| [x] | Sliced evaluation (accuracy, ECE, Brier, AURC) | `systemone.eval.typed` |
| [x] | One-minute end-to-end quickstart | `examples/01_quickstart.py` |
| [x] | The cookbook | `RECIPE.md` |
| [ ] | Second example: distillation from a teacher | `examples/02_*.py` |
| [ ] | Second example: bring-your-own JSONL | `examples/03_*.py` |
| [ ] | Confirm the recipe on a non-reranking task | — |

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

## Headline result so far

**Anchoring is insurance, not an upgrade.** It makes the utility scale
*invariant to slate composition* rather than better: `r_anchored` holds at
~0.43–0.49 whether slates are uniform or heterogeneous, while `r_naive`
collapses from ~0.48 to ~0.20. So under good slate construction anchors cost
1.7–2.5× the forward passes for nothing, and they earn their slots exactly
where composition cannot be controlled — streaming, merged pools after
`widen_retrieval`, and multi-round re-scoring after `read_more` or escalation.

**Default changed:** stratified slates *without* anchors (10 candidates/slate,
10 passes per 100), anchors switched on only for the multi-round and merged-pool
paths. That is a 1.7× local compute reduction against the plan's A=4 default.

## Decisions taken

| Date | Decision | Why |
|---|---|---|
| 2026-09-21 | STATE layout as default, not OPTIONS | F7: +0.27 nDCG@10 on the base checkpoint |
| 2026-09-21 | trec-covid as the Phase 0 corpus | only small BEIR set with explicit grade-0 judgements |
| 2026-09-21 | Jev via OpenRouter `/api/v1/systemone` | F5: works, reports real per-call cost |
| 2026-09-21 | Train on M4/MPS | user's constraint; caps the "strong specialist" scale in plan Part IX |
| 2026-09-21 | Teacher-graded pivots as default | F11: templates only weakly ordered; Jev gives continuous utilities on real text |
| 2026-09-21 | Anchors OFF by default; ON for multi-round/merged pools | F8: +0.181 when composition varies, null when it does not |
| 2026-09-21 | Phase 1 gate rewritten to falsification test 4 | plan Part X's gate contradicts its own Part VIII |
| 2026-09-21 | Train on nfcorpus, evaluate on trec-covid | 323 training queries vs 50; tests transfer, not fit |
| 2026-09-21 | Temperature OFF for Phase 0 | shipped map is fitted on the vendor's mix; rescaling changes what the anchor regression estimates |
