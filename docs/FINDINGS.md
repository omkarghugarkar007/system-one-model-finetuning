# Findings ledger

Every claim this project relies on, and its current evidential state. A claim
moves only when a run says so, and the run is named. `plan.md` is the design
document and is *not* silently edited to match results — where a finding
contradicts it, that is recorded here as a correction.

Status key: **confirmed** · **refuted** · **open** · **blocked**

---

## Corrections to `plan.md`

### F1 — Option text is capped at ~16 tokens at M=10 · **confirmed**

`plan.md` treats the ~320-token *state* budget as the binding constraint and
proposes signatures carrying "title, section path, metadata, the best lexical
window, the best dense window". That does not fit. Laya's `build_sequence`
caps each option at 48 tokens, and when the options overflow the 192-token head
budget it shrinks **every** option to `max(4, (192-16)//M)` tokens — 17 at
M = 10, of which one is the `[MASK]` marker.

So a candidate signature in the OPTIONS layout gets **16 tokens**. A title, and
not a long one.

Verified 16/16 exact against the shipped tokenizer (`make budget`).
Encoded in `core/budget.py`, checked by `models/laya/budget_probe.py`.

Two independent constraints now agree on M = 10: the `choice:11+ → T = 0.1006`
temperature cliff, and the point where option text falls to 16 tokens.

### F2 — The STATE layout truncates positionally and silently · **confirmed**

Putting candidates in the state instead of the options buys ~37 tokens each,
but `build_sequence` cuts the state with `st[:room]` — from the right. From
M = 6 upward with realistic passages the state overflows, so the **last**
candidates lose their text while keeping their markers: they are scored on
nothing, and nothing in the API says so.

`core/packing.StatePacker` pre-truncates every candidate to a fair share, so
the loss is even and reported rather than positional and invisible.

### F3 — "Marker noise" is not a real quantity · **confirmed**

`plan.md`'s simulation assumes marker noise σ = 0.35 and slate temperature
drift σ = 0.15. Laya is deterministic: identical input gives identical logits.
The real measurement floor is **option-order variation**, measured at
**0.38 nats** per candidate on trec-covid. Every residual in Phase 0 is judged
against that, not against a simulated noise term.

This matters because the plan's simulated SNR (noise four times smaller than
signal) is roughly inverted in reality: the c_S spread is ~1.0 nats against a
0.38-nat floor.

### F4 — Laya's order sensitivity is 33%, not 25% · **open**

`plan.md` cites 24.7% top-1 change under passage reordering for Jev's Choice,
against 92.6% for a generative reranker. Measured on Laya's Choice over
trec-covid slates: **32.8%** argmax change across 8 permutations. Same order of
magnitude, and still far better than a generative reranker, but it is not the
same number and Laya is not Jev. Sample is 8 queries.

### F5 — Jev is on OpenRouter, but invisible to the model listing · **confirmed**

`typesafe/jev-1.13` is served at `POST https://openrouter.ai/api/v1/systemone`
with the native TypeSafe `{state, model, questions}` shape, at the documented
$0.042/1M input with free output, and returns `usage.cost` per call. It does
**not** appear in `GET /api/v1/models`, because that listing only returns
`text->text` models and Jev's modality is `text->decisions`. Query
`/api/v1/models/typesafe/jev-1.13/endpoints` directly.

---

## Phase 0 gates

### F6 — `log p_{i,S} = l_i − c_S` holds on real text · **confirmed**

The load-bearing assumption. Fitted the two-way additive model over an
incomplete block design on trec-covid, 8 queries, 40 slates each.

| quantity | value |
|---|---|
| residual RMSE | 0.321 nats |
| order-noise floor | 0.382 nats |
| **residual / floor** | **0.84×** |
| c_S range (signal) | 1.018 nats |
| design power (signal / floor) | 2.67× |
| interaction R² (grade × slate-mean) | 0.0074 |

The residual sits *at* the measurement floor, and the interaction term — the
specific failure Part XI names as fatal, a candidate scoring differently
because of *which* others share its slate — explains 0.7% of what is left.

**Gate: PASS.** `c_S` exists, is a per-slate shift, and is additive.

Run: `runs/2026-09-21-phase0-trec-covid-options/`.

### F7 — The base checkpoint has signal; the OPTIONS layout throws it away · **confirmed (n=50, paired CIs)**

The first anchor sweep found anchoring bought nothing, with a naive
score/label correlation of only 0.14. Two explanations: no signal in the base
weights (which Laya's own card predicts — "a fast base to specialise, not a
zero-shot decision engine"), or the 16-token budget starving it. Distinguished
by varying the evidence budget.

trec-covid, **all 50 queries**, base checkpoint, no fine-tuning, no temperature.
Paired bootstrap over queries, 10k resamples:

| layout | signature | tok/cand | sig recall | corr(l, grade) | nDCG@10 | vs BM25 | 95% CI |
|---|---|---|---|---|---|---|---|
| **state** | **title+lexical** | **37** | **0.542** | **0.545** | **0.807** | **+0.232** | **[+0.151, +0.312]** |
| state | lexical | 37 | 0.498 | 0.426 | 0.706 | +0.131 | [+0.042, +0.221] |
| state | title | 37 | 0.424 | 0.366 | 0.693 | +0.118 | [+0.044, +0.194] |
| options | title | 16 | 0.334 | 0.247 | 0.585 | +0.010 | [−0.064, +0.082] |
| options | title+lexical | 16 | 0.251 | 0.141 | 0.541 | −0.034 | [−0.117, +0.049] |
| options | lexical | 16 | 0.270 | −0.020 | 0.461 | −0.114 | [−0.194, −0.033] |
| — | BM25 floor | — | — | — | 0.575 | — | — |

**Every STATE config beats BM25 with a CI excluding zero. No OPTIONS config
does.** The winner also beats every other config with CIs excluding zero.
Correlation spans −0.020 → 0.545 across evidence budgets. The bottleneck was
the slicer and the layout, not the weights.

**Read with care.** These nDCG@10 values are computed over a *balanced probe
set* of ~36 candidates per query drawn across all three grades, not a realistic
top-100 pool. They are not BEIR numbers and must not be compared to published
ones. The BM25 column is computed identically, so comparisons within the table
hold.

This is the largest design consequence found so far: `plan.md`'s architecture
implicitly assumes the OPTIONS layout, and that layout does not beat its own
first stage. **STATE is now the project default** (`docs/ROADMAP.md` decisions).

The n=8 pilot reported +0.27 against a BM25 floor of 0.455; at n=50 the floor
rises to 0.575 and the gap settles at +0.232 [+0.151, +0.312]. The direction
and significance held; the magnitude moved, which is why the pilot was rerun.

### F8 — Does anchoring buy the cross-slate scale? · **open (underpowered)**

Run three times, and the first two runs were invalid for reasons worth keeping.

**Run 1 (OPTIONS layout).** No benefit at any A. Invalid: F7 showed that layout
has almost no signal to calibrate (`r_naive` = 0.14).

**Run 2 (STATE layout).** Still no benefit, mean Δr ≈ −0.010. Invalid for two
defects in the test itself, both visible in the output as slopes collapsing
1.00 → 0.58 and residuals growing 0 → 0.64:

1. *The ridge prior was wrong.* `fit_slate` shrank the slope toward 1.0, which
   is only correct when utilities are expressed in the same units as
   log-probabilities. On a 0–2 grade scale against log-probs spanning ~3 nats
   the true slope is ~0.67, so every slate was pulled the same wrong way — a
   bias indistinguishable from "anchoring does not help". `slope_prior` is now
   a parameter and is estimated from the data.
2. *Anchors were spread over the latent logit, not the known utility.* Pivots
   picked for latent spread can all carry the same grade, leaving the
   regression one distinct x-value and no leverage. The plan says pivots should
   "span the utility range"; now they do.

**Run 3 (corrected).** trec-covid, STATE layout, 10 queries:

| A | distinct anchor utilities | r naive | r anchored | Δr |
|---|---|---|---|---|
| 1 | 1.0 | 0.274 | 0.290 | +0.015 |
| 2 | 2.0 | 0.142 | 0.145 | +0.003 |
| 3 | 2.8 | 0.152 | 0.147 | −0.005 |
| 4 | 2.9 | 0.187 | 0.197 | +0.010 |
| 6 | 3.0 | 0.189 | 0.216 | +0.026 |

Mean Δr = **+0.010**, per-query SE = **0.013**, n = 10 queries.

Consistently positive in 4 of 5 cells, and smaller than its own standard error.
**Test 4 is underpowered, not failed.** It neither confirms nor refutes the
claim.

**The binding limit is identified**, and it is not the mechanism. The last
column is the story: trec-covid has three grades, so the anchor regression has
at most **three distinct utility values** to fit an affine map through, against
a 0.38-nat measurement floor. `plan.md`'s simulation drew continuous latent
utilities — a regime no BEIR-style corpus provides, and the reason the
simulated effect (r 0.918 → 0.961) is far larger than anything measurable here.

**Concrete fix for the next run:** give pivots *continuous* utilities. Jev's
Score returns a fractional expected grade (2.85, not 3), so teacher-graded
pivots have real leverage where qrel grades have almost none — and that is what
production would use anyway. TREC DL's 4-level scale would also help.

### F11 — Templated pivots are only weakly ordered by the base model · **confirmed**

`training.anchors` proposed pivots whose grade is known by construction, since
no fixed document has a fixed relevance to every query. Probed on trec-covid,
20 queries, 120 slates:

| pivot | designed utility | mean log p | mean rank (of 4) |
|---|---|---|---|
| off_topic | −3.0 | −3.64 | 3.73 |
| topical_nonanswer | −1.0 | **−1.89** | **1.55** |
| partial | +1.0 | −2.51 | 2.34 |
| direct_answer | +3.0 | −2.39 | 2.38 |

Rank correlation with the designed order **0.33**; exact order in **2.5%** of
slates. Only the grade-0 pivot lands where intended.

Two causes, and only one is the model's fault. The base checkpoint responds to
topical overlap rather than to answering — consistent with F7, where its signal
is real but modest. But the grade-3 template is also badly designed: it
*claims* to answer the query without containing an answer, and a relevance
model is correct not to reward a self-referential claim. Synthetic text cannot
carry real relevance at the top of the scale, because a passage that answers a
question has to contain the answer, and only the corpus has that.

**Resolution: teacher-graded pivots.** Real documents, graded by Jev, whose
Score returns a *fractional* expected grade. Continuous utilities (fixing F8's
leverage problem), real relevance, one cached call per query at ~$0.0003, and
the scale is fixed corpus-wide even though the pivot documents differ per
query. Implemented as `AnchorPool(strategy="teacher")` and now the default.

The templated set is kept as an ablation, and `anchor_affine_loss` gives a
falsifiable prediction: if training for anchor alignment works, this rank
correlation should rise from 0.33 toward 1.0 on the tuned checkpoint.

---

## Measured engineering facts

### F12 — The first stage is validated against published BEIR numbers · **confirmed**

The plan's own thesis is that the first stage moves the metric more than the
reranker does, so a gain measured on a broken floor is not a gain.
`scripts/check_first_stage.py` checks BM25 against Thakur et al.:

| dataset | ours | published | delta |
|---|---|---|---|
| nfcorpus | 0.3235 | 0.325 | −0.002 |
| scifact | 0.6759 | 0.665 | +0.011 |
| trec-covid | 0.5794 | 0.656 | **−0.077** |

Adding Porter stemming (Anserini's default, and BEIR's published numbers are
Anserini's) moved nfcorpus +0.016 and trec-covid +0.016. Two of three now match
to within 0.011.

**trec-covid remains 0.077 low and this is not yet explained.** Since it is the
primary evaluation corpus, the consequence must be stated: a floor that is too
low makes every reranking gain measured against it look slightly better than it
should. The gap is constant across rows, so *comparisons between rerankers*
remain valid; *absolute* "beats BM25 by X" claims on trec-covid carry this
caveat until it is closed.



### F9 — Laya on an M4 · **confirmed**

421.3M parameters (394.8M encoder), loads in ~26 s, runs in fp32 on MPS.

| layout | ms per slate (batched 16) | slates/s |
|---|---|---|
| OPTIONS | 59 | 17.0 |
| STATE | 107 | 9.4 |

At 17 slates per 100-candidate query that is **1.0 s/query** (OPTIONS) or
**1.8 s/query** (STATE) on an M4. `plan.md` models a T4 at ~14 ms/question;
these are the numbers for this project's actual hardware.

### F10 — The reimplementation is faithful · **confirmed**

`models/laya/runtime.DecisionModel` loads the shipped checkpoint with
`strict=True` and reproduces TypeSafe's own documented example: Jev returns
noul 0.95 for "Help! My payouts have been failing for 3 days"; this
implementation returns **0.9237**.
