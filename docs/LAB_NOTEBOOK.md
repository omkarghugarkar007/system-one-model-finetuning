# Lab notebook

Dated record of what was run and what it said. Entries are append-only;
corrections go in a later entry rather than rewriting an earlier one.

---

## 2026-09-21 — project start, Phase 0

### Read the vendor source before writing any code

Pulled `rl_common.py`, `rl_agent_api.py` and the configs from
`convaiinnovations/laya` into `docs/vendor/`. Three things came out of that
which the design document does not have:

1. `build_sequence` caps option text at 48 tokens, then shrinks **all** options
   to `max(4, (192-16)//M)` when they overflow the head budget. At M = 10 a
   candidate gets 16 tokens. This is the constraint that decides the
   architecture, and it is not in `plan.md`. → F1
2. The state is truncated from the right, so a candidates-in-state layout
   silently deletes its **last** candidates. → F2
3. `rl_agent_config.json` confirms `max_len=512`, `head_max_len=192`,
   `head_layers=2`, and the `choice:11+ → T=0.1006` temperature that justifies
   M = 10.

Wrote `core/budget.py` to do the arithmetic without a tokenizer, then
`models/laya/budget_probe.py` to check it against the real one. **16/16 exact.**
The formula is trustworthy and the test will fail loudly if a Laya release
changes it.

### Faithful reimplementation

`models/laya/runtime.DecisionModel` reproduces the shipped module tree and
loads the checkpoint with `strict=True`. 421.3M params. Reproduces TypeSafe's
own documented example (noul 0.9237 vs Jev's 0.95). → F10

### Teacher: Jev is on OpenRouter after all

First pass over `GET /api/v1/models` found zero `jev`/`typesafe` hits and I
concluded it was unavailable — wrong. The listing only returns `text->text`
models; Jev's modality is `text->decisions`. It is served at
`POST /api/v1/systemone` with the native TypeSafe shape and returns
`usage.cost`. Verified with a live call ($0.0000119). → F5

Lesson: query the specific model endpoint, not the catalogue.

Built four teacher backends behind one protocol: Jev (OpenRouter or direct),
`InSessionTeacher` (file round-trip so a human grades on the same interface),
`SimulatedTeacher`, and a content-addressed `CachedTeacher` so no judgement is
ever paid for twice.

### Corpus choice

Checked grade distributions before committing:

| dataset | docs | queries | grades | judgements/query (median) |
|---|---|---|---|---|
| nfcorpus | 3.6k | 323 | {1, 2} only | 16 |
| scifact | 5.2k | 300 | binary | 1 |
| trec-covid | 171k | 50 | {0, 1, 2} | 1268 |

nfcorpus has **no explicit grade-0 judgements**, so slate composition barely
varies and c_S has nothing to move. A first Phase 0 run there came back with
c_S range 0.28 against a 0.29 noise floor — no contrast, no power.

That also exposed a bug in my own gate logic: it was failing on R², which is
`1 - SS_res/SS_tot`, and SS_tot is inflated by the spread of `l_i`. A design
where c_S barely moves gives a low R² however well the model holds. Gate
rewritten to check **design power first** and report INCONCLUSIVE when the
slates were not different enough to be informative.

trec-covid is the right Phase 0 corpus.

### T2 — the gate: PASS

Two-way additive fit `y_{i,S} = l_i - c_S`, 8 queries × 40 slates.
Residual 0.321 nats against a 0.382-nat order-noise floor — **0.84×**, i.e. the
model explains the data down to the measurement limit. Interaction term
explains 0.7% of the residual. → F6

Worth stating plainly: because Laya is deterministic, `plan.md`'s "marker
noise" does not exist. The honest floor is option-order variation, and every
residual here is judged against that. → F3

### T4 — anchoring: a false negative, then a real finding

First anchor sweep (OPTIONS layout, A ∈ {1,2,3,4,6}) showed anchoring buying
nothing — best Δr = +0.018, negative at most A. But `r_naive` was only 0.14,
which means there was almost no signal to put on a common scale.

Two candidate explanations, and they call for opposite work: the base
checkpoint is near-chance (Laya's card says so), or the 16-token budget starved
it. Ran the evidence-budget sweep to separate them.

**It was starvation.** corr(l, grade) moves 0.024 → 0.484 purely by changing
layout and slicer; nDCG@10 on the probe set goes 0.490 → 0.761 against a BM25
floor of 0.455 — on the *base* checkpoint, no fine-tuning. → F7

So `plan.md`'s implicit OPTIONS layout is the wrong one, and the anchor sweep
has to be re-run where signal exists.

Caveat recorded in F7 and not to be forgotten: those nDCG@10 values are over a
balanced 36-candidate probe set, not a top-100 pool. They are not BEIR numbers.
n = 8 queries.

### Open

- Re-run the anchor sweep on the STATE layout (in flight).
- The SNR concern stands regardless: order noise 0.38 nats against a c_S spread
  of ~1.0 means each anchor measures the offset with error comparable to the
  signal. `plan.md`'s simulation assumed the opposite regime.
- Nothing is fine-tuned yet. Phase 1 is the precondition for most of the plan.
