# FrontierRank

Reranking that spends compute in proportion to **ranking regret** rather than
candidate count. A small local typed-decision model (Laya, 421M, Apache-2.0)
scores every candidate; a hosted large-slate teacher (Jev) is called only on
the uncertain top-k frontier; every teacher call becomes training data.

`plan.md` is the design document. This README is how to run it.

## Status

| Phase | Gate | State |
|---|---|---|
| 0 | `c_S` is a per-slate shift; anchors recover it | **T2 passed, T4 blocked** — see [docs/FINDINGS.md](docs/FINDINGS.md) |
| 1 | anchored beats naive after fine-tuning | not started |
| 2 | matches a 4B reranker under 10% escalation | not started |
| 3 | controller beats the best fixed threshold | not started |
| 4 | quality rises at fixed escalation | not started |

The headline so far: the **mechanism** is confirmed on real text — the slate
offset behaves exactly as the theory says — but the **base checkpoint has
almost no relevance signal to put on a common scale**, which is what Laya's own
model card predicts. Fine-tuning is not an enhancement here, it is a
precondition.

## Setup

```bash
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e '.[model,data,eval,dev]'

# Laya weights (~1.7 GB) into data/cache/laya
python -c "from huggingface_hub import snapshot_download as d; d('convaiinnovations/laya', local_dir='data/cache/laya', allow_patterns=['model.safetensors','rl_agent_config.json','encoder/*','tokenizer/*','eval/results.json'])"

# BEIR corpora
make data
```

For the teacher, put `OPENROUTER_API_KEY` in `.env` (see `.env.example`).

## Running

```bash
make test           # unit tests, seconds, no weights
make budget         # the token arithmetic that drove the slate design
make phase0         # the go/no-go gate on trec-covid
make phase0-signal  # is there signal, or did we starve the model?
make sim            # the numpy-only simulation from plan.md Part VIII
```

Every run writes `runs/<date>-<name>/` with `manifest.json` (config, commit,
device, versions), `metrics.json` and a human-readable `report.txt`.

## Layout

```
src/frontierrank/
  core/          numpy-only theory. No model, no network, no corpus.
    budget.py      token accounting for Laya's build_sequence  <- drove the design
    packing.py     OPTIONS vs STATE slate layouts
    slates.py      slate construction under the option-slot budget
    calibrate.py   anchor estimators: shift, affine OLS, ridge
    scoring.py     AnchoredScorer: slates -> one comparable utility + sigma
    frontier.py    top-k regret frontier, swap probabilities
    controller.py  EVI action choice over stop / read_more / widen / teacher
  models/
    protocols.py   ScorerProtocol + TeacherProtocol. The only vendor boundary.
    laya/          the real checkpoint: vendored sequence builder, runtime, scorer
    teachers/      Jev (via OpenRouter), in-session human, cache, simulated
  data/          BEIR loaders, BM25, the evidence slicer
  training/      the composite objective
  eval/          metrics with TREC conventions enforced
  experiments/   one module per phase gate
docs/
  FINDINGS.md    the claims ledger: confirmed, refuted, open
  LAB_NOTEBOOK.md dated record of what was run and what it said
  vendor/        pinned third-party source and docs the design cites
```

## The two interfaces

Everything swappable sits behind two protocols, and nothing above them names a
vendor:

```python
class ScorerProtocol:      # the student
    def choice_logprobs(self, instructions, options, state) -> np.ndarray: ...

class TeacherProtocol:     # the escalation target
    def grade(self, query, signatures, rubric) -> TeacherVerdict: ...
```

Jev is closed-weights and early access, so all vendor risk sits on
`TeacherProtocol`. `InSessionTeacher` puts a human on the same interface, which
is how the small high-value sets get graded.

## Constraints encoded in code, not just prose

- **M = 10 option slots.** Two independent reasons agree: Laya ships
  `choice:11+ -> T = 0.1006`, a 10x sharpening that only fits near-uniform
  logits; and M = 10 is exactly where option text hits 16 tokens.
- **16 tokens of option text per candidate at M = 10.** Verified against the
  real tokenizer, 16/16 exact (`make budget`). This is the constraint that
  decides the architecture, and it is not in `plan.md`.
- **State is not shared across questions.** `build_sequence` re-appends the
  full state per question, so a 100-candidate query is 17 independent
  512-token sequences. Do not fan out per candidate.
- **Jev caps at 255 options** and its latency is flat in option count, so the
  teacher gets the whole frontier.
- **TREC DL grade 1 is NOT relevant.** `recall_at_k`/`mrr_at_k` take an
  explicit threshold, matching `trec_eval -l 2`.
