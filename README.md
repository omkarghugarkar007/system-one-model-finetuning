# systemone — a recipe for fine-tuning System One models

System One models ([Laya](https://huggingface.co/convaiinnovations/laya),
[Jev](https://docs.typesafe.ai/)) read a state and return **typed decisions with
calibrated probabilities** instead of text. They are small, fast and
self-hostable — and they are explicitly not useful out of the box. Laya's own
model card puts its base checkpoint *below the majority-class baseline* and says
plainly: *"a fast base to specialise, not a zero-shot decision engine."*

So the useful artifact is not the model. It is the recipe.

This repo is that recipe, extracted from a reranking research project and
generalised. It is not a leaderboard entry — see [FINDINGS.md](FINDINGS.md),
where a 22M cross-encoder beats our fine-tuned 422M model on ranking quality.
What it is: a **worked, measured, honestly-reported account of how to fine-tune
these models**, including the parts that did not work.

---

## Quickstart

```bash
git clone <this repo> && cd systemone
uv venv --python 3.12 .venv && source .venv/bin/activate
uv pip install -e '.[train]'

# Laya weights, ~1.7 GB
python -c "from huggingface_hub import snapshot_download as d; \
d('convaiinnovations/laya', local_dir='data/cache/laya', \
allow_patterns=['model.safetensors','rl_agent_config.json','encoder/*','tokenizer/*'])"

python examples/01_quickstart.py
```

About a minute on a laptop, no GPU. It fine-tunes on an ordinal severity rubric
where **the rubric deliberately disagrees with the tone** of the report — a
calmly worded outage is severity 3, a furious complaint about an internal
dashboard is severity 1. The base checkpoint reads tone, because that is what
general pretraining gives it. Training teaches it *your* rubric:

| | accuracy | ECE | Brier | AURC |
|---|---|---|---|---|
| base checkpoint | 0.679 | 0.172 | 0.580 | 0.204 |
| **after ~55s of fine-tuning** | **0.839** | **0.094** | **0.176** | **0.019** |
| majority-class baseline | 0.268 | | | |

## Your own data

Three question types, the same shapes the inference API takes:

```python
from systemone import choice, score, noul, TypedDataset

examples = [
    choice("Payouts have been failing for 3 days",
           "Which team should handle this?",
           {"billing": "payments, invoices, refunds",
            "technical": "bugs, outages, integrations"},
           label="technical"),

    score("Search takes 8 seconds for half of accounts",
          "Rate the severity.",
          ["no impact", "internal only", "some customers", "full outage"],
          label=2),                       # levels are ORDERED; index is the score

    noul("I want my money back", "Does this request a refund?", label=True),
]
```

or from JSONL with `systemone.data.load_jsonl`. Then:

```python
from systemone.model import LayaRuntime, SystemOneModel
from systemone.train.trainer import TypedTrainer, TypedTrainConfig

rt = LayaRuntime("data/cache/laya")
model = SystemOneModel(rt.model, n_levels=4)
ds = TypedDataset(examples, rt.tok)

trainer = TypedTrainer(model, rt.tok, TypedTrainConfig(warmup_epochs=3))
trainer.preflight(ds)      # ALWAYS. see below.
trainer.fit(ds)
```

## The five things that will bite you

Each of these cost us a run. [RECIPE.md](RECIPE.md) has the detail.

1. **Your data may never reach the model.** The head budget is 192 tokens
   shared across *all* options, and long options are silently shrunk — at 10
   options each keeps **16 tokens**. `trainer.preflight(ds)` measures this and
   refuses to train if the state is being truncated. Run it first, always.

2. **Where you put the evidence changes everything.** Putting candidates in the
   *options* versus in the *state* moved our correlation with ground truth from
   0.20 to 0.48 and nDCG@10 from 0.49 to 0.76 — same model, same weights.

3. **Ordinal levels are not categories.** `score` levels are ordered, so being
   off by one should cost less than being off by three. Plain cross-entropy
   cannot know that; `TypedDecisionLoss` applies the ranked probability score
   to `score` questions automatically, and never shuffles their order.

4. **Aggregate ECE lies.** Laya's own eval reports 0.030 aggregate while
   concealing a task family at 0.438. `systemone.eval.report()` always prints
   the worst slice and the ratio.

5. **Training does not calibrate.** Fit a temperature per `(type, option-count)`
   bucket on a held-out split afterwards. Minutes, and it is the difference
   between probabilities you can threshold and numbers that merely rank.

## What's here

```
src/systemone/
  data/        typed examples (choice/score/noul) + token-budget diagnostics
  model/       faithful sequence builder, decision head, an ordinal CORAL head
               the vendor does not ship, and the token arithmetic
  train/       the objective and a trainer sized for a laptop
  calibrate/   per-bucket temperature, conformal intervals
  eval/        accuracy, ECE per slice, Brier, risk-coverage
  teachers/    distillation sources behind one protocol (Jev, human, cached)
  reranking/   THE WORKED EXAMPLE — the research this came from
examples/      start here
docs/vendor/   pinned upstream source and docs, so every claim is checkable
```

- **[RECIPE.md](RECIPE.md)** — the cookbook. Read before your first real run.
- **[FINDINGS.md](FINDINGS.md)** — everything we measured, including the
  negative results and the four bugs we shipped and caught.
- **[docs/ROADMAP.md](docs/ROADMAP.md)** — what is done and what is not.

## When *not* to use a System One model

Stated plainly, because the evidence here says so:

- **You need the best possible accuracy.** A pretrained cross-encoder beats a
  fine-tuned Laya on ranking by +0.097 nDCG@10 at 1/19th the parameters
  (FINDINGS F16). If you only need a score, use one.
- **Your task needs reasoning, or per-query instructions.** Sub-1B models are
  unambiguously weak here; a 4B reranker is the right tool.
- **You cannot produce a few thousand labelled examples.** Base checkpoints are
  near or below the majority-class baseline. Fine-tuning is a precondition,
  not an enhancement.
- **Your inputs are long.** The context is 512 tokens total. Evidence selection
  becomes your bottleneck, and no amount of training fixes it.

Use one when you want **calibrated probabilities from a small self-hosted model
on your own decision boundary** — thresholding, routing, escalation, or any
place where "how sure are you?" has to be a number you can act on.

## Credits and licence

Apache-2.0. `systemone/model/sequence.py` and `runtime.py` are faithful
re-implementations of `rl_common.py` from
[`convaiinnovations/laya`](https://huggingface.co/convaiinnovations/laya)
(Apache-2.0); a pinned copy of the upstream source is in `docs/vendor/`.
The typed-decision API shape follows [TypeSafe's](https://docs.typesafe.ai/)
System One request format.
