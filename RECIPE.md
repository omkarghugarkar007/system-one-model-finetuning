# The recipe

How to fine-tune a System One model, in the order you should do it. Every
number here is measured in this repo; the runs are in [FINDINGS.md](FINDINGS.md).

---

## 0. The mental model

A System One model is a **bidirectional encoder with a decision head**, not a
language model. One sequence is built per question:

```
[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
```

Each option gets a `[MASK]` marker. A single shared scalar MLP scores every
marker's hidden state, and one softmax runs across that question's markers.
There is no decoding, no permutation string, and no format failures by
construction.

Three consequences you have to design around:

- **Options and instructions share a 192-token head budget**, and the state gets
  what is left of 512. This is the constraint that decides everything.
- **State is not shared across questions.** Asking five questions about one
  state runs the encoder five times, re-reading the state each time. Do not
  fan out per item if you can batch into one question.
- **The softmax is normalised within the question.** Scores are comparable
  *inside* one question and not across questions. If you need a globally
  comparable number, you need a pointwise ordinal head (§5) — this is why we
  added one.

---

## 1. Design the question before you collect data

Pick the type by what you need back, not by what feels natural:

| type | use when | returns | notes |
|---|---|---|---|
| `choice` | one of N named options | the option + full distribution | order is meaningless, so it gets shuffled |
| `score` | a rating on **ordered** levels | fractional expected score + per-level distribution | order is meaningful, never shuffled |
| `noul` | a yes/no question | one probability | two options under the hood |

**Ask one thing per question.** "Is this urgent and customer-facing?" is two
questions wearing one coat, and the model will answer whichever half is more
salient. Split it and combine in code.

**Write the criteria as decision boundaries, not as labels.** `"billing"` tells
the model nothing; `"billing: payments, invoices, refunds, subscription
charges"` tells it where the boundary is. The criteria text is *read*, and it is
the cheapest quality you can buy.

**Put the boundary cases in the criteria.** If you look at a wrong answer and
find yourself explaining what you really meant, that explanation is the missing
half of your instructions.

---

## 2. Check the token budget. Before anything else.

This is the step people skip and it is the one that silently wastes runs.

```python
trainer.preflight(dataset)     # raises if the state is being truncated
```

The arithmetic, from the shipped `build_sequence`:

```
opt_ids[i] = [MASK] + tokens(option_i)[:48]        # 48-token hard cap per option
if 192 - sum(len(opt_ids)) < 16:                   # options overflow the head
    per = max(4, (192 - 16) // n_options)          # EVERY option is shrunk
```

What survives, per option (verified exactly against the real tokenizer):

| options | tokens of option text each |
|---|---|
| 2–3 | 48 (the cap) |
| 4 | 43 |
| 6 | 28 |
| 8 | 21 |
| **10** | **16** |
| 16 | 10 |
| 20 | 7 |
| 30+ | 3 (the floor) |

**At 10 options each one keeps 16 tokens.** Not approximately — the shrink
applies to every option whether it needed it or not. If your options are long
descriptions, you are training on their first few words.

`systemone.model.budget.max_options_for_option_text(n)` inverts this: "I need
30 tokens per option, how many fit?" (answer: 6).

Two independent constraints happen to agree on **M = 10 as the usable ceiling**:
this budget, and the shipped `temperature_by_options` map, which needs
`T = 0.1006` for 11+ options — a 10× sharpening that only makes sense if the
raw logits were near-uniform.

---

## 3. Decide where the evidence goes

This mattered more than anything else we changed. Two layouts:

- **In the options** — the marker sits inside the item's own text. Local and
  direct, but capped by the head budget above.
- **In the state** — options are bare ids (`"c0"`), the state carries the
  content. ~2.5× more tokens per item, but the marker sits on a semantically
  empty token and everything must arrive by attention.

Measured on the same model, same weights, 50 queries:

| layout | tokens/item | correlation with truth | nDCG@10 |
|---|---|---|---|
| options | 16 | 0.20 | 0.49 |
| **state** | **37** | **0.48** | **0.76** |

Neither is obviously right a priori; **measure both on your task.** If your
items are short (labels, names, short spans) options is simpler and better. If
they are passages, state wins.

**One trap in the state layout:** the state is truncated from the *right*, so a
list that overflows loses its **last** items entirely — they still get markers
and are still scored, on nothing. `systemone.reranking.packing.StatePacker`
pre-truncates evenly so the loss is visible instead of positional.

---

## 4. Labels: how many, and how good

| stage | examples | what you get |
|---|---|---|
| does it learn at all | 300–1,000 | beats the majority-class baseline |
| useful specialist | 3k–10k | beats a generic model on your boundary |
| production | 30k+ | diminishing returns without harder data |

**Compare against the majority-class baseline, never against chance.** A model
below majority is worse than a constant. `eval.report()` prints it.

**Soft labels are conditional, not automatic.** If you have a teacher, you can
store its whole distribution:

```python
score(text, "Rate severity.", LEVELS, soft_label=[0.01, 0.04, 0.21, 0.74])
```

The usual argument is that `P(3)=0.74` with mass on 2 tells the student where
its uncertainty belongs, while the argmax throws that away. That is true *when
the teacher knows something your labels do not.* We measured it on the bundled
routing task and got a dead heat — ECE 0.1217 hard vs 0.1275 soft — because the
teacher agreed with the labels almost perfectly and so carried no extra
information. `examples/03_distill_from_a_teacher.py` runs the three-arm
comparison; run it on your data rather than assuming.

Two things to watch:

- **Distillation transfers the teacher's confidence LEVEL, not only its
  ordering.** A teacher that is uncertain about *your bespoke criteria* — as a
  zero-shot teacher usually is — will train a well-calibrated student to be
  underconfident. Sharpen the teacher distribution (`--sharpen`) when its
  spread reflects its own ignorance rather than the item's ambiguity.
- **Ask the teacher the same question you ask the student.** `teachers.JevTeacher.ask()`
  posts the actual typed question. The reranking-shaped `grade()` takes
  candidates and a rubric; pushing a classification question through it asks
  something nobody meant. We did exactly that once, and the resulting
  "distillation hurts calibration" finding was entirely an artifact of it.

`systemone.teachers` has Jev, a content-addressed cache, and an in-session
human grader on the same interface.

**Mine hard negatives from several sources.** Negatives from one retriever or
one heuristic teach that source's failure modes rather than your boundary. The
highest-value example is a confident disagreement between your model and your
teacher — and it is free, because you already paid for the teacher call.

---

## 5. Heads and loss

`SystemOneModel` gives you three outputs on one backbone:

- **slate head** — the shipped shared scalar MLP over markers. Trained with
  cross-entropy against your target distribution.
- **ordinal head** — a CORAL head emitting `K-1` cumulative logits `P(y > k)`.
  **Not in the upstream model.** The `score` primitive renders levels as
  ordinary Choice options and only the reward knows they are ordered, so the
  model can put mass on levels 0 and 3 and none on 1 and 2. A CORAL head makes
  grades monotone by construction and gives a globally comparable pointwise
  score, which a within-question softmax structurally cannot.
- **act head** — the shipped escalation head. Retarget it to whatever "should I
  escalate?" means for you.

The default objective is `TypedDecisionLoss`:

```
L = CE(target) + RPS(score questions only) [+ optional confidence penalty]
```

RPS is strictly proper *and* ordinal-aware. It is applied only to `score`
questions, because a CDF over shuffled `choice` options is meaningless.

---

## 6. Train

Two stages, and **the order is not cosmetic**:

```python
TypedTrainConfig(warmup_epochs=1,   # encoder FROZEN
                 full_epochs=2)     # everything trainable
```

Any head you add starts from noise. Letting its gradients into a pretrained
encoder on step one destroys what the backbone already knows.

**Memory is the binding constraint on a laptop.** AdamW keeps two fp32 moments
per trainable parameter:

| configuration | trainable | GB before activations |
|---|---|---|
| everything | 422M | **6.76** |
| bottom 14 of 28 layers frozen | 199M | 4.08 |
| encoder frozen | 28M | 2.02 |

On a 16 GB machine the first row swaps, and throughput collapses from
27 s/step to 210 s/step. Use `freeze_lower_layers=14` — the bottom of a
pretrained encoder does general language and the task-specific work happens
near the top. Set `save_every` too: a long run on constrained hardware that
only saves at the end will eventually lose everything.

Starting points: `lr_head=1e-4` (frozen: up to 5e-4), `lr_encoder=1e-5`,
effective batch 32 via accumulation, fp32. Autocast on MPS is not worth the
numerical risk for a model whose product is calibrated probabilities.

---

## 7. Calibrate. Separately, afterwards.

Training does not make a model calibrated and often makes it worse.

```python
from systemone.calibrate import TemperatureMap
temps = TemperatureMap.fit(records, min_samples=50)   # HELD-OUT data
```

**Fit per `(type, option-count)` bucket, not globally.** The shipped map spans
`T = 0.1006` to `T = 1.98`; one scalar cannot serve both ends. Buckets with too
few samples fall back to the global fit rather than fitting noise.

**Report per slice, never in aggregate.** Laya's own eval shows an aggregate
ECE of 0.030 concealing a family at 0.438. `eval.report()` prints the worst
slice and the ratio, and warns when the aggregate is flattering you.

If you need *intervals* rather than point probabilities — thresholding,
expected-value arithmetic, deciding whether another judgement is worth buying —
use `calibrate.ConformalIntervals`. ECE is about the probability of the argmax;
a model can have excellent ECE and intervals half as wide as they should be.
Conformal gives a distribution-free coverage guarantee instead.

---

## 8. Troubleshooting

| symptom | likely cause |
|---|---|
| accuracy near the majority baseline after training | options did not fit; run `preflight`. Or the criteria describe labels, not boundaries. |
| great aggregate ECE, bad behaviour in production | you reported the aggregate. Slice it. |
| model follows tone/surface form, not your rubric | not enough examples where they disagree. That contrast *is* the training signal. |
| ordinal model puts mass on levels 0 and 3 | you are using `choice` semantics. Use `score`, and keep the RPS term. |
| training 8× slower partway through | swap. Check `freeze_lower_layers` and the memory table in §6. |
| results move when you reorder options | expected on a base checkpoint (~33%). Option order is randomised each epoch to train it out. |
| loss in the hundreds | a masking bug — padded slots carry -1e4 logits and must carry zero target mass. See FINDINGS. |

---

## 9. The honest caveat

We measured this recipe on reranking, and on that task a 22M pretrained
cross-encoder beat our fine-tuned 422M System One model by +0.097 nDCG@10 at
7× the speed. The fine-tuned model *was* better calibrated (ECE 0.111 vs 0.192
after both were temperature-scaled), which is the property this class of model
is for — but if you only need a score, a cross-encoder is the better tool.

Fine-tune a System One model when you need **a calibrated probability from a
small self-hosted model on your own decision boundary**. Not when you need the
highest number.
