# Fine-tune a small model to make decisions, not text

**Stop asking an LLM to classify things and parsing the answer out of prose.**

There is a class of small models — called **System One models** — that never
write text. You tell them what the possible answers are, and they hand back
**how likely each answer is**. No JSON parsing. No retries on malformed output.
No guessing how confident the model was.

They are also close to useless until you train them. **This repo is the recipe
for training them**, with every number measured and the failures left in.

[![License](https://img.shields.io/badge/license-Apache--2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/python-3.10%2B-blue.svg)](https://www.python.org)
[![Model](https://img.shields.io/badge/model-421M%20params-green.svg)](https://huggingface.co/convaiinnovations/laya)
[![Hardware](https://img.shields.io/badge/trains%20on-a%20laptop-green.svg)](#try-it)
[![Tests](https://img.shields.io/badge/tests-84%20passing-brightgreen.svg)](tests)

---

## The problem this solves

You need to answer a small, repeated question. *Which team owns this ticket?
How severe is this incident? Is this a refund request?*

So you call a large language model, and now you own a pile of problems:

- It writes a sentence, and you write a parser for the sentence.
- Sometimes the format changes and the parser breaks.
- It says "billing" with total confidence whether it is sure or not.
- You pay per call, and your data leaves your machine.

A System One model removes all four. It cannot answer outside the options you
gave it, and it tells you **how sure it is** — a number your code can branch on.

![How a System One model answers](assets/how-it-works.svg)

---

## What you get

| | |
|---|---|
| **A number, not a sentence** | Act automatically at 0.9. Ask a human at 0.5. Your system's behaviour becomes something you can reason about. |
| **Runs on your laptop** | 421M parameters. No GPU needed. Your data never leaves your machine. |
| **No format failures** | The model physically cannot return anything outside your options. |
| **Fast** | Nothing is generated word by word, so there is nothing to wait for. |
| **Yours** | Apache-2.0, trained on your own judgement calls, no per-request bill. |

---

## Try it

You need Python and about 2 GB of disk.

```bash
git clone https://github.com/omkarghugarkar007/system-one-model-finetuning
cd system-one-model-finetuning

make install     # set up Python
make weights     # download the model, about 1.7 GB
make quickstart  # train it
```

The example teaches the model to rate how serious an incident report is. The
interesting part: **the rule deliberately disagrees with the tone.** A calmly
written note about a total outage is serious. A furious complaint about a broken
internal dashboard is not.

An untrained model goes by tone, because that is what it absorbed from general
text. After training it follows *your* rule — more answers right, and its
confidence far more trustworthy.

It is a demonstration on a small made-up task, not a result. The real numbers
are below.

---

## Does it actually work? Here are the real numbers

We trained on **2,000 examples from 250 search queries**, for **1 hour 45
minutes on a laptop**, then tested on a *completely different* collection of
documents to check it had learned something rather than memorised something.

Scores are nDCG@10 — a standard search-quality measure from 0 to 1, higher is
better.

| | score |
|---|---|
| Plain keyword search (the thing to beat) | 0.528 |
| The model **before** training | 0.455 &nbsp;*worse than keyword search* |
| The model **after** training | **0.607** &nbsp;*+0.152* |

So: untrained it actively hurt, and training took it from well below the
baseline to clearly above it, on data it had never seen.

### And here is where it lost

We then ran it against an ordinary off-the-shelf ranking model **twenty times
smaller**, on identical data.

| | score | speed |
|---|---|---|
| Our fine-tuned model (422M) | 0.607 | 1x |
| **Off-the-shelf model (22M)** | **0.720** | **7x faster** |

**It beat us, comfortably.** We are telling you this because you will find out
anyway, and because it sharpens what these models are actually for.

Where we won was **knowing when to doubt itself**. Calibration error — how far
stated confidence drifts from how often it is actually right, lower is better —
was **0.111** for ours against **0.192** for the smaller model, after giving both
the same fairness adjustment. Roughly twice as trustworthy.

**The trade:** if you want raw accuracy, use the ordinary model. If you want a
confidence number you can safely act on, this is what you are buying, and now
you know its exact size before spending an afternoon.

---

## The recipe

![The recipe](assets/the-recipe.svg)

Five steps. **Step 2 is the one everyone skips and the one that quietly ruins
runs** — the model only reads a small, fixed amount of text, and anything past
that is thrown away silently. We lost most of a day to this, convinced the model
was bad when it had simply never been shown the evidence.

---

## Using your own data

One line per example:

```json
{"state": "I was charged twice for my Pro plan",
 "type": "choice",
 "instructions": "Which team should handle this?",
 "criteria": {"billing": "payments, invoices, refunds",
              "technical": "bugs, outages, integrations"},
 "label": "billing"}
```

Three shapes of question:

- **choice** — pick one of several. *Which team owns this ticket?*
- **score** — rate on a scale where order matters. *How severe, 0 to 3?*
- **noul** — plain yes or no. *Is this a refund request?*

**The highest-leverage tip in this whole repo:** describe each answer as a
**boundary**, not a name. `"billing"` tells the model nothing.
`"billing: payments, invoices, refunds, subscription charges"` tells it where
the line falls. The model reads that description. Improving it is the cheapest
quality you will ever buy.

---

## Four things that will bite you

**1. Your text may never reach the model.** There is a small fixed budget. When
your answer descriptions are long they all get trimmed — sometimes to a few
words each. A check runs before training and stops you.

**2. Where you put the information changes everything.** We moved the same
content from one part of the input to another, changed nothing else, and the
results improved dramatically. Try both ways on your task.

**3. A good average hides bad groups.** The model's own published numbers
include an excellent overall confidence score that conceals one category where
it was badly wrong. Always read the breakdown.

**4. Training does not make confidence honest.** It makes the model more often
right, but its sense of *how sure am I* drifts. There is a quick final step that
fixes this, and it is not optional if you plan to act on the numbers.

---

## When you should *not* use this

Told up front, because finding out later is worse.

- **You only need accuracy.** Use an ordinary classifier. Ours lost to one
  twenty times smaller.
- **Your task needs reasoning or multi-step thinking.** These models are too
  small. Use a bigger one.
- **You have no labelled examples.** There is nothing to train on, and untrained
  these models are not useful.
- **Your inputs are long documents.** They will not fit. You would have to pick
  out the relevant part first, and that becomes the hard problem.

Use one when you want **a trustworthy confidence number from a small, fast,
private model trained on your own judgement calls.**

---

## Questions people ask

**How is this different from asking GPT with a JSON schema?**
A schema constrains the *shape* of the answer. It does not give you a calibrated
probability, and you still pay per call and send your data out. Here the
probability is the product.

**Do I need a GPU?**
No. Everything here was built and measured on a laptop.

**How many labelled examples do I need?**
A few hundred to see it learn. A few thousand for something you would deploy.

**Which models does this work with?**
**Laya** (open, 421M, Apache-2.0 — used throughout) and **Jev** (hosted, paid).
Both take the same question format, so the recipe transfers.

**Is this state of the art?**
No, and we show you exactly where it loses. It is a working, honest method with
the measurements attached.

**What is it built on?**
A ModernBERT encoder with a decision head. The training code, evaluation and
calibration here are ours.

---

## Read more

- **RECIPE.md** — the detailed cookbook, with all the numbers.
- **FINDINGS.md** — everything measured, including the failures and the two bugs
  that produced convincing results that turned out to be wrong.
- **docs/research/** — the original research this grew out of, and which of its
  predictions survived contact with reality.

---

Apache-2.0. Built on [Laya](https://huggingface.co/convaiinnovations/laya) by
ConvAI Innovations, using the question format from
[TypeSafe](https://docs.typesafe.ai/).

If this saved you time, a star helps other people find it.
