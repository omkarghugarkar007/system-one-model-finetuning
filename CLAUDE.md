# Working notes for Claude

## Commits

- **Do not add Claude attribution to commits.** No `Co-Authored-By: Claude`,
  no `Generated with Claude Code`. Same for PR descriptions.
- Commit messages explain *why*, and say plainly when something was wrong
  before. A message that records a corrected mistake is more useful than one
  that only describes the final state.

## What this repo is

A recipe for fine-tuning **System One models** — small models that return typed
decisions with calibrated probabilities instead of text. It grew out of a
reranking research project; that work is kept as the worked example.

Public at `github.com/omkarghugarkar007/system-one-model-finetuning`.

Three docs carry the substance, and they have different jobs:
- `README.md` — for a general reader. Plain language, no module paths, no jargon
  without a one-clause explanation.
- `RECIPE.md` — the cookbook, with the numbers.
- `FINDINGS.md` — the ledger. Every claim names the run that produced it,
  including the negative results and the bugs that produced false findings.

Keep them in sync when results change. `docs/ROADMAP.md` tracks what is done.

## House rules learned the hard way

- **Do not edit `docs/research/plan.md` to match results.** It is the original
  proposal. Where measurement contradicted it, that goes in `FINDINGS.md` —
  the disagreements are the interesting part.
- **Report ECE per slice, never only in aggregate.** An aggregate hid a badly
  calibrated group in this very project.
- **Compare accuracy against the majority-class baseline, not chance.**
- **Run the token budget preflight before training.** The most common silent
  failure is that the data never reached the model, and it looks exactly like
  the model being bad.
- Numbers in the README are stated plainly with the metric explained. No
  vague "a large improvement".

## Environment

- Apple M4, 16 GB. Full fp32 fine-tuning of the 422M model **swaps** — use the
  layer-freezing lever. Throughput collapses ~8x when it does.
- `.env` holds `OPENROUTER_API_KEY` (Jev teacher) and GitHub credentials.
  Never commit it, never echo the values.
- `make help` lists every entry point.

## Testing

`make test` is unit-only: seconds, no weights, no network. Tests that need
weights or an API are marked `slow` / `network` and live in `tests/integration`.
