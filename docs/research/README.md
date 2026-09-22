# The research this recipe came from

`plan.md` is the original design document: a reranking architecture
(FrontierRank) built on a fine-tuned System One model, with anchored slates, a
regret frontier and a value-of-information controller.

The recipe in [../../RECIPE.md](../../RECIPE.md) is what generalised out of
building it. The reranking system itself lives in `systemone.reranking` and is
kept as the worked example, because it is the honest demonstration of what
these models can and cannot do.

**Read `plan.md` as a proposal, not as findings.** Several of its claims did not
survive contact with the model, and those are recorded in
[../../FINDINGS.md](../../FINDINGS.md) rather than edited into the document:

| plan.md says | we measured |
|---|---|
| the ~320-token state is the binding constraint | the 192-token *head* budget is; at 10 options each keeps 16 tokens (F1) |
| marker noise σ = 0.35 | the model is deterministic; the real floor is option-order variation, 0.38 nats (F3) |
| anchored A=4 beats naive on cross-query scale (+0.043) | null when slates are uniform; **+0.181** only when composition varies (F8) |
| Phase 1 gate: anchored beats naive on nDCG@10 | that contradicts the document's own Part VIII; measured +0.0004, a tie (F13) |
| — | a 22M cross-encoder beats the fine-tuned 422M model by +0.097 nDCG@10 (F16) |

What did survive, and is the substance of the recipe: the identifiability
argument (`log p = l_i − c_S` holds on real text, residual at the measurement
floor), the value of training *for* the anchor fit rather than only correcting
at inference, and the whole calibration story.
