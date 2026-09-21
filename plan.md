LLM Reranking: Theory, SOTA, and the CALIPER Architecture
21 Sept 2026 · @Omkar
Executive summary
The remaining headroom in reranking is not in better relevance models. It is in deciding which relevance judgments are worth buying. Three years of work on BEIR moved the classic BM25-top-100 lineage from 43.4 to 54.7 nDCG@10, and the last two years of that contributed roughly one point. Meanwhile the cost of a reranking pass spans four orders of magnitude, from $0.05 to $256 per 1,000 queries for nominally the same task.
FrontierRank is a reranking system that spends compute in proportion to ranking regret rather than candidate count. A fine-tuned Laya (421M, Apache-2.0, self-hosted) scores every candidate locally; Jev is called only on the uncertain top-k frontier; and every Jev call becomes training data that reduces the need for future Jev calls.
The mechanism that makes this affordable is anchored slates. Laya's Choice primitive emits a softmax normalised within a slate, so log p_i = l_i − c_S and the per-slate offset c_S is unobservable. Ranking globally by p_i is therefore broken — and broken worst in exactly the configuration everyone would choose, slating by first-stage rank. Injecting a few pivot documents of known grade into every slate identifies c_S in closed form, which lets a 10-option model emulate a 255-option one.
Claim
Figure
Basis
Naive concatenation, slates blocked by first-stage rank
0.679 nDCG@10
simulation
Stratifying the slates — free, no anchors
0.895 nDCG@10
simulation
Anchored, 4 pivots per slate
0.902 nDCG@10
simulation
Cross-query score correlation, naive → anchored
0.918 → 0.961
simulation
FrontierRank ranking cost, 5% escalation, 30% GPU util
$0.096 / 1k queries
list prices + measured throughput
Cohere Rerank 4 Pro, same workload
$2.51 / 1k queries
measured
Jev batched 4-level rubric, 30 candidates in one call
0.692 nDCG@10 @ $0.45 / 1k
measured
The honest scope: anchoring, regret-aware escalation, and teacher distillation each have prior art. The claim here is that the combination produces a materially better quality/cost frontier, and that the anchor mechanism is what makes a sub-500M local model viable as the frontier's scoring substrate. Both are hypotheses with a defined falsification test in Part VIII.
Part I — Theory
Reranking exists because relevance is a joint function of query and document, and first-stage retrieval cannot afford to compute joint functions. A bi-encoder must commit to a document vector before it sees the query; whatever the query would have made salient is already averaged away. The cross-encoder pays O(N) joint forward passes to recover it, over a candidate set small enough to afford them.
What the reranker is estimating
Given query q and candidates D = {d_1 … d_N}, the reranker estimates s_i ≈ U(q, d_i), the downstream utility of d_i. The Probability Ranking Principle says that ordering by probability of relevance is optimal for any metric that is a positive decreasing function of rank — but only under two conditions that reranking routinely violates: that relevance is judged independently per document, and that the probabilities are correct, not merely correctly ordered.
The second condition is what makes calibration load-bearing rather than cosmetic. Any monotone transform of the scores gives the same nDCG, so a reranker used purely to sort does not need calibration. The moment you want to threshold ("send the top-k above 0.8 to the LLM"), merge ("combine two slates"), or decide ("is another judgment worth buying?"), you need the numbers to mean something. FrontierRank is built entirely on the third of those, which is why Part VIII spends its effort on identifiability rather than on accuracy.
The four paradigms
Paradigm
Model sees
Forward passes
Score is
Main weakness
Pointwise
(q, d_i)
N
absolute, comparable
no candidate-vs-candidate context
Pairwise
(q, d_i, d_j)
up to N(N−1)/2
relative
cost; needs an aggregation sort
Setwise
(q, {d_i…d_m})
O(N log N) with a sort
relative within set
ordering algorithm required
Listwise
(q, d_1…d_N)
⌈N/w⌉ windows
relative within window
position bias; window non-comparability
The complexity column is the one people quote; the "score is" column is the one that matters for system design. Pointwise is the only paradigm that natively produces a globally comparable number. Every other paradigm produces scores that are only meaningful inside the comparison unit, and every listwise system must therefore solve a stitching problem that is usually left implicit.
Sliding windows are the standard dodge: RankGPT reranks 100 passages with window 20 and stride 10, so consecutive windows overlap by 10 documents and the overlap propagates an ordering. This works, but it is an O(N) sequential bubble pass with a 1.8× re-encode tax, it cannot be parallelised across windows, and it never produces a score — only a permutation. You cannot threshold a permutation.
How a score is actually produced
Three mechanisms are in play across the current model zoo, and they have very different cost and robustness profiles.
1. Generative permutation decoding (RankGPT, RankZephyr). The model writes [4] > [11] > [2] > …. Cost is dominated by input tokens — the output permutation is ~80 tokens against ~5,300 input tokens per window, so 93–98% of the bill is input. Format failures are real: RankGPT variants show 76.6–98.9% output-format success.
2. First-token logit decoding (FIRST). Read the ranking off the logits of the first generated identifier and stop. Roughly 50% faster, and it turns the output into a distribution rather than a string.
3. Marker softmax (Laya, and by strong inference Jev). Place a [MASK] marker before each option inside one bidirectional sequence, gather the marker hidden states, score each with a shared scalar MLP, and softmax across that question's markers. One forward pass, no decoding, no format failures by construction.
Mechanism 3 is a 2020-era MMLU log-prob scoring trick moved from a decoder to an encoder, and community commentary is right to say so. What is not priced in is its order robustness. On the one public head-to-head, reversing passage order changed the top-ranked result on 92.6% of queries for a generative Qwen reranker but only 24.7% for Jev's Choice primitive (jev-rerank-bench). Bidirectional attention with position-symmetric markers has no autoregressive left-to-right prior to overcome. For a system that will slate candidates thousands of times per query, that robustness is worth more than a point of nDCG.
The catch nobody writes down
Mechanism 3's softmax is normalised over the slate. Writing l_i for the pre-softmax marker logit:
\log p_i = l_i - \underbrace{\log \sum_{j \in S} e^{l_j}}_{c_S}
c_S depends only on the slate, not on the document. Within a slate it cancels and the ordering is correct. Across slates it does not cancel, and it is not observable from p alone — the softmax is invariant to adding any constant to every logit in the slate. So concatenating per-slate probabilities into a global ranking compares apples to a different slate's apples. Part VIII quantifies how badly, and fixes it.
Part II — SOTA, September 2026
There is no single SOTA reranker, and the reason is not modesty. The first-stage retriever moves the number more than the reranker does. RankZephyr scores DL20 nDCG@10 of 0.7086 reranking BM25 top-100 and 0.8159 reranking SPLADE++ ED top-100 — the same model, a 10.7-point swing (RankZephyr). Any cross-paper comparison that does not hold the first stage fixed is measuring the first stage.
BEIR "averages" in the literature range over 7, 8, 9, 11, 13, 15 and 18 datasets. One model, mxbai-rerank-large-v2, is reported at 57.49, 61.44 and 62.45 in three different papers. Read the protocol column or read nothing.
What is actually strong
Model
Params
Weights
BEIR nDCG@10
First stage
Note
jina-reranker-v3.5
0.6B
CC BY-NC-SA
63.20 (13 ds)
jina-emb-v5-small top-100
highest verified; non-commercial licence
jina-reranker-v3
0.6B
open
61.94 (13 ds)
jina-emb-v3 top-100
beats Qwen3-4B on identical inputs
Qwen3-Reranker-4B
4B
Apache-2.0
61.16 (13 ds)
jina-emb-v3 top-100
MTEB-R 69.76, FollowIR +14.84
mxbai-rerank-large-v2
1.5B
Apache-2.0
57.49 (BM25)
BM25
best BM25-first-stage vendor figure
Qwen3-Reranker-0.6B
0.6B
Apache-2.0
56.28 (13 ds)
jina-emb-v3 top-100
FollowIR +5.41
BGE-reranker-v2-m3
0.6B
open
56.51 (13 ds)
jina-emb-v3 top-100
FollowIR −0.01
BracketRank-20 (GPT-4)
closed
no
54.66 (BEIR-8)
BM25 top-100
best controlled BEIR-8 result
RankGPT-4
closed
no
53.68 (BEIR-8)
BM25 top-100
the 2023 reference point
monoT5-3B
3B
open
51.36 (BEIR-8)
BM25 top-100

BM25
—
—
43.42 (BEIR-8)
itself
floor
On the one lineage that is internally consistent — BEIR-8 over BM25 top-100 — three years moved 43.42 → 54.66, and the last two years of that contributed about one point. BEIR is saturated as a discriminator.
Parameter count is not the axis
The cleanest result in the whole landscape: on identical inputs, jina-reranker-v3 at 0.6B scores 61.94 and Qwen3-Reranker-4B at 4B scores 61.16. Within the Qwen3 family, 4B → 8B is negative on MTEB-R (69.76 → 69.02). A 23M-parameter MiniLM-L6 cross-encoder hits 74.30 on TREC DL19 against monoT5-3B's 71.83, and L6 → L12 buys 0.01 nDCG@10 for a 1.9× slowdown.
Where size does buy something, it buys it sharply:
• Reasoning (BRIGHT): Rank1's size ablation runs roughly 11 → 22 nDCG@10 from 0.5B to 32B, with the jump concentrated at 3B→7B. A MiniLM cross-encoder actively degrades BRIGHT results.
• Instruction following (FollowIR p-MRR): a sign flip, not a slope. Sub-1B models cluster at −1.64 to −0.01 — they ignore the instruction. Qwen3-0.6B is +5.41, Qwen3-4B +14.84.
• Code (MTEB-Code): Qwen3 0.6B = 73.42, 4B = 81.20.
Architecture and training method dominate on classical relevance; parameter count dominates on reasoning, instructions and code. That is the single most important fact for a design built on a 421M local model, and Part XI turns it into a decision rule.
The efficiency literature has already made the argument
E2R-FLOPs proposes Ranking-metric Per PetaFLOP and measures the frontier's steepness directly on TREC DL19: pointwise Flan-T5-Large gets nDCG 0.654 at RPP 72.67; pairwise Flan-T5-XL gets 0.713 at RPP 0.10. +0.059 nDCG for a 727× worse quality-per-FLOP ratio.
Everyone who has measured this concludes the same thing. HeadRank truncates at layer 16 of 28 for a 57% FLOP cut and still beats RankGPT (46.73 vs 44.89). MICE masks query-document attention below a critical layer for a 4× speedup and improves BEIR OOD from 44.7 to 46.3–50.2. AcuRank (NeurIPS 2025) runs TrueSkill over document relevance and reranks only where posterior uncertainty is high. Rerank Before You Reason finds d=10–20 reranking beats spending the same budget on reasoning, at ~6× fewer tokens, with diminishing returns past d=20.
FrontierRank sits in this lineage, not against it. What it adds is that the adaptive unit is a typed, calibrated distribution rather than a permutation — which is what makes expected-regret arithmetic possible at all.
One training detail worth stealing
Jina mines hard negatives from BM25, Jina, BGE, GTE, E5 and ColBERT-style retrievers. Negatives from a single retriever teach the reranker that retriever's failure modes rather than relevance. Their structured-retrieval training goes further: generate near-identical records where exactly one constrained field is perturbed. Part IX makes this the centrepiece of data construction, because it is almost certainly worth more than the loss function.
Part III — Benchmarks
Dimension
Benchmark
Scale
Metric
What it catches
Conventional relevance
BEIR
18 datasets, zero-shot
nDCG@10
broad baseline; saturated
Graded relevance
TREC DL 2019–22
43 / ~54 judged queries
nDCG@10
the only clean 4-level qrels
Reasoning
BRIGHT
12 datasets, 1,384 queries
nDCG@10
semantic matching vs actual inference
Instructions
MAIR / FollowIR
126 tasks / 102 queries
nDCG@10 / p-MRR
does the model read the instruction at all
Negation
NevIR
1,383 paired questions
pairwise acc
surface-form matching
Structured constraints
Struct-IR, RTEB
—
nDCG@10
dates, numbers, equality
Contamination
FutureQueryEval
post-cutoff queries
nDCG@10
5–15% drop across all reranker families
The pitfalls that will bite
TREC DL grade 1 is not relevant. The scale is 3 perfectly relevant, 2 highly relevant, 1 "related but does not answer", 0 irrelevant. nDCG@10 uses all four grades, but binary metrics must be run with trec_eval -l 2. Forgetting the flag silently inflates MAP and MRR, and it is one of the most common reporting errors in the literature. With 43 judged queries on DL19, differences under ~2 nDCG@10 points are noise.
BRIGHT's binding constraint is recall, not reranking. Reranker-Guided Search reports that only 31% of BRIGHT ground-truth answers appear in the top-100 by embedding similarity, against 87% on FollowIR. A rerank-top-100 pipeline is structurally capped near that ceiling. Deepening helps directly: 28.8 at rerank-100 → 33.0 at rerank-500. Any BRIGHT result that does not state its candidate depth is uninterpretable.
Also: the original BRIGHT "BM25" baseline is actually query-side BM25, undocumented in the paper; a reproducibility audit found duplicate documents, degenerate passages under 5 tokens in up to 36.5% of some datasets, and zero-length documents up to 3.6%. The public leaderboard takes submissions by email with optional code links — its top two entries (66.9 and 63.4) have no paper and no verification. Do not benchmark against them. The highest paper-documented pipeline is DIVER-v3 at 46.8; the best reproducible open baseline is 40.3.
MS MARCO dev is retired as a signal. 6,980 queries at ~1.06 relevant passages each, binary judgments, extreme false-negative rate, task retired in 2023. Report it for continuity, decide nothing on it.
BEIR's subset problem is the biggest comparability trap. Fix your dataset list, publish it, and never compare across papers that do not share it.
What to report
nDCG@10 alone is an incomplete result for this system, because the claim is about the frontier rather than the peak. Every row should carry:
• Quality: nDCG@10, Recall@10, MRR@10
• Calibration: ECE and Brier, per option-count bucket — not just aggregate, since Laya's own eval shows an aggregate ECE of 0.030 hiding a family at 0.438
• Cost: dollars per 1,000 queries, split into local compute and teacher API
• Latency: P50 / P95 / P99. No commercial vendor publishes P99; you should
• Escalation rate: the fraction of queries that bought a teacher judgment
The headline artifact is a scatter of nDCG@10 against dollars per million searches, with each configuration a point. FrontierRank is interesting only if it moves that frontier outward — not if it merely scores highest.
Part IV — Cost and latency
Normalised workload throughout: 1,000 queries × 100 candidates × 250 tokens, query 20 tokens. Prices observed 2026-09-21.
Tier
Approach
$ / 1k queries
× cheapest
Self-hosted encoder
400M cross-encoder, L40S @ 100% util
$0.05
1×
Self-hosted decoder
Qwen3-Reranker-0.6B, same
$0.08
1.6×
Hosted open weights
DeepInfra Qwen3-Reranker-0.6B
$0.28
6×
Commercial, cheapest
ZeroEntropy zerank-2
$0.63
13×
Commercial, mid
Voyage rerank-2.5
$1.35
27×
Typed decision API
TypeSafe Jev, batched
$1.13
23×
Commercial, per-search
Cohere Rerank 4 Pro
$2.50
50×
Cheap-LLM listwise
RankGPT on Groq GPT-OSS 20B
$3.79
76×
Frontier-LLM listwise
RankGPT on a frontier model
$256
5,100×
Four structural facts
1. Listwise LLM reranking is input-dominated. RankGPT's window-20/stride-10 over 100 passages is 9 windows × ~5,300 input tokens = 47,700 input against 720 output tokens per query. Input is 93–98% of the bill, and prompt caching barely helps because only the ~200-token instruction prefix repeats. The 9×20 = 180 passage encodings for 100 passages is a 1.8× re-encode tax that is pure waste.
2. Cohere bills a step function. A search unit is one query with up to 100 documents, and any document over 500 tokens including the query is auto-chunked, each chunk counting as a document. At 250-token passages you pay $0.100 per 1M effective tokens; at 480-token passages, $0.052. A 20-token increase from 480 to 500 doubles the bill. Chunk size is a pricing lever, not only a quality lever.
3. Self-hosting a small encoder pays back at ~2% GPU utilisation. A 400M cross-encoder on a $796/month L40S beats Cohere Rerank 4 Pro at 320k queries/month. A 4B decoder only beats the cheapest token-billed APIs at ~90–100% sustained utilisation, which no real workload achieves. The make-or-buy line is set by model size, not volume.
4. Jev's free output tokens matter more than its rate. $0.042/1M input with output free removes the term that dominates generative reranking, and the 64k context fits all 100 passages in one request. That is the whole reason a typed-decision API can compete with a dedicated cross-encoder API.
The one public head-to-head
jev-rerank-bench — 8 English datasets, 1,617 scored queries, BM25 top-30, 2,000-char truncation held constant:
Configuration
nDCG@10
$ / 1k queries
Jev, 4-level rubric, 30 candidates in one call
0.692
$0.45
Cohere Rerank 4 Pro
0.691
$2.51
Jev, 30 yes/no in one call
0.685
$0.41
Jev, one Choice over 30
0.684
$0.33
ZeroEntropy zerank-2
0.682
$0.22
Jev, prune then pairwise cascade
0.674
$0.63
Jev, tournament, 6 groups + final
0.668
$0.43
Jev, 45 duels in one call over top-10
0.580
$0.21
BM25 floor
0.486
$0
Read the caveats before the numbers. The author states the Jev–Cohere gap is +0.001 with a 95% interval of −0.009 to +0.012, "neither a winner nor equivalence," and that weighting equally per query rather than per dataset flips the order — Cohere 0.756, Jev 0.738. Public datasets may sit in any model's training data.
Three things survive the caveats and matter for the design:
• Batched beats per-pair, decisively. One call over 30 candidates at $0.33–$0.45 beats per-pair configurations at $0.81, and the all-duels configuration is the worst result in the table at 0.580. TypeSafe's own reranking cookbook uses one request per candidate — 1,200 calls for 40 queries, $1.61 per 1k queries normalised. The vendor's own recipe is the expensive path.
• The 4-level rubric edges the single Choice (0.692 vs 0.684). A pointwise ordinal grade is globally comparable; a Choice softmax is not. That difference is the entire subject of Part VIII.
• Order robustness: reversing passage order changed the top result on 92.6% of queries for a generative Qwen reranker and 24.7% for Jev's Choice.
Latency
System
Latency, ~100 candidates
Setup
400M cross-encoder, pure compute
62 ms
H100, modelled at 35% MFU
jina-reranker-v3 (0.56B)
188 ms
H100 PCIe, self-hosted
zerank-2
149.7 ms P50
vendor, 12 KB payload
Cohere Rerank 4 Fast
447 ms
third-party
Cohere Rerank 4 Pro
614 ms
third-party
Jev
264–349 ms P50, flat 2→255 options
independent; vendor claims 70–500 ms
Qwen3-Reranker-4B
>1,000 ms
H100 PCIe
Self-hosted jina-v3 at 188 ms against hosted Cohere Pro at 614 ms is a ~3× gap that is mostly queuing, batching window and network — not compute. No vendor publishes P99. Jev's latency is flat in option count, which is the property FrontierRank exploits: escalating with 30 candidates costs the same wall-clock as escalating with 5.
Part V — Where reranking pays
Reranking earns its keep when the first stage has recall but not precision in its top-k, and when something downstream is expensive or sensitive enough to care which document arrives first.
Use case
Why reranking helps
Where the value actually is
RAG grounding
the generator reads 5–20 passages; wrong ones cause confident wrong answers
precision@5, not nDCG@100
Legal and medical search
recall-first retrieval by design; huge candidate pools
graded relevance, long documents
Entity resolution, record linkage
near-duplicates differ on one field
constraint satisfaction, not topical similarity
Code search
lexical overlap is actively misleading
reasoning; needs a larger model
Agent tool selection
a small closed set, one wrong pick derails a trajectory
calibration — the agent must know when to ask
Recommendation re-scoring
cheap candidate generation, expensive slate
listwise interactions between items
When to not bother
• The first stage already puts the answer at rank 1 most of the time. Measure MRR of the retriever alone. If it is above ~0.85 for your k, a reranker buys tie-breaking and little else.
• Fewer than ~20 candidates. Below that the pool is the ceiling; spend on recall.
• Binary retrieve-or-not decisions. That is a classifier, not a ranker.
• Latency budget under ~50 ms end-to-end. Even a 400M cross-encoder needs ~62 ms of pure H100 compute for 100 pairs.
The one that keeps being underrated
Reranking is often cheaper than the generation it feeds. Rerank Before You Reason measured a deep-search pipeline at 42.17% accuracy for 2,132.72M tokens with no reranking, against 44.15% accuracy for 364.16M tokens with d=20 reranking — better accuracy at roughly 6× fewer tokens. The reranker is not a cost centre in a RAG system. It is the thing that stops you paying a frontier model to read garbage.
Part VI — What Jev and Laya actually are
Both are non-autoregressive "System One" models: they read a state and return typed decisions with probabilities instead of generating text. Jev shipped 2026-09-15; the Laya repo was created 2026-09-18 and is a deliberate API-compatible reimplementation — same three primitive names, same request and response shapes, including the coined term Noul.

Jev 1.13.0
Laya (English root)
Weights
closed, hosted only, early access
Apache-2.0, 421M
Backbone
undisclosed
ModernBERT-large 395M + 2-layer head
Context
64k request / 32k state
512 tokens (192 head, 320 state)
Choice options
255 (hard reject at 256)
no enforced cap; degrades past ~10
Fine-tuning
none, same weights all accounts
yes, published recipe
Price
$0.042/1M input, output free
GPU time only
Latency
264–349 ms P50, flat in option count
38.4 ms (1 question, T4)
The three primitives
Choice — criteria is a map of option keys to descriptions. Returns the argmax key, the full distribution, and a confidence derived from distribution concentration. Score — criteria is an ordered array of level descriptions, 0-indexed by position; Jev allows 2–10 levels and returns a fractional expected score plus the per-level distribution. Noul — "is this statement true?", returns one probability.
In Laya, score and noul are not separate heads. score renders levels as "level %d: %s" and treats them as ordinary options; noul builds a two-option choice. Only the ordinal reward differs. That matters for Part IX: if you want a genuinely ordinal head, you add it.
Laya's mechanism, read from the source
One sequence per question:
[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]
The option marker is the tokenizer's ordinary [MASK] — no custom special tokens, no vocabulary extension, and a defensive .replace(mask_tok, ' ') strips any literal [MASK] from option text. There is no per-option head: a single shared scalar MLP (LayerNorm → Linear → GELU → Linear(d,1)) scores every marker position, padding markers are filled with -1e4, and one softmax runs across that question's markers. Because options sit in the same bidirectional sequence as the state, each marker's hidden state is contextualised by both its own option text and the state. That is where the discrimination comes from, and it is why the mechanism is position-robust.
Correction to the model card. The card says "every question in a call is answered in one single forward pass." The source does not do that: build_sequence runs once per question, re-appending the full state each time, and collate_items stacks the results into a batch. Measured on T4: 1 question 38.4 ms, 10 questions 156.0 ms, 50 questions 721.4 ms — ~14 ms marginal per question, 18.8× for 50×. Batching helps; it does not amortise the state. Any design that asks Laya five factorised questions per candidate pays five encoder passes per candidate.
Training scale, from the shipped config: updates: 7313, epochs_completed: 1, hours: 1.96, world_size: 1. Under two hours on one GPU.
RLCD, and what it actually contains
Reinforcement Learning for Calibrated Decisions: the policy emits a distribution and the reward is a strictly proper scoring rule, so honest probabilities are reward-maximising. Laya's implementation applies log score and spherical score (w_sph = 0.5) to all types, and ranked probability score (w_rps = 1.0) only to score questions — the correct choice, since RPS is the ordinal-aware proper rule. Optimisation is REINFORCE with a group-mean baseline (GRPO-style) and Gaussian noise on the logits for exploration.
One detail the vendor materials do not foreground: the typed-decisions recipe also uses soft cross-entropy against a teacher's distributions. The specialised checkpoint is partly distilled. There is no arXiv paper for RLCD from either vendor — and the acronym collides with an unrelated 2023 alignment paper, Reinforcement Learning from Contrastive Distillation (arXiv 2307.12950). Different problem, different method.
The honest limits
• Zero-shot, Laya is below the majority-class baseline. The card states it plainly: base checkpoints score 0.362 (English) and 0.352 (multilingual) on typed-decisions against 0.318 random and 0.461 majority-class. Fine-tuned, the same benchmark reaches 0.766. Its own conclusion: "Laya is a fast base to specialise, not a zero-shot decision engine."
• High cardinality breaks it. On Banking77, Jev scores 0.870 on 72 labels and Laya 0.425 on 77, because (256 − 16) // 77 ≈ 3 tokens per label. The shipped temperature_by_options map is the tell: every bucket softens (T > 1) except choice:11+, which needs T = 0.1006, a 10× sharpening. A fitted temperature that extreme means the raw logits were near-uniform. The cliff is at 11 options, not 20.
• Both ship over-confident. Temperature refit moves Laya's ECE 0.466 → 0.081 and the multilingual model 0.314 → 0.106. The specialised checkpoint's ECE is 0.213, not 0.081.
• Jev has no structural invariants. TypeSafe publishes this themselves: complementary Nouls summing to 1.19 instead of 1.0, and the same refund question returning Noul 0.22 against Choice yes=0.01/no=0.99. The format is guaranteed; the coherence is not. Jev is also explicitly not hardened against prompt injection — "state is data, and jev-1.13 does not treat it as hostile by default."
The vendor comparison does not survive a matched protocol
ConvAI's comparison page reports Laya beating Jev by +11.5% on DAIR Emotion. That figure uses AbdelStark's n=100 pilot, which produced the single worst Jev emotion number in existence (0.480). elcronos ran the full 2,000-row official test set with a protocol published before test inference:

Accuracy
ECE
Brier
Jev 1.13
0.587
0.281
0.667
Laya
0.587
0.307
0.707
PrismNLI-0.4B
0.725
0.174
0.441
McNemar p = 1.000, paired difference CI [−0.022, +0.020]. Laya is worse calibrated than Jev here, the opposite of the headline claim, and a third open model neither vendor mentions beats both at 58 ms local. The latency comparison on that page is also a category error — Jev's remote round-trip from France against Laya's local T4 forward pass. Treat the comparison table as unusable; the limitations section on the same page is honest and worth reading.
Why these are secretly rerankers anyway
Strip the marketing and three things line up with reranking exactly:
1. Score with a 4-level rubric is the TREC graded-relevance scale. 3 perfectly relevant, 2 highly relevant, 1 related but does not answer, 0 irrelevant. Not an analogy — the same scale, and it is the best-performing Jev reranking configuration measured (0.692).
2. Choice over N candidates is a listwise reranker in one forward pass. No decoding, no permutation string, no format failures, and 24.7% order sensitivity against 92.6% for a generative reranker.
3. Calibrated probabilities make expected-regret arithmetic possible. A permutation cannot tell you whether buying another judgment is worth it. A distribution can.
And the division of labour falls out of the specs rather than out of a quality ranking. Under a matched protocol Jev and Laya are statistically indistinguishable on a normal classification task — so "Jev is the smarter teacher" is not supported. What is supported is that Jev handles 255 options and Laya handles about 10. The teacher/student boundary is cardinality, not intelligence. That single fact determines the entire architecture in Part VII.
Part VII — The FrontierRank architecture
Existing rerankers spend compute in proportion to candidate count. FrontierRank spends it in proportion to ranking regret: the expected nDCG@k you lose by stopping now.
flowchart TD
  Q[Query] --> R[Hybrid retrieval<br/>BM25 + dense + LI]
  R -->|100-300 candidates| S[Evidence slicer<br/>signatures, no LLM]
  S --> L[Anchored Laya slates<br/>13 passes x 10 options]
  L --> C[Anchor calibration<br/>one global utility scale]
  C --> F[Top-k regret frontier]
  F -->|regret below price| OUT[Final top-k]
  F -->|regret above price| CTL{EVI controller}
  CTL -->|more evidence| S
  CTL -->|more candidates| R
  CTL -->|buy a judgment| J[Jev: one batched rubric<br/>over the frontier]
  J --> C
  J --> B[(Distillation buffer)]
  B -.periodic retrain.-> L
1. Signatures, not documents
Laya's English checkpoint has ~320 tokens of state budget. Feeding it documents is an architectural mistake. Build a deterministic signature per candidate: title, section path, metadata, the best lexical window, the best dense window. No LLM, no extra model — you already have the BM25 term positions and the chunk embeddings from retrieval. Under this framing the short context stops being a limitation and becomes an evidence-selection problem, which is separately optimisable and separately testable.
2. Anchored slates — the scoring substrate
Every slate carries A pivot documents of known grade alongside M − A real candidates. The pivots are fixed across the whole corpus, held out of evaluation, and chosen to span the utility range.
Why: the slate softmax hides c_S (Part I). The anchors observe it. Fit per slate
\hat{u}_i = \alpha_S \log p_i + \beta_S, \qquad
(\alpha_S, \beta_S) = \arg\min_{\alpha,\beta} \sum_{a \in \mathcal{A}_S} \big(\alpha \log p_a + \beta - u_a\big)^2 + \lambda(\alpha - 1)^2
and every slate's candidates land on one global scale. The ridge term on the slope matters: with A=2 an unregularised fit interpolates two points exactly and the slope explodes. Part VIII has the numbers.
Budget and slating. Laya's cliff is at 11 options, so M = 10. Deal candidates into slates round-robin by first-stage rank, not in blocks — stratification is free and it is what carries the within-query ranking (Part VIII). A = 4 anchors then leaves 6 real candidates per slate, 17 passes for 100 candidates, ~240 ms on a T4. The anchors are not paying for the ranking; they are paying for the absolute scale that the frontier, the thresholds and the re-scoring after read_more all require.
This replaces the five-axis factorisation, and that is the single largest cost correction in the design. Asking Laya P_rel, P_direct, P_constraints, P_contradiction, P_authority per candidate is 5 questions × 100 candidates = 500 encoder passes, because Laya does not share state across questions. One anchored Choice slate pass per slate is 13. If you want the extra axes, add one ordinal Score pass per slate (26 total) or fold the distinctions into the anchor grades — do not fan out per candidate.
3. The regret frontier
With calibrated u_i and standard errors σ_i, candidate i is on the frontier if its interval overlaps the interval of the candidate at rank K, or its swap probability clears a floor. Everything strictly above stays in, everything strictly below stays out, and no compute spent on them can change nDCG@K.
Then the quantity that actually drives the system:
\mathbb{E}[\mathrm{Regret}] = \mathbb{E}_{\tilde{u} \sim \mathcal{N}(\hat{u}, \sigma)}\Big[ \max\big(\mathrm{nDCG}@k(\tilde{u}) - \mathrm{nDCG}@k(\hat{u}), 0\big) \Big]
Estimated by sampling utilities and re-ranking. This is why calibration is load-bearing rather than decorative: if σ is wrong, the regret estimate is wrong, and every downstream decision inherits the error.
4. The controller
a^{*} = \arg\max_{a \in \mathcal{A}} \Big( \mathbb{E}[\Delta \mathrm{nDCG}@k \mid s, a] - \lambda_C \, C(a) - \lambda_L \, L(a) \Big), \qquad \text{stop if } a^{*} \le 0
Action
Marginal cost
Latency
What it buys
stop
$0
0 ms
—
read_more(d)
$0.0000042
~45 ms
more information
widen_retrieval
$0.00002
~120 ms
a better pool
teacher_choice
$0.000252
~300 ms
relative ordering, 255 slots
teacher_rubric
$0.000336
~300 ms
absolute grades, 255 slots
λ_C converts dollars into nDCG points and is the operating point — sweep it to draw the Pareto curve rather than tuning it to hit a target escalation rate, which inverts cause and effect.
Start with supervised counterfactual estimation, not RL: log every action's realised outcome offline, fit f(s,a) → V(s,a) with LightGBM, act greedily. Bandits only once production feedback exists.
read_more is the underrated action. A Laya score of 0.54 may mean the candidate is genuinely ambiguous — or that the evidence was not in the 320-token signature. Those are different failures and the second is ~80× cheaper to fix. The controller choosing between more intelligence and more information is the part of this design with the least prior art.
5. Jev as a large-slate teacher
Escalation sends the top ~30 frontier candidates in one shared state and asks a 4-level rubric question each — the configuration measured at 0.692 nDCG@10 for $0.45/1k, which beat both the single Choice and every per-pair variant. Jev's latency is flat in option count, so a 30-candidate escalation costs the same wall-clock as a 5-candidate one; there is no reason to send fewer than the frontier.
This is also where the cardinality argument pays off. Laya at M=10 must slate; Jev at M=255 does not. Anchoring is what lets the 10-option model approximate the 255-option one, and escalation is what you do when the approximation is not good enough.
Do not treat Jev as ground truth. Under a matched protocol it is statistically indistinguishable from Laya on ordinary classification (Part VI). The hierarchy is human > consensus of several teachers > Jev > weak labels, and the highest-value training examples are confident disagreements between student and teacher.
6. Cost annealing, with its floor stated
Every escalation writes (query, slate, teacher distribution) to a buffer. Periodic fine-tuning absorbs it, the student gets confident where the teacher used to be needed, and escalation falls.
Labels accumulated
Escalation
$ / 1k queries
$ / 1M searches
0 (day 1)
30%
$0.180
$180
25k
15%
$0.130
$130
100k
8%
$0.106
$106
500k
4%
$0.093
$93
2M
2%
$0.086
$86
floor (no escalation at all)
0%
$0.080
$80
Laya at A=4 on a T4 at 30% utilisation, Jev at 8,000 tokens per escalation. The floor row is the point of the table.
The honest read: annealing saturates. From 30% to 8% escalation saves $74 per million searches; from 8% to 2% saves $20. Below ~5% the local GPU floor dominates and further annealing is noise. The value of the flywheel past that point is quality, not cost — the student gets better, so the same escalation budget buys more. Framing cost annealing as the headline result overstates it; framing it as quality annealing at fixed cost is defensible.
Complexity
Stage
Passes / query
Wall clock
$ / 1k queries
Retrieval + slicing
—
~20 ms
~0
Laya anchored slates, A=4, stratified
17
~240 ms (T4)
$0.024 @100% util
Frontier + regret, 256 samples
0
~2 ms CPU
~0
Jev escalation @ 5%
0.05 calls
+300 ms on 5% of queries
$0.017
Total

~260 ms P50
$0.041
At 30% GPU utilisation: $0.096 / 1k queries, against $0.22 for the cheapest commercial reranker and $2.51 for Cohere Rerank 4 Pro. An L40S instead of a T4 cuts the local term ~5.6×, bringing the total to about $0.051 / 1k at the same utilisation.
Part VIII — Why anchored slates work
Running the implementation corrected my own first claim. The corrected version is sharper: stratification and anchoring fix two different problems, and only one of them is the one FrontierRank actually needs.
The identifiability argument
A slate softmax is invariant to adding a constant to every logit in the slate, so log p determines l only up to c_S:
\log p_i = l_i - c_S \quad \Longrightarrow \quad \{l_i\}_{i \in S} \text{ is identified only modulo an additive constant}
Within a slate this is harmless. Across slates it is fatal, and the damage scales with how much c_S varies between slates. A anchors with known u_a turn the unidentified family into a regression: one anchor recovers the shift, two or more also recover slate-dependent scale.
Simulation
Latent utilities u ~ N(0, 1.5) over 100 candidates, grades 0–3 at the 70/90/97th percentiles, marker noise σ = 0.35, slate temperature drift σ = 0.15, M = 10 option slots, ridge estimator λ = 1.0, 300–600 queries per cell. Two slating policies: blocked (slate k gets ranks 8k…8k+7, the obvious implementation) and stratified (round-robin deal, so every slate spans the rank range).
A. Within-query ranking, nDCG@10
Slating
Anchors
Naive
Anchored
Δ
blocked
2
0.478
0.855
+0.377
blocked
4
0.679
0.902
+0.223
stratified
2
0.869
0.845
−0.024
stratified
4
0.895
0.897
+0.002
B. Cross-query absolute scale — correlation with true latent utility, pooled over every query
Slating
Anchors
Naive r
Anchored r
Anchored RMSE
blocked
2
0.525
0.933
0.565
blocked
4
0.853
0.961
0.425
stratified
2
0.897
0.935
0.560
stratified
4
0.918
0.961
0.424
Naive log-probabilities have no absolute scale at all, so RMSE against a true utility is undefined for them.
What the numbers actually say
Stratifying the slates is a free fix for within-query ranking. Round-robin dealing takes blocked-slate nDCG@10 from 0.478 to 0.869 at zero cost, because it makes slate composition — and therefore c_S — nearly constant. If all you want is a ranking inside one query's pool, stratify and stop; you do not need anchors. My first draft claimed anchoring was worth +0.41 nDCG@10. It is, but only against the naive blocked baseline, which is a strawman once you know to stratify.
Anchors buy the absolute scale, which stratification cannot. Pooled across queries, the correlation with true utility goes 0.918 → 0.961 at A=4, and you get an interpretable utility with RMSE 0.42 rather than an uncalibrated log-probability. That is the property FrontierRank depends on, in four places:
1. Regret estimation needs σ in utility units, not in per-slate log-probability units.
2. Thresholding — "send everything above grade 2 to the generator" — requires an absolute scale by definition.
3. Re-scoring across rounds. After read_more or an escalation, the frontier is re-slated. Round-1 and round-2 scores are incomparable without a fixed reference; anchors are that reference.
4. Merging pools after widen_retrieval.
Anchors at low A cost a little ranking quality under stratification (−0.024 at A=2) — the two-point fit adds variance where there was little offset to remove. At A=4 it is free (+0.002) and the scale is materially better. Use A = 4 with stratified slates, at 17 passes per 100 candidates.
If you must slate blocked — streaming, incremental reranking, or a pool you cannot see in full — anchors are not optional. That is the +0.223 row.
What this is not
A simulation under a Luce-with-drift generative model. It establishes that the mechanism is sound, that the magnitudes are worth chasing, and — usefully — that one of the obvious framings of it was wrong. It does not establish that Laya's marker softmax behaves like a Luce model on real text. That is the assumption everything rests on.
Falsification
Run these in order. They are cheap, and they are the first go/no-go gate.
1. Is the shift real? Score the same candidate in slates of deliberately different mean quality. If log p_i moves by a roughly constant amount per slate, c_S exists as modelled. If it moves idiosyncratically per candidate, anchoring cannot work.
2. Do anchors recover it? Regress known-grade pivots on their observed log-probabilities and report the residual. Residuals on the order of the marker noise mean the affine model is adequate; systematically curved residuals mean you need a monotone spline.
3. Does anchoring beat stratification alone? This is now the real control, and it is the test my first draft was missing. On nDCG@10 expect roughly a tie; the claim to defend is the cross-query scale and the multi-round re-scoring, so measure those.
4. Does the absolute scale transfer across queries? Pool calibrated utilities over many queries and correlate against graded labels. If pooled correlation is no better than naive, anchoring has bought nothing that matters.
5. Does anchor choice matter more than it should? Swap the pivot set. Swings above ~1 nDCG@10 point across reasonable pivot sets mean the method is fragile and needs many pivots sampled per query.
Test 1 decides whether any of this survives. Test 3 decides whether it is a contribution or a strawman.
Part IX — Fine-tuning recipe
Laya must be fine-tuned. Its own card says base checkpoints are below the majority-class baseline on typed decisions. The published path works — laya_finetune_typed_decisions_2xT4_kaggle.ipynb, roughly 4–5 hours on Kaggle's free 2×T4, ~4 epochs over ~30k questions — so the question is what to feed it, not whether it can learn.
Heads
Three outputs, one backbone:
1. Slate head — the existing shared scalar MLP over [MASK] markers. Trained listwise. This is the ranking signal.
2. Ordinal head — a new CORAL head emitting K−1 cumulative logits P(y > k) on the 0–3 TREC scale. Laya's score primitive is currently a Choice with ordinal-shaped rewards, not a genuinely ordinal head; adding one gives monotone grades and a globally comparable pointwise score for free.
3. Act head — already present, taking top-1 probability, top-1/top-2 margin, entropy, and normalised option count. Retarget it from "escalate to a human" to "will a teacher call change the top-k?", which is what the controller actually needs.
The loss
\mathcal{L} = w_{\text{pair}}\mathcal{L}_{\lambda\text{-RankNet}} + w_{\text{list}}\mathcal{L}_{\text{ListNet}} + w_{\text{kd}}\,\mathrm{KL}(t \,\|\, s) + w_{\text{ord}}\mathcal{L}_{\text{CORAL}} + w_{\text{cal}}\mathcal{L}_{\text{RPS}}
Starting weights: pair 1.0, list 1.0, kd 2.0, ord 0.5, cal 1.0.
• λ-RankNet weights each pair by |ΔnDCG| if swapped, so confusing rank 1 with rank 100 costs far more than 76 with 77.
• ListNet aligns the slate softmax with a gain-softmax over graded labels — the listwise term the marker mechanism is naturally shaped for.
• KL on the teacher's distribution, not its argmax. P(0)=.01, P(1)=.04, P(2)=.21, P(3)=.74 carries the teacher's uncertainty, which is exactly the signal telling the student where its own uncertainty should live. Distilling hard labels throws that away and is the most common way to get a confident, badly calibrated student.
• RPS is strictly proper and ordinal-aware — being off by one level should cost less than three. Laya's own RLCD already applies it to score questions at w_rps = 1.0. Keep it.
Data construction matters more than the loss
This is where the effort goes. A negative like "Paris is the capital of France" teaches topic detection. The negative that teaches relevance is the one that differs in exactly one logical commitment:
Positive: Device warranty applies for 24 months except batteries, which are limited to 12 months. Hard negative: Device warranty applies for 24 months including batteries.
Build the pool deliberately across: near-duplicate constraint violations, negations, wrong dates, wrong numbers, right entity with wrong relationship, superseded versions, partial answers, indirect answers, same-topic non-answers, contradictory evidence.
Two rules from Jina's v3.5 work, which is the strongest public evidence on this:
• Mine negatives from several retrievers — BM25, a dense model, a late-interaction model, at minimum. Negatives from one retriever teach that retriever's failure modes rather than relevance.
• For structured fields, generate perturbations programmatically — near-identical records with exactly one constrained field changed. Cheap, and it targets the failure mode general rerankers are worst at.
Slate construction for training
Training slates must mirror inference slates or the calibration will not transfer:
• M = 10 options: M − A real candidates plus the same A anchors used at inference
• Anchors held out of evaluation entirely, and never used as positives or negatives
• Vary slate difficulty deliberately — some all-hard, some mixed, some easy — so the ordinal head sees the full grade range and the act head learns when the frontier is genuinely contested
• Randomise option order within a slate every epoch; the mechanism is position-robust, but do not assume it
Calibration stage
Non-negotiable, and separate from training. Laya ships over-confident (ECE 0.466 → 0.081 after refit), and its shipped temperature_by_options map is per (type, option-count) bucket. Do the same on your domain: fit a temperature per bucket on held-out data, then verify per-slice rather than in aggregate. Laya's own eval reports 0.030 aggregate ECE hiding a family at 0.438 — an aggregate ECE is not evidence of calibration.
For the frontier you need interval coverage, not just ECE. Use split conformal on held-out queries to turn σ_i into intervals with a guaranteed marginal coverage rate. The regret estimate is only as good as those intervals.
Data volumes
Stage
Queries
Judgments
Goal
Proof of concept
300–500
10k–25k
does the anchor mechanism transfer at all
Useful domain model
2k–5k
50k–250k
beat a generic 0.6B cross-encoder
Strong specialist
10k–30k
500k–2M
approach a 4B generalist
Production flywheel
continuous
millions
quality annealing at fixed escalation
Humans label queries and the contested candidates. The teacher expands to soft labels. Then mine disagreements — a confident student against a confident teacher is the highest-information example you can buy, and it is free, because you already paid for the teacher call.
Curriculum
1. Warm start on public graded data (MS MARCO with distilled grades, TREC DL) to teach the slate head what ranking is — the base checkpoint effectively does not know.
2. Anchor-aware stage: introduce pivots, freeze nothing, train all three heads jointly.
3. Domain stage: your data, hard negatives, teacher distributions.
4. Calibration stage: freeze the backbone, fit temperatures and conformal quantiles.
Budget: stages 1–3 at roughly the published scale (a few hours on 2×T4 per stage for tens of thousands of slates); stage 4 is minutes. This is a cheap model to iterate on, which is its main advantage over a 4B student.
Ablation ladder
Remove one thing at a time from the full system and report ΔnDCG@10, Δcost, ΔP95:
no anchors · anchors but no calibration fit · shift-only instead of ridge · no ordinal head · no teacher distillation (hard labels only) · no hard-negative mining · single-retriever negatives · fixed escalation threshold instead of the controller · no read_more · no widen_retrieval · no frontier (escalate everything) · no escalation at all.
The two that decide whether the paper exists are no anchors and fixed threshold instead of the controller. If either ablation costs little, the corresponding contribution is not real.
Part X — Evaluation protocol
The deliverable is a Pareto frontier, not a leaderboard row. FrontierRank wins only if it pushes the quality/cost curve outward. Highest nDCG@10 alone is not the claim and should not be reported as if it were.
Held constant across every row
First stage fixed: hybrid BM25 + a dense retriever, RRF-fused, top-100, identical pool for every system. Passage truncation identical. Datasets fixed and named. Anchors held out of all qrels. Report the first stage's own nDCG@10 and Recall@100 as the floor and the ceiling.
Without this, you are measuring the retriever — as Part II's 10.7-point RankZephyr swing shows.
Grid
Row
What it isolates
BM25, dense, hybrid
the floor
Laya zero-shot
how far below useful the base checkpoint is
Laya fine-tuned, naive slate concatenation
the bug, quantified on real data
Laya fine-tuned, anchored slates
the core contribution
+ fixed-threshold escalation
is the controller doing anything
+ full controller (FrontierRank)
the system
Jev only, batched 4-level rubric
the measured 0.692 configuration
Qwen3-Reranker-0.6B, -4B
the honest open-weight competition
jina-reranker-v3.5
current best verified, note the non-commercial licence
ZeroEntropy zerank-2, Cohere Rerank 4 Pro
the commercial cost anchors
Sweep λ_C across at least six values per FrontierRank row. Each value is a point on the curve; the curve is the result.
Metrics, per row
nDCG@10 · Recall@10 · MRR@10 · ECE and Brier per option-count bucket · conformal interval coverage · escalation rate · $/1k queries split local vs teacher · P50/P95/P99 · Kendall tau against the teacher's own ranking on escalated queries.
For binary metrics on TREC DL, run trec_eval -l 2. Grade 1 is not relevant.
Statistics
Paired bootstrap over queries, 10,000 resamples, report the 95% interval on every difference. Holm–Bonferroni across the grid. Three seeds minimum for anything involving fine-tuning, reporting seed variance separately from query variance — on TREC DL's 43 judged queries, differences under ~2 nDCG@10 points are noise and should be stated as such rather than claimed.
Borrow the jev-rerank-bench discipline: report both equal-weight-per-dataset and equal-weight-per-query averages. On that benchmark the two weightings reverse the Jev/Cohere ordering. If your two weightings disagree, say so rather than picking the flattering one.
Phases and gates
Phase 0 — the mechanism (1 week). Run falsification tests 1 and 2 from Part VIII on TREC DL19 with a base Laya checkpoint. No training. → Gate: c_S behaves as a per-slate shift, and anchor regression residuals are on the order of marker noise. If not, stop. The rest of the design does not survive.
Phase 1 — anchored scoring (2–3 weeks). Fine-tune on public graded data. Compare anchored vs naive vs pointwise-rubric at fixed compute. → Gate: anchored beats naive by a large margin, and beats the pointwise rubric at equal or lower pass count. If anchoring only matches the rubric, use the rubric — it is simpler.
Phase 2 — the frontier (2 weeks). Add calibration, conformal intervals, regret estimation, fixed-threshold escalation to Jev. → Gate: matches Qwen3-Reranker-4B quality at under 10% escalation. This is the result that makes the system interesting on its own.
Phase 3 — the controller (3–4 weeks). Offline counterfactual logging of all actions, fit V(s,a), replace the threshold. → Gate: the controller beats the best fixed threshold at matched cost. If it does not, the multi-action claim is unsupported — report that, and ship the threshold.
Phase 4 — annealing (ongoing). Retrain on the distillation buffer at intervals; track escalation and quality. → Gate: quality rises at fixed escalation rate. Note from Part VII that the cost curve saturates below ~5% escalation; the claim to test is quality annealing, not cost annealing.
The figure
One scatter: dollars per million searches on x (log scale), nDCG@10 on y, one point per configuration, FrontierRank's λ_C sweep as a connected curve. Every competitor is a single point. If the curve does not pass above and to the left of those points, there is no result.
Part XI — Risks and limitations
The ones that can kill it
The Luce assumption may not hold. Everything rests on the marker softmax behaving as a shift of a latent logit. If Laya's slate behaviour is composition-dependent in ways an affine map cannot absorb — a candidate scoring differently because of which other candidates are present, not just how many or how good — then anchoring recovers less than the simulation suggests. Phase 0 tests exactly this, and it is the first thing to run.
Sub-1B is a hard ceiling on reasoning and instructions. Part II's evidence is unambiguous: a 0.5B→7B jump roughly doubles BRIGHT, and FollowIR p-MRR flips sign between sub-1B models and 4B. If your queries carry per-query instructions or need inference rather than matching, a 421M student is the wrong substrate and no amount of calibration fixes it. Escalation partly covers this — but if most queries need reasoning, you are escalating most queries, and the economics collapse into "Jev with extra steps".
Laya is near-random zero-shot. 0.362 against a 0.461 majority-class baseline. Every claim here assumes domain fine-tuning works on your data. If you cannot produce a few thousand graded queries, this is the wrong architecture and a pretrained cross-encoder is strictly better.
The teacher may not be much of a teacher. Under the one matched-protocol independent test, Jev and Laya are statistically indistinguishable on ordinary classification, and Laya is worse calibrated. The defensible justification for escalation is cardinality (255 vs ~10 options) and Banking77-style evidence — not general superiority. If Phase 2 shows escalation adds little, the honest conclusion is that the cascade is unnecessary, and the anchored scorer is the whole contribution.
The ones you manage
Risk
Mitigation
512-token context; signature quality becomes the bottleneck
make read_more a first-class action; measure signature recall separately from ranking quality
Anchor drift as the corpus changes
re-fit anchor utilities on a schedule; monitor per-slate residuals and alert on drift
Calibration decays under distribution shift
conformal intervals with periodic re-fit; track coverage, not just ECE
Slate composition bias
stratify slates by first-stage rank rather than blocking; randomise within-slate order
Jev is early access, closed, no weights, no SLA
keep the teacher behind an interface; a Qwen3-Reranker-4B or an LLM judge must be swappable in one config line
Jev is not hardened against prompt injection — vendor's own words
never put untrusted document text where it can read as instructions; treat retrieved content as hostile
Jev has no structural invariants (complementary Nouls summing to 1.19)
do not build logic that assumes coherence across primitives; use one primitive per decision
Licence traps
jina-reranker-v3.5 is CC BY-NC-SA. Fine as a benchmark row, not as a production component
Novelty, stated honestly
Budget-aware allocation exists (EcoRank). Uncertainty-driven adaptive reranking exists (AcuRank). Exploiting the initial ranking to skip comparisons exists (Setwise Insertion). Value-of-information control exists in adjacent agent work. Teacher-student rank distillation is old. None of the components is new.
The two things that look genuinely unclaimed: the anchor mechanism for cross-slate identifiability in typed-decision listwise scoring, and a controller choosing between buying more intelligence and buying more information under one regret budget. Both are testable and both have a named falsification in Part VIII. No patent search was done here, so treat this as a plausible hypothesis, not a priority claim.
When to not build this
flowchart TD
  A{Per-query instructions<br/>or reasoning queries?} -->|yes| B[Use Qwen3-Reranker-4B.<br/>Sub-1B fails here.]
  A -->|no| C{Can you produce<br/>3k+ graded queries?}
  C -->|no| D[Use a pretrained<br/>cross-encoder.]
  C -->|yes| E{Volume above<br/>~300k queries/month?}
  E -->|no| F[Rent zerank-2.<br/>Not worth the ops.]
  E -->|yes| G[FrontierRank.<br/>Run Phase 0 first.]
The uncomfortable comparison to keep in view: a 6-layer MiniLM cross-encoder reranking BM25 top-100 lifts BEIR from 0.408 to 0.475 — the largest per-parameter gain anywhere in this document, available this afternoon, at 1,800 documents/second. Everything here is chasing the last few points beyond that. Worth doing at scale; not worth doing first.
Appendix — What ships alongside this
frontierrank/ — 1,426 lines, numpy-only for everything except the losses, 9 tests passing, runs on CPU in seconds with no model weights and no API key.
Module
Contents
slates.py
slate construction under a hard option-slot budget; blocked vs stratified dealing
calibrate.py
anchor estimators — shift, affine OLS, ridge — recovering the per-slate offset
score.py
AnchoredScorer: slates → one globally comparable utility + per-candidate σ
frontier.py
top-k regret frontier, swap probabilities, expected nDCG@k regret by sampling
controller.py
EVI action choice over stop / read_more / widen / teacher, with seeded priors
teacher.py
TeacherProtocol + a Jev batched-rubric implementation and the 4-level TREC rubric
losses.py
λ-RankNet, ListNet, teacher-distribution KL, CORAL, RPS, and the composite
evaluate.py
nDCG with the TREC convention, ECE, paired bootstrap
scripts/smoke_test.py
end-to-end on a synthetic Luce scorer; reproduces Part VIII's tables
scripts/anchor_budget_sweep.py
anchors against slate capacity
scripts/cost_model.py
every figure in Part IV, with the arithmetic in-file
Plugging in a real model
ScorerProtocol is the entire contract — choice_logprobs(instructions, options, state) -> np.ndarray. Laya's SDK Router fits it directly, as does a Jev Choice call. TeacherProtocol is equally thin, so Jev swaps for a Qwen3-Reranker or an LLM judge in one line. Nothing above teacher.py imports a vendor name, deliberately: Jev is closed-weights, early access, and cannot be fine-tuned, so all vendor risk sits on that one interface.
Constraints encoded in the code, not just the prose
• M = 10 option slots for Laya, from the choice:11+ → T = 0.1006 tell. The cliff is at 11, not 20.
• State is not shared across questions — ~14 ms marginal per question on T4. The API makes fanning out per candidate awkward on purpose.
• Jev caps at 255 options and its latency is flat in option count, so JevTeacher sends the whole frontier.
• recall_at_k and mrr_at_k default to threshold=2, matching trec_eval -l 2. TREC DL grade 1 is not relevant.
• Badly-fitting slates inflate σ rather than being silently trusted — a slate whose anchors misbehave produces low-confidence candidates, which is what pushes them onto the frontier.
Out of scope
Retrieval, the evidence slicer, and the BEIR/TREC harness depend on your index and are deliberately absent. AnchoredScorer.score() takes signatures plus a first-stage order; produce those however you like.
Sources
Laya model card · laya-typed-decisions · TypeSafe models & pricing · TypeSafe reranking cookbook · jev-rerank-bench · jev-vs-open-decision-models · BEIR · BRIGHT · BRIGHT reproducibility audit · Reranker-Guided Search · FollowIR · Qwen3 Embedding & Reranker · jina-reranker-v3 · jina-reranker-v3.5 · RankZephyr · RankGPT · Rank1 · E2R-FLOPs · AcuRank · HeadRank · MICE · Rerank Before You Reason · EcoRank · Setwise Insertion · DIVER-v3 · ModernBERT · Voyage pricing · ZeroEntropy pricing