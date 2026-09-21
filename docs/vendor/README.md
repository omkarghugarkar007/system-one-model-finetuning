# Vendored reference material

Pinned copies of third-party material the design depends on, so a claim in
`plan.md` can always be checked against what the vendor actually shipped.

| Path | What | Retrieved |
|---|---|---|
| `laya_src/rl_common.py` | Laya's `build_sequence`, `DecisionModel`, RLCD rewards. Apache-2.0, `convaiinnovations/laya`. | 2026-09-21 |
| `laya_src/rl_agent_api.py` | Laya's `system_one` inference path — the Jev-compatible surface. | 2026-09-21 |
| `laya_src/rl_agent_config.json` | `max_len=512`, `head_max_len=192`, the `temperature_by_options` map. | 2026-09-21 |
| `laya_src/encoder/config.json` | ModernBERT-large, 28 layers, hidden 1024. | 2026-09-21 |
| `laya_src/eval/results.json` | Laya's own per-family accuracy/ECE and the T4 latency numbers. | 2026-09-21 |
| `typesafe_docs.txt` | Full TypeSafe docs (`llms-full.txt`), including the System One HTTP API and the re-ranking cookbook. | 2026-09-21 |

These are reference only. Nothing in `src/` imports them.
