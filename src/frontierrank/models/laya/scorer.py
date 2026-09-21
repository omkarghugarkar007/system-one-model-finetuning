"""Laya as a `ScorerProtocol`: one slate in, one log-probability vector out.

Two entry points, and the batched one is not an optimisation detail. Laya does
not share state across questions -- `build_sequence` re-appends the full state
per question -- so a 100-candidate query is 17 independent 512-token sequences.
Running them one at a time on MPS wastes most of the device. `score_many`
batches them into one forward pass; `choice_logprobs` exists to satisfy the
protocol and for single-slate probing.
"""
from __future__ import annotations

from typing import Sequence

import numpy as np

from ...core.packing import PackedSlate
from .runtime import LayaRuntime

__all__ = ["LayaScorer"]


class LayaScorer:
    """Wraps a `LayaRuntime` and exposes the scoring contract.

    `max_batch_slates` is a memory knob, not a throughput one: 512 tokens x
    1024 hidden x 28 layers of activations adds up on 16 GB of unified memory.
    16 slates is comfortable on an M4; raise it on a real GPU.
    """

    def __init__(self, runtime: LayaRuntime, qtype: str = "choice",
                 max_batch_slates: int = 16):
        self.rt = runtime
        self.qtype = qtype
        self.max_batch_slates = max_batch_slates
        self.build_info: list[dict] = []

    def token_counter(self):
        """Exact token counts for the packers, using Laya's own tokenizer."""
        tok = self.rt.tok
        return lambda s: len(tok(s, add_special_tokens=False)["input_ids"])

    # --------------------------------------------------------------- protocol
    def choice_logprobs(self, instructions: str, options: Sequence[str],
                        state) -> np.ndarray:
        built = self.rt.build(state, self.qtype, instructions, list(options),
                              rendered=True)
        self.build_info.append(built[2])
        z = self.rt.run_batch([built], self.qtype)[0]
        return self.rt.log_softmax(z)

    # ---------------------------------------------------------------- batched
    def score_many(self, slates: Sequence[PackedSlate]) -> list[np.ndarray]:
        """Log-probabilities for many slates, batched under the memory cap."""
        out: list[np.ndarray] = []
        for i in range(0, len(slates), self.max_batch_slates):
            chunk = list(slates[i:i + self.max_batch_slates])
            built = [self.rt.build(s.state, self.qtype, s.instructions,
                                   list(s.options), rendered=True) for s in chunk]
            self.build_info.extend(b[2] for b in built)
            out.extend(self.rt.log_softmax(z)
                       for z in self.rt.run_batch(built, self.qtype))
        return out

    # ------------------------------------------------------------ diagnostics
    def budget_report(self) -> dict:
        """What the token budget actually did across every slate built so far.

        Worth printing on any new corpus. `state_truncated` firing means the
        signature builder is producing evidence the model never reads, which
        looks exactly like a model quality problem and is not one.
        """
        if not self.build_info:
            return {}
        opt = [t for b in self.build_info for t in b["option_text_tokens"]]
        raw = [t for b in self.build_info for t in b["option_text_tokens_untruncated"]]
        return {
            "slates": len(self.build_info),
            "option_tokens_mean": round(float(np.mean(opt)), 2),
            "option_tokens_min": int(np.min(opt)),
            "option_text_wanted_mean": round(float(np.mean(raw)), 2),
            "option_shrink_fired": float(np.mean(
                [b["option_shrink_fired"] for b in self.build_info])),
            "instruction_tokens_mean": round(float(np.mean(
                [b["instruction_tokens"] for b in self.build_info])), 2),
            "state_tokens_mean": round(float(np.mean(
                [b["state_tokens"] for b in self.build_info])), 2),
            "state_truncated_frac": round(float(np.mean(
                [b["state_truncated"] for b in self.build_info])), 3),
            "total_tokens_mean": round(float(np.mean(
                [b["total_tokens"] for b in self.build_info])), 1),
        }

    def reset_diagnostics(self):
        self.build_info.clear()
