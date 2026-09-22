"""Qwen3-Reranker as a baseline. A decoder scoring yes/no, not a cross-encoder.

Part II puts Qwen3-Reranker-0.6B at 56.28 BEIR (13 datasets) and, more
importantly for this project, at FollowIR p-MRR +5.41 where every other sub-1B
model is negative. If per-query instructions ever matter, this is the family
that can read them and Laya is not.

Mechanism: the model is asked a yes/no question and the answer is read off the
logits of the "yes" and "no" tokens at the first generated position. That makes
it *first-token logit decoding* (Part I, mechanism 2) rather than generative
permutation decoding, so there are no format failures -- and the softmax over
those two tokens is a genuine probability, which makes its calibration directly
comparable to Laya's.
"""
from __future__ import annotations

import time

import numpy as np
import torch

__all__ = ["QwenReranker"]

PREFIX = ("<|im_start|>system\nJudge whether the Document meets the requirements "
          "based on the Query and the Instruct provided. Note that the answer "
          "can only be \"yes\" or \"no\".<|im_end|>\n<|im_start|>user\n")
SUFFIX = ("<|im_end|>\n<|im_start|>assistant\n<think>\n\n</think>\n\n")
DEFAULT_INSTRUCT = ("Given a web search query, retrieve relevant passages that "
                    "answer the query")


class QwenReranker:
    def __init__(self, model_dir: str, device=None, max_length: int = 512,
                 batch_size: int = 16, instruct: str = DEFAULT_INSTRUCT,
                 name: str = "qwen3-reranker-0.6b", dtype=torch.float16):
        """fp16 here, fp32 for Laya, and the asymmetry is deliberate.

        Laya's whole claim is calibrated probabilities, so it is scored in
        fp32 to keep the numbers exact. Qwen is being measured as a *quality*
        baseline and is 27x slower on this hardware, so fp16 is the difference
        between a 30-minute run and a 2.3-hour one. If Qwen's calibration ends
        up load-bearing for a decision, re-run it in fp32.
        """
        from transformers import AutoModelForCausalLM, AutoTokenizer

        from ..laya.runtime import pick_device
        self.device = pick_device(device)
        self.tok = AutoTokenizer.from_pretrained(model_dir, padding_side="left")
        self.model = AutoModelForCausalLM.from_pretrained(
            model_dir, dtype=dtype).to(self.device).eval()
        self.dtype = dtype
        self.max_length = max_length
        self.batch_size = batch_size
        self.instruct = instruct
        self.name = name
        self.yes_id = self.tok.convert_tokens_to_ids("yes")
        self.no_id = self.tok.convert_tokens_to_ids("no")
        if self.yes_id is None or self.no_id is None:
            raise ValueError("tokenizer has no single-token 'yes'/'no'")
        self.prefix_ids = self.tok.encode(PREFIX, add_special_tokens=False)
        self.suffix_ids = self.tok.encode(SUFFIX, add_special_tokens=False)
        self.n_pairs = 0
        self.total_ms = 0.0

    def _build(self, query: str, doc: str) -> list[int]:
        body = f"<Instruct>: {self.instruct}\n<Query>: {query}\n<Document>: {doc}"
        room = self.max_length - len(self.prefix_ids) - len(self.suffix_ids)
        ids = self.tok.encode(body, add_special_tokens=False)[:max(8, room)]
        return self.prefix_ids + ids + self.suffix_ids

    @torch.no_grad()
    def probability(self, query: str, documents) -> np.ndarray:
        """P(yes) from the softmax over the yes/no logits at the next position."""
        docs = list(documents)
        out = np.empty(len(docs), dtype=float)
        pad = self.tok.pad_token_id or self.tok.eos_token_id
        for i in range(0, len(docs), self.batch_size):
            chunk = [self._build(query, d) for d in docs[i:i + self.batch_size]]
            L = max(len(c) for c in chunk)
            # left padding, so the last position is the answer slot for every row
            ids = torch.full((len(chunk), L), pad, dtype=torch.long)
            att = torch.zeros((len(chunk), L), dtype=torch.long)
            for r, c in enumerate(chunk):
                ids[r, L - len(c):] = torch.tensor(c)
                att[r, L - len(c):] = 1
            t0 = time.perf_counter()
            logits = self.model(input_ids=ids.to(self.device),
                                attention_mask=att.to(self.device)).logits[:, -1, :]
            if self.device.type == "mps":
                torch.mps.synchronize()
            self.total_ms += (time.perf_counter() - t0) * 1e3
            self.n_pairs += len(chunk)
            pair = torch.stack([logits[:, self.no_id], logits[:, self.yes_id]], -1)
            out[i:i + len(chunk)] = torch.softmax(pair.float(), -1)[:, 1].cpu().numpy()
        return out

    def score(self, query: str, documents) -> np.ndarray:
        return self.probability(query, documents)

    def throughput(self) -> dict:
        return {"pairs": self.n_pairs, "total_ms": round(self.total_ms, 1),
                "ms_per_pair": round(self.total_ms / max(1, self.n_pairs), 3),
                "pairs_per_s": round(self.n_pairs / max(1e-9, self.total_ms / 1e3), 1),
                "device": str(self.device)}
