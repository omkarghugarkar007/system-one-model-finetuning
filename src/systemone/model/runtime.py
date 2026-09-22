"""Loading and running the real Laya checkpoint.

`DecisionModel` reproduces the shipped module tree exactly, because the
checkpoint is loaded with `strict=True` and any drift shows up as a key error
rather than as quietly wrong numbers. Derived from `rl_common.py` in
`convaiinnovations/laya` (Apache-2.0).

Device notes for this project's target (Apple M4, 16 GB unified):
  * fp32 on MPS. Autocast on MPS is not worth the numerical risk for a 395M
    encoder whose whole value proposition is calibrated probabilities.
  * `encoder.config.reference_compile = False` -- upstream disables it too;
    torch.compile loses on small batches.
  * Sequences are a fixed 512 tokens, so throughput is flat and easy to model.
"""
from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .sequence import QTYPES, build_sequence, render_options, temp_bucket

__all__ = ["DecisionModel", "LayaRuntime", "LayaConfig", "pick_device"]


def pick_device(prefer: str | None = None) -> torch.device:
    if prefer:
        return torch.device(prefer)
    if torch.cuda.is_available():
        return torch.device("cuda")
    if torch.backends.mps.is_available():
        return torch.device("mps")
    return torch.device("cpu")


class DecisionModel(nn.Module):
    """Bidirectional encoder + from-scratch decision head, shaped like the checkpoint."""

    def __init__(self, encoder: nn.Module, head_layers: int = 2, n_act: int = 2,
                 dropout: float = 0.1):
        super().__init__()
        self.encoder = encoder
        d = encoder.config.hidden_size
        nhead = max(1, d // 64)
        layer = nn.TransformerEncoderLayer(d, nhead, 4 * d, dropout,
                                           batch_first=True, norm_first=True)
        self.head = (nn.TransformerEncoder(layer, head_layers,
                                           enable_nested_tensor=False)
                     if head_layers > 0 else None)
        self.type_emb = nn.Embedding(3, d)
        # ONE shared scalar MLP over every marker -- there is no per-option head,
        # which is what makes the mechanism position-robust.
        self.scorer = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, d), nn.GELU(),
                                    nn.Linear(d, 1))
        self.act_head = nn.Sequential(nn.Linear(d + 4, 256), nn.GELU(),
                                      nn.Linear(256, n_act))
        self.register_buffer("temperature", torch.ones(3))

    def forward(self, input_ids, attention_mask, marker_pos, marker_mask, qtype):
        h = self.encoder(input_ids=input_ids,
                         attention_mask=attention_mask).last_hidden_state
        h = h + self.type_emb(qtype)[:, None, :]
        if self.head is not None:
            pad = ~attention_mask.bool()
            for layer in self.head.layers:
                h = layer(h, src_key_padding_mask=pad)
        idx = marker_pos.clamp(min=0)[:, :, None].expand(-1, -1, h.size(-1))
        m = torch.gather(h, 1, idx)
        logits = self.scorer(m).squeeze(-1).float()
        logits = logits.masked_fill(~marker_mask, -1e4)

        p = torch.softmax(logits.detach(), -1)
        k = marker_mask.sum(-1).clamp(min=2).float()
        ent = -(p * torch.log(p.clamp_min(1e-9))).sum(-1) / torch.log(k)
        top2 = p.topk(2, -1).values
        feats = torch.stack([top2[:, 0], top2[:, 0] - top2[:, 1], ent, k / 255.0], -1)
        act_logits = self.act_head(torch.cat([h[:, 0].float(), feats], -1))
        return logits, act_logits


@dataclass
class LayaConfig:
    max_len: int = 512
    head_max_len: int = 192
    head_layers: int = 2
    n_act: int = 2
    temperature: tuple = (1.0, 1.0, 1.0)
    temperature_by_options: dict = None

    @classmethod
    def from_dir(cls, d: str | Path) -> LayaConfig:
        cfg = json.loads((Path(d) / "rl_agent_config.json").read_text())
        return cls(max_len=cfg["max_len"], head_max_len=cfg["head_max_len"],
                   head_layers=cfg["head_layers"],
                   n_act=len(cfg.get("act_costs", {"escalate": 0.5})) + 1,
                   temperature=tuple(cfg.get("temperature", [1.0, 1.0, 1.0])),
                   temperature_by_options=cfg.get("temperature_by_options", {}) or {})


class LayaRuntime:
    """Batched marker-softmax inference over pre-built slates.

    `apply_temperature=False` is the setting Phase 0 needs. The shipped
    temperature map is fitted per (type, option-count) bucket on the vendor's
    own eval mix, and dividing the logits rescales every slate by the same
    constant -- which is fine for ranking but changes what the anchor
    regression is estimating. Fit calibration on your own data instead; the
    anchored affine map absorbs a global temperature into its slope anyway.
    """

    def __init__(self, model_dir: str | Path, device: str | None = None,
                 apply_temperature: bool = False, dtype: torch.dtype | None = None):
        from safetensors.torch import load_file
        from transformers import AutoConfig, AutoModel, AutoTokenizer

        self.dir = Path(model_dir)
        self.cfg = LayaConfig.from_dir(self.dir)
        self.device = pick_device(device)
        self.dtype = dtype or torch.float32
        self.apply_temperature = apply_temperature

        self.tok = AutoTokenizer.from_pretrained(str(self.dir / "tokenizer"))
        ecfg = AutoConfig.from_pretrained(str(self.dir / "encoder"))
        enc = AutoModel.from_config(ecfg, attn_implementation="sdpa")
        self.model = DecisionModel(enc, self.cfg.head_layers, self.cfg.n_act)
        missing = self.model.load_state_dict(
            load_file(str(self.dir / "model.safetensors")), strict=True)
        self.model.encoder.config.reference_compile = False
        self.model.to(self.device, dtype=self.dtype).eval()
        self._load_report = missing
        self.n_forward = 0
        self.n_slates = 0
        self.total_ms = 0.0

    # ------------------------------------------------------------------ build
    def build(self, state, qtype: str, instructions: str, options,
              rendered: bool = False, strict: bool = True):
        opts = options if rendered else render_options(qtype, options)
        return build_sequence(self.tok, state, qtype, instructions, opts,
                              self.cfg.max_len, self.cfg.head_max_len,
                              strict=strict)

    # -------------------------------------------------------------- inference
    @torch.no_grad()
    def run_batch(self, built: list, qtype: str = "choice") -> list[np.ndarray]:
        """`built` is a list of (ids, markers, info). Returns raw logits per slate."""
        if not built:
            return []
        n = len(built)
        L = max(len(b[0]) for b in built)
        kmax = max(len(b[1]) for b in built)
        pad = self.tok.pad_token_id

        ids = torch.full((n, L), pad, dtype=torch.long)
        att = torch.zeros((n, L), dtype=torch.long)
        mpos = torch.zeros((n, kmax), dtype=torch.long)
        mmask = torch.zeros((n, kmax), dtype=torch.bool)
        for i, (seq, mk, _) in enumerate(built):
            ids[i, :len(seq)] = torch.tensor(seq)
            att[i, :len(seq)] = 1
            mpos[i, :len(mk)] = torch.tensor(mk)
            mmask[i, :len(mk)] = True
        qt = torch.full((n,), QTYPES[qtype], dtype=torch.long)

        t0 = time.perf_counter()
        logits, _ = self.model(ids.to(self.device), att.to(self.device),
                               mpos.to(self.device), mmask.to(self.device),
                               qt.to(self.device))
        if self.device.type == "mps":
            torch.mps.synchronize()
        self.total_ms += (time.perf_counter() - t0) * 1e3
        self.n_forward += 1
        self.n_slates += n

        logits = logits.float().cpu().numpy()
        out = []
        for i, (_, mk, _) in enumerate(built):
            z = logits[i, :len(mk)].astype(float)
            if self.apply_temperature:
                t = (self.cfg.temperature_by_options or {}).get(
                    temp_bucket(qtype, len(mk)),
                    self.cfg.temperature[QTYPES[qtype]])
                z = z / t
            out.append(z)
        return out

    @staticmethod
    def log_softmax(z: np.ndarray) -> np.ndarray:
        z = np.asarray(z, dtype=float)
        m = z.max()
        return z - (m + np.log(np.exp(z - m).sum()))

    def throughput(self) -> dict:
        return {"forwards": self.n_forward, "slates": self.n_slates,
                "total_ms": round(self.total_ms, 1),
                "ms_per_slate": round(self.total_ms / max(1, self.n_slates), 2),
                "slates_per_s": round(self.n_slates / max(1e-9, self.total_ms / 1e3), 2),
                "device": str(self.device)}
