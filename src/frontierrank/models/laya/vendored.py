"""Faithful re-implementation of Laya's sequence construction and decision head.

Derived from `rl_common.py` in `convaiinnovations/laya` (Apache-2.0); a pinned
copy lives in `docs/vendor/laya_src/`. This is a re-implementation rather than
an import because the upstream file is a flat script that expects to be run
from the model directory, and because we need two things it does not offer:
the marker positions surfaced for inspection, and a hard error instead of
silent truncation when options do not fit.

The one behaviour to internalise, because it drives the whole slate design:

    opt_ids[i] = [MASK] + tokens(" " + option_i)[:48]
    if head_max_len - sum(len(opt_ids)) < 16:
        per = max(4, (head_max_len - 16) // n_options)
        opt_ids = [o[:per] for o in opt_ids]

At M = 10 options that leaves **16 tokens of option text per option**. See
`core.budget` for the arithmetic and what it forces.
"""
from __future__ import annotations

import json
from typing import Sequence

__all__ = ["QTYPES", "QTYPE_NAMES", "serialize_state", "render_options",
           "build_sequence", "temp_bucket", "OPTION_TOKEN_CAP"]

QTYPES = {"choice": 0, "score": 1, "noul": 2}
QTYPE_NAMES = {v: k for k, v in QTYPES.items()}

OPTION_TOKEN_CAP = 48      # the `[:48]` in build_sequence
MIN_OPT_BUDGET = 16        # the `< 16` that triggers the shrink
MIN_PER_OPTION = 4         # the `max(4, ...)` floor
MIN_HEAD_TOKENS = 8        # the `max(8, opt_budget)` floor


def serialize_state(state) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, ensure_ascii=False)


def render_options(qtype: str, criteria) -> list[str]:
    """Option texts in label-index order, exactly as upstream renders them.

    `score` levels become "level %d: %s" and are then treated as ordinary
    options -- Laya has no separate ordinal head, which is precisely why the
    fine-tuning recipe adds one.
    """
    if qtype == "choice":
        if isinstance(criteria, (list, tuple)):
            return [str(c) for c in criteria]
        return [k if not v else f"{k}: {v}" for k, v in criteria.items()]
    if qtype == "score":
        return [f"level {i}: {c}" for i, c in enumerate(criteria)]
    crit = criteria or {}
    return ["false: " + (crit.get("false") or "no, the statement does not hold"),
            "true: " + (crit.get("true") or "yes, the statement holds")]


def build_sequence(tok, state, qtype: str, instructions: str,
                   options: Sequence[str], max_len: int = 512,
                   head_max_len: int = 192,
                   option_order: Sequence[int] | None = None,
                   truncate_left: bool = False,
                   strict: bool = True):
    """[CLS] <type> instructions [SEP] [MASK] opt0 [MASK] opt1 ... [SEP] state [SEP]

    Returns (input_ids, marker_positions, info). `info` records what the token
    budget actually did -- how much option text survived, how much instruction,
    how much state -- because in this architecture that is the experiment.

    strict=True raises when a marker is pushed past `max_len`. Upstream drops it
    silently, which would hand the caller a slate with fewer markers than
    options and corrupt every anchor fit downstream.
    """
    mask_tok = tok.mask_token
    opts = [str(o).replace(mask_tok, " ") for o in options]
    order = list(option_order) if option_order is not None else list(range(len(opts)))
    ins = str(instructions).replace(mask_tok, " ")

    head_ids = tok(f"{qtype} question: {ins}", add_special_tokens=False)["input_ids"]
    opt_ids = [[tok.mask_token_id]
               + tok(" " + opts[i], add_special_tokens=False)["input_ids"][:OPTION_TOKEN_CAP]
               for i in order]

    untruncated = [len(o) - 1 for o in opt_ids]
    opt_budget = head_max_len - sum(len(o) for o in opt_ids)
    shrank = False
    if opt_budget < MIN_OPT_BUDGET:
        per = max(MIN_PER_OPTION, (head_max_len - MIN_OPT_BUDGET) // max(1, len(opt_ids)))
        opt_ids = [o[:per] for o in opt_ids]
        opt_budget = head_max_len - sum(len(o) for o in opt_ids)
        shrank = True
    head_ids = head_ids[:max(MIN_HEAD_TOKENS, opt_budget)]

    ids = [tok.cls_token_id] + head_ids + [tok.sep_token_id]
    markers = []
    for o in opt_ids:
        markers.append(len(ids))
        ids.extend(o)
    ids.append(tok.sep_token_id)

    room = max(0, max_len - len(ids) - 1)
    st = tok(serialize_state(state).replace(mask_tok, " "),
             add_special_tokens=False)["input_ids"]
    state_untruncated = len(st)
    st = st[-room:] if truncate_left else st[:room]
    ids = ids + st + [tok.sep_token_id]
    ids = ids[:max_len]
    kept = [m for m in markers if m < max_len]

    if strict and len(kept) != len(opts):
        raise ValueError(
            f"{len(opts)} options but only {len(kept)} markers fit in max_len="
            f"{max_len}. Shorten the instructions or the slate.")

    info = {
        "n_options": len(opts),
        "option_text_tokens": [len(o) - 1 for o in opt_ids],
        "option_text_tokens_untruncated": untruncated,
        "option_shrink_fired": shrank,
        "instruction_tokens": len(head_ids),
        "state_tokens": len(st),
        "state_tokens_untruncated": state_untruncated,
        "state_truncated": state_untruncated > len(st),
        "total_tokens": len(ids),
        "order": order,
    }
    return ids, kept, info


def temp_bucket(qtype: str, k: int) -> str:
    """Key into `temperature_by_options`. The cliff is at 11 options, not 20.

    Laya ships `choice:11+ -> T = 0.1006`, a 10x sharpening. A fitted
    temperature that extreme only makes sense if the raw logits were close to
    uniform, which is the evidence that M = 10 is the usable ceiling.
    """
    size = "2" if k <= 2 else "3-5" if k <= 5 else "6-10" if k <= 10 else "11+"
    return f"{qtype}:{size}"
