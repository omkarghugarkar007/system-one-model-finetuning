"""Check `core.budget`'s arithmetic against the real tokenizer.

`core.budget` predicts, from token counts alone, how much option text, how much
instruction and how much state survive `build_sequence`. Those predictions
drive the slate design, so they need to be verified against the shipped
tokenizer rather than trusted -- and re-verified if a Laya release changes the
constants. `verify()` returns the mismatches; the test suite fails on any.
"""
from __future__ import annotations

from ...core.budget import DEFAULT_BUDGET, SlateLayout, plan_slate_budget
from .vendored import build_sequence

__all__ = ["verify", "probe_table"]

# comfortably past the 48-token option cap, so every cap and shrink fires
LONG = ("The device warranty applies for twenty four months from the date of "
        "purchase, except for batteries and consumable parts, which are limited "
        "to twelve months, and excludes damage caused by unauthorised repair "
        "or by use outside the documented operating range. Claims must be filed "
        "through the regional service desk within thirty days of the fault being "
        "observed, accompanied by the original proof of purchase and the serial "
        "number printed on the underside of the chassis.")


def _measure(tok, n_options: int, layout: str, instructions: str):
    if layout == SlateLayout.OPTIONS:
        opts, state = [LONG] * n_options, "what is the battery warranty period?"
    else:
        opts = [f"c{i}" for i in range(n_options)]
        state = {"query": "what is the battery warranty period?",
                 "candidates": {f"c{i}": LONG for i in range(n_options)}}
    _, markers, info = build_sequence(tok, state, "choice", instructions, opts,
                                      DEFAULT_BUDGET.max_len,
                                      DEFAULT_BUDGET.head_max_len, strict=False)
    return markers, info


def _ntok(tok, s: str) -> int:
    return len(tok(s, add_special_tokens=False)["input_ids"])


def verify(tok, sizes=(2, 4, 6, 8, 10, 12, 16, 20),
           instructions: str = "Which passage best answers the query?") -> list[dict]:
    """Return one record per (size, layout) with predicted vs measured.

    The predictions are fed the *real* token counts of the option ids and the
    instructions. `plan_slate_budget` answers "given inputs this long, what
    survives?", so feeding it a guess and then calling the mismatch a prediction
    error would be testing the guess rather than the formula.
    """
    out = []
    ins_tok = _ntok(tok, instructions)
    for m in sizes:
        for layout in (SlateLayout.OPTIONS, SlateLayout.STATE):
            markers, info = _measure(tok, m, layout, instructions)
            # " c0" etc, with the leading space build_sequence prepends
            id_tok = max(_ntok(tok, f" c{i}") for i in range(m))
            pred = plan_slate_budget(m, layout, option_id_tokens=id_tok,
                                     instruction_tokens=ins_tok,
                                     budget=DEFAULT_BUDGET)
            measured_opt = min(info["option_text_tokens"])
            out.append({
                "n_options": m, "layout": layout,
                "pred_option_tokens": pred.option_text_tokens,
                "measured_option_tokens": measured_opt,
                "option_match": pred.option_text_tokens == measured_opt,
                "pred_instruction_tokens": pred.instruction_tokens,
                "measured_instruction_tokens": info["instruction_tokens"],
                "pred_state_capacity": pred.state_tokens,
                "measured_state_used": info["state_tokens"],
                "measured_state_wanted": info["state_tokens_untruncated"],
                "state_truncated": info["state_truncated"],
                "markers_kept": len(markers),
                "markers_expected": m,
                "shrink_fired": info["option_shrink_fired"],
            })
    return out


def probe_table(tok, **kw) -> str:
    rows = verify(tok, **kw)
    lines = ["Predicted (core.budget) vs measured (real Laya tokenizer)", "",
             f"{'M':>3} {'layout':>8} {'opt pred':>9} {'opt meas':>9} {'ok':>3} "
             f"{'state cap':>10} {'wanted':>8} {'used':>6} {'trunc':>6} {'mark':>6}",
             "-" * 82]
    for r in rows:
        lines.append(
            f"{r['n_options']:>3} {r['layout']:>8} {r['pred_option_tokens']:>9} "
            f"{r['measured_option_tokens']:>9} {'OK' if r['option_match'] else 'XX':>3} "
            f"{r['pred_state_capacity']:>10} {r['measured_state_wanted']:>8} "
            f"{r['measured_state_used']:>6} {str(r['state_truncated']):>6} "
            f"{r['markers_kept']}/{r['markers_expected']:>4}")
    bad = [r for r in rows if not r["option_match"]]
    lines += ["",
              f"{len(rows) - len(bad)}/{len(rows)} option-budget predictions exact.",
              "",
              "'trunc' is the trap: in the STATE layout the state overflows from M=10 up,",
              "and build_sequence cuts it with st[:room] -- from the RIGHT. The last",
              "candidates lose their text but keep their markers, so they are scored on",
              "nothing. core.packing.StatePacker pre-truncates each candidate to a fair",
              "share so the loss is even and reported instead of positional and silent."]
    return "\n".join(lines)
