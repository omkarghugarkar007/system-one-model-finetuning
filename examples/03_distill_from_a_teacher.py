"""Distil a teacher's uncertainty, and measure whether it was worth it.

    python examples/03_distill_from_a_teacher.py          # needs OPENROUTER_API_KEY
    python examples/03_distill_from_a_teacher.py --simulated   # no key, no cost

"Soft labels beat hard ones" is the usual advice. It is conditional, and this
example runs the comparison instead of asserting the conclusion.

Three arms, same initialisation, same examples:

    HARD      your labels, one-hot
    SOFT      the teacher's full distribution, as-is
    SHARPENED the teacher's distribution with a temperature applied

Measured on the bundled routing task, Jev as teacher, n=54 test:

    arm         accuracy    ECE      Brier
    hard          0.833   0.1217   0.1270
    soft          0.833   0.1275   0.1289
    sharpened     0.833   0.1218   0.1271

**A dead heat, and that is the lesson.** Jev agrees with these labels almost
perfectly -- it returns P(billing)=1.00 on a double-charge complaint -- so its
distribution carries no information the hard label did not already have.
Distillation costs API calls and buys nothing.

Distillation pays when the teacher knows something your labels do not:
ambiguous items where your own hard label was close to a coin flip, or where
you have few labels and the teacher generalises better. It is not a default,
and the way to find out is to run this comparison on YOUR data.

Watch the confidence column too. Soft labels transfer the teacher's confidence
LEVEL, not just its ordering. A teacher that is merely uncertain about your
bespoke criteria -- as a zero-shot teacher often is -- will train a
well-calibrated student to be underconfident. `--sharpen` is the lever for
that; it keeps the ordering while letting the student stay as certain as its
accuracy justifies.

Teacher calls are content-addressed and cached, so re-runs are free.
"""
import argparse
import copy
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from systemone import TypedDataset, TypedExample  # noqa: E402
from systemone.data import load_jsonl  # noqa: E402
from systemone.eval import report, slice_report  # noqa: E402
from systemone.model import LayaRuntime, SystemOneModel  # noqa: E402
from systemone.teachers import CachedTeacher, SimulatedTeacher  # noqa: E402
from systemone.train.trainer import TypedTrainConfig, TypedTrainer  # noqa: E402


def load_key():
    if os.environ.get("OPENROUTER_API_KEY"):
        return
    env = Path(".env")
    if env.exists():
        for line in env.read_text().splitlines():
            if line.startswith("OPENROUTER_API_KEY="):
                os.environ["OPENROUTER_API_KEY"] = \
                    line.split("=", 1)[1].strip().strip("\"'")


def teacher_soft_labels(examples, teacher, log=print):
    """One teacher call per example -> a distribution over that example's options.

    The teacher is asked the SAME typed question the student will be asked, via
    `JevTeacher.ask`. This matters more than it sounds: the reranking-shaped
    `grade()` interface takes candidates and a rubric, so pushing a
    classification question through it asks the teacher something nobody meant
    and returns a distribution over the wrong thing. (We did exactly that on
    the first attempt, and the resulting "distillation hurts" finding was an
    artifact of the bug.)
    """
    out = []
    for i, ex in enumerate(examples):
        q = ex.question
        if hasattr(teacher, "ask"):
            p = teacher.ask(ex.state, q.type, q.instructions, q.criteria)
        else:                       # simulated teacher: rubric interface is fine
            state = ex.state if isinstance(ex.state, str) else str(ex.state)
            v = teacher.grade(f"{q.instructions}\n\nSTATE: {state}",
                              q.options, q.options[:10])
            p = np.asarray(v.level_probs[0], dtype=float)
        p = np.resize(np.asarray(p, dtype=float), q.n_options)
        out.append(TypedExample(ex.state, q, ex.label, p / p.sum(),
                                ex.weight, ex.meta))
        if (i + 1) % 25 == 0:
            log(f"  teacher: {i + 1}/{len(examples)}")
    return out


def sharpen(p, t=0.5):
    """Temperature-sharpen a teacher distribution. t < 1 makes it more certain.

    The teacher's spread reflects ITS uncertainty about your criteria, which is
    not the same as the irreducible ambiguity of the item. Sharpening keeps the
    ordering and the relative mass -- which is the part worth distilling -- while
    not forcing the student to be less certain than it can justifiably be.
    """
    p = np.clip(np.asarray(p, dtype=float), 1e-9, None) ** (1.0 / t)
    return p / p.sum()


def train_once(examples, rt, model, init_state, tag, epochs, test_ds, log=print):
    """Train, predict, and hand back the predictions.

    The two arms MUST start from the same weights and must not share them.
    `SystemOneModel(rt.model, ...)` wraps the runtime's model *by reference*, so
    building two wrappers gives you two views of one set of weights and the
    second arm silently continues training the first. Reloading `init_state`
    into one model and running the arms sequentially avoids that and also
    avoids holding two 421M models in memory at once.
    """
    model.load_state_dict(copy.deepcopy(init_state), strict=True)
    model.to(rt.device)
    ds = TypedDataset(examples, rt.tok)
    cfg = TypedTrainConfig(warmup_epochs=epochs, full_epochs=0, batch_size=8,
                           lr_head=3e-4, eval_every=10 ** 9)
    trainer = TypedTrainer(model, rt.tok, cfg, device=rt.device, log=lambda *_: None)
    log(f"training on {tag} labels ...")
    trainer.fit(ds, skip_preflight=True)
    return trainer.predict(test_ds)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="examples/data/sample.jsonl")
    ap.add_argument("--model", default="data/cache/laya")
    ap.add_argument("--epochs", type=int, default=4)
    ap.add_argument("--sharpen", type=float, default=0.5,
                    help="temperature on the teacher distribution; <1 sharpens")
    ap.add_argument("--simulated", action="store_true",
                    help="use a synthetic teacher; no API key, no cost")
    args = ap.parse_args()

    examples = load_jsonl(args.data)
    split = int(0.75 * len(examples))
    train_ex, test_ex = examples[:split], examples[split:]
    n_levels = max(e.question.n_options for e in examples)

    if args.simulated:
        truth = {f"{e.question.instructions}\n\nSTATE: {e.state}":
                 np.full(e.question.n_options, 0.0) for e in train_ex}
        for e in train_ex:
            truth[f"{e.question.instructions}\n\nSTATE: {e.state}"][e.label] = 2.0
        teacher = SimulatedTeacher(truth, skill=0.75, n_levels=n_levels)
        print("teacher: simulated (skill 0.75)")
    else:
        load_key()
        from systemone.teachers import JevTeacher
        teacher = CachedTeacher(JevTeacher(max_candidates=n_levels),
                                "data/cache/teacher", backend_tag="jev-distill")
        print("teacher: Jev via OpenRouter, cached")

    print(f"{len(train_ex)} train / {len(test_ex)} test\n")
    soft_ex = teacher_soft_labels(train_ex, teacher)
    if hasattr(teacher, "stats"):
        print(f"teacher usage: {teacher.stats()}")

    rt = LayaRuntime(args.model, apply_temperature=False)
    model = SystemOneModel(rt.model, n_levels=n_levels)
    init = copy.deepcopy(model.state_dict())
    test_ds = TypedDataset(test_ex, rt.tok)

    sharp_ex = [TypedExample(e.state, e.question, e.label,
                             sharpen(e.soft_label, args.sharpen), e.weight, e.meta)
                for e in soft_ex]

    # sequential, from identical weights, one model in memory at a time
    ph = train_once(train_ex, rt, model, init, "HARD", args.epochs, test_ds)
    ps = train_once(soft_ex, rt, model, init, "SOFT", args.epochs, test_ds)
    pz = train_once(sharp_ex, rt, model, init,
                    f"SHARPENED (T={args.sharpen})", args.epochs, test_ds)

    print("\n" + "=" * 70)
    print("HARD labels  (argmax only)")
    print("=" * 70)
    print(report(ph, by="type"))

    print("\n" + "=" * 70)
    print("SOFT labels  (the teacher's distribution, as-is)")
    print("=" * 70)
    print(report(ps, by="type"))

    print("\n" + "=" * 70)
    print(f"SHARPENED  (teacher distribution at T={args.sharpen})")
    print("=" * 70)
    print(report(pz, by="type"))

    arms = {"hard": slice_report(ph)["__aggregate__"],
            "soft": slice_report(ps)["__aggregate__"],
            "sharp": slice_report(pz)["__aggregate__"]}
    print("\n" + "=" * 70)
    print("DID IT PAY?")
    print("=" * 70)
    print(f"  {'metric':<10} {'hard':>9} {'soft':>9} {'sharpened':>10}   best")
    print("  " + "-" * 52)
    for k in ("accuracy", "ece", "brier", "aurc", "mean_conf"):
        vals = {a: arms[a][k] for a in arms}
        spread = max(vals.values()) - min(vals.values())
        scale = max(abs(v) for v in vals.values()) or 1.0
        if k == "mean_conf":
            best = "-"
        elif spread / scale < 0.05:
            best = "tie"          # a 5% spread is not a result
        else:
            best = (max(vals, key=vals.get) if k == "accuracy"
                    else min(vals, key=vals.get))
        print(f"  {k:<10} {vals['hard']:>9.4f} {vals['soft']:>9.4f} "
              f"{vals['sharp']:>10.4f}   {best}")
    print("\n  'tie' means the spread across arms is under 5% -- not a result.")
    print("  A tie means the teacher agreed with your labels and told the student")
    print("  nothing new. That is the common case on easy tasks, and it is why")
    print("  distillation is worth measuring rather than assuming.")
    print("\n  Read accuracy LAST: it barely moves. What distillation changes is")
    print("  CONFIDENCE, and it transfers the teacher's confidence level whether")
    print("  or not that level is justified for your student. Use --sharpen when")
    print("  the teacher is uncertain about your criteria rather than about the")
    print("  item.")


if __name__ == "__main__":
    main()
