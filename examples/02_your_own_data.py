"""Fine-tune on your own JSONL. The path most people want.

    python examples/02_your_own_data.py --data examples/data/sample.jsonl

Your file is one JSON object per line:

    {"state": "...",                      a string, or any JSON object
     "type": "choice",                    "choice" | "score" | "noul"
     "instructions": "...",               what the model is deciding
     "criteria": {"a": "...", "b": "..."} choice: option -> description
                 ["low", "high"]          score:  ORDERED levels, low first
                 null,                    noul:   optional {true:..., false:...}
     "label": "a" | 2 | true,             the answer
     "soft_label": [0.1, 0.9],            optional teacher distribution
     "meta": {"source": "..."}}           optional; used to slice the report

`state` can be a JSON object and usually should be. A System One model reads
structure, and `{"customer": {...}, "message": "..."}` lets the instructions
point at a field by name instead of hoping the model finds it in prose.

The script splits train/val/test, runs the preflight budget check, trains,
evaluates sliced, and fits a temperature on the VALIDATION split -- not on
test, which would report a calibration you do not have.
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from systemone import TypedDataset                                     # noqa: E402
from systemone.calibrate import TemperatureMap, ece_by_bucket          # noqa: E402
from systemone.data import load_jsonl                                  # noqa: E402
from systemone.eval import report                                      # noqa: E402
from systemone.model import LayaRuntime, SystemOneModel                # noqa: E402
from systemone.train.trainer import TypedTrainConfig, TypedTrainer     # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="examples/data/sample.jsonl")
    ap.add_argument("--model", default="data/cache/laya")
    ap.add_argument("--warmup-epochs", type=int, default=4)
    ap.add_argument("--full-epochs", type=int, default=0,
                    help=">0 unfreezes the encoder; see RECIPE.md section 6 on memory")
    ap.add_argument("--freeze-lower", type=int, default=14,
                    help="only used when --full-epochs > 0")
    ap.add_argument("--lr-head", type=float, default=3e-4)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--slice-by", default="type",
                    help="'type', 'options', or any key in your meta")
    ap.add_argument("--out", default="runs/my_model.pt")
    args = ap.parse_args()

    examples = load_jsonl(args.data)
    n = len(examples)
    a, b = int(0.7 * n), int(0.85 * n)
    train, val, test = examples[:a], examples[a:b], examples[b:]
    print(f"{n} examples -> {len(train)} train / {len(val)} val / {len(test)} test")

    rt = LayaRuntime(args.model, apply_temperature=False)
    n_levels = max(e.question.n_options for e in examples)
    model = SystemOneModel(rt.model, n_levels=n_levels).to(rt.device)

    train_ds = TypedDataset(train, rt.tok)
    val_ds = TypedDataset(val, rt.tok)
    test_ds = TypedDataset(test, rt.tok)

    cfg = TypedTrainConfig(warmup_epochs=args.warmup_epochs,
                           full_epochs=args.full_epochs,
                           freeze_lower_layers=args.freeze_lower,
                           batch_size=args.batch_size, lr_head=args.lr_head,
                           eval_every=10 ** 9)
    trainer = TypedTrainer(model, rt.tok, cfg, device=rt.device)

    # 1. the data, before the model
    trainer.preflight(train_ds)

    print("\n=== before ===")
    print(report(trainer.predict(test_ds), by=args.slice_by))

    trainer.fit(train_ds, skip_preflight=True)

    print("\n=== after ===")
    preds = trainer.predict(test_ds)
    print(report(preds, by=args.slice_by))

    # 2. calibrate on VALIDATION, then check on test
    val_records = [(p["logits"], p["example"].label, p["example"].question.type)
                   for p in trainer.predict(val_ds)]
    temps = TemperatureMap.fit(val_records, min_samples=20)
    print("\n=== calibration (fitted on val) ===")
    print(temps.report())

    test_records = [(p["logits"], p["example"].label, p["example"].question.type)
                    for p in preds]
    before = ece_by_bucket(test_records)["__aggregate__"]
    after = ece_by_bucket(test_records, temps)["__aggregate__"]
    print(f"\ntest ECE {before['ece']:.4f} -> {after['ece']:.4f} after temperature")
    if after.get("hiding_ratio", 1) > 2:
        print(f"WARNING: worst bucket is {after['worst_bucket_ece']:.4f}, "
              f"{after['hiding_ratio']}x the aggregate. Look at the slices.")

    trainer.save(args.out)
    print(f"\ncheckpoint -> {args.out}")
    print("Serve it with systemone.model.LayaRuntime + SystemOneModel, and apply")
    print("the fitted temperature before you threshold anything.")


if __name__ == "__main__":
    main()
