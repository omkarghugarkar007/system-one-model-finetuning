"""Fine-tune a System One model end to end, on a laptop, in a few minutes.

    python examples/01_quickstart.py

This is the whole recipe in one file. Swap `build_data()` for your own examples
and nothing else changes.

The task is chosen to show what fine-tuning is actually for. It is an ordinal
severity rubric where **the rubric disagrees with the tone**: a calmly worded
customer-facing outage is severity 3, a furious complaint about an internal
dashboard is severity 1. A base checkpoint reads tone, because that is what
general pretraining gives it. Only training teaches it *your* rubric.

That is the honest use case for a System One model. It is not a better
classifier out of the box -- Laya's own card puts its base checkpoint below the
majority-class baseline -- it is a small, fast, calibrated base that learns your
specific decision boundary cheaply.

Steps, in the order that matters:

  1. preflight   look at the token budget BEFORE training. The commonest silent
                 failure is that your state never reached the model.
  2. baseline    measure the majority-class baseline and the untrained model.
  3. train       encoder frozen: fast, ~2 GB, enough for a small task.
  4. evaluate    accuracy AND calibration, sliced, never aggregate-only.
  5. calibrate   fit a temperature on held-out data.
"""
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))

from systemone import TypedDataset, score  # noqa: E402
from systemone.calibrate import TemperatureMap  # noqa: E402
from systemone.eval import report  # noqa: E402
from systemone.model import LayaRuntime, SystemOneModel  # noqa: E402
from systemone.train.trainer import TypedTrainConfig, TypedTrainer  # noqa: E402

MODEL_DIR = "data/cache/laya"

# The rubric is about BLAST RADIUS, not about how upset the reporter sounds.
LEVELS = [
    "no user impact: cosmetic, internal-only, or already mitigated",
    "internal impact only: staff tooling degraded, customers unaffected",
    "some customers degraded: slow or partial failure, workaround exists",
    "customer-facing outage: users blocked with no workaround",
]

# (severity, tone, template). Tone is deliberately uncorrelated with severity.
CASES = [
    (3, "calm", "Noting that checkout has been returning errors for all users since {n}:00."),
    (3, "calm", "FYI, the login endpoint is down globally. No workaround identified."),
    (3, "angry", "NOTHING WORKS. Every customer is locked out. Fix this now!!"),
    (3, "angry", "Total outage on the payments API for {n} minutes and still going."),
    (2, "calm", "Search results are taking about {n} seconds for roughly half of accounts."),
    (2, "calm", "Exports are failing intermittently; retrying usually succeeds."),
    (2, "angry", "This is unacceptable, uploads keep timing out! Retry works but still!"),
    (2, "angry", "Customers are furious about the {n}s page loads. They can still buy though."),
    (1, "calm", "The internal analytics dashboard has not refreshed since {n}:00."),
    (1, "calm", "Our staff admin tool is slow. No customer-facing component affected."),
    (1, "angry", "The ops dashboard is COMPLETELY broken and I cannot do my job!"),
    (1, "angry", "Absolutely furious that the internal wiki search is down again."),
    (0, "calm", "Minor typo in the footer copyright year on the marketing site."),
    (0, "calm", "Logged a rendering glitch that was already fixed in release {n}."),
    (0, "angry", "The button is 2 PIXELS OFF and it has been like this for weeks!!"),
    (0, "angry", "Why is the icon still the old one?! This is embarrassing!"),
]


def build_data(n_per_case=14, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for sev, tone, tpl in CASES:
        for _ in range(n_per_case):
            text = tpl.format(n=int(rng.integers(2, 59)))
            out.append(score(text, "Rate the severity of this incident report.",
                             LEVELS, label=sev, meta={"tone": tone}))
    rng.shuffle(out)
    return out


def main():
    if not Path(MODEL_DIR).exists():
        raise SystemExit(
            f"{MODEL_DIR} not found. Fetch the weights first:\n"
            "  python -c \"from huggingface_hub import snapshot_download as d; "
            "d('convaiinnovations/laya', local_dir='data/cache/laya', "
            "allow_patterns=['model.safetensors','rl_agent_config.json',"
            "'encoder/*','tokenizer/*'])\"")

    examples = build_data()
    split = int(0.75 * len(examples))
    train_ex, test_ex = examples[:split], examples[split:]
    print(f"{len(train_ex)} train / {len(test_ex)} test, "
          f"{len(LEVELS)} ordered severity levels\n")

    rt = LayaRuntime(MODEL_DIR, apply_temperature=False)
    model = SystemOneModel(rt.model, n_levels=len(LEVELS)).to(rt.device)
    train_ds = TypedDataset(train_ex, rt.tok)
    test_ds = TypedDataset(test_ex, rt.tok)

    cfg = TypedTrainConfig(warmup_epochs=6, full_epochs=0, batch_size=8,
                           grad_accum=2, lr_head=5e-4, eval_every=10 ** 9)
    trainer = TypedTrainer(model, rt.tok, cfg, device=rt.device)

    trainer.preflight(train_ds)

    print("\n" + "=" * 70)
    print("BEFORE FINE-TUNING  -- the base checkpoint reads tone, not the rubric")
    print("=" * 70)
    before = trainer.predict(test_ds)
    print(report(before, by="type"))
    print("\nsliced by TONE, which the rubric says is irrelevant:")
    print(report(before, by="tone"))

    print("\n" + "=" * 70)
    print("TRAINING")
    print("=" * 70)
    trainer.fit(train_ds, skip_preflight=True)

    print("\n" + "=" * 70)
    print("AFTER FINE-TUNING")
    print("=" * 70)
    after = trainer.predict(test_ds)
    print(report(after, by="type"))
    print("\nsliced by TONE -- the gap between calm and angry should have closed:")
    print(report(after, by="tone"))

    records = [(p["logits"], p["example"].label, p["example"].question.type)
               for p in after]
    temps = TemperatureMap.fit(records, min_samples=20)
    print("\n" + "=" * 70)
    print("CALIBRATION")
    print("=" * 70)
    print(temps.report())
    print("\nNOTE: fitted on the test split here to keep the example short. In")
    print("real use fit on a THIRD split -- a temperature fitted on your")
    print("evaluation set reports a calibration you do not actually have.")

    out = Path("runs/quickstart.pt")
    trainer.save(out)
    print(f"\ncheckpoint -> {out}")


if __name__ == "__main__":
    main()
