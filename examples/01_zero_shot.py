"""Zero-shot coronary classification -- no task-specific training.

    python examples/01_zero_shot.py scan.nii.gz --modality NCCT
"""
import argparse

from cardiofm import CardioFM, prompts


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", help="cardiac CT in NIfTI format")
    ap.add_argument("--modality", choices=["NCCT", "CCTA"], default="NCCT")
    args = ap.parse_args()

    model = CardioFM.from_pretrained()

    # Which tasks apply depends on the protocol: stenosis, obstructive disease and
    # plaque composition need contrast; vessel-level calcium is read on NCCT.
    tasks = (["burden", "stenosis", "obstructive", "plaque"] if args.modality == "CCTA"
             else ["burden", "vessel"])

    for task in tasks:
        print(f"\n=== {task}: {prompts.TASKS[task]} ===")
        example = prompts.templates(task, prompts.class_names(task)[0], args.modality)
        print(f"example prompt: {example[0]!r}   (+3 paraphrases, averaged)")

        out = model.zero_shot(args.scan, task=task, modality=args.modality)
        if task in ("burden", "stenosis"):
            for c, p in zip(out["classes"], out["probs"]):
                mark = " <--" if c == out["prediction"] else ""
                print(f"  {c:<9} {p:.3f}{mark}")
            if task == "burden":
                print(f"  predicted Agatston range: {out['agatston_bin']}")
        else:
            # score = sim(positive) - sim(negative); threshold 0, not a probability
            for k, v in out["score"].items():
                print(f"  {k:<16} score {v:+.4f}  -> {'positive' if v > 0 else 'negative'}")


if __name__ == "__main__":
    main()
