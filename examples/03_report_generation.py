"""Report generation with the RadLLaMA + LoRA decoder.

    python examples/03_report_generation.py scan.nii.gz

LICENCE: this is the one entry point in CardioFM that is not Apache-2.0 at run
time. The decoder is a LoRA adapter over StanfordAIMI/RadLLaMA-7b, a Llama 2
derivative, and is downloaded separately on first use. See MODEL_LICENSE.md.
"""
import argparse

from cardiofm import CardioFM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan", nargs="+", help="one or more cardiac CT scans")
    ap.add_argument("--max_new_tokens", type=int, default=256)
    ap.add_argument("--num_beams", type=int, default=4)
    ap.add_argument("--weights", default=None,
                    help="local directory with adapter.pt / lora/ / tokenizer/; "
                         "omit to download and cache")
    args = ap.parse_args()

    model = CardioFM.from_pretrained()
    # Loading the 7B decoder takes a while; do it once, then reuse it.
    model.load_report_generator(weights=args.weights)

    for path, text in zip(args.scan, model.generate_reports(
            args.scan, max_new_tokens=args.max_new_tokens, num_beams=args.num_beams)):
        print(f"\n=== {path} ===\n{text}")

    # The decoder is fluent but not numerically reliable: a generated Agatston score
    # or CT-FFR value may be wrong even when the category it implies is right.
    # Treat generated numbers as unverified.


if __name__ == "__main__":
    main()
