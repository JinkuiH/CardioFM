"""Report generation from the image, via the pretrained captioning decoder.

    python examples/03_report_generation.py scan.nii.gz
"""
import argparse

from cardiofm import CardioFM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--max_new_tokens", type=int, default=160)
    ap.add_argument("--num_beams", type=int, default=4)
    args = ap.parse_args()

    model = CardioFM.from_pretrained()
    text = model.generate_report(args.scan, max_new_tokens=args.max_new_tokens,
                                 num_beams=args.num_beams)
    print(text)

    # The decoder is fluent but not guaranteed numerically correct: a generated
    # Agatston score or CT-FFR value may be wrong even when the category it implies
    # is right. Treat generated numbers as unverified.


if __name__ == "__main__":
    main()
