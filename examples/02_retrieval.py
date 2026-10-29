"""Cross-modal retrieval in the shared image-text embedding space.

    python examples/02_retrieval.py scan.nii.gz
"""
import argparse

from cardiofm import CardioFM

GALLERY = [
    "Non-contrast coronary calcium CT. Total Agatston 0 (none). No coronary calcification.",
    "Non-contrast coronary calcium CT. Total Agatston 7 (minimal). Minimal calcium in the LAD.",
    "Non-contrast coronary calcium CT. Total Agatston 64 (mild). Calcium in the LAD and RCA.",
    "Non-contrast coronary calcium CT. Total Agatston 288 (moderate). Calcium in the LM, LAD and RCA.",
    "Non-contrast coronary calcium CT. Total Agatston 912 (severe). Extensive three-vessel calcification.",
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scan")
    ap.add_argument("--top_k", type=int, default=3)
    args = ap.parse_args()

    model = CardioFM.from_pretrained()

    print("=== image -> report ===")
    for hit in model.retrieve(args.scan, GALLERY, top_k=args.top_k)[0]:
        print(f"  {hit['rank']}. [{hit['score']:.4f}] {hit['item']}")

    # The same space runs the other way: rank images for a text query.
    print("\n=== report -> image ===")
    query = "extensive three-vessel coronary calcification"
    for hit in model.retrieve([query], [args.scan], query_is_image=False, top_k=1)[0]:
        print(f"  query {query!r} -> scan index {hit['index']} [{hit['score']:.4f}]")


if __name__ == "__main__":
    main()
