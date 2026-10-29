"""Extract embeddings for your own downstream model.

    python examples/04_embeddings.py scan1.nii.gz scan2.nii.gz --out emb.npy
"""
import argparse

import numpy as np

from cardiofm import CardioFM


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("scans", nargs="+")
    ap.add_argument("--out", default="embeddings.npy")
    args = ap.parse_args()

    model = CardioFM.from_pretrained()
    emb = np.stack([model.encode_image(s).cpu().numpy()[0] for s in args.scans])
    np.save(args.out, emb)
    print(f"wrote {args.out}  shape {emb.shape}")
    # These 512-d vectors are what the paper's frozen-encoder probes were trained
    # on: fit a small head on them for a new task rather than fine-tuning the tower.


if __name__ == "__main__":
    main()
