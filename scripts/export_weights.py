#!/usr/bin/env python
"""Convert a training checkpoint into the released inference-only checkpoint.

Drops the optimizer, scheduler and scaler state, and drops the objective
aggregator entirely -- which is where the SimMIM reconstruction head and the CAC
regression head live. What remains is the image encoder, the pooling taps and the
text tower: everything the inference API uses and nothing else.

    python scripts/export_weights.py --ckpt best.ckpt --out cardiofm_pretrained.pt
"""
import argparse
from collections import Counter

import torch

KEEP_PREFIXES = ("encoder.", "attn_pool.", "token_proj.", "text_tower.")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ckpt", required=True, help="training checkpoint")
    ap.add_argument("--out", required=True, help="destination .pt")
    ap.add_argument("--half", action="store_true",
                    help="store float16 (halves the file; embeddings differ in the 4th decimal)")
    args = ap.parse_args()

    ck = torch.load(args.ckpt, map_location="cpu")
    sd = ck.get("model", ck)

    kept, dropped = {}, []
    for k, v in sd.items():
        if k.startswith(KEEP_PREFIXES):
            kept[k] = v.half() if (args.half and v.is_floating_point()) else v
        else:
            dropped.append(k)

    # The aggregator is not carried over at all; report what it held so the
    # omission is visible rather than silent.
    agg = list((ck.get("aggregator") or {}).keys())
    if agg:
        print(f"dropping aggregator ({len(agg)} tensors): "
              + ", ".join(sorted({a.split('.')[1] for a in agg if a.startswith('objs.')})))
    if dropped:
        print(f"dropping {len(dropped)} non-inference tensors")
    for k in kept:
        assert "cac" not in k.lower(), f"CAC tensor leaked into the export: {k}"

    pref = Counter(k.split(".")[0] for k in kept)
    n = sum(v.numel() for v in kept.values())
    print(f"keeping {len(kept)} tensors, {n/1e6:.1f}M params, prefixes: {dict(pref)}")

    torch.save({"model": kept,
                "format": "cardiofm-inference-v1",
                "dtype": "float16" if args.half else "float32"}, args.out)
    print(f"wrote {args.out}")


if __name__ == "__main__":
    main()
