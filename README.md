# CardioFM

A vision–language foundation model for coronary artery disease from cardiac CT.

CardioFM pairs a 3D image encoder with a clinical text tower in a shared
512-dimensional embedding space. Give it a cardiac CT — non-contrast calcium CT (NCCT) or
CT angiography (CCTA) — and it will classify coronary findings **with no task-specific
training**, retrieve matching reports, generate a findings narrative, or hand you an
embedding to build on.

This repository contains the model architecture and the inference path.

### Tasks

| Task | `task=` | Protocol | Output |
|---|---|---|---|
| Coronary calcium burden | `burden` | NCCT, CCTA | 5-class ordinal + Agatston range |
| Maximal stenosis | `stenosis` | CCTA | 5-class ordinal |
| Obstructive CAD | `obstructive` | CCTA | binary |
| Vessel-level calcium | `vessel` | NCCT | binary per LM/LAD/LCx/RCA/PDA |
| Plaque composition | `plaque` | CCTA | binary, calcified / non-calcified |

Plus cross-modal retrieval, report generation and embedding extraction.

## Installation

```bash
pip install git+https://github.com/JinkuiH/CardioFM.git
```


## Usage

```python
from cardiofm import CardioFM

model = CardioFM.from_pretrained()          # downloads and caches the weights once

# zero-shot classification
out = model.zero_shot("scan.nii.gz", task="burden", modality="NCCT")
print(out["prediction"], out["agatston_bin"])     # 'moderate'  '101-400'

# embeddings in the shared image-text space
emb = model.encode_image("scan.nii.gz")           # [1, 512]
txt = model.encode_text("moderate coronary calcium burden")

# report generation (downloads the Llama-2-licensed decoder on first call)
print(model.generate_report("scan.nii.gz"))

# retrieval
model.retrieve("scan.nii.gz", list_of_reports, top_k=5)
```

Runnable versions of each:

```bash
python examples/01_zero_shot.py         scan.nii.gz --modality NCCT
python examples/02_retrieval.py         scan.nii.gz
python examples/03_report_generation.py scan.nii.gz
python examples/04_embeddings.py        scan1.nii.gz scan2.nii.gz --out emb.npy
```

Weights are cached in `~/.cache/cardiofm` (override with `CARDIOFM_HOME`). To use a local
file, pass `CardioFM.from_pretrained(weights="...")` or set `CARDIOFM_WEIGHTS`.

## Preprocessing

`preprocess_ct` turns a NIfTI volume into the `[1, 4, 96, 256, 256]` tensor the model
expects. The order matters — reorientation and resampling happen *before* the crop, so
the margin is applied in resampled voxels:

1. heart mask from TotalSegmentator (`roi_subset=["heart"]`, `fast=True`)
2. reorient to **SAR**
3. resample to `(1.5, 0.5, 0.5)` mm
4. crop to the heart bounding box + margin `(12, 20, 20)` voxels = `(18, 10, 10)` mm
5. centre crop or pad to `96 × 256 × 256`
6. four HU windows → four channels

Steps 2–4 are a MONAI transform chain identical to the one that built the training
volumes.

### The heart crop is required

CardioFM was trained **exclusively** on heart-cropped volumes. An uncropped scan is far
outside the distribution the model saw: the embedding stops being interpretable and the
zero-shot scores stop meaning anything. `preprocess_ct` therefore raises rather than
falling back to a whole-volume crop — a loud failure beats a plausible-looking wrong
answer.

Pass `ts_cache="dir"` to reuse the heart mask across repeated calls on the same scan;
segmentation is the slow part of the pipeline.

### Channel order

Fixed. Do not change it — the encoder's four input channels are these windows:

| channel | window (level, width) | emphasises |
|---|---|---|
| 0 | (40, 400) | soft tissue |
| 1 | (130, 300) | calcium |
| 2 | (300, 800) | contrast-filled lumen |
| 3 | (500, 1500) | wide / dense structures |

Already have preprocessed volumes? Pass a tensor straight to `encode_image` — any
`[B, 4, D, H, W]` input following this channel order is accepted, and preprocessing is
skipped.

## Prompts

Zero-shot scores are similarities between the image embedding and class prompts. Each
class uses **four** templates; the class embedding is the mean of their L2-normalised
embeddings, re-normalised.

```python
from cardiofm import prompts

prompts.templates("burden", "moderate", "NCCT")
# ['Non-contrast coronary calcium CT: coronary artery calcification burden is moderate',
#  'Non-contrast coronary calcium CT: moderate coronary calcium burden',
#  'Non-contrast coronary calcium CT: the calcium score indicates moderate burden',
#  'Non-contrast coronary calcium CT: overall calcification is moderate']

prompts.all_prompts("vessel", "NCCT")       # every class of a task
```

Prefixes are `"Non-contrast coronary calcium CT: "` and `"Coronary CT angiography: "`;
the trailing colon and space are part of the prefix. The complete bank is in
[`docs/prompts.md`](docs/prompts.md).

**Scoring is not uniform across tasks.** The two ordinal tasks return `probs`, a softmax
over the five class similarities, and `prediction`, their arg-max. The binary tasks
return `score = sim(positive) − sim(negative)`, whose decision threshold is 0 — a
similarity difference, **not a calibrated probability**.

## Advanced

**Agatston categories.** `burden` maps to the standard five bins: `0` none, `1–10`
minimal, `11–100` mild, `101–400` moderate, `>400` severe. `zero_shot` returns the
predicted bin as `agatston_bin`.

**Frozen-encoder probes.** The 512-d embeddings from `encode_image` are what the paper's
downstream probes were trained on. For a new task, fit a small head on these rather than
fine-tuning the tower.

**Exporting weights.** `scripts/export_weights.py` converts a training checkpoint into
the released inference checkpoint, dropping the optimizer state and the objective
aggregator.


## Citation

```bibtex
@article{cardiofm,
  title  = {A vision--language foundation model for coronary artery disease from cardiac CT},
  author = {Hao, Jinkui and Bajela, Victoria and Hoori, Ammar and Wilson, David L.
            and Shah, Nilay S. and Zhou, Bo},
  year   = {2026}
}
```

## License

Source code is under the [Apache License 2.0](LICENSE); third-party attributions are in
[NOTICE](NOTICE).

Weights are not all under the same licence — check [MODEL_LICENSE.md](MODEL_LICENSE.md)
before redistributing any of them:

| Artefact | Licence |
|---|---|
| Source code | Apache-2.0 |
| `cardiofm_pretrained.pt` | Apache-2.0 |
| Report-generation LoRA adapter | **Llama 2 Community License** |

Everything in this README except `generate_report()` uses Apache-2.0 artefacts only.
`generate_report()` loads the RadLLaMA LoRA adapter, a derivative of a Llama 2 model,
published as a separate release asset for that reason; calling it means accepting the
Llama 2 Community License and its Acceptable Use Policy.

## References

- Swin UNETR encoder and the self-supervised warm-start: [MONAI](https://github.com/Project-MONAI/MONAI)
- Heart localisation: [TotalSegmentator](https://github.com/wasserth/TotalSegmentator)
- Text tower: [T5](https://huggingface.co/t5-small)
