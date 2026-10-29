# CardioFM

A vision–language foundation model for coronary artery disease from cardiac CT.

CardioFM pairs a 3D Swin UNETR image encoder with a clinical text tower in a shared
512-dimensional embedding space. It takes a heart-cropped cardiac CT — non-contrast
calcium CT (NCCT) or CT angiography (CCTA) — and supports **zero-shot classification**,
**cross-modal retrieval**, **report generation** and **embedding extraction** without
task-specific training.

This repository contains the model architecture and the inference path only.

## Install

```bash
git clone https://github.com/JinkuiH/CardioFM.git
cd CardioFM
pip install -e .
```

Optional, for automatic heart localisation (recommended — see *Preprocessing*):

```bash
pip install TotalSegmentator
```

## Quick start

```python
from cardiofm import CardioFM

model = CardioFM.from_pretrained()          # downloads and caches the weights once

emb = model.encode_image("scan.nii.gz")     # [1, 512]
txt = model.encode_text("moderate coronary calcium burden")

out = model.zero_shot("scan.nii.gz", task="burden", modality="NCCT")
print(out["prediction"], out["agatston_bin"])   # e.g. 'moderate'  '101-400'

print(model.generate_report("scan.nii.gz"))
```

Weights are fetched to `~/.cache/cardiofm` (override with `CARDIOFM_HOME`), or pass a
local path with `CardioFM.from_pretrained(weights="...")` / the `CARDIOFM_WEIGHTS`
environment variable.

## Tasks

| Task | `task=` | Protocol | Output |
|---|---|---|---|
| Coronary calcium burden | `burden` | NCCT, CCTA | 5-class ordinal + Agatston range |
| Maximal stenosis | `stenosis` | CCTA | 5-class ordinal |
| Obstructive CAD | `obstructive` | CCTA | binary |
| Vessel-level calcium | `vessel` | NCCT | binary per LM/LAD/LCx/RCA/PDA |
| Plaque composition | `plaque` | CCTA | binary, calcified and non-calcified |

### Prompts

Zero-shot scores are similarities between the image embedding and class prompts. Every
class uses **four** templates; the class embedding is the mean of their L2-normalised
embeddings, re-normalised. The prompt is the modality prefix concatenated directly with
the template body:

```python
from cardiofm import prompts

prompts.templates("burden", "moderate", "NCCT")
# ['Non-contrast coronary calcium CT: coronary artery calcification burden is moderate',
#  'Non-contrast coronary calcium CT: moderate coronary calcium burden',
#  'Non-contrast coronary calcium CT: the calcium score indicates moderate burden',
#  'Non-contrast coronary calcium CT: overall calcification is moderate']

prompts.all_prompts("vessel", "NCCT")       # every class of a task
```

Prefixes are `"Non-contrast coronary calcium CT: "` and `"Coronary CT angiography: "` —
the trailing colon and space are part of the prefix. See [`docs/prompts.md`](docs/prompts.md)
for the complete bank.

**Scoring is not uniform across tasks.** The two ordinal tasks return `probs`, a softmax
over the five class similarities, and `prediction`, their arg-max. The binary tasks
return `score = sim(positive) − sim(negative)`, whose decision threshold is 0. That
difference is a similarity, **not a calibrated probability**, and should not be read as one.

## Preprocessing

`preprocess_ct` expects a NIfTI volume and produces `[1, 4, 96, 256, 256]`. The steps run
in this order, which matters — reorientation and resampling happen *before* the crop, so
the margin is applied in resampled voxels:

1. heart mask from TotalSegmentator (`roi_subset=["heart"]`, `fast=True`)
2. reorient to **SAR**
3. resample to `(1.5, 0.5, 0.5)` mm
4. crop to the heart bounding box + margin `(12, 20, 20)` voxels = `(18, 10, 10)` mm
5. centre crop or pad to `96 × 256 × 256`
6. four HU windows → four channels, **in this fixed order**:

   | channel | window (level, width) | emphasises |
   |---|---|---|
   | 0 | (40, 400) | soft tissue |
   | 1 | (130, 300) | calcium |
   | 2 | (300, 800) | contrast-filled lumen |
   | 3 | (500, 1500) | wide / dense structures |

Steps 2–4 are a MONAI transform chain identical to the one that built the training
volumes. If TotalSegmentator is unavailable the chain still reorients and resamples but
skips the heart crop, and a warning is raised — CardioFM was trained on heart-cropped
input, so **embeddings and zero-shot scores will not reproduce the published results
without it**.

Pass `ts_cache="dir"` to reuse the heart mask across repeated calls on the same scan.

Already have preprocessed volumes? Pass a tensor directly — any `[B, 4, D, H, W]` input
whose channels follow the window order above is accepted.

## Examples

```bash
python examples/01_zero_shot.py        scan.nii.gz --modality NCCT
python examples/02_retrieval.py        scan.nii.gz
python examples/03_report_generation.py scan.nii.gz
python examples/04_embeddings.py       scan1.nii.gz scan2.nii.gz --out emb.npy
```

## What is and is not in this repository

**Included** — the image encoder, pooling taps and text tower; preprocessing; the full
zero-shot prompt bank; inference for classification, retrieval, report generation and
embedding extraction.

**Not included** — training code, data pipelines, and the downstream task heads
(outcome prediction, segmentation, supervised probes) reported in the paper. The
released checkpoint carries no CAC-regression head: that pretraining objective's
parameters live in the objective aggregator, which `scripts/export_weights.py` drops.

`generate_report()` runs through the **pretrained T5 captioning decoder** carried in
`cardiofm_pretrained.pt`, and needs no further download. The LoRA-adapted RadLLaMA report
decoder benchmarked in the paper is optional, published separately, and governed by the
Llama 2 Community License — see [MODEL_LICENSE.md](MODEL_LICENSE.md).

## Hardware

A GPU is strongly recommended. A full-resolution `96 × 256 × 256 × 4` volume through the
encoder needs roughly 8 GB of GPU memory; on CPU it is slow and may exhaust system RAM.

## Limitations

- **Research use only.** Not a medical device, not validated for clinical decision-making.
- Trained on gated cardiac CT cropped to the heart. Behaviour on other fields of view,
  ungated chest CT or non-cardiac protocols is untested.
- Generated reports are fluent but numerically unreliable: an Agatston score or CT-FFR
  value in generated text may be wrong even when the category it implies is right.
- Zero-shot binary scores are uncalibrated similarity differences; the threshold of 0 is
  a convention, not a tuned operating point.

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

The source code is under the [Apache License 2.0](LICENSE); third-party
attributions are in [NOTICE](NOTICE).

Weights are not all under the same licence, so check
[MODEL_LICENSE.md](MODEL_LICENSE.md) before redistributing any of them:

| Artefact | Licence |
|---|---|
| Source code | Apache-2.0 |
| `cardiofm_pretrained.pt` | Apache-2.0 |
| Report-generation LoRA adapter | **Llama 2 Community License** |

Everything shown in this README — embeddings, zero-shot classification, retrieval
and `generate_report()` — uses Apache-2.0 artefacts only. The Llama 2 licence
applies only if you separately download the optional RadLLaMA LoRA adapter, which
is a derivative of a Llama 2 model and is published as a separate release asset
for that reason.
