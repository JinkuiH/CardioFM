# Licensing of code and model weights

Different artefacts in this project are governed by different licences. The
repository's `LICENSE` file applies to the **source code only**. Read this file
before redistributing any weights.

| Artefact | Licence | Where |
|---|---|---|
| Source code in this repository | Apache License 2.0 | [`LICENSE`](LICENSE) |
| `cardiofm_pretrained.pt` (image encoder, taps, text tower) | Apache License 2.0 | Release asset |
| Report-generation LoRA adapter | **Llama 2 Community License** | Released separately |

## Source code

Apache License 2.0. Third-party attributions are in [`NOTICE`](NOTICE).

## CardioFM pretrained weights

`cardiofm_pretrained.pt` is released under the Apache License 2.0.

Its upstream initialisations are Apache-2.0 and impose no further restriction:
the image encoder backbone was warm-started from MONAI's self-supervised Swin
UNETR weights, and the text tower from `t5-small`.

The weights were trained on de-identified cardiac CT and the structured text
derived from the corresponding radiology reports. No patient data, report text or
identifiers are contained in or recoverable from the released parameters, and
none are distributed with them.

**Intended use.** Research only. These weights are not a medical device, have not
been cleared or approved by any regulatory body, and must not be used to inform
clinical decisions. See the Limitations section of the README.

## Report-generation adapter

The optional report-generation adapter is a LoRA adapter trained on top of
[`StanfordAIMI/RadLLaMA-7b`](https://huggingface.co/StanfordAIMI/RadLLaMA-7b),
which is distributed under the **Llama 2 Community License** (`license: llama2`
on its model card). A LoRA adapter is a derivative work of the base model, so the
adapter inherits that licence and **cannot** be redistributed under Apache-2.0.

    Llama 2 is licensed under the LLAMA 2 Community License,
    Copyright (c) Meta Platforms, Inc. All Rights Reserved.

Consequences for users:

- The adapter is published as a **separate** release asset from
  `cardiofm_pretrained.pt`, so that an Apache-2.0 download and a Llama-2 download
  are never bundled together.
- The adapter is useless on its own. The package fetches RadLLaMA-7b from the
  Hugging Face Hub on first use; doing so means accepting the Llama 2 Community
  License, including its Acceptable Use Policy.
- The Llama 2 licence's restrictions, including its commercial-use threshold, flow
  through to the adapter.

The `generate_report()` method in the Python package **does** use this adapter.
Calling it downloads the adapter and the RadLLaMA-7b base model, and means
accepting the Llama 2 Community License. No other method in the package does.

## Summary

| Call | Licence it invokes |
|---|---|
| `encode_image`, `encode_text`, `zero_shot`, `retrieve`, `encoder_tokens` | Apache-2.0 only |
| `generate_report`, `generate_reports`, `load_report_generator` | **Llama 2** |

Using CardioFM for embeddings, zero-shot classification or retrieval involves
Apache-2.0 artefacts only. Report generation does not.
