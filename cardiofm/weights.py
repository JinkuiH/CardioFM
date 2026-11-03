"""Locating and caching the released CardioFM weights."""
from __future__ import annotations

import os
from pathlib import Path
from typing import Optional

#: Release asset holding the inference-only checkpoint.
WEIGHTS_URL = ("https://github.com/JinkuiH/CardioFM/releases/download/"
               "v1.0.0/cardiofm_pretrained.pt")
WEIGHTS_NAME = "cardiofm_pretrained.pt"


def cache_dir() -> Path:
    d = Path(os.environ.get("CARDIOFM_HOME", Path.home() / ".cache" / "cardiofm"))
    d.mkdir(parents=True, exist_ok=True)
    return d


def resolve_weights(weights: Optional[str] = None) -> str:
    """Return a local path to the checkpoint, downloading it on first use."""
    if weights:
        if not os.path.exists(weights):
            raise FileNotFoundError(weights)
        return weights

    env = os.environ.get("CARDIOFM_WEIGHTS")
    if env:
        if not os.path.exists(env):
            raise FileNotFoundError(f"CARDIOFM_WEIGHTS={env} does not exist")
        return env

    dest = cache_dir() / WEIGHTS_NAME
    if dest.exists():
        return str(dest)

    import urllib.request

    tmp = dest.with_suffix(".part")
    print(f"Downloading CardioFM weights to {dest} (this happens once) ...")
    try:
        urllib.request.urlretrieve(WEIGHTS_URL, tmp)
    except Exception as exc:                                    # pragma: no cover
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download weights from {WEIGHTS_URL} ({exc}). Download the "
            f"asset manually and either pass its path to from_pretrained(weights=...) "
            f"or set CARDIOFM_WEIGHTS to it.") from exc
    tmp.rename(dest)
    return str(dest)


# --------------------------------------------------------------------------- #
# Report-generation decoder -- Llama 2 licensed, published as a separate asset
# --------------------------------------------------------------------------- #
REPORT_URL = ("https://github.com/JinkuiH/CardioFM/releases/download/"
              "v1.0.0-reportgen-llama2/cardiofm_reportgen_lora_llama2.tar.gz")
REPORT_DIRNAME = "reportgen"

_LLAMA2_NOTICE = """
The report-generation decoder is a LoRA adapter over StanfordAIMI/RadLLaMA-7b and
is distributed under the Llama 2 Community License, NOT the Apache License 2.0
that covers the rest of CardioFM.

    Llama 2 is licensed under the LLAMA 2 Community License,
    Copyright (c) Meta Platforms, Inc. All Rights Reserved.

Downloading and using it means accepting that licence, including its Acceptable
Use Policy. The 7B base model is fetched separately from the Hugging Face Hub.
See MODEL_LICENSE.md.
"""


def resolve_report_weights(weights: Optional[str] = None) -> str:
    """Return a local directory holding ``adapter.pt``, ``lora/`` and ``tokenizer/``."""
    if weights:
        if not os.path.isdir(weights):
            raise NotADirectoryError(weights)
        return weights

    env = os.environ.get("CARDIOFM_REPORT_WEIGHTS")
    if env:
        if not os.path.isdir(env):
            raise NotADirectoryError(f"CARDIOFM_REPORT_WEIGHTS={env} is not a directory")
        return env

    dest = cache_dir() / REPORT_DIRNAME
    if (dest / "adapter.pt").exists():
        return str(dest)

    import tarfile
    import urllib.request

    print(_LLAMA2_NOTICE)
    tmp = cache_dir() / "reportgen.tar.gz.part"
    print(f"Downloading the report-generation decoder to {dest} (this happens once) ...")
    try:
        urllib.request.urlretrieve(REPORT_URL, tmp)
        with tarfile.open(tmp) as tf:
            tf.extractall(cache_dir())
    except Exception as exc:                                    # pragma: no cover
        tmp.unlink(missing_ok=True)
        raise RuntimeError(
            f"Could not download the report decoder from {REPORT_URL} ({exc}). Download "
            f"the asset manually, extract it, and pass the directory to "
            f"load_report_generator(weights=...) or set CARDIOFM_REPORT_WEIGHTS.") from exc
    tmp.unlink(missing_ok=True)
    if not (dest / "adapter.pt").exists():
        raise RuntimeError(f"archive extracted but {dest/'adapter.pt'} is missing")
    return str(dest)
