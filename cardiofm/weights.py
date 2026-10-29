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
