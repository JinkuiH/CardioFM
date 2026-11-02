"""CT preprocessing for CardioFM.

This reproduces the geometry the model was trained on, in this order:

    1. TotalSegmentator heart mask      (roi_subset=["heart"], fast=True)
    2. reorient to SAR
    3. resample to (1.5, 0.5, 0.5) mm
    4. crop to the heart bounding box + margin (12, 20, 20) voxels
    5. centre crop or pad to 96 x 256 x 256
    6. four HU windows -> four channels

Steps 2-4 run as a MONAI transform chain, identical to the one used to build the
training volumes. The order matters: reorientation and resampling happen *before*
the crop, so the margin is applied in resampled voxels.

The heart crop is required, not optional. CardioFM was trained exclusively on
heart-cropped volumes, and an uncropped scan puts the model far outside the
distribution it saw -- the embedding is then not interpretable and the zero-shot
scores are meaningless. Rather than silently degrade, this module raises.
"""
from __future__ import annotations

import os
import tempfile
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

#: (level, width) per input channel. Channel order is fixed and must not change:
#: 0 soft tissue, 1 calcium, 2 lumen, 3 wide.
WINDOWS: Tuple[Tuple[float, float], ...] = ((40.0, 400.0), (130.0, 300.0),
                                            (300.0, 800.0), (500.0, 1500.0))
TARGET_SPACING: Tuple[float, float, float] = (1.5, 0.5, 0.5)
#: Heart-crop margin in resampled voxels = (18, 10, 10) mm.
MARGIN_VOX: Tuple[int, int, int] = (12, 20, 20)
ORIENTATION: str = "SAR"
TARGET_SHAPE: Tuple[int, int, int] = (96, 256, 256)
HU_PAD_VALUE: float = -1024.0


def apply_windows(hu: np.ndarray,
                  windows: Sequence[Sequence[float]] = WINDOWS) -> torch.Tensor:
    """HU volume ``[D,H,W]`` -> ``[C,D,H,W]`` float32 in ``[0,1]``, one channel per window."""
    chans = []
    for level, width in windows:
        lo, hi = level - width / 2.0, level + width / 2.0
        x = np.clip(hu, lo, hi)
        chans.append(((x - lo) / max(hi - lo, 1e-6)).astype(np.float32))
    return torch.from_numpy(np.stack(chans, axis=0))


def crop_or_pad_to_shape(vol: np.ndarray,
                         shape: Sequence[int] = TARGET_SHAPE,
                         pad_value: float = HU_PAD_VALUE) -> np.ndarray:
    """Centre crop or pad ``vol`` to ``shape``."""
    out = np.full(tuple(shape), pad_value, dtype=vol.dtype)
    src, dst = [], []
    for sv, sz in zip(vol.shape, shape):
        if sv >= sz:
            s = (sv - sz) // 2
            src.append(slice(s, s + sz)); dst.append(slice(0, sz))
        else:
            s = (sz - sv) // 2
            src.append(slice(0, sv)); dst.append(slice(s, s + sv))
    out[tuple(dst)] = vol[tuple(src)]
    return out


def heart_mask(image_path: str, cache_dir: Optional[str] = None) -> str:
    """Run TotalSegmentator and return the path to ``heart.nii.gz``.

    ``cache_dir`` persists the mask so repeated calls on the same scan are cheap.
    Raises ``ImportError`` if TotalSegmentator is missing and ``RuntimeError`` if it
    produces no heart mask.
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError as exc:
        raise ImportError(
            "CardioFM requires TotalSegmentator to localise the heart. Install it with "
            "`pip install TotalSegmentator`. The model was trained exclusively on "
            "heart-cropped volumes, so this step cannot be skipped."
        ) from exc

    out_dir = cache_dir or tempfile.mkdtemp(prefix="cardiofm_ts_")
    heart = os.path.join(out_dir, "heart.nii.gz")
    if os.path.exists(heart):
        return heart
    os.makedirs(out_dir, exist_ok=True)
    totalsegmentator(image_path, out_dir, roi_subset=["heart"],
                     fast=True, quiet=True, ml=False)
    if not os.path.exists(heart):
        raise RuntimeError(
            f"TotalSegmentator produced no heart mask for {image_path}. CardioFM "
            f"expects a cardiac CT covering the heart; check the field of view.")
    return heart


def _heart_crop_chain(heart_path: str) -> np.ndarray:
    """MONAI chain: load -> SAR -> resample -> crop to the heart box + margin."""
    from monai.transforms import (Compose, CropForegroundd, EnsureChannelFirstd,
                                  LoadImaged, Orientationd, Spacingd)
    return Compose([
        LoadImaged(keys=["image", "heart"]),
        EnsureChannelFirstd(keys=["image", "heart"]),
        Orientationd(keys=["image", "heart"], axcodes=ORIENTATION),
        Spacingd(keys=["image", "heart"], pixdim=TARGET_SPACING,
                 mode=("bilinear", "nearest")),
        CropForegroundd(keys=["image"], source_key="heart",
                        margin=list(MARGIN_VOX), allow_smaller=True),
    ])


def preprocess_ct(path: str, ts_cache: Optional[str] = None) -> torch.Tensor:
    """NIfTI path -> ``[1, 4, 96, 256, 256]`` ready for :meth:`CardioFM.encode_image`.

    Requires TotalSegmentator; see module docstring. Set ``ts_cache`` to a directory
    to reuse the heart mask across repeated calls on the same scan.
    """
    heart = heart_mask(path, ts_cache)
    out = _heart_crop_chain(heart)({"image": path, "heart": heart})

    vol = out["image"][0]
    vol = vol.cpu().numpy() if hasattr(vol, "cpu") else np.asarray(vol)
    vol = crop_or_pad_to_shape(vol.astype(np.float32))
    return apply_windows(vol).unsqueeze(0)
