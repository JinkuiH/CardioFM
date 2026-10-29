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
"""
from __future__ import annotations

import os
import tempfile
import warnings
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


def heart_mask(image_path: str, cache_dir: Optional[str] = None) -> Optional[str]:
    """Run TotalSegmentator and return the path to ``heart.nii.gz``, or ``None``.

    ``cache_dir`` persists the mask so repeated calls on the same scan are cheap.
    """
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError:
        return None

    out_dir = cache_dir or tempfile.mkdtemp(prefix="cardiofm_ts_")
    heart = os.path.join(out_dir, "heart.nii.gz")
    if os.path.exists(heart):
        return heart
    os.makedirs(out_dir, exist_ok=True)
    totalsegmentator(image_path, out_dir, roi_subset=["heart"],
                     fast=True, quiet=True, ml=False)
    return heart if os.path.exists(heart) else None


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


def _fallback_chain() -> np.ndarray:
    """Same geometry minus the heart crop, for when TotalSegmentator is unavailable."""
    from monai.transforms import (Compose, EnsureChannelFirstd, LoadImaged,
                                  Orientationd, Spacingd)
    return Compose([
        LoadImaged(keys=["image"]),
        EnsureChannelFirstd(keys=["image"]),
        Orientationd(keys=["image"], axcodes=ORIENTATION),
        Spacingd(keys=["image"], pixdim=TARGET_SPACING, mode="bilinear"),
    ])


def preprocess_ct(path: str, use_totalsegmentator: bool = True,
                  ts_cache: Optional[str] = None) -> torch.Tensor:
    """NIfTI path -> ``[1, 4, 96, 256, 256]`` ready for :meth:`CardioFM.encode_image`.

    Set ``ts_cache`` to a directory to reuse the heart mask across calls.
    """
    heart = heart_mask(path, ts_cache) if use_totalsegmentator else None

    if heart is None:
        if use_totalsegmentator:
            warnings.warn(
                "TotalSegmentator produced no heart mask (not installed, or segmentation "
                "failed). Falling back to a centre crop of the reoriented, resampled volume. "
                "CardioFM was trained on heart-cropped input, so embeddings and zero-shot "
                "scores will NOT reproduce the published results. Install it with "
                "`pip install TotalSegmentator`.",
                RuntimeWarning, stacklevel=2)
        out = _fallback_chain()({"image": path})
    else:
        out = _heart_crop_chain(heart)({"image": path, "heart": heart})

    vol = out["image"][0]
    vol = vol.cpu().numpy() if hasattr(vol, "cpu") else np.asarray(vol)
    vol = crop_or_pad_to_shape(vol.astype(np.float32))
    return apply_windows(vol).unsqueeze(0)
