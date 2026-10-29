"""CT preprocessing for CardioFM.

The model expects a heart-cropped volume resampled to (1.5, 0.5, 0.5) mm in SAR
orientation, cropped or padded to 96 x 256 x 256, and expanded to four channels
by four HU windows. ``preprocess_ct`` does all of this from a NIfTI path.

Heart localisation uses TotalSegmentator when it is installed. If it is not, the
volume is centre-cropped instead and a warning is emitted -- results will differ
from the paper, because the model was trained on heart-cropped input.
"""
from __future__ import annotations

import warnings
from typing import Optional, Sequence, Tuple

import numpy as np
import torch

#: (level, width) per input channel. Channel order is fixed and must not change:
#: 0 soft tissue, 1 calcium, 2 lumen, 3 wide.
WINDOWS: Tuple[Tuple[float, float], ...] = ((40.0, 400.0), (130.0, 300.0),
                                            (300.0, 800.0), (500.0, 1500.0))
TARGET_SPACING: Tuple[float, float, float] = (1.5, 0.5, 0.5)
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


def _resample(vol: np.ndarray, spacing: Sequence[float],
              target: Sequence[float] = TARGET_SPACING) -> np.ndarray:
    from scipy.ndimage import zoom

    factors = [float(s) / float(t) for s, t in zip(spacing, target)]
    if all(abs(f - 1.0) < 1e-3 for f in factors):
        return vol
    return zoom(vol, factors, order=1, mode="nearest")


def _heart_bbox(path: str, margin: int = 16) -> Optional[Tuple[slice, slice, slice]]:
    """Heart bounding box from TotalSegmentator, or ``None`` if unavailable."""
    try:
        from totalsegmentator.python_api import totalsegmentator
    except ImportError:
        return None
    import nibabel as nib
    import tempfile, os

    with tempfile.TemporaryDirectory() as td:
        totalsegmentator(path, td, roi_subset=["heart"], quiet=True)
        f = os.path.join(td, "heart.nii.gz")
        if not os.path.exists(f):
            return None
        m = np.asanyarray(nib.load(f).dataobj) > 0
    if not m.any():
        return None
    out = []
    for ax in range(3):
        idx = np.where(m.any(axis=tuple(i for i in range(3) if i != ax)))[0]
        lo = max(0, int(idx[0]) - margin)
        hi = min(m.shape[ax], int(idx[-1]) + 1 + margin)
        out.append(slice(lo, hi))
    return tuple(out)


def preprocess_ct(path: str, use_totalsegmentator: bool = True) -> torch.Tensor:
    """NIfTI path -> ``[1, 4, 96, 256, 256]`` tensor ready for :meth:`CardioFM.encode_image`."""
    import nibabel as nib

    img = nib.load(path)
    vol = np.asanyarray(img.dataobj).astype(np.float32)
    spacing = tuple(float(z) for z in img.header.get_zooms()[:3])

    box = _heart_bbox(path) if use_totalsegmentator else None
    if box is None:
        if use_totalsegmentator:
            warnings.warn(
                "TotalSegmentator unavailable or found no heart; falling back to a centre "
                "crop. CardioFM was trained on heart-cropped volumes, so embeddings and "
                "zero-shot scores will not match the published results.",
                RuntimeWarning, stacklevel=2)
    else:
        vol = vol[box]

    vol = _resample(vol, spacing)
    vol = crop_or_pad_to_shape(vol)
    return apply_windows(vol).unsqueeze(0)
