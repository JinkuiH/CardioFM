"""CardioFM -- a vision-language foundation model for coronary artery disease from cardiac CT."""
from .model import CardioFM
from .preprocess import apply_windows, preprocess_ct, WINDOWS, TARGET_SHAPE, TARGET_SPACING
from . import prompts

__all__ = ["CardioFM", "preprocess_ct", "apply_windows", "prompts",
           "WINDOWS", "TARGET_SHAPE", "TARGET_SPACING"]
__version__ = "1.0.0"
