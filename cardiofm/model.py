"""CardioFM: a vision-language foundation model for coronary artery disease from cardiac CT.

    from cardiofm import CardioFM
    model = CardioFM.from_pretrained()                      # downloads weights once
    emb   = model.encode_image("scan.nii.gz")               # [1, 512]
    out   = model.zero_shot("scan.nii.gz", task="burden", modality="NCCT")
    text  = model.generate_report("scan.nii.gz")

Only the inference path is included. The published model was pretrained with a
contrastive objective, report captioning and coronary-guided masked image
modelling; the released weights carry the image encoder, the pooling taps and the
text tower, which is everything the calls above need.
"""
from __future__ import annotations

import os
from typing import Dict, List, Optional, Sequence, Union

import torch
import torch.nn.functional as F

from . import prompts as P
from .encoder import SwinUNETREncoder
from .preprocess import preprocess_ct
from .taps import AttentionPool3d, TokenProjection
from .text_tower import ClinicalTextTower
from .weights import resolve_weights

ImageLike = Union[str, torch.Tensor]

DEFAULT_CFG = {
    "d_proj": 512,
    "d_txt": 512,
    "attn_pool_heads": 8,
    "encoder": {"in_channels": 4, "feature_size": 48,
                "depths": (2, 2, 2, 2), "num_heads": (3, 6, 12, 24)},
    "text_tower": {"checkpoint": "t5-small"},
}


class CardioFM(torch.nn.Module):
    """Image tower + text tower sharing a 512-d embedding space."""

    def __init__(self, cfg: Optional[dict] = None, tokenizer_name: str = "t5-small") -> None:
        super().__init__()
        cfg = {**DEFAULT_CFG, **(cfg or {})}
        self.cfg = cfg
        d_proj, d_txt = int(cfg["d_proj"]), int(cfg["d_txt"])

        self.encoder = SwinUNETREncoder(**cfg["encoder"])
        c = self.encoder.bottleneck_dim
        self.attn_pool = AttentionPool3d(c, d_proj, int(cfg["attn_pool_heads"]))
        self.token_proj = TokenProjection(c, d_txt)
        self.text_tower = ClinicalTextTower(d_proj=d_proj, d_txt=d_txt, **cfg["text_tower"])

        from transformers import AutoTokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
        self.eval()

    # ------------------------------------------------------------------ loading
    @classmethod
    def from_pretrained(cls, weights: Optional[str] = None, device: Optional[str] = None,
                        **kw) -> "CardioFM":
        """Build the model and load the released inference weights.

        ``weights`` may be a local path; when omitted the checkpoint is downloaded
        and cached under ``$CARDIOFM_HOME`` (default ``~/.cache/cardiofm``).
        """
        model = cls(**kw)
        path = resolve_weights(weights)
        sd = torch.load(path, map_location="cpu")
        sd = sd.get("model", sd)
        missing, unexpected = model.load_state_dict(sd, strict=False)
        # The tokenizer and T5 buffers are rebuilt from the HF checkpoint, so a
        # handful of non-persistent keys are expected to be missing; anything in
        # `unexpected` means the checkpoint does not match this architecture.
        if unexpected:
            raise RuntimeError(f"checkpoint has {len(unexpected)} unexpected keys, "
                               f"first few: {unexpected[:5]}")
        model._missing_keys = missing
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        return model.to(device).eval()

    @property
    def device(self) -> torch.device:
        return next(self.parameters()).device

    # ------------------------------------------------------------------ encoding
    def _as_batch(self, image: ImageLike) -> torch.Tensor:
        if isinstance(image, str):
            image = preprocess_ct(image)
        if image.dim() == 4:
            image = image.unsqueeze(0)
        if image.shape[1] != self.encoder.in_channels:
            raise ValueError(
                f"expected {self.encoder.in_channels} channels (one per HU window), "
                f"got {image.shape[1]}; use cardiofm.preprocess.apply_windows")
        return image.to(self.device)

    @torch.no_grad()
    def encode_image(self, image: ImageLike, normalize: bool = True) -> torch.Tensor:
        """Image -> ``[B, 512]`` embedding in the shared space."""
        tokens = self.encoder.encode(self._as_batch(image))["tokens"]
        emb = self.attn_pool(tokens)
        return F.normalize(emb, dim=-1) if normalize else emb

    @torch.no_grad()
    def image_tokens(self, image: ImageLike) -> torch.Tensor:
        """Image -> ``[B, N, d_txt]`` tokens, the decoder memory for report generation."""
        return self.token_proj(self.encoder.encode(self._as_batch(image))["tokens"])

    @torch.no_grad()
    def encode_text(self, texts: Union[str, Sequence[str]],
                    normalize: bool = True, max_length: int = 128) -> torch.Tensor:
        """Text -> ``[B, 512]`` embedding in the shared space."""
        if isinstance(texts, str):
            texts = [texts]
        tok = self.tokenizer(list(texts), padding=True, truncation=True,
                             max_length=max_length, return_tensors="pt").to(self.device)
        emb = self.text_tower.encode_text(tok["input_ids"], tok["attention_mask"])
        return F.normalize(emb, dim=-1) if normalize else emb

    # ------------------------------------------------------------------ zero-shot
    @torch.no_grad()
    def _class_embedding(self, task: str, class_name: str, modality: str) -> torch.Tensor:
        """Mean of the four L2-normalised template embeddings, re-normalised."""
        e = self.encode_text(P.templates(task, class_name, modality), normalize=True)
        return F.normalize(e.mean(0), dim=-1)

    @torch.no_grad()
    def zero_shot(self, image: ImageLike, task: str, modality: str) -> Dict[str, object]:
        """Score one image against a task's class prompts.

        Returns ``{"task", "modality", "classes", "similarity", ...}``. For the two
        ordinal tasks the result also carries ``probs`` (softmax over classes) and
        ``prediction``; for the binary tasks it carries ``score`` =
        sim(positive) - sim(negative), whose decision threshold is 0. That score is
        a similarity difference, not a calibrated probability.
        """
        img = self.encode_image(image, normalize=True)                   # [1, 512]
        names = P.class_names(task)
        cls = torch.stack([self._class_embedding(task, c, modality) for c in names])
        sim = (img @ cls.T).squeeze(0)                                   # [n_classes]
        out: Dict[str, object] = {"task": task, "modality": modality,
                                  "classes": names,
                                  "similarity": sim.cpu().tolist()}
        if task in ("burden", "stenosis"):
            probs = sim.softmax(0)
            out["probs"] = probs.cpu().tolist()
            out["prediction"] = names[int(sim.argmax())]
            if task == "burden":
                out["agatston_bin"] = P.AGATSTON_BINS[int(sim.argmax())]
        else:
            pos = {n: i for i, n in enumerate(names)}
            pairs = ([("obstructive", "positive", "negative")] if task == "obstructive"
                     else [(n.split(":")[0], f"{n.split(':')[0]}:present",
                            f"{n.split(':')[0]}:absent")
                           for n in names if n.endswith(":present")])
            out["score"] = {label: float(sim[pos[p]] - sim[pos[a]]) for label, p, a in pairs}
            out["prediction"] = {k: bool(v > 0) for k, v in out["score"].items()}
        return out

    # ------------------------------------------------------------------ generation
    @torch.no_grad()
    def generate_report(self, image: ImageLike, max_new_tokens: int = 160,
                        num_beams: int = 4, **kw) -> str:
        """Generate a findings narrative from the image, via the pretrained decoder."""
        ids = self.text_tower.generate(self.image_tokens(image),
                                       max_new_tokens=max_new_tokens,
                                       num_beams=num_beams, **kw)
        return self.tokenizer.decode(ids[0], skip_special_tokens=True)

    # ------------------------------------------------------------------ retrieval
    @torch.no_grad()
    def retrieve(self, queries: Union[ImageLike, Sequence[str]],
                 gallery: Sequence[Union[str, torch.Tensor]],
                 query_is_image: bool = True, top_k: int = 5) -> List[List[dict]]:
        """Rank ``gallery`` against ``queries`` by cosine similarity in the shared space.

        ``query_is_image=True`` ranks reports (``gallery`` = strings) for an image
        query; ``False`` ranks images (``gallery`` = paths or tensors) for text
        queries.
        """
        if query_is_image:
            q = self.encode_image(queries)
            g = self.encode_text(list(gallery))
        else:
            q = self.encode_text(list(queries))
            g = torch.cat([self.encode_image(x) for x in gallery])
        sims = q @ g.T
        k = min(top_k, g.shape[0])
        out = []
        for row in sims:
            vals, idx = row.topk(k)
            out.append([{"rank": r + 1, "index": int(i), "score": float(v),
                         "item": gallery[int(i)] if isinstance(gallery[int(i)], str) else None}
                        for r, (v, i) in enumerate(zip(vals, idx))])
        return out
