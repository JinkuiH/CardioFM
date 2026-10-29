"""Clinical text tower: a T5 encoder-decoder.

The encoder yields the contrastive text embedding used for zero-shot prompting
and retrieval; the decoder performs report generation by cross-attending to
projected image tokens (image tokens replace the text-encoder memory).
"""
from __future__ import annotations

import torch
import torch.nn as nn


class ClinicalTextTower(nn.Module):
    def __init__(
        self,
        checkpoint: str = "t5-small",
        d_proj: int = 512,
        d_txt: int = 512,
        **_ignore,
    ) -> None:
        super().__init__()
        from transformers import T5ForConditionalGeneration

        self.model = T5ForConditionalGeneration.from_pretrained(checkpoint)
        self.d_model = self.model.config.d_model
        self.d_proj = d_proj
        self.d_txt = d_txt
        # Contrastive projection (encoder pooled -> shared dim) and image-memory
        # projection (image tokens d_txt -> decoder hidden d_model).
        self.text_proj = nn.Linear(self.d_model, d_proj)
        self.img_mem_proj = nn.Linear(d_txt, self.d_model)

    def encode_text(self, input_ids: torch.Tensor, attention_mask: torch.Tensor) -> torch.Tensor:
        enc = self.model.encoder(input_ids=input_ids,
                                 attention_mask=attention_mask).last_hidden_state
        mask = attention_mask.unsqueeze(-1).to(enc.dtype)
        pooled = (enc * mask).sum(1) / mask.sum(1).clamp(min=1e-6)   # mean-pool
        return self.text_proj(pooled)                                 # [B, d_proj]

    def _image_memory(self, image_tokens: torch.Tensor):
        from transformers.modeling_outputs import BaseModelOutput

        if image_tokens.shape[-1] != self.d_txt:
            raise ValueError(
                f"image tokens dim {image_tokens.shape[-1]} != d_txt {self.d_txt}")
        mem = self.img_mem_proj(image_tokens)                         # [B, N, d_model]
        attn = torch.ones(mem.shape[:2], device=mem.device, dtype=torch.long)
        return BaseModelOutput(last_hidden_state=mem), attn

    @torch.no_grad()
    def generate(self, image_tokens: torch.Tensor, max_new_tokens: int = 160, **kw):
        enc_out, attn = self._image_memory(image_tokens)
        return self.model.generate(encoder_outputs=enc_out, attention_mask=attn,
                                   max_new_tokens=max_new_tokens, **kw)
