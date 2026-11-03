"""Report generation with the RadLLaMA-7b + LoRA decoder.

The CardioFM image encoder stays frozen. Its bottleneck tokens pass through a
trained adapter, are prepended to the instruction prompt as a visual prefix, and
the LoRA-adapted language model decodes the report.

    CardioFM encoder -> [B, N, 768] -> Adapter -> [B, N, 4096] -> visual prefix
                                                                       |
                            prompt embeddings ---------------------> concat -> LLM

LICENCE. This module is the one part of CardioFM that is not Apache-2.0 at run
time. The decoder is StanfordAIMI/RadLLaMA-7b, a Llama 2 derivative under the
Llama 2 Community License, and the LoRA adapter is a derivative of that. Using
report generation means obtaining RadLLaMA-7b yourself and accepting the Llama 2
Community License, including its Acceptable Use Policy.

    Llama 2 is licensed under the LLAMA 2 Community License,
    Copyright (c) Meta Platforms, Inc. All Rights Reserved.

See MODEL_LICENSE.md. Nothing else in the package pulls in this dependency.
"""
from __future__ import annotations

import os
from typing import List, Optional, Sequence, Union

import torch
import torch.nn as nn

#: Base decoder. Not redistributed with CardioFM -- fetched from the Hub on first use.
LLM_ID = "StanfordAIMI/RadLLaMA-7b"
#: The instruction the adapter was trained against. Changing it degrades output.
PROMPT = "Generate a detailed radiology report for the cardiac CT scan.\n###\n"


class Adapter(nn.Module):
    """Image tokens -> LLM embedding space. Must match the trained checkpoint."""

    def __init__(self, in_dim: int, out_dim: int) -> None:
        super().__init__()
        self.net = nn.Sequential(nn.LayerNorm(in_dim), nn.Linear(in_dim, out_dim),
                                 nn.GELU(), nn.Linear(out_dim, out_dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class ReportGenerator(nn.Module):
    """RadLLaMA + LoRA decoder driven by CardioFM image tokens.

    Built by :meth:`CardioFM.load_report_generator`; you rarely construct it directly.
    """

    def __init__(self, weights_dir: str, device: Optional[str] = None,
                 dtype: torch.dtype = torch.bfloat16) -> None:
        super().__init__()
        from peft import PeftModel
        from transformers import AutoModelForCausalLM, AutoTokenizer

        lora_dir = os.path.join(weights_dir, "lora")
        tok_dir = os.path.join(weights_dir, "tokenizer")
        adapter_pt = os.path.join(weights_dir, "adapter.pt")
        for p in (lora_dir, tok_dir, adapter_pt):
            if not os.path.exists(p):
                raise FileNotFoundError(
                    f"{p} is missing. Download the report-generation asset from "
                    f"https://github.com/JinkuiH/CardioFM/releases/tag/v1.0.0-reportgen-llama2 "
                    f"and extract it; it is published separately because it is "
                    f"Llama 2 licensed, not Apache-2.0.")

        ck = torch.load(adapter_pt, map_location="cpu")
        self.prompt = ck.get("prompt", PROMPT)
        llm_id = ck.get("llm_id", LLM_ID)

        self.tok = AutoTokenizer.from_pretrained(tok_dir, use_fast=True, trust_remote_code=True)
        if self.tok.pad_token is None:
            self.tok.pad_token = self.tok.eos_token

        base = AutoModelForCausalLM.from_pretrained(llm_id, torch_dtype=dtype)
        self.llm = PeftModel.from_pretrained(base, lora_dir)
        self.llm.eval()

        # Infer the adapter geometry from the trained tensors rather than assuming it,
        # so a checkpoint with a different encoder width still loads.
        sd = ck["adapter"]
        in_dim, out_dim = sd["net.1.weight"].shape[1], sd["net.1.weight"].shape[0]
        self.dim = self.llm.get_input_embeddings().weight.shape[-1]
        if out_dim != self.dim:
            raise ValueError(f"adapter outputs {out_dim}-d but {llm_id} embeds {self.dim}-d; "
                             f"the adapter and the base model do not match")
        self.adapter = Adapter(in_dim, out_dim)
        self.adapter.load_state_dict(sd, strict=True)
        self.adapter.eval()

        self._prompt_ids = self.tok(self.prompt, add_special_tokens=False).input_ids
        device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.to(device)

    @property
    def device(self) -> torch.device:
        return next(self.adapter.parameters()).device

    @torch.no_grad()
    def generate(self, image_tokens: torch.Tensor, max_new_tokens: int = 256,
                 num_beams: int = 4, **kw) -> List[str]:
        """``[B, N, 768]`` CardioFM image tokens -> one report per item."""
        img = self.adapter(image_tokens.to(self.device).float()).to(
            self.llm.get_input_embeddings().weight.dtype)
        b = img.shape[0]
        pid = torch.tensor(self._prompt_ids, device=self.device)
        pe = self.llm.get_input_embeddings()(pid).to(img.dtype).unsqueeze(0).expand(b, -1, -1)
        ie = torch.cat([img, pe], 1)
        attn = torch.ones(ie.shape[:2], dtype=torch.long, device=self.device)
        out = self.llm.generate(inputs_embeds=ie, attention_mask=attn,
                                max_new_tokens=max_new_tokens, num_beams=num_beams,
                                pad_token_id=self.tok.pad_token_id, **kw)
        return self.tok.batch_decode(out, skip_special_tokens=True)
