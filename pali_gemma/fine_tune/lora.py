# simple_peft_lora.py
from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Iterable, Sequence, Set

import torch
import torch.nn as nn
import torch.nn.functional as F


# ========= LoRA Linear (single adapter) =========
class LoraLinear(nn.Linear):
    """
    y = x W^T + (alpha/r) * (B @ A) contribution, where A: [r,in], B: [out,r]
    Base weights are frozen; only A/B are trainable.
    """

    def __init__(
        self, in_features, out_features, bias=True, r: int = 8, alpha: float = 16.0, dropout: float = 0.0
    ):
        super().__init__(in_features, out_features, bias=bias)
        assert r > 0, "LoRA rank r must be > 0"
        self.r = r
        self.alpha = float(alpha)
        self.scaling = self.alpha / r
        self.lora_enabled = True
        self.lora_dropout = nn.Dropout(dropout) if dropout and dropout > 0 else nn.Identity()

        # LoRA params
        self.lora_A = nn.Parameter(torch.empty(r, in_features))
        self.lora_B = nn.Parameter(torch.empty(out_features, r))
        self.reset_lora_parameters()

        # freeze base
        self.weight.requires_grad_(False)
        if self.bias is not None:
            self.bias.requires_grad_(False)

        # track merge state
        self._merged = False

    def reset_lora_parameters(self):
        nn.init.kaiming_uniform_(self.lora_A, a=math.sqrt(5))
        nn.init.zeros_(self.lora_B)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = F.linear(x, self.weight, self.bias)
        if self.lora_enabled and not self._merged:
            h = F.linear(self.lora_dropout(x), self.lora_A)  # [*, r]
            out = out + F.linear(h, self.lora_B) * self.scaling
        return out

    @torch.no_grad()
    def merge_lora(self):
        if self._merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling  # [out, in]
        self.weight.add_(delta.to(self.weight.dtype))
        self._merged = True

    @torch.no_grad()
    def unmerge_lora(self):
        if not self._merged:
            return
        delta = (self.lora_B @ self.lora_A) * self.scaling
        self.weight.sub_(delta.to(self.weight.dtype))
        self._merged = False


# ========= Simple PEFT-like interface =========
@dataclass
class LoraConfig:
    r: int = 8
    lora_alpha: float = 16.0
    lora_dropout: float = 0.0
    target_modules: Sequence[str] = field(
        default_factory=lambda: (
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "up_proj",
            "down_proj",
            "gate_proj",
            "proj",
            "fc",
        )
    )
    exclude_modules: Sequence[str] = field(default_factory=lambda: ("lm_head", "embed_tokens"))


class LoraModel(nn.Module):
    """
    Thin wrapper to hold base model and provide helpers.
    """

    def __init__(self, base: nn.Module, cfg: LoraConfig):
        super().__init__()
        self.base = base
        self.cfg = cfg

    def forward(self, *args, **kw):
        return self.base(*args, **kw)

    # ---- trainable selection ----
    def lora_parameters(self) -> Iterable[nn.Parameter]:
        for m in self.base.modules():
            if isinstance(m, LoraLinear):
                yield m.lora_A
                yield m.lora_B

    def mark_only_lora_as_trainable(self):
        for p in self.base.parameters():
            p.requires_grad_(False)
        for p in self.lora_parameters():
            p.requires_grad_(True)

    # ---- runtime controls ----
    def enable_adapter(self):
        for m in self.base.modules():
            if isinstance(m, LoraLinear):
                m.lora_enabled = True

    def disable_adapter(self):
        for m in self.base.modules():
            if isinstance(m, LoraLinear):
                m.lora_enabled = False

    def merge_adapter(self):
        for m in self.base.modules():
            if isinstance(m, LoraLinear):
                m.merge_lora()

    def unmerge_adapter(self):
        for m in self.base.modules():
            if isinstance(m, LoraLinear):
                m.unmerge_lora()

    # ---- save / load adapter-only ----
    def save_adapter(self, path: str):
        blob = {}
        for name, m in self.base.named_modules():
            if isinstance(m, LoraLinear):
                blob[f"{name}.A"] = m.lora_A.detach().cpu()
                blob[f"{name}.B"] = m.lora_B.detach().cpu()
                blob[f"{name}.alpha"] = torch.tensor(m.alpha)
                blob[f"{name}.r"] = torch.tensor(m.r)
        torch.save(dict(config=self.cfg, state=blob), path)

    @torch.no_grad()
    def load_adapter(self, path: str, strict_shape: bool = True):
        obj = torch.load(path, map_location="cpu")
        state = obj["state"]
        named = dict(self.base.named_modules())
        # copy tensors back
        for key, tensor in state.items():
            if not (key.endswith(".A") or key.endswith(".B") or key.endswith(".alpha") or key.endswith(".r")):
                continue
            mod_name, field = key.rsplit(".", 1)
            m = named.get(mod_name, None)
            if not isinstance(m, LoraLinear):
                continue
            if field == "A":
                if strict_shape and tensor.shape != m.lora_A.shape:
                    raise ValueError(f"Shape mismatch for {key}: {tensor.shape} vs {m.lora_A.shape}")
                m.lora_A.data.copy_(tensor.to(m.lora_A.dtype))
            elif field == "B":
                if strict_shape and tensor.shape != m.lora_B.shape:
                    raise ValueError(f"Shape mismatch for {key}: {tensor.shape} vs {m.lora_B.shape}")
                m.lora_B.data.copy_(tensor.to(m.lora_B.dtype))
            elif field == "alpha":
                m.alpha = float(tensor)
                m.scaling = m.alpha / m.r
            elif field == "r":
                # informational; we don't resize layers automatically in this simple version
                pass


def _should_wrap_linear(name: str, cfg: LoraConfig) -> bool:
    if any(ex in name for ex in cfg.exclude_modules):
        return False
    return any(t in name for t in cfg.target_modules)


def get_lora_model(model: nn.Module, cfg: LoraConfig) -> LoraModel:
    # replace targeted Linear layers in-place
    for parent_name, parent in list(model.named_modules()):
        for child_name, child in list(parent.named_children()):
            fqname = f"{parent_name}.{child_name}" if parent_name else child_name
            if isinstance(child, nn.Linear) and _should_wrap_linear(fqname, cfg):
                new = LoraLinear(
                    child.in_features,
                    child.out_features,
                    bias=(child.bias is not None),
                    r=cfg.r,
                    alpha=cfg.lora_alpha,
                    dropout=cfg.lora_dropout,
                )
                with torch.no_grad():
                    new.weight.copy_(child.weight)
                    if child.bias is not None:
                        new.bias.copy_(child.bias)
                setattr(parent, child_name, new)

    wrapped = LoraModel(model, cfg)
    wrapped.mark_only_lora_as_trainable()
    return wrapped
