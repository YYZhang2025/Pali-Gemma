from __future__ import annotations

import os
from typing import Any, Dict, List, Optional

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm
from datasets import load_dataset

from pali_gemma.data.paligemma_preprocess import PaliGemmaProcessor
from pali_gemma.data.utils import move_inputs_to_device
from pali_gemma.fine_tune.lora import LoraConfig, get_lora_model
from pali_gemma.load_weight import load_hf_model
from pali_gemma.utils import get_device
import json
import random


# =====================
# Label building (mask image/pad tokens)
# =====================
def build_labels(
    input_ids: torch.Tensor, attention_mask: torch.Tensor, pad_token_id: int, image_token_id: Optional[int]
) -> torch.Tensor:
    """
    Prepare next-token prediction labels from `input_ids` by masking out
    padding and image-token positions with -100.
    labels[t] corresponds to predicting token at position t from logits t-1.
    We'll shift during loss computation.
    """
    labels = input_ids.clone()
    # mask pads
    labels[attention_mask == 0] = -100
    # mask image tokens if we can identify them
    if image_token_id is not None and image_token_id >= 0:
        labels[labels == image_token_id] = -100
    return labels


# =====================
# CORD dataset (image -> JSON sequence)
# =====================
class CORDDataset(torch.utils.data.Dataset):
    """Loads Donut-formatted CORD (naver-clova-ix/cord-v2).
    Each sample yields (image, target_sequence_string).
    """

    def __init__(self, dataset_name_or_path: str, split: str = "train", sort_json_key: bool = True):
        super().__init__()
        self.sort_json_key = sort_json_key
        self.data = load_dataset(dataset_name_or_path, split=split)
        # Pre-build target sequences
        self.targets: List[List[str]] = []
        for sample in self.data:
            gt = (
                json.loads(sample["ground_truth"])
                if isinstance(sample.get("ground_truth"), str)
                else sample["ground_truth"]
            )
            if "gt_parses" in gt:  # multiple ground truths
                gt_list = gt["gt_parses"]
            else:
                gt_list = [gt.get("gt_parse", gt)]
            self.targets.append([self._json2token(gj) for gj in gt_list])

    def _json2token(self, obj: Any) -> str:
        """Convert a JSON object into a linearized token string using Donut-style tags."""
        if isinstance(obj, dict):
            if len(obj) == 1 and "text_sequence" in obj:
                return obj["text_sequence"]
            keys = sorted(obj.keys(), reverse=True) if self.sort_json_key else obj.keys()
            out = ""
            for k in keys:
                out += rf"<s_{k}>" + self._json2token(obj[k]) + rf"</s_{k}>"
            return out
        if isinstance(obj, list):
            return r"<sep/>".join([self._json2token(x) for x in obj])
        return str(obj)

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, idx: int):
        sample = self.data[idx]
        image = sample["image"]  # PIL.Image
        # If multiple gts exist, pick one at random each epoch
        tgt = random.choice(self.targets[idx])
        return image, tgt


# =====================
# Training loop
# =====================


def train_step(
    model: nn.Module, batch: dict, pad_token_id: int, image_token_id: Optional[int]
) -> torch.Tensor:
    input_ids = batch["input_ids"]  # [B, S]
    attention_mask = batch["attention_mask"]  # [B, S]
    pixel_values = batch["pixel_values"]  # [B, C, H, W]

    labels = build_labels(input_ids, attention_mask, pad_token_id, image_token_id)

    # Forward
    outputs = model(
        input_ids=input_ids,
        pixel_values=pixel_values,
        attention_mask=attention_mask,
        kv_cache=None,
    )
    logits = outputs["logits"]  # [B, S, V]

    # Shift for next-token prediction
    shift_logits = logits[:, :-1, :].contiguous()
    shift_labels = labels[:, 1:].contiguous()

    loss = F.cross_entropy(
        shift_logits.view(-1, shift_logits.size(-1)),
        shift_labels.view(-1),
        ignore_index=-100,
    )
    return loss


# =====================
# DataLoader helpers (CORD)
# =====================
def make_train_collate(processor: PaliGemmaProcessor, prompt_text: str):
    def _fn(batch):
        images = [b[0] for b in batch]
        # teacher-forcing: concatenate prompt and target JSON
        texts = [prompt_text + tgt for (_, tgt) in batch]
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            tokenize_newline_separately=False,
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
        }

    return _fn


def make_eval_collate(processor: PaliGemmaProcessor, prompt_text: str):
    # Same as train, but keeps the raw target for optional inspection
    def _fn(batch):
        images = [b[0] for b in batch]
        texts = [prompt_text + tgt for (_, tgt) in batch]
        inputs = processor(
            text=texts,
            images=images,
            return_tensors="pt",
            padding=True,
            tokenize_newline_separately=False,
        )
        return {
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
            "pixel_values": inputs["pixel_values"],
        }

    return _fn


# =====================
# Main: LoRA fine-tune on one (prompt,image,answer)
# =====================
def main(
    model_path: str,
    prompt: str,
    image_file_path: str,
    answer: str = "",
    # LoRA
    lora_r: int = 64,
    lora_alpha: float = 16.0,
    lora_dropout: float = 0.05,
    # Train
    lr: float = 1e-4,
    epochs: int = 1,
    steps: int = 100,
    grad_accum: int = 1,
    only_cpu: bool = False,
    save_adapter_path: str = "pali_lora_adapter.pt",
    adapters_dir: str = "./lora_adapters",
    # Data
    dataset_name_or_path: str = "naver-clova-ix/cord-v2",
    train_split: str = "train",
    val_split: str = "validation",
    batch_size: int = 4,
    num_workers: int = 2,
):
    device = get_device(only_cpu)
    writer = SummaryWriter()

    # 1) Load base model & tokenizer
    print("Loading model...")
    base_model, tokenizer = load_hf_model(model_path, device)

    # 2) Build processor (image tokenizer helper)
    num_image_tokens = base_model.config.vision_num_image_tokens
    image_size = base_model.config.vision_image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    # 3) Wrap with LoRA (PEFT-like)
    cfg = LoraConfig(
        r=lora_r,
        lora_alpha=lora_alpha,
        lora_dropout=lora_dropout,
        target_modules=("q_proj", "k_proj", "v_proj", "o_proj", "up_proj", "down_proj", "gate_proj", "proj"),
        exclude_modules=("lm_head", "embed_tokens"),
    )
    lora_model = get_lora_model(base_model, cfg, device)

    # Inserted code: count parameters and log to TensorBoard
    total_params = sum(p.numel() for p in lora_model.parameters())
    trainable_params = sum(p.numel() for p in lora_model.parameters() if p.requires_grad)
    trainable_percent = 100 * trainable_params / total_params
    print(f"Total parameters:     {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_percent:.2f}%)")

    # 4) Optimizer (only LoRA params)
    if hasattr(lora_model, "lora_parameters"):
        params = lora_model.lora_parameters()
    else:
        # fallback: parameters that contain lora_A/B
        params = (p for n, p in lora_model.named_parameters() if ("lora_A" in n or "lora_B" in n))
    optimizer = torch.optim.AdamW(params, lr=lr)

    # 5) Try get image token id for masking
    image_token_id = None
    try:
        image_token_id = tokenizer.convert_tokens_to_ids(processor.IMAGE_TOKEN)
        if image_token_id is None:
            image_token_id = -1
    except Exception:
        image_token_id = None

    pad_token_id = getattr(tokenizer, "pad_token_id", None)
    if pad_token_id is None:
        # some tokenizers use eos as pad; fallback
        pad_token_id = getattr(tokenizer, "eos_token_id", -100)

    # 6) Build datasets & dataloaders (CORD image->JSON)
    print("Loading CORD dataset...")
    train_dataset = CORDDataset(dataset_name_or_path, split=train_split, sort_json_key=True)
    val_dataset = CORDDataset(dataset_name_or_path, split=val_split, sort_json_key=True)

    train_collate = make_train_collate(processor, prompt)
    eval_collate = make_eval_collate(processor, prompt)

    from torch.utils.data import DataLoader

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        collate_fn=train_collate,
        pin_memory=True,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        collate_fn=eval_collate,
        pin_memory=True,
    )

    # 7) Training loop
    print("Start LoRA fine-tuning on CORD...")
    global_step = 0
    for epoch in range(epochs):
        lora_model.train()
        running = 0.0
        pbar = tqdm(train_loader, desc=f"Epoch {epoch + 1}/{epochs}")
        for step, batch in enumerate(pbar, start=1):
            batch = move_inputs_to_device(batch, device)
            loss = train_step(lora_model, batch, pad_token_id, image_token_id)
            (loss / grad_accum).backward()

            if step % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running += loss.item()
            if step % max(1, len(train_loader) // 10) == 0:
                avg = running / step
                pbar.set_postfix(loss=loss.item(), avg=avg)
                writer.add_scalar("train/loss", loss.item(), global_step)

        # Optional validation (loss-only)
        lora_model.eval()
        with torch.no_grad():
            val_running = 0.0
            val_steps = 0
            for vb in val_loader:
                vb = move_inputs_to_device(vb, device)
                vloss = train_step(lora_model, vb, pad_token_id, image_token_id)
                val_running += vloss.item()
                val_steps += 1
            if val_steps > 0:
                val_avg = val_running / val_steps
                print(f"[Val] epoch {epoch + 1} avg_loss {val_avg:.4f}")
                writer.add_scalar("val/loss", val_avg, epoch + 1)

    # Close the SummaryWriter
    writer.close()

    # Normalize save paths: always use <base>.pt and <base>.config
    lora_dir = os.path.join(adapters_dir, save_adapter_path)
    os.makedirs(lora_dir, exist_ok=True)  # ensure directory exists
    adapter_path = os.path.join(lora_dir, "adapter.pt")
    config_path = os.path.join(lora_dir, "config.json")

    print(f"Saving LoRA adapter -> {adapter_path}")
    if hasattr(lora_model, "save_adapter"):
        lora_model.save_adapter(adapter_path)
        cfg_dict = {
            "r": cfg.r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "target_modules": cfg.target_modules,
            "exclude_modules": cfg.exclude_modules,
        }
        torch.save(cfg_dict, config_path)
        print(f"Saved LoRA config -> {config_path}")
    else:
        # generic save: collect lora_A/B by module name
        blob = {}
        for name, m in lora_model.named_modules():
            if hasattr(m, "lora_A") and hasattr(m, "lora_B"):
                blob[f"{name}.A"] = m.lora_A.detach().cpu()
                blob[f"{name}.B"] = m.lora_B.detach().cpu()
        torch.save({"state": blob}, adapter_path)
        cfg_dict = {
            "r": cfg.r,
            "lora_alpha": cfg.lora_alpha,
            "lora_dropout": cfg.lora_dropout,
            "target_modules": cfg.target_modules,
            "exclude_modules": cfg.exclude_modules,
        }
        torch.save(cfg_dict, config_path)
        print(f"Saved LoRA config -> {config_path}")
    print("Done.")


if __name__ == "__main__":
    fire.Fire(main)
