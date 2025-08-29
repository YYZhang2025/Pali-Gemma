from __future__ import annotations

import os
from typing import Optional

import fire
import torch
import torch.nn as nn
import torch.nn.functional as F
from PIL import Image
from torch.utils.tensorboard import SummaryWriter
from tqdm import tqdm

from pali_gemma.data_process.paligemma_preprocess import PaliGemmaProcessor
from pali_gemma.fine_tune.lora import LoraConfig, get_lora_model
from pali_gemma.load_weight import load_hf_model
from pali_gemma.utils import get_device, move_inputs_to_device


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
# Single-sample dataset builder
# =====================
def build_single_sample(
    processor: PaliGemmaProcessor, prompt: str, answer: str, image_path: str, device: torch.device
) -> dict:
    image = Image.open(image_path).convert("RGB")
    model_inputs = processor(prefix_prompt=[prompt], suffix_prompt=[answer], images=[image])
    # Using concatenated prompt+answer as targets; you can switch to special formatting if needed.
    return move_inputs_to_device(model_inputs, device)


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
# Main: LoRA fine-tune on one (prompt,image,answer)
# =====================
def main(
    model_path: str,
    prompt: str,
    image_file_path: str,
    answer: str = "",
    # LoRA
    lora_r: int = 16,
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
):
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
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

    # 6) Build one-sample batch
    batch = build_single_sample(processor, prompt, answer, image_file_path, device)

    # 7) Training loop (toy): repeat the same sample
    print("Start LoRA fine-tuning...")
    global_step = 0
    for epoch in range(epochs):
        running = 0.0
        for step in tqdm(range(steps), desc=f"Epoch {epoch + 1}/{epochs}"):
            loss = train_step(lora_model, batch, pad_token_id, image_token_id)
            loss = loss / grad_accum
            loss.backward()

            if (step + 1) % grad_accum == 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                global_step += 1

            running += loss.item()
            if (step + 1) % max(1, steps // 10) == 0:
                avg = running / max(1, step + 1)
                print(f"epoch {epoch + 1} step {step + 1}/{steps} loss {loss.item():.4f} (avg {avg:.4f})")

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
