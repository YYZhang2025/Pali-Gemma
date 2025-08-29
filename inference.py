# inference_fixed.py
from __future__ import annotations

import os

import fire
import torch
import torch.nn as nn

from pali_gemma.data_process.paligemma_preprocess import (
    PaliGemmaProcessor,
    get_model_inputs,
)
from pali_gemma.fine_tune.lora import get_lora_model
from pali_gemma.inference.sampling import get_sampler
from pali_gemma.load_weight import load_hf_model
from pali_gemma.model.kv_cache import KVCache
from pali_gemma.utils import (
    get_device,
    load_lora_config_from_file_or_args,
    print_color,
)

# Where your LoRA adapters are saved
LORA_BASE_DIR = "./lora_adapters"


@torch.no_grad()
def generate_with_cache(
    model: nn.Module,
    processor: PaliGemmaProcessor,
    device: torch.device,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int = 256,
    temperature: float = 0.9,
    top_p: float = 0.95,
    do_sample: bool = True,
    min_steps_no_eos: int = 8,
    verbose: bool = True,
) -> str:
    """
    Greedy/top-p decoding with KV cache. Blocks EOS for the first `min_steps_no_eos` steps
    so the model can't immediately end after "Extract JSON".
    """
    # Build inputs (prompt + <image> tokens + pixel_values + attention_mask)
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]  # [1, S]
    attention_mask = model_inputs["attention_mask"]  # [1, S]
    pixel_values = model_inputs["pixel_values"]  # [1, C, H, W]

    # Quick sanity checks
    eos_id = processor.tokenizer.eos_token_id
    if eos_id is None:
        raise RuntimeError("Your tokenizer has no eos_token_id set.")

    if verbose:
        ends_with_eos = input_ids[0, -1].item() == eos_id
        print_color(f"ends_with_eos_in_prompt? {ends_with_eos}", "yellow")
        print_color(f"pixel_values: {tuple(pixel_values.shape)}", "yellow")

    # KV cache init
    kv_cache = KVCache()

    # Prepare sampling
    sampler = get_sampler("top_p" if do_sample else "naive")
    generated_tokens = []  # list of tensors shape [1, 1]

    # Decode loop
    for step in range(max_tokens_to_generate):
        outputs = model(
            input_ids=input_ids,  # full prompt on first step, 1 token afterwards
            pixel_values=pixel_values,  # can be None after first step; safe to pass once too
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]  # [1, V]

        # Block EOS for a few steps to force some content
        if step < min_steps_no_eos:
            next_token_logits[:, eos_id] = -float("inf")

        # Sample or take argmax
        next_token = sampler(next_token_logits, temperature=temperature, p=top_p)  # [1, 1]

        generated_tokens.append(next_token)

        # If EOS and we've allowed it, stop
        if step >= min_steps_no_eos and next_token.item() == eos_id:
            break

        # Incremental decoding with cache:
        # Feed only the last generated token in the next step.
        input_ids = next_token  # shape [1,1]
        # Grow the attention mask by one position of 1
        attention_mask = torch.cat(
            [attention_mask, torch.ones((1, 1), device=attention_mask.device, dtype=attention_mask.dtype)],
            dim=-1,
        )

    # Concatenate generated tokens -> [1, T]
    gen = (
        torch.cat(generated_tokens, dim=1)
        if generated_tokens
        else torch.empty((1, 0), dtype=torch.long, device=device)
    )

    # Return decoded string (generated part only)
    decoded = processor.tokenizer.decode(gen[0], skip_special_tokens=False)
    if verbose:
        print_color("=== PROMPT ===", "cyan")
        print_color(prompt)
        print_color("=== GENERATED ===", "cyan")
        print_color(decoded, "green")
    return decoded


def main(
    model_path: str,
    prompt: str,
    image_file_path: str,
    # decoding
    max_tokens_to_generate: int = 256,
    temperature: float = 0.9,
    top_p: float = 0.95,
    do_sample: bool = True,
    min_steps_no_eos: int = 8,
    # runtime
    only_cpu: bool = False,
    # LoRA
    lora_adapter_name: str | None = None,  # e.g. "pali_lora_adapter" (directory name under LORA_BASE_DIR)
    lora_merge: bool = False,
    # logging
    verbose: bool = True,
):
    """
    Example:
    python inference_fixed.py --model_path <HF_OR_LOCAL_MODEL> \
        --prompt "extract JSON.\n" \
        --image_file_path ./receipt.jpg \
        --lora_adapter_name pali_lora_adapter
    """
    device = get_device(only_cpu)
    print_color(f"Loading model from: {model_path}", "yellow")
    model, tokenizer = load_hf_model(model_path, device)

    # LoRA adapter (optional)
    if lora_adapter_name:
        lora_dir = os.path.join(LORA_BASE_DIR, lora_adapter_name)
        if not os.path.isdir(lora_dir):
            raise FileNotFoundError(f"LoRA adapter not found at {lora_dir}")
        lora_cfg = load_lora_config_from_file_or_args(lora_dir)
        model = get_lora_model(model, lora_cfg, device)
        model.load_adapter(lora_dir)
        if lora_merge:
            model.merge_adapter()
        print_color(f"Loaded LoRA adapter from: {lora_dir}. Merged: {lora_merge}", "yellow")

    model.eval().to(device)

    # Build processor
    num_image_tokens = model.config.vision_num_image_tokens
    image_size = model.config.vision_image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print_color("Running inference", "yellow")
    _ = generate_with_cache(
        model=model,
        processor=processor,
        device=device,
        prompt=prompt,
        image_file_path=image_file_path,
        max_tokens_to_generate=max_tokens_to_generate,
        temperature=temperature,
        top_p=top_p,
        do_sample=do_sample,
        min_steps_no_eos=min_steps_no_eos,
        verbose=verbose,
    )


if __name__ == "__main__":
    fire.Fire(main)
