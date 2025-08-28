import os

import fire
import torch
import torch.nn as nn

from pali_gemma.data_process.paligemma_preprocess import PaliGemmaProcessor, get_model_inputs
from pali_gemma.fine_tune.lora import get_lora_model
from pali_gemma.inference.sampling import get_sampler
from pali_gemma.load_weight import load_hf_model
from pali_gemma.model.kv_cache import KVCache
from pali_gemma.utils import get_device, load_lora_config_from_file_or_args, print_color

# CONST VARIABLE
LORA_BASE_DIR = "./lora_adapters"


def inference(
    model: nn.Module,
    processor: PaliGemmaProcessor,
    device: torch.device,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int,
    temperature: float,
    top_p: float,
    do_sample: bool,
):
    model_inputs = get_model_inputs(processor, prompt, image_file_path, device)
    input_ids = model_inputs["input_ids"]
    attention_mask = model_inputs["attention_mask"]
    pixel_values = model_inputs["pixel_values"]

    kv_cache = KVCache()

    # Generate tokens until you see the stop token
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    for _ in range(max_tokens_to_generate):
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attention_mask,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]

        # Sample the next token
        sampler = get_sampler("top_p" if do_sample else "naive")
        next_token = sampler(next_token_logits, temperature=temperature, p=top_p)
        assert next_token.size() == (1, 1)
        next_token = next_token.squeeze(0)  # Remove batch dimension
        generated_tokens.append(next_token)
        # Stop if the stop token has been generated
        if next_token.item() == stop_token:
            break
        # Append the next token to the input
        input_ids = next_token.unsqueeze(0)
        attention_mask = torch.cat([attention_mask, torch.ones((1, 1), device=input_ids.device)], dim=-1)

    generated_tokens = torch.cat(generated_tokens, dim=-1)

    # Decode the generated tokens
    decoded = processor.tokenizer.decode(generated_tokens, skip_special_tokens=True)

    # print(prompt)
    print_color(prompt)
    print_color(decoded, "green")


def main(
    model_path: str,
    prompt: str,
    image_file_path: str,
    max_tokens_to_generate: int = 100,
    temperature: float = 0.8,
    top_p: float = 0.9,
    do_sample: bool = True,
    only_cpu: bool = False,
    lora_adapter_name: str | None = None,
    lora_merge: bool = False,
):
    device = get_device(only_cpu)

    print("Loading model from:", model_path)
    model, tokenizer = load_hf_model(model_path, device)

    # Optionally load LoRA adapter
    if lora_adapter_name:
        lora_adapter_path = os.path.join(LORA_BASE_DIR, lora_adapter_name)

        if not os.path.isdir(lora_adapter_path):
            raise FileNotFoundError(f"LoRA adapter not found: {lora_adapter_path}")

        # Wrap base model with LoRA layers (must match training target modules)
        lora_cfg = load_lora_config_from_file_or_args(lora_adapter_path)
        model = get_lora_model(model, lora_cfg, device)
        model.load_adapter(
            lora_adapter_path,
        )
        if lora_merge:
            model.merge_adapter()
        print(f"Loaded LoRA adapter from: {lora_adapter_path}. Merged: {lora_merge}")

    model = model.eval()
    model = model.to(device)

    num_image_tokens = model.config.vision_num_image_tokens
    image_size = model.config.vision_image_size
    processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

    print("Running inference")
    with torch.no_grad():
        inference(
            model,
            processor,
            device,
            prompt,
            image_file_path,
            max_tokens_to_generate,
            temperature,
            top_p,
            do_sample,
        )


if __name__ == "__main__":
    fire.Fire(main)
