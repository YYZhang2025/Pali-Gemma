import os

import gradio as gr
import torch
from PIL import Image

from pali_gemma.data.paligemma_preprocess import PaliGemmaProcessor

# --- LoRA (simple PEFT-like API) ---
from pali_gemma.fine_tune.lora import LoraConfig, get_lora_model
from pali_gemma.load_weight import load_hf_model
from pali_gemma.model.kv_cache import KVCache
from pali_gemma.model.pali_gemma_model import PaliGemmaForConditionalGeneration


# -------- Utilities -------- #
def pick_device(only_cpu: bool = False) -> str:
    if only_cpu:
        return "cpu"
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def ensure_rgb(image: Image.Image) -> Image.Image:
    return image if image.mode == "RGB" else image.convert("RGB")


def _sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_indices = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = probs_indices.gather(-1, next_token)
    return next_token


def _format_param_counts(model: torch.nn.Module) -> str:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    pct = 100.0 * trainable / max(total, 1)
    return f"Total parameters:     {total:,}\nTrainable parameters: {trainable:,} ({pct:.2f}%)"


# -------- Gradio stateful objects -------- #
class AppState:
    def __init__(self):
        self.model: PaliGemmaForConditionalGeneration | None = None
        self.tokenizer = None
        self.processor: PaliGemmaProcessor | None = None
        self.device: str = "cpu"
        # LoRA-related
        self.lora_wrapped: bool = False

    def is_ready(self) -> bool:
        return self.model is not None and self.processor is not None


STATE = AppState()


# -------- Core actions -------- #
def _apply_lora_if_any(
    model_path_or_msg: str,
    use_lora: bool,
    lora_files,  # list[gr.File] or None
    chosen_adapter_name: str,  # radio choice label (basename) or ""
    merge_adapters: bool,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
) -> tuple[str, str]:
    """
    Returns: (status_text, param_counts_text)
    """
    if not use_lora:
        STATE.lora_wrapped = False
        return (f"{model_path_or_msg}\nLoRA: disabled.", "")

    # Resolve file paths
    file_objs = lora_files or []
    filepaths = [f.name if hasattr(f, "name") else getattr(f, "path", None) for f in file_objs]
    filepaths = [p for p in filepaths if p and os.path.exists(p)]
    if not filepaths:
        return (f"{model_path_or_msg}\nLoRA: no adapter files provided.", "")

    # Wrap the base model with LoRA once
    cfg = LoraConfig(r=lora_r, lora_alpha=lora_alpha, lora_dropout=lora_dropout)
    STATE.model = get_lora_model(STATE.model, cfg)
    STATE.model.to(STATE.device)  # ensure A/B follow device
    STATE.lora_wrapped = True

    status_lines = [model_path_or_msg, f"LoRA: {len(filepaths)} file(s) provided."]

    if merge_adapters:
        # Merge all adapters sequentially into base
        applied = 0
        for p in filepaths:
            try:
                STATE.model.load_adapter(p, strict_shape=True)
                STATE.model.merge_adapter()  # fold delta W
                STATE.model.disable_adapter()  # keep base-only after merge
                applied += 1
            except Exception as e:
                status_lines.append(f"Merge failed for {os.path.basename(p)}: {e}")
        status_lines.append(f"Merged adapters: {applied}/{len(filepaths)}")
        param_txt = _format_param_counts(STATE.model)
        return ("\n".join(status_lines), param_txt)

    # Not merging: pick ONE to activate by name (radio)
    # Fallback: if radio empty or not matched, use the first
    target_basename = chosen_adapter_name or os.path.basename(filepaths[0])
    picked = None
    for p in filepaths:
        if os.path.basename(p) == target_basename:
            picked = p
            break
    if picked is None:
        picked = filepaths[0]
        status_lines.append(f"Adapter choice not found; defaulting to {os.path.basename(picked)}")

    try:
        STATE.model.load_adapter(picked, strict_shape=True)
        STATE.model.enable_adapter()
        param_txt = _format_param_counts(STATE.model)
        status_lines.append(f"Loaded adapter: {os.path.basename(picked)} (unmerged)")
        return ("\n".join(status_lines), param_txt)
    except Exception as e:
        status_lines.append(f"Load adapter failed: {e}")
        return ("\n".join(status_lines), "")


def load_model_action(
    model_path: str,
    only_cpu: bool,
    use_lora: bool,
    lora_files,
    chosen_adapter_name: str,
    merge_adapters: bool,
    lora_r: int,
    lora_alpha: float,
    lora_dropout: float,
):
    try:
        STATE.device = pick_device(only_cpu)
        model, tokenizer = load_hf_model(model_path, STATE.device)
        model = model.to(STATE.device).eval()

        # Build processor from model config
        num_image_tokens = model.config.vision_num_image_tokens
        image_size = model.config.vision_image_size
        processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))

        STATE.model = model
        STATE.tokenizer = tokenizer
        STATE.processor = processor

        base_msg = f"Loaded on {STATE.device}"
        status_text, param_counts = _apply_lora_if_any(
            base_msg,
            use_lora,
            lora_files,
            chosen_adapter_name,
            merge_adapters,
            lora_r,
            lora_alpha,
            lora_dropout,
        )
        # Enable Generate button if ready
        return (
            gr.update(value=status_text),
            gr.update(visible=True),
            gr.update(value=param_counts, visible=True),
        )
    except Exception as e:
        return gr.update(value=f"Load error: {e}"), gr.update(visible=False), gr.update(visible=False)


def update_adapter_choices(lora_files):
    """
    When files are dropped/removed, refresh the Radio choices with their basenames.
    """
    files = lora_files or []
    names = [os.path.basename(f.name if hasattr(f, "name") else getattr(f, "path", "")) for f in files if f]
    # If nothing, keep empty
    return gr.update(choices=names, value=(names[0] if names else None))


def generate_action(
    prompt: str, image: Image.Image, max_tokens: int, do_sample: bool, temperature: float, top_p: float
):
    if not STATE.is_ready():
        return "Model not loaded yet."
    if image is None or not prompt:
        return "Please provide both an image and a prompt."

    image = ensure_rgb(image)
    device = STATE.device
    model = STATE.model
    processor = STATE.processor

    inputs = processor(text=[prompt], images=[image])
    input_ids = inputs["input_ids"].to(device)
    attn_mask_ones = inputs["attention_mask"].to(device)
    pixel_values = inputs["pixel_values"].to(device)

    kv_cache = KVCache()
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    with torch.no_grad():
        outputs = model(
            input_ids=input_ids,
            pixel_values=pixel_values,
            attention_mask=attn_mask_ones,
            kv_cache=kv_cache,
        )
        kv_cache = outputs["kv_cache"]
        next_token_logits = outputs["logits"][:, -1, :]
        if do_sample:
            probs = torch.softmax(next_token_logits / max(1e-6, float(temperature)), dim=-1)
            next_token = _sample_top_p(probs, float(top_p))
        else:
            next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
        generated_tokens.append(next_token)

        for _ in range(max_tokens - 1):
            if next_token.item() == stop_token:
                break
            input_ids = next_token
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=torch.ones_like(input_ids, device=device),
                kv_cache=kv_cache,
            )
            kv_cache = outputs["kv_cache"]
            next_token_logits = outputs["logits"][:, -1, :]
            if do_sample:
                probs = torch.softmax(next_token_logits / max(1e-6, float(temperature)), dim=-1)
                next_token = _sample_top_p(probs, float(top_p))
            else:
                next_token = torch.argmax(next_token_logits, dim=-1, keepdim=True)
            generated_tokens.append(next_token)

    tokens = torch.cat(generated_tokens, dim=-1)
    text = processor.tokenizer.decode(tokens.squeeze(0), skip_special_tokens=True)
    return text


# -------- Gradio UI -------- #
with gr.Blocks(title="PaliGemma Simple Demo with LoRA") as demo:
    gr.Markdown(
        "# PaliGemma Inference Demo\nUpload an image, enter a prompt, and generate a response.\n\n**LoRA support:** Toggle it on, drag & drop one or more adapters, pick one to use or merge them into base."
    )

    with gr.Row():
        model_path = gr.Textbox(label="Model path or repo id", value="./models_weight", scale=3)
        only_cpu = gr.Checkbox(label="Force CPU", value=False)

    # --- LoRA controls ---
    with gr.Group():
        gr.Markdown("### LoRA Options")
        with gr.Row():
            use_lora = gr.Checkbox(label="Use LoRA", value=False)
            merge_adapters = gr.Checkbox(
                label="Merge adapters into base",
                value=False,
                info="If checked, all provided adapters will be merged sequentially.",
            )
        with gr.Row():
            lora_files = gr.Files(
                label="LoRA adapter files (.pt, .bin, etc.) — drag & drop multiple", file_count="multiple"
            )
            chosen_adapter = gr.Radio(choices=[], label="Choose adapter to use (if not merging)")
        with gr.Row():
            lora_r = gr.Slider(1, 256, value=8, step=1, label="LoRA rank (r)")
            lora_alpha = gr.Slider(1, 512, value=16.0, step=1.0, label="LoRA alpha")
            lora_dropout = gr.Slider(0.0, 0.5, value=0.0, step=0.01, label="LoRA dropout")

    load_btn = gr.Button("Load model", variant="primary")
    status = gr.Textbox(label="Status", value="Model not loaded", interactive=False)
    param_counts = gr.Textbox(label="Parameter counts", value="", interactive=False, visible=False)

    with gr.Row(variant="compact"):
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="What is in this image?", lines=2)
            image = gr.Image(type="pil", label="Image")
        with gr.Column():
            do_sample = gr.Checkbox(label="Use sampling (top-p)", value=False)
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            max_tokens = gr.Slider(1, 256, value=100, step=1, label="Max new tokens")
            run_btn = gr.Button("Generate", variant="primary", visible=False)

    output_text = gr.Textbox(label="Output", lines=8)

    # Wiring
    lora_files.change(update_adapter_choices, inputs=[lora_files], outputs=[chosen_adapter])
    load_btn.click(
        load_model_action,
        inputs=[
            model_path,
            only_cpu,
            use_lora,
            lora_files,
            chosen_adapter,
            merge_adapters,
            lora_r,
            lora_alpha,
            lora_dropout,
        ],
        outputs=[status, run_btn, param_counts],
    )
    run_btn.click(
        generate_action,
        inputs=[prompt, image, max_tokens, do_sample, temperature, top_p],
        outputs=output_text,
    )


if __name__ == "__main__":
    demo.launch()
