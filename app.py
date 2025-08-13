import gradio as gr
from PIL import Image
import torch

from pali_gemma.data.paligemma_preprocess import PaliGemmaProcessor
from pali_gemma.model.pali_gemma_model import PaliGemmaForConditionalGeneration
from pali_gemma.model.kv_cache import KVCache
from pali_gemma.load_weight import load_hf_model


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
    if image.mode != "RGB":
        return image.convert("RGB")
    return image


def _sample_top_p(probs: torch.Tensor, p: float) -> torch.Tensor:
    probs_sort, probs_indices = torch.sort(probs, dim=-1, descending=True)
    probs_sum = torch.cumsum(probs_sort, dim=-1)
    mask = probs_sum - probs_sort > p
    probs_sort[mask] = 0.0
    probs_sort.div_(probs_sort.sum(dim=-1, keepdim=True))
    next_token = torch.multinomial(probs_sort, num_samples=1)
    next_token = probs_indices.gather(-1, next_token)
    return next_token


# -------- Gradio stateful objects -------- #
class AppState:
    def __init__(self):
        self.model: PaliGemmaForConditionalGeneration | None = None
        self.tokenizer = None
        self.processor: PaliGemmaProcessor | None = None
        self.device: str = "cpu"

    def is_ready(self) -> bool:
        return self.model is not None and self.processor is not None


STATE = AppState()


# -------- Core actions -------- #
def load_model_action(model_path: str, only_cpu: bool):
    try:
        STATE.device = pick_device(only_cpu)
        model, tokenizer = load_hf_model(model_path, STATE.device)
        model = model.to(STATE.device).eval()

        # Build processor from model config
        num_image_tokens = model.config.vision_num_image_tokens
        image_size = model.config.vision_image_size
        processor = PaliGemmaProcessor(tokenizer, num_image_tokens, image_size)

        # If vocab grew (special/extra tokens), resize embeddings
        if hasattr(model, "resize_token_embeddings"):
            model.resize_token_embeddings(len(tokenizer))

        STATE.model = model
        STATE.tokenizer = tokenizer
        STATE.processor = processor
        return gr.update(value=f"Loaded on {STATE.device}"), gr.update(visible=True)
    except Exception as e:
        return gr.update(value=f"Load error: {e}"), gr.update(visible=False)


def generate_action(prompt: str, image: Image.Image, max_tokens: int, do_sample: bool, temperature: float, top_p: float):
    if not STATE.is_ready():
        return "Model not loaded yet."
    if image is None or not prompt:
        return "Please provide both an image and a prompt."

    image = ensure_rgb(image)
    device = STATE.device
    model = STATE.model
    processor = STATE.processor

    # Prepare inputs (no padding expected by your model forward)
    inputs = processor(text=[prompt], images=[image])
    input_ids = inputs["input_ids"].to(device)
    attn_mask_ones = inputs["attention_mask"].to(device)  # should be ones for no padding
    pixel_values = inputs["pixel_values"].to(device)

    # KV cache + decode loop
    kv_cache = KVCache()
    stop_token = processor.tokenizer.eos_token_id
    generated_tokens = []

    with torch.no_grad():
        # First pass: full sequence
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

        # Incremental steps
        for _ in range(max_tokens - 1):
            if next_token.item() == stop_token:
                break
            input_ids = next_token  # shape (1,1)
            outputs = model(
                input_ids=input_ids,
                pixel_values=pixel_values,
                attention_mask=torch.ones_like(input_ids, device=device),  # LM asserts no padding
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
with gr.Blocks(title="PaliGemma Simple Demo") as demo:
    gr.Markdown("# PaliGemma Inference Demo\nUpload an image, enter a prompt, and generate a response.")

    with gr.Row():
        model_path = gr.Textbox(label="Model path or repo id", value="./models_weight", scale=3)
        only_cpu = gr.Checkbox(label="Force CPU", value=False)
        load_btn = gr.Button("Load model", variant="primary")
    status = gr.Textbox(label="Status", value="Model not loaded", interactive=False)

    with gr.Row(variant="compact"):
        with gr.Column():
            prompt = gr.Textbox(label="Prompt", value="What is in this image?", lines=2)
            image = gr.Image(type="pil", label="Image")
        with gr.Column():
            do_sample = gr.Checkbox(label="Use sampling (top-p)", value=False)
            temperature = gr.Slider(0.1, 2.0, value=0.8, step=0.05, label="Temperature")
            top_p = gr.Slider(0.1, 1.0, value=0.9, step=0.05, label="Top-p")
            max_tokens = gr.Slider(1, 256, value=100, step=1, label="Max new tokens")
            run_btn = gr.Button("Generate", variant="primary")

    output_text = gr.Textbox(label="Output", lines=8)

    load_btn.click(load_model_action, inputs=[model_path, only_cpu], outputs=[status, run_btn])
    run_btn.click(generate_action, inputs=[prompt, image, max_tokens, do_sample, temperature, top_p], outputs=output_text)


if __name__ == "__main__":
    demo.launch()
