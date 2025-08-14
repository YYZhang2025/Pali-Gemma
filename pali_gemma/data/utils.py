import torch
from PIL import Image

from pali_gemma.data.paligemma_preprocess import PaliGemmaProcessor


def move_inputs_to_device(model_inputs: dict, device: torch.device) -> dict:
    # model_inputs = {k: v.to(device) for k, v in model_inputs.items()}
    model_inputs = {k: (v.to(device) if torch.is_tensor(v) else v) for k, v in model_inputs.items()}

    return model_inputs


def get_model_inputs(processor: PaliGemmaProcessor, prompt: str, image_file_path: str, device: torch.device):
    image = Image.open(image_file_path)
    image = image.convert("RGB")  # Ensure the image is in RGB format
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)

    return model_inputs
