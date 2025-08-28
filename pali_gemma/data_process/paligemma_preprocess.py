from typing import List

import numpy as np
import torch
from PIL import Image
from transformers import PreTrainedTokenizerBase

from pali_gemma.data_process.image_preprocess import process_images
from pali_gemma.utils import move_inputs_to_device, numpy_to_torch


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer: PreTrainedTokenizerBase, num_image_tokens: int, image_size: int):
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        EXTRA_TOKENS = [self.IMAGE_TOKEN]
        EXTRA_TOKENS += [f"<loc{i:04d}>" for i in range(1024)]  # For Object Detection (Bounding Boxes)
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]  # For Object Segmentation

        tokenizer.add_tokens(EXTRA_TOKENS, special_tokens=True)

        self.image_token_ids = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = True
        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = False,
    ):
        pixel_values = process_images(
            images,
            size=(self.image_size, self.image_size),
            resample=Image.Resampling.BICUBIC,
            rescale_factor=1 / 255.0,
        )

        # Stack images
        pixel_values = np.stack(pixel_values, axis=0)  # (B, 3, H, W)
        pixel_values = numpy_to_torch(pixel_values)

        input_strings = [
            add_image_tokens_to_prompt(
                prefix_prompt=prompt,
                bos_token=self.tokenizer.bos_token,
                image_seq_len=self.image_seq_length,
                image_token=self.IMAGE_TOKEN,
            )
            for prompt in text
        ]

        inputs = self.tokenizer(
            input_strings,
            padding=padding,
            return_tensors="pt",
            add_special_tokens=False,
            truncation=truncation,
        )

        return {
            "pixel_values": pixel_values,
            "input_ids": inputs["input_ids"],
            "attention_mask": inputs["attention_mask"],
        }


def get_model_inputs(processor, prompt: str, image_file_path: str, device: torch.device):
    image = Image.open(image_file_path)
    image = image.convert("RGB")  # Ensure the image is in RGB format
    images = [image]
    prompts = [prompt]
    model_inputs = processor(text=prompts, images=images)
    model_inputs = move_inputs_to_device(model_inputs, device)

    return model_inputs
