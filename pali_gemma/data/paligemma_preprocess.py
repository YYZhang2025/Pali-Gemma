from typing import List

import numpy as np
from PIL import Image
from transformers import PreTrainedTokenizerBase

from pali_gemma.data.image_preprocess import process_images
from pali_gemma.utils import numpy_to_torch


def add_image_tokens_to_prompt(prefix_prompt, bos_token, image_seq_len, image_token):
    return f"{image_token * image_seq_len}{bos_token}{prefix_prompt}\n"


class PaliGemmaProcessor:
    IMAGE_TOKEN = "<image>"

    def __init__(self, tokenizer: PreTrainedTokenizerBase, num_image_tokens: int, image_size: int):
        self.image_seq_length = num_image_tokens
        self.image_size = image_size

        tokens_to_add = {"additional_special_tokens": [self.IMAGE_TOKEN]}

        # tokenizer.add_special_tokens(tokens_to_add)
        # Add the image token as a special token without tripping strict type checkers
        tokenizer.add_tokens([self.IMAGE_TOKEN], special_tokens=True)

        EXTRA_TOKENS = [f"<loc{i:04d}>" for i in range(1024)]  # For Object Detection (Bounding Boxes)
        EXTRA_TOKENS += [f"<seg{i:03d}>" for i in range(128)]  # For Object Segmentation

        tokenizer.add_tokens(EXTRA_TOKENS)

        self.image_token_ids = tokenizer.convert_tokens_to_ids(self.IMAGE_TOKEN)

        tokenizer.add_bos_token = False
        tokenizer.add_eos_token = False

        self.tokenizer = tokenizer

    def __call__(
        self,
        text: List[str],
        images: List[Image.Image],
        padding: str = "longest",
        truncation: bool = True,
    ):
        # TODO: Extend to take several images
        # assert len(images) == 1 and len(text) == 1, f"Received {len(images)} images for {len(text)} prompts."

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
        )

        return_data = {"pixel_values": pixel_values, **inputs}

        return return_data
