import glob
import json
import os
from typing import Any, Dict, Tuple

import torch
from safetensors import safe_open
from transformers import AutoTokenizer

from pali_gemma.config import ModelConfig
from pali_gemma.model.pali_gemma_model import PaliGemmaForConditionalGeneration


def hf_to_model_config(hf_cfg: Dict[str, Any]) -> ModelConfig:
    """Convert a HF-style PaliGemma config dict into our ModelConfig.

    This maps fields from top-level, `vision_config`, and `text_config`.
    Any missing fields fall back to the dataclass defaults.
    """
    v = hf_cfg.get("vision_config", {}) or {}
    t = hf_cfg.get("text_config", {}) or {}

    # Helper to pull a default from the dataclass when a key is absent
    defaults = {k: f.default for k, f in ModelConfig.__dataclass_fields__.items()}

    return ModelConfig(
        # Vision
        vision_hidden_size=v.get("hidden_size", defaults["vision_hidden_size"]),
        vision_intermediate_size=v.get("intermediate_size", defaults["vision_intermediate_size"]),
        vision_num_hidden_layers=v.get("num_hidden_layers", defaults["vision_num_hidden_layers"]),
        vision_num_attention_heads=v.get("num_attention_heads", defaults["vision_num_attention_heads"]),
        vision_num_channels=v.get("num_channels", defaults["vision_num_channels"]),
        vision_image_size=v.get("image_size", defaults["vision_image_size"]),
        vision_patch_size=v.get("patch_size", defaults["vision_patch_size"]),
        vision_layer_norm_eps=v.get("layer_norm_eps", defaults["vision_layer_norm_eps"]),
        vision_attention_dropout=v.get("attention_dropout", defaults["vision_attention_dropout"]),
        vision_num_image_tokens=v.get("num_image_tokens", None),
        # LM
        lm_vocab_size=t.get("vocab_size", hf_cfg.get("vocab_size", defaults["lm_vocab_size"])),
        lm_hidden_size=t.get("hidden_size", defaults["lm_hidden_size"]),
        lm_intermediate_size=t.get("intermediate_size", defaults["lm_intermediate_size"]),
        lm_num_hidden_layers=t.get("num_hidden_layers", defaults["lm_num_hidden_layers"]),
        lm_num_attention_heads=t.get("num_attention_heads", defaults["lm_num_attention_heads"]),
        lm_num_key_value_heads=t.get("num_key_value_heads", defaults["lm_num_key_value_heads"]),
        lm_num_heads=t.get("num_attention_heads", defaults["lm_num_heads"]),
        lm_max_position_embeddings=t.get("max_position_embeddings", defaults["lm_max_position_embeddings"]),
        lm_rms_norm_eps=t.get("rms_norm_eps", defaults["lm_rms_norm_eps"]),
        lm_rope_theta=hf_cfg.get("lm_rope_theta", defaults["lm_rope_theta"]),
        lm_attention_bias=t.get("attention_bias", defaults["lm_attention_bias"]),
        lm_attention_dropout=t.get("attention_dropout", defaults["lm_attention_dropout"]),
        pad_token_id=hf_cfg.get("pad_token_id", defaults["pad_token_id"]),
        # PaliGemma-specific
        ignore_index=hf_cfg.get("ignore_index", defaults["ignore_index"]),
        image_token_index=hf_cfg.get("image_token_index", defaults["image_token_index"]),
        projection_dim=hf_cfg.get("projection_dim", defaults["projection_dim"]),
        is_encoder_decoder=hf_cfg.get("is_encoder_decoder", defaults["is_encoder_decoder"]),
    )


def load_model_config_from_json(path: str) -> ModelConfig:
    """Load a HF JSON file from disk and convert it to ModelConfig."""
    with open(path, "r") as f:
        hf_cfg = json.load(f)
    return hf_to_model_config(hf_cfg)


def load_hf_model(
    model_path: str = "models_weight",
    device: torch.device = torch.device("cpu"),
) -> Tuple[PaliGemmaForConditionalGeneration, AutoTokenizer]:
    tokenizer = AutoTokenizer.from_pretrained(model_path, padding_side="right")

    ## LOAD Model Weight
    # Find all the *.safetensors files
    safetensors_files = glob.glob(os.path.join(model_path, "*.safetensors"))
    tensors = {}
    for safetensors_file in safetensors_files:
        with safe_open(safetensors_file, framework="pt", device="cpu") as f:  # type: ignore
            for key in f.keys():
                tensors[key] = f.get_tensor(key)

    # Load the model's config
    config_path = os.path.join(model_path, "config.json")
    config = load_model_config_from_json(config_path)

    # Create the model using the configuration
    model = PaliGemmaForConditionalGeneration(config).to(device)
    model.load_state_dict(tensors, strict=False)

    return (model, tokenizer)
