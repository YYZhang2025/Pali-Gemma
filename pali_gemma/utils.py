import os

import numpy as np
import torch

from pali_gemma.fine_tune.lora import LoraConfig


def numpy_to_torch(numpy_array: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.from_numpy(numpy_array).to(dtype)


def get_device(only_cpu: bool = False) -> torch.device:
    if only_cpu:
        return torch.device("cpu")
    if torch.cuda.is_available():
        print("Using GPU")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Apple Silicon(MPS)")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")


def unsqueeze_tensor(tensor: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    while tensor.dim() < size:
        tensor = tensor.unsqueeze(dim)

    return tensor


def load_lora_config_from_file_or_args(adapter_path: str) -> LoraConfig:
    """
    Load a LoRAConfig from a saved .config file next to adapter_path.
    Raises FileNotFoundError if the config file is missing.
    """
    config_path = os.path.join(adapter_path, "config.json")
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"LoRA config file not found: {config_path}")
    saved_cfg = torch.load(config_path, map_location="cpu")
    if not isinstance(saved_cfg, dict):
        raise ValueError(f"Invalid LoRA config format in {config_path}")
    return LoraConfig(
        r=int(saved_cfg["r"]),
        lora_alpha=float(saved_cfg["lora_alpha"]),
        lora_dropout=float(saved_cfg["lora_dropout"]),
        target_modules=tuple(saved_cfg["target_modules"]),
        exclude_modules=tuple(saved_cfg["exclude_modules"]),
    )
