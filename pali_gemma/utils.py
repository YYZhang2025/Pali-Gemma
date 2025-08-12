


import torch 
import numpy as np


def numpy_to_torch(numpy_array: np.ndarray, dtype: torch.dtype = torch.float32) -> torch.Tensor:
    return torch.from_numpy(numpy_array).to(dtype)


def get_device():
    if torch.cuda.is_available():
        print("Using GPU")
        return torch.device("cuda")
    elif torch.backends.mps.is_available():
        print("Using Apple Silicon")
        return torch.device("mps")
    else:
        print("Using CPU")
        return torch.device("cpu")

def unsqueeze_tensor(tensor: torch.Tensor, size: int, dim: int) -> torch.Tensor:
    while tensor.dim() < size:
        tensor = tensor.unsqueeze(dim)
        
    return tensor


