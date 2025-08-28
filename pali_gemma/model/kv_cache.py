from typing import List, Tuple

import torch


class KVCache:
    """
    Key-Value Cache for storing attention keys and values for every layer
    The Key/Value shape is:
    [batch_size, num_heads_kv, seq_length, head_dim]
    """

    def __init__(self):
        self.key_cache: List[torch.Tensor] = []
        self.value_cache: List[torch.Tensor] = []

    def num_items(self) -> int:
        if len(self.key_cache) == 0:
            return 0
        return self.key_cache[0].shape[-2]

    def update(
        self, key_states: torch.Tensor, value_states: torch.Tensor, layer_idx: int
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        if len(self.key_cache) <= layer_idx:
            # In the Prefilling stage,
            self.key_cache.append(key_states)
            self.value_cache.append(value_states)
        else:
            self.key_cache[layer_idx] = torch.cat([self.key_cache[layer_idx], key_states], dim=-2)
            self.value_cache[layer_idx] = torch.cat([self.value_cache[layer_idx], value_states], dim=-2)

        return self.key_cache[layer_idx], self.value_cache[layer_idx]


def repeat_kv(hidden_states: torch.Tensor, n_rep: int) -> torch.Tensor:
    B, H, S, D = hidden_states.shape

    if n_rep == 1:
        return hidden_states

    expanded = hidden_states[:, :, None, :, :].expand(B, H, n_rep, S, D)

    return expanded.reshape(B, H * n_rep, S, D)
