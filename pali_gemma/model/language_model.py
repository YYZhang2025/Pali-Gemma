from typing import Dict, Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F

from pali_gemma.config import ModelConfig
from pali_gemma.model.kv_cache import KVCache, repeat_kv


# TODO: Confirm the behavior of (1.0 + self.weight) and self.weight
class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()

        self.eps = eps
        self.weight = nn.Parameter(torch.zeros(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        div = torch.rsqrt(torch.sum(x**2, dim=-1, keepdim=True) / x.shape[-1] + self.eps)

        return x * div * (1.0 + self.weight)


class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embedding: int = 2048, base: float = 10_000):
        super().__init__()

        self.dim = dim
        self.max_position_embedding = max_position_embedding
        self.base = base

        # Calculate the theta
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def forward(self, x, position_ids):
        device = x.device
        device_type = x.device.type
        dtype = x.dtype

        self.inv_freq = self.inv_freq.to(device)
        inv_freq_expanded = self.inv_freq[None, :, None].float().expand(position_ids.shape[0], -1, 1)
        position_ids_expanded = position_ids[:, None, :].float()

        device_type = device_type if isinstance(device_type, str) and device_type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, enabled=False):
            freqs = (inv_freq_expanded.float() @ position_ids_expanded.float()).transpose(1, 2)
            # emb: [Batch_Size, Seq_Len, Head_Dim]
            emb = torch.cat((freqs, freqs), dim=-1)
            # cos, sin: [Batch_Size, Seq_Len, Head_Dim]
            cos = emb.cos()
            sin = emb.sin()
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rotate_half(x):
    # Build the [-x2, x1, -x4, x3, ...] tensor for the sin part of the positional encoding.
    x1 = x[..., : x.shape[-1] // 2]  # Takes the first half of the last dimension
    x2 = x[..., x.shape[-1] // 2 :]  # Takes the second half of the last dimension
    return torch.cat((-x2, x1), dim=-1)


def apply_rotary_embedding(q, k, cos, sin, unsqueeze_dim=1):
    cos = cos.unsqueeze(unsqueeze_dim)  # Unsqueeze for head dimension
    sin = sin.unsqueeze(unsqueeze_dim)  # Unsqueeze for head dimension

    q_embed = q * cos + rotate_half(q) * sin
    k_embed = k * cos + rotate_half(k) * sin

    return q_embed, k_embed


class GemmaAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.q_proj = nn.Linear(
            config.lm_hidden_size,
            config.lm_num_heads * config.lm_head_dim,
            bias=config.lm_attention_bias,
        )
        self.k_proj = nn.Linear(
            config.lm_hidden_size,
            config.lm_num_key_value_heads * config.lm_head_dim,
            bias=config.lm_attention_bias,
        )
        self.v_proj = nn.Linear(
            config.lm_hidden_size,
            config.lm_num_key_value_heads * config.lm_head_dim,
            bias=config.lm_attention_bias,
        )
        self.o_proj = nn.Linear(
            config.lm_num_heads * config.lm_head_dim,
            config.lm_hidden_size,
            bias=config.lm_attention_bias,
        )

        self.rotary_emb = GemmaRotaryEmbedding(
            config.lm_head_dim,
            max_position_embedding=config.lm_max_position_embeddings,
            base=config.lm_rope_theta,
        )

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
        **kwargs,
    ):
        B, S, D = hidden_states.shape
        d_k = D // self.config.lm_num_heads

        q = (
            self.q_proj(hidden_states)
            .view(B, S, self.config.lm_num_heads, self.config.lm_head_dim)
            .transpose(1, 2)
        )

        k = (
            self.k_proj(hidden_states)
            .view(B, S, self.config.lm_num_key_value_heads, self.config.lm_head_dim)
            .transpose(1, 2)
        )

        v = (
            self.v_proj(hidden_states)
            .view(B, S, self.config.lm_num_key_value_heads, self.config.lm_head_dim)
            .transpose(1, 2)
        )

        cos, sin = self.rotary_emb(v, position_ids)  # HIGHLIGHT: Get cos and sin using value state.
        q, k = apply_rotary_embedding(q, k, cos, sin)

        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        k = repeat_kv(k, self.config.lm_num_heads // self.config.lm_num_key_value_heads)
        v = repeat_kv(v, self.config.lm_num_heads // self.config.lm_num_key_value_heads)

        attn = torch.matmul(q, k.transpose(-2, -1)) / (d_k**0.5)

        assert attention_mask is not None
        attn = attn + attention_mask

        attn = attn.softmax(dim=-1)
        attn = F.dropout(attn, p=self.config.lm_attention_dropout, training=self.training)

        out = torch.matmul(attn, v)
        out = out.transpose(1, 2).contiguous().view(B, S, D)
        out = self.o_proj(out)

        return out, attn


class GemmaMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.hidden_size = config.lm_hidden_size
        self.intermediate_size = config.lm_intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        return self.down_proj(nn.functional.gelu(self.gate_proj(x), approximate="tanh") * self.up_proj(x))


class GemmaDecoderLayer(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        self.self_attn = GemmaAttention(config, layer_idx)
        self.input_layernorm = GemmaRMSNorm(config.lm_hidden_size, eps=config.lm_rms_norm_eps)

        self.mlp = GemmaMLP(config)
        self.post_attention_layernorm = GemmaRMSNorm(config.lm_hidden_size, eps=config.lm_rms_norm_eps)

    def forward(
        self,
        hidden_states: torch.Tensor,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.Tensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states, _ = self.self_attn(
            hidden_states,
            attention_mask=attention_mask,
            position_ids=position_ids,
            kv_cache=kv_cache,
        )
        hidden_states = residual + hidden_states

        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states


class GemmaModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        self.padding_idx = config.pad_token_id
        self.vocab_size = config.lm_vocab_size

        self.embed_tokens = nn.Embedding(self.vocab_size, config.lm_hidden_size, self.padding_idx)

        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.lm_num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.lm_hidden_size, eps=config.lm_rms_norm_eps)

    def get_input_embeddings(self):
        return self.embed_tokens

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        kv_cache: KVCache,
    ):
        # Normalized
        hidden_states = inputs_embeds * (self.config.lm_hidden_size**0.5)

        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        return self.norm(hidden_states)


class GemmaForCausalLM(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.model = GemmaModel(config)
        self.vocab_size = config.lm_vocab_size
        self.lm_head = nn.Linear(config.lm_hidden_size, config.lm_vocab_size, bias=False)

    def get_input_embeddings(self):
        return self.model.embed_tokens

    def forward(
        self,
        attention_mask: torch.Tensor,
        position_ids: torch.Tensor,
        inputs_embeds: torch.Tensor,
        kv_cache: KVCache,
    ) -> Dict[str, Union[torch.Tensor, KVCache]]:
        # input_embeds: [Batch_Size, Seq_Len, Hidden_Size]
        # outputs: [Batch_Size, Seq_Len, Hidden_Size]
        outputs = self.model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        hidden_states = outputs
        logits = F.linear(hidden_states, self.model.embed_tokens.weight)
        logits = logits.float()

        return_data = {
            "logits": logits,
        }

        if kv_cache is not None:
            # Return the updated cache
            return_data["kv_cache"] = kv_cache

        return return_data
