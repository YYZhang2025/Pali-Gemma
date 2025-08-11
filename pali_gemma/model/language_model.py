from pali_gemma.config import ModelConfig
from pali_gemma.model.kv_cache import KVCache, repeat_kv
from pali_gemma.utils import unsqueeze_tensor

import torch 
import torch.nn as nn
import torch.nn.functional as F


from typing import Optional


class GemmaRMSNorm(nn.Module):
    def __init__(self, dim: int, eps: float = 1e-6):
        super().__init__()
        
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        div = torch.rsqrt(torch.sum(x ** 2, dim=-1, keepdim=True) / x.shape[-1] + self.eps)
        
        return x * div * self.weight

class GemmaRotaryEmbedding(nn.Module):
    def __init__(self, dim: int, max_position_embedding:int = 2048, base: float = 10_000):
        super().__init__()
        
        self.dim = dim 
        self.max_position_embedding = max_position_embedding
        self.base = base
        
        # Calculate the theta 
        inv_freq = 1.0 / (self.base ** (torch.arange(0, self.dim, 2).float() / self.dim))
        self.register_buffer("inv_freq", inv_freq)


    def forward(self, x, position_ids):
        device = x.device 
        dtype = x.dtype
        

        self.inv_freq = self.inv_freq.to(device)
        inv_freq = self.inv_freq[None, None, :] # (1, 1, D)
        position_ids = unsqueeze_tensor(position_ids, 2, -1) # (B, S, 1)

        device_type = device.type if isinstance(device.type, str) and device.type != "mps" else "cpu"
        with torch.autocast(device_type=device_type, dtype=dtype):
            freqs = position_ids * inv_freq
            emb = torch.cat((freqs, freqs), dim = -1)
            
            cos = emb.cos()
            sin = emb.sin()
        
        return cos.to(dtype=dtype), sin.to(dtype=dtype)


def rotate_half(x):
    x1 = x[...,: x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2:]
    x = torch.cat((-x2, x1), dim=-1)
    return x


def apply_rotary_embedding(q, k, cos, sin, unsqueeze_dim = 1):
    cos = cos.unsqueeze(unsqueeze_dim) # Unsqueeze for head dimension
    sin = sin.unsqueeze(unsqueeze_dim) # Unsqueeze for head dimension

    q = q * cos + rotate_half(q) * sin
    k = k * cos + rotate_half(k) * sin

    return q, k





class GemmaAttention(nn.Module):
    def __init__(self, config: ModelConfig, layer_idx: int):
        super().__init__()
        self.config = config
        self.layer_idx = layer_idx

        
        
        self.q_proj = nn.Linear(config.lm_hidden_size, config.lm_num_heads * config.lm_head_dim, bias=config.lm_attention_bias)
        self.k_proj = nn.Linear(config.lm_hidden_size, config.lm_num_key_value_heads * config.lm_head_dim, bias=config.lm_attention_bias)
        self.v_proj = nn.Linear(config.lm_hidden_size, config.lm_num_key_value_heads * config.lm_head_dim, bias=config.lm_attention_bias)
        self.o_proj = nn.Linear(config.lm_num_heads * config.lm_head_dim, config.lm_hidden_size, bias=config.lm_attention_bias)
        
        self.rotary_emb = GemmaRotaryEmbedding(
            config.lm_head_dim, 
            max_position_embedding=config.lm_max_position_embeddings,
            base=config.lm_rope_theta,
        )

    def forward(self, 
                hidden_states: torch.Tensor, 
                attention_mask: Optional[torch.Tensor] = None,
                position_ids: Optional[torch.Tensor] = None,
                kv_cache: Optional[KVCache] = None,
                **kwargs
                ):

        B, S, D = hidden_states.shape
        d_k = D // self.config.lm_num_heads

        q = self.q_proj(hidden_states).view(B, S, self.config.lm_num_heads, self.config.lm_head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.config.lm_num_key_value_heads, self.config.lm_head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.config.lm_num_key_value_heads, self.config.lm_head_dim).transpose(1, 2)


        cos, sin = self.rotary_emb(q, position_ids)
        q, k = apply_rotary_embedding(q, k, cos, sin)
        
        if kv_cache is not None:
            k, v = kv_cache.update(k, v, self.layer_idx)

        k = repeat_kv(k, self.config.lm_num_heads // self.config.lm_num_key_value_heads)
        v = repeat_kv(v, self.config.lm_num_heads // self.config.lm_num_key_value_heads)

        attn = torch.matmul(q, k.transpose(-2, -1)) * d_k

        assert attention_mask is not None
        attn = attn + attention_mask

        attn = attn.softmax(dim=-1)

        attn = F.dropout(attn, p=self.config.lm_attention_dropout, training=self.training)

        out = torch.matmul(attn, v)
        out = out.transpose(1,2).contiguous().view(B, S, D)
        out = self.o_proj(out)

        return out, attn

class GemmaMLP(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        self.hidden_size = config.hidden_size
        self.intermediate_size = config.intermediate_size
        self.gate_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.up_proj = nn.Linear(self.hidden_size, self.intermediate_size, bias=False)
        self.down_proj = nn.Linear(self.intermediate_size, self.hidden_size, bias=False)

    def forward(self, x):
        # Equivalent to:
        # y = self.gate_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # y = torch.gelu(y, approximate="tanh") # [Batch_Size, Seq_Len, Intermediate_Size]
        # j = self.up_proj(x) # [Batch_Size, Seq_Len, Hidden_Size] -> [Batch_Size, Seq_Len, Intermediate_Size]
        # z = y * j # [Batch_Size, Seq_Len, Intermediate_Size]
        # z = self.down_proj(z) # [Batch_Size, Seq_Len, Intermediate_Size] -> [Batch_Size, Seq_Len, Hidden_Size]
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
        hidden_states, _ = self.self_attn(hidden_states, attention_mask=attention_mask, position_ids=position_ids, kv_cache=kv_cache)
        hidden_states = residual + hidden_states
        
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        return residual + hidden_states
    



class GemmaModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()
        self.config = config
        
        self.padding_idx = config.lm_pad_token_id
        self.vocab_size = config.lm_vocab_size

        self.embed_tokens = nn.Embedding(self.vocab_size, config.lm_hidden_size, self.padding_idx)
        self.layers = nn.ModuleList(
            [GemmaDecoderLayer(config, layer_idx) for layer_idx in range(config.lm_num_hidden_layers)]
        )
        self.norm = GemmaRMSNorm(config.lm_hidden_size, eps=config.lm_rms_norm_eps)

    def forward(
        self,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        kv_cache: Optional[KVCache] = None,
    ):
        # Embed the input tokens
        hidden_states = inputs_embeds
        # [Batch_Size, Seq_Len, Hidden_Size]
        normalizer = torch.tensor(self.config.lm_hidden_size**0.5, dtype=hidden_states.dtype)
        hidden_states = hidden_states * normalizer

        for decoder_layer in self.layers:
            # [Batch_Size, Seq_Len, Hidden_Size]
            hidden_states = decoder_layer(
                hidden_states,
                attention_mask=attention_mask,
                position_ids=position_ids,
                kv_cache=kv_cache,
            )

        # [Batch_Size, Seq_Len, Hidden_Size]
        hidden_states = self.norm(hidden_states)

        # [Batch_Size, Seq_Len, Hidden_Size]
        return hidden_states
