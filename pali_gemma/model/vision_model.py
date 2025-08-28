import torch
import torch.nn as nn
import torch.nn.functional as F

from pali_gemma.config import ModelConfig


class SiglipVisionEmbeddings(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.embed_dim = config.vision_hidden_size
        self.image_size = config.vision_image_size
        self.patch_size = config.vision_patch_size

        self.patch_embedding = nn.Conv2d(
            in_channels=config.vision_num_channels,
            out_channels=config.vision_hidden_size,
            kernel_size=config.vision_patch_size,
            stride=config.vision_patch_size,
            padding="valid",
        )

        assert self.image_size % self.patch_size == 0, "Image size must be divisible by patch size"
        num_patches = (self.image_size // self.patch_size) ** 2
        self.num_positions = num_patches
        self.position_embedding = nn.Embedding(self.num_positions, self.embed_dim)
        self.register_buffer(
            "position_ids",
            torch.arange(self.num_positions, dtype=torch.int64).expand((1, -1)),
            persistent=False,
        )

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        # (B, C, H, W) -> (B, C', H', W')
        patch_embeds = self.patch_embedding(imgs)
        # (B, C', H', W') -> (B, H' * W', C')
        patch_embeds = patch_embeds.flatten(2).transpose(1, 2)

        position_embeds = self.position_embedding(self.position_ids)

        return patch_embeds + position_embeds


class SiglipAttention(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.embed_dim = config.vision_hidden_size
        self.num_heads = config.vision_num_attention_heads

        assert self.embed_dim % self.num_heads == 0, (
            "Embedding dimension must be divisible by number of heads"
        )
        self.head_dim = self.embed_dim // self.num_heads

        self.q_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.k_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.v_proj = nn.Linear(self.embed_dim, self.embed_dim)
        self.out_proj = nn.Linear(self.embed_dim, self.embed_dim)

        self.dropout_prob = config.vision_attention_dropout

    def forward(self, hidden_states: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        B, S, D = hidden_states.shape

        q = self.q_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        k = self.k_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)
        v = self.v_proj(hidden_states).view(B, S, self.num_heads, self.head_dim).transpose(1, 2)

        attn_weights = (q @ k.transpose(-2, -1)) * self.head_dim**-0.5
        attn_weights = attn_weights.softmax(dim=-1)

        if self.dropout_prob > 0:
            attn_weights = nn.functional.dropout(attn_weights, p=self.dropout_prob, training=self.training)

        attn_output = (attn_weights @ v).transpose(1, 2).contiguous().view(B, S, D)

        attn_output = self.out_proj(attn_output)

        return attn_output, attn_weights


class SiglipMLP(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.fc1 = nn.Linear(config.vision_hidden_size, config.vision_intermediate_size)
        self.fc2 = nn.Linear(config.vision_intermediate_size, config.vision_hidden_size)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        hidden_states = self.fc1(hidden_states)
        hidden_states = F.gelu(hidden_states, approximate="tanh")
        hidden_states = self.fc2(hidden_states)

        return hidden_states


class SiglipEncoderLayer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.self_attn = SiglipAttention(config)
        self.layer_norm1 = nn.LayerNorm(config.vision_hidden_size, eps=config.vision_layer_norm_eps)

        self.mlp = SiglipMLP(config)
        self.layer_norm2 = nn.LayerNorm(config.vision_hidden_size, eps=config.vision_layer_norm_eps)

    def forward(self, hidden_states: torch.Tensor) -> torch.Tensor:
        residual = hidden_states
        hidden_states = self.layer_norm1(hidden_states)
        attn_output, _ = self.self_attn(hidden_states)
        hidden_states = residual + attn_output

        residual = hidden_states
        hidden_states = self.layer_norm2(hidden_states)
        mlp_output = self.mlp(hidden_states)
        hidden_states = residual + mlp_output

        return hidden_states


class SiglipEncoder(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.layers = nn.ModuleList(
            [SiglipEncoderLayer(config) for _ in range(config.vision_num_hidden_layers)]
        )

    def forward(self, inputs_embeds: torch.Tensor) -> torch.Tensor:
        hidden_states = inputs_embeds
        for layer in self.layers:
            hidden_states = layer(hidden_states)

        return hidden_states


class SiglipVisionTransformer(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.embeddings = SiglipVisionEmbeddings(config)
        self.encoder = SiglipEncoder(config)

        self.post_layernorm = nn.LayerNorm(config.vision_hidden_size, eps=config.vision_layer_norm_eps)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.embeddings(imgs)
        hidden_states = self.encoder(hidden_states)
        hidden_states = self.post_layernorm(hidden_states)
        return hidden_states


class SiglipVisionModel(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.vision_model = SiglipVisionTransformer(config)

    def forward(self, imgs: torch.Tensor) -> torch.Tensor:
        hidden_states = self.vision_model(imgs)

        return hidden_states
