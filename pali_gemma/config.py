from dataclasses import dataclass 
from typing import Optional 


@dataclass
class ModelConfig:
    # Vision Configuration
    vision_hidden_size: int = 768
    vision_intermediate_size: int = 3072
    vision_num_hidden_layers: int = 12
    vision_num_attention_heads: int = 12
    vision_num_channels: int = 3
    vision_image_size: int = 224
    vision_patch_size: int = 16
    vision_layer_norm_eps: float = 1e-6
    vision_attention_dropout: float = 0.0
    vision_num_image_tokens: Optional[int] = None

    # LM Configuration
    lm_vocab_size: int = 32000
    lm_hidden_size: int = 1024
    lm_intermediate_size: int = 4096
    lm_num_hidden_layers: int = 24
    lm_num_attention_heads: int = 16
    lm_num_key_value_heads: int = 16
    lm_num_heads: int = 16
    lm_head_dim: int = 256
    lm_max_position_embeddings: int = 8192
    lm_rms_norm_eps: float = 1e-6
    lm_rope_theta: float = 10000.0
    lm_attention_bias: bool = False
    lm_attention_dropout: float = 0.0
    lm_pad_token_id: Optional[int] = None

    
    
    def __post_init__(self):
        if self.vision_num_image_tokens is None:
            self.vision_num_image_tokens = (self.vision_image_size // self.vision_patch_size) ** 2

        # Ensure that the number of image tokens is a perfect square
        assert (self.vision_num_image_tokens ** 0.5).is_integer(), "Number of image tokens must be a perfect square"