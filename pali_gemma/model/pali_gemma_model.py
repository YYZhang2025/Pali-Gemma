from pali_gemma.config import ModelConfig
from pali_gemma.model.vision_model import SiglipVisionModel
from pali_gemma.model.language_model import GemmaForCausalLM
from pali_gemma.model.kv_cache import KVCache


import torch
import torch.nn as nn

from typing import Optional, Tuple


class PaliGemmaMultiModalProjector(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.linear = nn.Linear(config.vision_hidden_size, config.projection_dim, bias=True)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear(x)


class PaliGemmaForConditionalGeneration(nn.Module):
    def __init__(self, config: ModelConfig):
        super().__init__()

        self.config = config
        self.vision_tower = SiglipVisionModel(config)
        self.multi_modal_projector = PaliGemmaMultiModalProjector(config)

        self.vocab_size = config.lm_vocab_size
        self.pad_token_id = self.config.lm_pad_token_id

        self.language_model = GemmaForCausalLM(config)

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        scaled_image_features = image_features / (self.config.projection_dim**0.5)

        # Combine the embeddings of the
        # - image tokens
        # - text tokens
        # and mask out all the padding tokens
        final_embeddings = torch.zeros(
            batch_size, sequence_length, embed_dim, dtype=dtype, device=device
        )

        # Create 3 mask:
        # 1. Text mask: True for text tokens
        # 2. Image mask: True for image tokens
        # 3. Padding mask: True for padding tokens
        # Both has shape: (batch_size, sequence_length)
        text_mask = (input_ids != self.pad_token_id) & (input_ids != self.config.image_token_index)
        image_mask = input_ids == self.config.image_token_index
        padding_mask = input_ids == self.pad_token_id

        text_mask = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        image_mask = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)
        padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Apply masks to final embeddings
        final_embeddings = torch.where(text_mask, inputs_embeds, final_embeddings)
        final_embeddings = final_embeddings.masked_scatter(image_mask, scaled_image_features)
        final_embeddings = torch.where(
            padding_mask, torch.zeros_like(final_embeddings), final_embeddings
        )

        # Create mask for the attention
        q_len = inputs_embeds.shape[1]

        if kv_cache is None or kv_cache.num_items() == 0:
            # Do not mask any token, because we're in the prefill phase
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, q_len), fill_value=0, dtype=dtype, device=device
            )
        else:
            # Since we are generating tokens, the query must be one single token
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len
            # Also in this case we don't need to mask anything, since each query should be able to attend all previous tokens.
            # This only works when we have no padding
            causal_mask = torch.full(
                (batch_size, q_len, kv_len), fill_value=0, dtype=dtype, device=device
            )

        # Add the head dimension
        # [Batch_Size, Q_Len, KV_Len] -> [Batch_Size, Num_Heads_Q, Q_Len, KV_Len]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (
                (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)
            )

        return final_embeddings, causal_mask, position_ids

    def tie_weights(self):
        return self.language_model.tie_weights()

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        assert torch.all(attention_mask == 1), "The input cannot be padded"

        # Get Text embedding
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Get Image Embedding
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        image_features = self.multi_modal_projector(selected_image_feature)

        # Concat together
        inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
            image_features, inputs_embeds, input_ids, attention_mask, kv_cache
        )

        # Feed to Language Model
        outputs = self.language_model(
            attention_mask=attention_mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
