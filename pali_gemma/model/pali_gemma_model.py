from typing import Optional, Tuple

import torch
import torch.nn as nn

from pali_gemma.config import ModelConfig
from pali_gemma.model.kv_cache import KVCache
from pali_gemma.model.language_model import GemmaForCausalLM
from pali_gemma.model.vision_model import SiglipVisionModel


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
        self.pad_token_id = self.config.pad_token_id

        self.language_model = GemmaForCausalLM(config)

    def _merge_input_ids_with_image_features(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ):
        """Merge image features with text inputs.

        Args:
            image_features (torch.Tensor): (B, num_image_tokens, D) The image features to merge.
            inputs_embeds (torch.Tensor): (B, S, D) The text inputs embeddings.
            input_ids (torch.Tensor): (B, S) The input IDs.
            attention_mask (torch.Tensor): (B, S) The attention mask.
            kv_cache (Optional[KVCache], optional): The key-value cache. Defaults to None.

        Returns:
            _type_: _description_
        """

        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # (B, D)
        scaled_image_features = image_features / (self.config.projection_dim**0.5)
        scaled_image_features = scaled_image_features.to(dtype)

        # Combine the embeddings of the
        # - image tokens
        # - text tokens
        # - mask out all the padding tokens
        final_embeddings = torch.zeros(batch_size, sequence_length, embed_dim, dtype=dtype, device=device)

        # Create 3 mask:
        # 1. Text mask: True for text tokens
        # 2. Image mask: True for image tokens
        # 3. Padding mask: True for padding tokens
        # Both has shape: (batch_size, sequence_length)
        # (B, S) => (B, S, D)

        text_mask = (input_ids != self.pad_token_id) & (input_ids != self.config.image_token_index)
        text_mask = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        image_mask = input_ids == self.config.image_token_index
        image_mask = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        padding_mask = input_ids == self.pad_token_id
        padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Apply masks to final embeddings
        final_embeddings = torch.where(text_mask, inputs_embeds, final_embeddings)
        final_embeddings = final_embeddings.masked_scatter(image_mask, scaled_image_features)
        # final_embeddings = torch.masked_scatter(final_embeddings, image_mask, scaled_image_features)
        final_embeddings = torch.where(padding_mask, torch.zeros_like(final_embeddings), final_embeddings)

        # Create mask for the attention
        q_len = inputs_embeds.shape[1]

        # q_len := inputs_embeds.shape[1], K := q_len if no cache else kv_cache.num_items() + q_len
        if kv_cache is None or kv_cache.num_items() == 0:
            # PREFILL Stage: can see all tokens up to current token
            # causal_lower: [Q, Q] True on and below diagonal
            causal_lower = torch.tril(torch.ones(q_len, q_len, device=device, dtype=torch.bool))

            # key padding mask: [B, K] True for real tokens (not pad)
            key_keep = attention_mask.to(torch.bool)  # [B, Q] here (K == Q)

            # allow[i, q, k] = causal_lower[q, k] & key_keep[i, k]
            allow = causal_lower.unsqueeze(0) & key_keep.unsqueeze(1)  # [B, Q, Q]
            causal_mask = allow  # [B, Q, Q]
        else:
            # Decode: single query can attend to all cached keys + itself (no future exists)
            assert q_len == 1
            kv_len = kv_cache.num_items() + q_len

            # If your cache never stores pads, you can allow all keys:
            # causal_mask = torch.zeros((batch_size, q_len, kv_len), dtype=dtype, device=device)
            causal_mask = torch.ones(
                (batch_size, q_len, kv_len),  # <-- bool mask, all allowed
                dtype=torch.bool,
                device=device,
            )

        # Add head dim: [B, 1, Q, K]
        causal_mask = causal_mask.unsqueeze(1)

        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)

        return final_embeddings, causal_mask, position_ids

    def _concat_image_text_embeddings(
        self,
        image_features: torch.Tensor,
        inputs_embeds: torch.Tensor,
        input_ids: torch.Tensor,
    ) -> torch.Tensor:
        _, _, embed_dim = image_features.shape
        batch_size, sequence_length = input_ids.shape
        dtype, device = inputs_embeds.dtype, inputs_embeds.device

        # (B, D)
        scaled_image_features = image_features / (self.config.projection_dim**0.5)
        scaled_image_features = scaled_image_features.to(dtype)

        # Combine the embeddings of the
        # - image tokens
        # - text tokens
        # - mask out all the padding tokens
        final_embeddings = torch.zeros(batch_size, sequence_length, embed_dim, dtype=dtype, device=device)

        # Create 3 mask:
        # 1. Text mask: True for text tokens
        # 2. Image mask: True for image tokens
        # 3. Padding mask: True for padding tokens
        # Both has shape: (batch_size, sequence_length)
        # (B, S) => (B, S, D)

        text_mask = (input_ids != self.pad_token_id) & (input_ids != self.config.image_token_index)
        text_mask = text_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        image_mask = input_ids == self.config.image_token_index
        image_mask = image_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        padding_mask = input_ids == self.pad_token_id
        padding_mask = padding_mask.unsqueeze(-1).expand(-1, -1, embed_dim)

        # Apply masks to final embeddings
        final_embeddings = torch.where(text_mask, inputs_embeds, final_embeddings)
        final_embeddings = final_embeddings.masked_scatter(image_mask, scaled_image_features)
        # final_embeddings = torch.masked_scatter(final_embeddings, image_mask, scaled_image_features)
        final_embeddings = torch.where(padding_mask, torch.zeros_like(final_embeddings), final_embeddings)
        return final_embeddings

    def _get_mask_and_position_ids(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        q_len = input_ids.shape[1]
        device = input_ids.device

        sep_token_id = self.config.sep_token_id

        # Two cases:
        # 1. No kv_cache or kv_cache is empty: PREFILL stage
        # 2. kv_cache is not empty: DECODE stage

        # PREFILL STAGE
        if kv_cache is None or kv_cache.num_items() == 0:
            # The sep_token_is indicate the position of Prefix/Suffix separator tokens
            # 1. Find the position of the first sep token in each sequence
            is_seq = input_ids == sep_token_id  # (B, S)
            first_idx = torch.argmax(is_seq.to(torch.int64), dim=-1)  # (B,)
            prefix_len = first_idx + 1  # (B,)

            mask = torch.zeros((input_ids.shape[0], 1, q_len, q_len), dtype=torch.bool, device=device)
            idx = torch.arange(q_len, device=device)
            causal = idx[None, :] <= idx[:, None]  # (S, S)

            for b in range(input_ids.shape[0]):
                p = int(prefix_len[b].item())
                mask[b, 0, :p, :p] = True
                mask[b, 0, p:, :p] = True
                mask[b, 0, p:, p:] = causal[p:, p:]

        else:  # DECODE STAGE
            # In decode stage, allow all tokens to attend to all cached tokens + itself
            batch_size = input_ids.shape[0]
            q_len = input_ids.shape[1]
            kv_len = kv_cache.num_items() + q_len
            mask = torch.ones(
                (batch_size, 1, q_len, kv_len),  # <-- bool mask, all allowed
                dtype=torch.bool,
                device=device,
            )

        # Create position_ids
        if kv_cache is not None and kv_cache.num_items() > 0:
            # The position of the query is just the last position
            position_ids = attention_mask.cumsum(-1)[:, -1]
            if position_ids.dim() == 1:
                position_ids = position_ids.unsqueeze(0)
        else:
            # Create a position_ids based on the size of the attention_mask
            # For masked tokens, use the number 1 as position.
            position_ids = (attention_mask.cumsum(-1)).masked_fill_((attention_mask == 0), 1).to(device)
        return mask, position_ids

    def forward(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: torch.Tensor,
        kv_cache: Optional[KVCache] = None,
    ) -> Tuple:
        # assert torch.all(attention_mask == 1), "The input cannot be padded"

        # Get Text embedding
        inputs_embeds = self.language_model.get_input_embeddings()(input_ids)

        # Get Image Embedding
        selected_image_feature = self.vision_tower(pixel_values.to(inputs_embeds.dtype))
        image_features = self.multi_modal_projector(selected_image_feature)

        # # Concat together
        # inputs_embeds, attention_mask, position_ids = self._merge_input_ids_with_image_features(
        #     image_features, inputs_embeds, input_ids, attention_mask, kv_cache
        # )

        # Concat image embeddings and text embeddings
        inputs_embeds = self._concat_image_text_embeddings(image_features, inputs_embeds, input_ids)

        # Create Mask and Position IDs
        mask, position_ids = self._get_mask_and_position_ids(input_ids, attention_mask, kv_cache)

        # Feed to Language Model
        outputs = self.language_model(
            attention_mask=mask,
            position_ids=position_ids,
            inputs_embeds=inputs_embeds,
            kv_cache=kv_cache,
        )

        return outputs
