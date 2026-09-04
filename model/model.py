"""
model/model.py
--------------
SLMModel and SLMForCausalLM — the full model registered with HuggingFace.

SLMModel: the core transformer (embeddings + decoder stack + final norm).
SLMForCausalLM: adds the language model head and loss computation.

Design:
    - No bias anywhere
    - Pre-norm throughout
    - KV cache support for efficient autoregressive generation
    - Compatible with HuggingFace generate(), trl, lm-evaluation-harness, vLLM

Important implementation detail:
    SLMModel is a plain nn.Module.
    SLMForCausalLM is the only PreTrainedModel.

This follows the standard HuggingFace architecture pattern used by Llama,
Mistral, GPT-NeoX, Phi, etc.:

    SLMForCausalLM(PreTrainedModel)
        └── SLMModel(nn.Module)

Only the outer class calls post_init(), so initialization and HF
save/load behavior are controlled from one PreTrainedModel.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from transformers import PreTrainedModel, initialization as init
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.masking_utils import create_causal_mask
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .attention import RotaryEmbedding
from .block import SLMDecoderBlock
from .config import SLMConfig
from .norm import RMSNorm


LegacyCache = list[tuple[torch.Tensor, torch.Tensor]]


class SLMModel(nn.Module):
    """
    The core SLM transformer — embeddings, decoder stack, final norm.

    Does not include the LM head — use SLMForCausalLM for language modelling.

    Important:
        This is intentionally a plain nn.Module, not a PreTrainedModel.
        The outer SLMForCausalLM owns HF initialization, saving, and loading.
    """

    def __init__(self, config: SLMConfig):
        super().__init__()
        self.config = config
        attention_implementation = getattr(config, "_attn_implementation", None)
        if attention_implementation not in (None, "sdpa"):
            raise ValueError(
                "SLM supports attn_implementation='sdpa' only, got "
                f"{attention_implementation!r}"
            )
        if attention_implementation is None:
            config._attn_implementation_internal = "sdpa"

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [SLMDecoderBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.rotary_emb = RotaryEmbedding(config)
        self.gradient_checkpointing = False

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def _convert_legacy_cache(self, past_key_values: LegacyCache) -> DynamicCache:
        if len(past_key_values) != len(self.layers):
            raise ValueError(
                "Legacy past_key_values must contain exactly one entry per "
                f"decoder layer: expected {len(self.layers)}, got "
                f"{len(past_key_values)}"
            )

        cache = DynamicCache(config=self.config)
        for layer_idx, layer_cache in enumerate(past_key_values):
            if not isinstance(layer_cache, (tuple, list)) or len(layer_cache) != 2:
                raise ValueError(
                    "Each legacy cache entry must be a (key, value) pair; "
                    f"invalid entry at layer {layer_idx}"
                )
            cache.update(layer_cache[0], layer_cache[1], layer_idx)
        return cache

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, LegacyCache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> Union[BaseModelOutputWithPast, tuple]:
        if (input_ids is None) == (inputs_embeds is None):
            raise ValueError("Specify exactly one of input_ids or inputs_embeds")

        if use_cache is None:
            use_cache = self.config.use_cache and not self.training
        if output_hidden_states is None:
            output_hidden_states = self.config.output_hidden_states
        if return_dict is None:
            return_dict = self.config.return_dict

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        if isinstance(past_key_values, list):
            past_key_values = self._convert_legacy_cache(past_key_values)
        elif past_key_values is not None and not isinstance(past_key_values, Cache):
            raise TypeError(
                "past_key_values must be a Transformers Cache or a legacy "
                f"list of key/value pairs, got {type(past_key_values).__name__}"
            )

        if use_cache and past_key_values is None:
            past_key_values = DynamicCache(config=self.config)

        past_seen_tokens = (
            past_key_values.get_seq_length()
            if past_key_values is not None
            else 0
        )
        if position_ids is None:
            position_ids = (
                torch.arange(
                    inputs_embeds.shape[1],
                    device=inputs_embeds.device,
                    dtype=torch.long,
                )
                + past_seen_tokens
            ).unsqueeze(0)

        causal_mask = create_causal_mask(
            config=self.config,
            inputs_embeds=inputs_embeds,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            position_ids=position_ids,
        )
        position_embeddings = self.rotary_emb(inputs_embeds, position_ids)

        hidden_states = inputs_embeds
        all_hidden_states: list | None = [] if output_hidden_states else None

        for layer in self.layers:
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if self.gradient_checkpointing and self.training:
                hidden_states = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    causal_mask,
                    position_embeddings,
                    None,
                    False,
                )
            else:
                hidden_states = layer(
                    hidden_states,
                    attention_mask=causal_mask,
                    position_embeddings=position_embeddings,
                    past_key_values=past_key_values,
                    use_cache=use_cache,
                )

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if not return_dict:
            return tuple(
                value
                for value in [hidden_states, past_key_values, all_hidden_states]
                if value is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=past_key_values,
            hidden_states=all_hidden_states,
        )


class SLMForCausalLM(PreTrainedModel, GenerationMixin):
    """
    SLM with a language modelling head for causal language modelling.

    This is the only PreTrainedModel in the architecture. It owns:
    - initialization via post_init()
    - standard save_pretrained() checkpoint serialization
    - tied embedding behavior
    - HF generation compatibility

    Checkpoint loading uses the native PreTrainedModel.from_pretrained() path.
    Weight initialization must therefore use transformers.initialization helpers:
    they preserve tensors materialized from a checkpoint while still initializing
    newly constructed models from scratch.
    """

    config_class = SLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["SLMDecoderBlock"]
    _skip_keys_device_placement = ["past_key_values"]

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.model = SLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights with config.initializer_range.

        This runs on every submodule when post_init() recurses, including
        modules inside SLMModel. SLMModel intentionally does not define its
        own _init_weights; this is the single source of init policy.

        Use Transformers initialization helpers rather than direct ``.data``
        writes so native loading can preserve already materialized checkpoint
        tensors while initializing only the modules that still require it.
        """
        std = self.config.initializer_range

        if isinstance(module, nn.Linear):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.bias is not None:
                init.zeros_(module.bias)

        elif isinstance(module, nn.Embedding):
            init.normal_(module.weight, mean=0.0, std=std)
            if module.padding_idx is not None:
                init.zeros_(module.weight[module.padding_idx])

    def get_input_embeddings(self) -> nn.Embedding:
        return self.model.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.model.embed_tokens = value

    def get_output_embeddings(self) -> nn.Linear:
        return self.lm_head

    def set_output_embeddings(self, new_embeddings: nn.Linear) -> None:
        self.lm_head = new_embeddings

    def get_decoder(self) -> SLMModel:
        return self.model

    def set_decoder(self, decoder: SLMModel) -> None:
        self.model = decoder

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        position_ids: Optional[torch.LongTensor] = None,
        past_key_values: Optional[Union[Cache, LegacyCache]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        logits_to_keep: Union[int, torch.Tensor] = 0,
        **kwargs,
    ) -> Union[CausalLMOutputWithPast, tuple]:
        if labels is not None and use_cache is None:
            use_cache = False
        if return_dict is None:
            return_dict = self.config.return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
        )

        hidden_states = outputs.last_hidden_state
        slice_indices = (
            slice(-logits_to_keep, None)
            if isinstance(logits_to_keep, int)
            else logits_to_keep
        )
        logits = self.lm_head(hidden_states[:, slice_indices, :])

        loss = None
        if labels is not None:
            loss = self.loss_function(
                logits=logits,
                labels=labels,
                vocab_size=self.config.vocab_size,
                **kwargs,
            )

        if not return_dict:
            output = tuple(
                value
                for value in [
                    logits,
                    outputs.past_key_values,
                    outputs.hidden_states,
                ]
                if value is not None
            )
            return ((loss,) + output) if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    def _reorder_cache(
        self,
        past_key_values: Union[Cache, list[tuple[torch.Tensor, torch.Tensor]]],
        beam_idx: torch.Tensor,
    ) -> Union[Cache, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Reorder KV cache for beam search.
        """
        if isinstance(past_key_values, Cache):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values

        return [
            (k.index_select(0, beam_idx), v.index_select(0, beam_idx))
            for k, v in past_key_values
        ]
