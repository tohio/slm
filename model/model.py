"""
model/model.py
--------------
SLMModel and SLMForCausalLM — the full model registered with HuggingFace.

SLMModel: the core transformer (embeddings + decoder stack + final norm).
SLMForCausalLM: adds the language model head and loss computation.

Registering with AutoModel/AutoModelForCausalLM means the model can be
loaded, saved, and used with the full HuggingFace ecosystem:

    from transformers import AutoModelForCausalLM
    model = AutoModelForCausalLM.from_pretrained("tohio/slm-125m")

Tied embeddings: the LM head weight is shared with the input embedding
weight. This reduces parameters and has been shown to improve performance
at small model scales.

Design:
    - No bias anywhere
    - Pre-norm throughout
    - KV cache support for efficient autoregressive generation
    - Compatible with HuggingFace generate(), trl, lm-evaluation-harness, vLLM

Compatibility (transformers v5):
    - SLMForCausalLM inherits from GenerationMixin directly — PreTrainedModel
      no longer inherits from GenerationMixin from v4.50 onwards.
    - Weight tying: tie_weights() does the actual sharing; _tied_weights_keys
      is set as a dict so save_pretrained correctly handles shared tensors.
    - past_key_values supports both legacy list[tuple] format and v5
      DynamicCache objects for full compatibility with trl and vLLM.
    - transformers v5 DynamicCache uses a .layers attribute containing
      DynamicLayer objects with .keys and .values tensors. Older versions
      used .key_cache / .value_cache list attributes. Both are handled.
    - prepare_inputs_for_generation consumes cache_position (v5) to slice
      input_ids correctly for single-token and multi-token resume.
    - use_cache is forced False when gradient checkpointing is enabled.
"""

from typing import Optional, Union

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.cache_utils import Cache, DynamicCache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .block import SLMDecoderBlock
from .config import SLMConfig
from .norm import RMSNorm


def _extract_kv_from_dynamic_cache(
    cache: Cache,
    n_layers: int,
) -> list[Optional[tuple[torch.Tensor, torch.Tensor]]]:
    """
    Extract per-layer (k, v) tuples from a DynamicCache object.

    Handles two DynamicCache formats:
    - transformers v5: cache.layers is a list of DynamicLayer objects
      with .keys and .values tensor attributes.
    - older versions: cache.key_cache / cache.value_cache are lists of tensors.

    Returns a list of length n_layers where each entry is either a
    (k, v) tuple or None if no cached state exists for that layer.
    """
    result = []

    # transformers v5 — DynamicLayer with .keys / .values
    cache_layers = getattr(cache, "layers", None)
    if cache_layers is not None:
        for i in range(n_layers):
            if i < len(cache_layers) and cache_layers[i].is_initialized:
                result.append((cache_layers[i].keys, cache_layers[i].values))
            else:
                result.append(None)
        return result

    # older DynamicCache — .key_cache / .value_cache lists
    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None:
        for i in range(n_layers):
            if i < len(key_cache):
                result.append((key_cache[i], value_cache[i]))
            else:
                result.append(None)
        return result

    # Unknown format — return empty cache
    return [None] * n_layers


class SLMModel(PreTrainedModel):
    """
    The core SLM transformer — embeddings, decoder stack, final norm.

    Does not include the LM head — use SLMForCausalLM for language modelling.

    Args:
        config (SLMConfig): Model configuration.
    """

    config_class = SLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.config = config

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [SLMDecoderBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

        # Apply configured weight init. SLMModel has no LM head and no tied
        # weights, so the dict-form _tied_weights_keys pathway in v5 is not
        # exercised here — post_init() is safe to call.
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with config.initializer_range."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def get_input_embeddings(self) -> nn.Embedding:
        return self.embed_tokens

    def set_input_embeddings(self, value: nn.Embedding) -> None:
        self.embed_tokens = value

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        if isinstance(module, SLMModel):
            module.gradient_checkpointing = value

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, list[tuple[torch.Tensor, torch.Tensor]]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else getattr(
            self.config, 'return_dict', getattr(self.config, 'use_return_dict', True)
        )

        # Gradient checkpointing and use_cache are mutually exclusive
        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        # Normalise past_key_values — accept both legacy list[tuple] and
        # v5 DynamicCache. Convert to per-layer (k, v) tuples or None.
        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        elif isinstance(past_key_values, Cache):
            past_key_values = _extract_kv_from_dynamic_cache(
                past_key_values, len(self.layers)
            )

        next_cache: list | None = [] if use_cache else None
        all_hidden_states: list | None = [] if output_hidden_states else None

        for layer, past_kv in zip(self.layers, past_key_values):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if self.gradient_checkpointing and self.training:
                layer_out = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    None,   # past_key_value
                    False,  # use_cache
                )
                # Layer returns (hidden_states, kv) — unpack safely
                hidden_states = layer_out[0]
                layer_kv = layer_out[1] if len(layer_out) > 1 else None
            else:
                hidden_states, layer_kv = layer(
                    hidden_states,
                    attention_mask=attention_mask,
                    past_key_value=past_kv,
                    use_cache=use_cache,
                )

            if use_cache:
                next_cache.append(layer_kv)

        hidden_states = self.norm(hidden_states)

        if output_hidden_states:
            all_hidden_states.append(hidden_states)

        if not return_dict:
            return tuple(
                v for v in [hidden_states, next_cache, all_hidden_states] if v is not None
            )

        return BaseModelOutputWithPast(
            last_hidden_state=hidden_states,
            past_key_values=next_cache,
            hidden_states=all_hidden_states,
        )


class SLMForCausalLM(PreTrainedModel, GenerationMixin):
    """
    SLM with a language modelling head for causal (autoregressive) generation.

    Adds a linear LM head on top of SLMModel. When tie_word_embeddings=True
    (default), the LM head weight is tied to the input embedding weight.

    Inherits from GenerationMixin directly — required from transformers v4.50+
    where PreTrainedModel no longer inherits from GenerationMixin.

    Compatible with:
        - HuggingFace generate() — autoregressive text generation
        - trl SFTTrainer / DPOTrainer — supervised fine-tuning and alignment
        - lm-evaluation-harness — benchmark evaluation
        - vLLM — production serving

    Args:
        config (SLMConfig): Model configuration.

    Example::

        from model.config import SLM_125M
        from model.model import SLMForCausalLM

        model = SLMForCausalLM(SLM_125M)
        inputs = tokenizer("Hello world", return_tensors="pt")
        output = model.generate(**inputs, max_new_tokens=50)
    """

    config_class = SLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    # _tied_weights_keys must be a dict in transformers v5:
    #   {target_weight_name: source_weight_name}
    # This tells save_pretrained which weights are shared so it can
    # correctly handle shared tensors during serialisation.
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.model = SLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def _init_weights(self, module: nn.Module) -> None:
        """Initialize weights with config.initializer_range."""
        std = self.config.initializer_range
        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.bias is not None:
                module.bias.data.zero_()
        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=std)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()

    def tie_weights(self, **kwargs) -> None:
        """
        Tie LM head weights to input embeddings when tie_word_embeddings=True.

        Called in two places:
        - post_init() during __init__, after weight initialization
        - from_pretrained(), after loading weights from disk, to restore the
          tie that save_pretrained() broke when serialising independent copies

        Accepts **kwargs for forward compatibility (v5 passes recompute_mapping=False).
        """
        if self.config.tie_word_embeddings:
            self.lm_head.weight = self.model.embed_tokens.weight

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

    def save_pretrained(self, save_directory, **kwargs) -> None:
        """
        Override save_pretrained to handle tied weights correctly.

        When tie_word_embeddings=True, lm_head.weight and model.embed_tokens.weight
        share the same storage. safetensors does not allow tensor aliasing so
        transformers removes lm_head.weight from the state dict before saving.
        When loading, from_pretrained then can't find lm_head.weight and reports
        it as MISSING — it will be re-tied after load, but only if the model
        knows about the tying.

        We temporarily untie the weights before saving by making lm_head.weight
        an independent copy, then restore the tie after. try/finally guarantees
        the tie is always restored even if save raises an exception.
        """
        if self.config.tie_word_embeddings:
            self.lm_head.weight = nn.Parameter(self.model.embed_tokens.weight.data.clone())
        try:
            super().save_pretrained(save_directory, **kwargs)
        finally:
            if self.config.tie_word_embeddings:
                self.lm_head.weight = self.model.embed_tokens.weight

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[Union[Cache, list[tuple[torch.Tensor, torch.Tensor]]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
        cache_position: Optional[torch.LongTensor] = None,
    ) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else getattr(
            self.config, 'return_dict', getattr(self.config, 'use_return_dict', True)
        )

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
            cache_position=cache_position,
        )

        hidden_states = outputs[0]
        logits = self.lm_head(hidden_states)
        logits = logits.float()  # upcast to float32 for loss stability

        loss = None
        if labels is not None:
            shift_logits = logits[..., :-1, :].contiguous()
            shift_labels = labels[..., 1:].contiguous()
            loss = CrossEntropyLoss()(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
            )

        if not return_dict:
            output = (logits,) + outputs[1:]
            return (loss,) + output if loss is not None else output

        return CausalLMOutputWithPast(
            loss=loss,
            logits=logits,
            past_key_values=outputs.past_key_values,
            hidden_states=outputs.hidden_states,
        )

    def prepare_inputs_for_generation(
        self,
        input_ids: torch.LongTensor,
        past_key_values: Optional[Union[Cache, list]] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        cache_position: Optional[torch.LongTensor] = None,
        **kwargs,
    ) -> dict:
        """
        Called by HuggingFace generate() at each decoding step.

        Slices input_ids to only the positions that haven't been processed yet:
        - If cache_position is provided (transformers v5), trim to len(cache_position)
          tokens from the end. This correctly handles both single-token
          generation (cache_position has length 1) and multi-token resume
          used by assisted/speculative decoding (cache_position has length > 1).
        - If cache_position is absent (older callers) and past_key_values is
          set, fall back to trimming to the last token, which matches the
          common autoregressive case.
        """
        if cache_position is not None:
            # v5 path: slice to exactly the new positions implied by cache_position.
            input_ids = input_ids[:, -cache_position.shape[0]:]
        elif past_key_values is not None:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache", True),
            "attention_mask": attention_mask,
            "cache_position": cache_position,
        })
        return model_inputs

    def _reorder_cache(
        self,
        past_key_values: Union[Cache, list[tuple[torch.Tensor, torch.Tensor]]],
        beam_idx: torch.Tensor,
    ) -> Union[Cache, list[tuple[torch.Tensor, torch.Tensor]]]:
        """
        Reorder KV cache for beam search.

        Handles both v5 DynamicCache objects and legacy list[tuple] format.
        Instance method (not static) as required by v5.
        """
        if isinstance(past_key_values, Cache):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values
        return [
            (k.index_select(0, beam_idx), v.index_select(0, beam_idx))
            for k, v in past_key_values
        ]