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

Compatibility:
    - transformers >= 4.50: SLMForCausalLM inherits from GenerationMixin
      directly to preserve generate() and related methods, as PreTrainedModel
      no longer inherits from GenerationMixin from v4.50 onwards.
    - post_init() is called only in SLMForCausalLM (which has the LM head
      and tied weights), not in SLMModel, to avoid tied weights resolution
      errors in the base model.
"""

from typing import Optional

import torch
import torch.nn as nn
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .block import SLMDecoderBlock
from .config import SLMConfig
from .norm import RMSNorm


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

        # Note: post_init() is intentionally NOT called here. SLMModel has no
        # LM head and no tied weights — calling post_init() triggers tied weight
        # resolution in transformers >= 4.50 which errors on the base model.
        # post_init() is called in SLMForCausalLM where tied weights are defined.

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
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> BaseModelOutputWithPast:
        use_cache = use_cache if use_cache is not None else self.config.use_cache
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)
        hidden_states = inputs_embeds

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)

        next_cache = [] if use_cache else None
        all_hidden_states = [] if output_hidden_states else None

        for layer, past_kv in zip(self.layers, past_key_values):
            if output_hidden_states:
                all_hidden_states.append(hidden_states)

            if self.gradient_checkpointing and self.training:
                hidden_states, layer_kv = self._gradient_checkpointing_func(
                    layer.__call__,
                    hidden_states,
                    attention_mask,
                    None,
                    False,
                )
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

    Inherits from GenerationMixin directly to preserve generate() and related
    methods — required from transformers >= 4.50 where PreTrainedModel no
    longer inherits from GenerationMixin.

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

    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.model = SLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    def tie_weights(self) -> None:
        """
        Tie LM head weights to input embeddings when tie_word_embeddings=True.

        Called automatically by post_init(). Overrides the default HuggingFace
        implementation to avoid _tied_weights_keys resolution which changed
        behaviour in transformers v5 and errors on list-based tied key specs.
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

    def forward(
        self,
        input_ids: Optional[torch.LongTensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        past_key_values: Optional[list[tuple[torch.Tensor, torch.Tensor]]] = None,
        inputs_embeds: Optional[torch.FloatTensor] = None,
        labels: Optional[torch.LongTensor] = None,
        use_cache: Optional[bool] = None,
        output_hidden_states: Optional[bool] = None,
        return_dict: Optional[bool] = None,
    ) -> CausalLMOutputWithPast:
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
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
        past_key_values: Optional[list] = None,
        attention_mask: Optional[torch.Tensor] = None,
        inputs_embeds: Optional[torch.Tensor] = None,
        **kwargs,
    ) -> dict:
        """Called by HuggingFace generate() at each decoding step."""
        if past_key_values is not None:
            input_ids = input_ids[:, -1:]

        if inputs_embeds is not None and past_key_values is None:
            model_inputs = {"inputs_embeds": inputs_embeds}
        else:
            model_inputs = {"input_ids": input_ids}

        model_inputs.update({
            "past_key_values": past_key_values,
            "use_cache": kwargs.get("use_cache"),
            "attention_mask": attention_mask,
        })
        return model_inputs

    @staticmethod
    def _reorder_cache(
        past_key_values: list[tuple[torch.Tensor, torch.Tensor]],
        beam_idx: torch.Tensor,
    ) -> list[tuple[torch.Tensor, torch.Tensor]]:
        """Reorder KV cache for beam search."""
        return [
            (k.index_select(0, beam_idx), v.index_select(0, beam_idx))
            for k, v in past_key_values
        ]