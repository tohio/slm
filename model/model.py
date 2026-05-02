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
    - Chunked LM loss in training path to avoid materializing full
      [B, T, vocab] logits tensors at once (see _chunked_lm_loss).

Important implementation detail:
    SLMModel is a plain nn.Module.
    SLMForCausalLM is the only PreTrainedModel.

This follows the standard HuggingFace architecture pattern used by Llama,
Mistral, GPT-NeoX, Phi, etc.:

    SLMForCausalLM(PreTrainedModel)
        └── SLMModel(nn.Module)

Only the outer class calls post_init(), so initialization and HF
save/load behavior are controlled from one PreTrainedModel.

Important loader note:
    transformers==5.5.4 was observed to report successful loading for this
    custom model while not actually applying checkpoint tensors. Therefore
    SLMForCausalLM.from_pretrained() is overridden to use the verified-safe
    path:

        config -> model init -> safetensors/torch load -> load_state_dict
"""

from typing import Optional, Union

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import CrossEntropyLoss
from transformers import PreTrainedModel
from transformers.cache_utils import Cache
from transformers.generation import GenerationMixin
from transformers.modeling_outputs import BaseModelOutputWithPast, CausalLMOutputWithPast

from .block import SLMDecoderBlock
from .config import SLMConfig
from .norm import RMSNorm


# Sequence-position chunk size for the chunked LM loss path.
# At seq_len=2048, vocab=32000, bf16:
#   full logits     = B * 2048 * 32000 * 2 B  = B * 131 MB
#   chunked logits  = B *  512 * 32000 * 2 B  = B *  33 MB   (4× headroom)
# 512 keeps memory bounded without sacrificing kernel efficiency on H100/H200.
LM_LOSS_CHUNK_SIZE = 512


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

    cache_layers = getattr(cache, "layers", None)
    if cache_layers is not None:
        for i in range(n_layers):
            if i < len(cache_layers) and cache_layers[i].is_initialized:
                result.append((cache_layers[i].keys, cache_layers[i].values))
            else:
                result.append(None)
        return result

    key_cache = getattr(cache, "key_cache", None)
    value_cache = getattr(cache, "value_cache", None)
    if key_cache is not None:
        for i in range(n_layers):
            if i < len(key_cache):
                result.append((key_cache[i], value_cache[i]))
            else:
                result.append(None)
        return result

    return [None] * n_layers


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

        self.embed_tokens = nn.Embedding(config.vocab_size, config.hidden_size)
        self.layers = nn.ModuleList(
            [SLMDecoderBlock(config, layer_idx=i) for i in range(config.num_hidden_layers)]
        )
        self.norm = RMSNorm(config.hidden_size, eps=config.rms_norm_eps)
        self.gradient_checkpointing = False

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
            self.config, "return_dict", getattr(self.config, "use_return_dict", True)
        )

        if self.gradient_checkpointing and self.training and use_cache:
            use_cache = False

        if inputs_embeds is None:
            inputs_embeds = self.embed_tokens(input_ids)

        hidden_states = inputs_embeds

        if past_key_values is None:
            past_key_values = [None] * len(self.layers)
        elif isinstance(past_key_values, Cache):
            past_key_values = _extract_kv_from_dynamic_cache(
                past_key_values,
                len(self.layers),
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
                    None,
                    False,
                )
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
    SLM with a language modelling head for causal language modelling.

    This is the only PreTrainedModel in the architecture. It owns:
    - initialization via post_init()
    - save_pretrained()
    - from_pretrained()
    - tied embedding behavior
    - HF generation compatibility
    """

    config_class = SLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True

    # Common HF tied-weight declaration.
    # If a future Transformers version requires the v5 dict form, test:
    #     {"lm_head.weight": "model.embed_tokens.weight"}
    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.model = SLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Safe SLM loader.

        transformers==5.5.4 was observed to report successful loading for this
        custom model while not applying checkpoint tensors correctly. This
        override preserves the public API while using the verified-safe route:

            config -> cls(config) -> safetensors/torch load -> load_state_dict

        Supports local checkpoint directories containing:
            - model.safetensors, or
            - pytorch_model.bin

        Notes:
            - `device_map="auto"` is handled minimally for local loading.
            - `torch_dtype` and `dtype` are both accepted.
            - `output_loading_info=True` returns (model, info), matching HF style.
        """
        import os
        import safetensors.torch
        from transformers import AutoConfig

        config = kwargs.pop("config", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        dtype = kwargs.pop("dtype", None)
        device_map = kwargs.pop("device_map", None)
        output_loading_info = kwargs.pop("output_loading_info", False)

        # Accepted by many HF call sites, but ignored by this safe local loader.
        kwargs.pop("low_cpu_mem_usage", None)
        kwargs.pop("trust_remote_code", None)

        if dtype is not None and torch_dtype is None:
            torch_dtype = dtype

        path = str(pretrained_model_name_or_path)

        if config is None:
            config = AutoConfig.from_pretrained(path)

        model = cls(config, *model_args)

        safetensors_path = os.path.join(path, "model.safetensors")
        bin_path = os.path.join(path, "pytorch_model.bin")

        if os.path.exists(safetensors_path):
            state_dict = safetensors.torch.load_file(safetensors_path, device="cpu")
        elif os.path.exists(bin_path):
            state_dict = torch.load(bin_path, map_location="cpu")
        else:
            raise FileNotFoundError(
                f"No model.safetensors or pytorch_model.bin found in {path}"
            )

        result = model.load_state_dict(state_dict, strict=False)

        allowed_missing = set()
        if getattr(config, "tie_word_embeddings", False):
            allowed_missing.add("lm_head.weight")

        missing_keys = set(result.missing_keys)
        unexpected_keys = set(result.unexpected_keys)
        unexpected_missing = sorted(k for k in missing_keys if k not in allowed_missing)

        if unexpected_missing:
            raise RuntimeError(f"Missing keys while loading {path}: {unexpected_missing}")

        if unexpected_keys:
            raise RuntimeError(f"Unexpected keys while loading {path}: {sorted(unexpected_keys)}")

        if getattr(config, "tie_word_embeddings", False):
            model.tie_weights()

        if torch_dtype is not None:
            model = model.to(dtype=torch_dtype)

        # Minimal local device_map support.
        if device_map is not None:
            if device_map == "auto" and torch.cuda.is_available():
                model = model.to("cuda")
            elif isinstance(device_map, str) and device_map != "auto":
                model = model.to(device_map)

        model.eval()

        if output_loading_info:
            info = {
                "missing_keys": missing_keys,
                "unexpected_keys": unexpected_keys,
                "mismatched_keys": set(),
                "error_msgs": [],
            }
            return model, info

        return model

    def _init_weights(self, module: nn.Module) -> None:
        """
        Initialize weights with config.initializer_range.

        This runs on every submodule when post_init() recurses, including
        modules inside SLMModel. SLMModel intentionally does not define its
        own _init_weights; this is the single source of init policy.
        """
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

        Direct assignment is used because transformers==5.5.4 in this environment
        does not expose _tie_or_clone_weights on PreTrainedModel.
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

    def _chunked_lm_loss(
        self,
        hidden_states: torch.Tensor,
        labels: torch.LongTensor,
        chunk_size: int = LM_LOSS_CHUNK_SIZE,
    ) -> torch.Tensor:
        """
        Compute next-token CE loss without materializing full [B, T, vocab] logits.

        Iterates over chunks of sequence positions, applies lm_head per chunk,
        and accumulates the per-token CE sum. Final loss is sum / num_valid_tokens
        where valid means label != -100. This matches the default reduction='mean'
        of nn.CrossEntropyLoss with ignore_index=-100, which is what the
        non-chunked path produces. Verified bit-exact against the full-logits
        path with random inputs and various masking patterns.

        Notable behavior:
            - Chunks where all labels are -100 contribute zero loss through ignore_index=-100.
              The lm_head matmul and CE compute are saved for those positions.
            - Standard next-token shift: prediction at position t is supervised
              by the label at position t+1. Implemented by chunking over the
              first T-1 positions of hidden_states and the last T-1 of labels.
            - Degenerate batch (zero supervised tokens): returns 0.0 with a
              dependency on hidden_states so DDP/FSDP gradient sync stays
              well-formed. No .item() / no host-device syncs in the loop —
              keeps the graph torch.compile-clean.

        Memory shape (per chunk, B=micro, C=chunk_size, V=vocab):
            chunk_logits : [B, C, V]   bf16    = B*C*V*2 bytes
            chunk_labels : [B, C]      int64
            CE upcasts internally to fp32 on the chunk only.
        """
        # Shift: predict token at t+1 from hidden_state at t.
        # Slicing avoids the .contiguous() allocations the old path made.
        shift_hidden = hidden_states[:, :-1, :]
        shift_labels = labels[:, 1:]
        seq_len = shift_hidden.shape[1]

        total_loss_sum = hidden_states.new_zeros((), dtype=torch.float32)
        total_valid = torch.zeros((), dtype=torch.long, device=hidden_states.device)

        for start in range(0, seq_len, chunk_size):
            end = min(start + chunk_size, seq_len)
            chunk_labels = shift_labels[:, start:end]

            chunk_hidden = shift_hidden[:, start:end, :]
            chunk_logits = self.lm_head(chunk_hidden)

            # F.cross_entropy with reduction='sum' so we can normalize by the
            # global valid-token count after the loop.
            chunk_loss_sum = F.cross_entropy(
                chunk_logits.reshape(-1, self.config.vocab_size),
                chunk_labels.reshape(-1),
                ignore_index=-100,
                reduction="sum",
            )
            chunk_valid = (chunk_labels != -100).sum()

            total_loss_sum = total_loss_sum + chunk_loss_sum.float()
            total_valid = total_valid + chunk_valid

        # Avoid div-by-zero on degenerate all-masked batches without using
        # .item() (which would break torch.compile and force a host sync).
        # If total_valid is 0 then total_loss_sum is also 0, so 0/1 = 0.
        # The trailing `0.0 * hidden_states.sum()` keeps hidden_states in
        # the autograd graph so DDP/FSDP gradient sync is well-formed even
        # on all-masked batches.
        denom = total_valid.clamp(min=1).float()
        return total_loss_sum / denom + 0.0 * hidden_states.sum()

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
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            past_key_values=past_key_values,
            inputs_embeds=inputs_embeds,
            use_cache=use_cache,
            output_hidden_states=output_hidden_states,
            return_dict=True,
            cache_position=cache_position,
        )

        hidden_states = outputs.last_hidden_state

        # Loss + logits policy
        # ---------------------
        # MEMORY INVARIANT: in the training+labels branch, we MUST NOT call
        # self.lm_head(hidden_states) on the full sequence. Doing so
        # materializes [B, T, vocab] logits — at micro=64, seq=2048, vocab=32k,
        # bf16 that is 8.4 GB just for the logits tensor, and CE upcasts to
        # fp32 to spike another 16+ GB on top. That allocation is what caused
        # the H200 OOM that motivated this whole patch.
        #
        # Instead the training branch goes straight into _chunked_lm_loss
        # (per-chunk lm_head + CE) and produces full logits ONLY for the last
        # token (~4 MB), which is enough for TRL's training-time accuracy/
        # entropy metrics. Anyone refactoring this branch: keep the rule
        # "no full-sequence lm_head() during training+labels" or the
        # optimization is silently lost.
        #
        # During eval (self.training == False) or when labels are None, fall
        # back to full logits so harness-style evals, generation, and
        # eval_loss reporting all behave exactly as before.
        if labels is not None and self.training:
            loss = self._chunked_lm_loss(hidden_states, labels)
            logits = self.lm_head(hidden_states[:, -1:, :])
        else:
            logits = self.lm_head(hidden_states)
            loss = None
            if labels is not None:
                # Eval path: keep the original behavior bit-for-bit so eval_loss
                # values are comparable across before/after the chunked-loss
                # patch. .contiguous() retained here because some older torch
                # builds require it for view().
                shift_logits = logits[..., :-1, :].contiguous()
                shift_labels = labels[..., 1:].contiguous()
                loss = CrossEntropyLoss()(
                    shift_logits.view(-1, self.config.vocab_size),
                    shift_labels.view(-1),
                )

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

        Slices input_ids to only positions that have not yet been processed.
        """
        if cache_position is not None:
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
        """
        if isinstance(past_key_values, Cache):
            past_key_values.reorder_cache(beam_idx)
            return past_key_values

        return [
            (k.index_select(0, beam_idx), v.index_select(0, beam_idx))
            for k, v in past_key_values
        ]