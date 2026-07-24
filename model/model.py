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
from transformers import PreTrainedModel
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

    def _set_gradient_checkpointing(self, module: nn.Module, value: bool = False) -> None:
        if isinstance(module, SLMModel):
            module.gradient_checkpointing = value

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

    Loading uses a custom safe from_pretrained() path that loads SLMConfig,
    instantiates the model, reads model.safetensors or pytorch_model.bin, applies
    the state dict directly, and re-ties lm_head when embeddings are tied. This
    avoids local AutoModel loading issues observed with this custom architecture.
    """

    config_class = SLMConfig
    base_model_prefix = "model"
    supports_gradient_checkpointing = True
    _supports_sdpa = True
    _no_split_modules = ["SLMDecoderBlock"]
    _skip_keys_device_placement = ["past_key_values"]

    _tied_weights_keys = {"lm_head.weight": "model.embed_tokens.weight"}

    # lm_head.weight is intentionally omitted from safetensors when tied
    # to model.embed_tokens.weight.
    _keys_to_ignore_on_load_missing = [r"lm_head\.weight"]

    def __init__(self, config: SLMConfig):
        super().__init__(config)
        self.model = SLMModel(config)
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)
        self.post_init()
    
    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path, *model_args, **kwargs):
        """
        Safe SLM loader.

        Transformers local AutoModel loading has been observed to instantiate
        this custom architecture while leaving most checkpoint tensors at fresh
        initialization. This loader uses the verified path:

            SLMConfig -> cls(config) -> safetensors/torch load -> load_state_dict

        Supports:
            - local checkpoint directories
            - Hub repo IDs
            - model.safetensors
            - pytorch_model.bin
            - dtype / torch_dtype strings from CLI tools
            - tied lm_head.weight missing from checkpoint
        """
        import os
        from pathlib import Path

        import safetensors.torch

        config = kwargs.pop("config", None)
        torch_dtype = kwargs.pop("torch_dtype", None)
        dtype = kwargs.pop("dtype", None)
        device_map = kwargs.pop("device_map", None)
        output_loading_info = kwargs.pop("output_loading_info", False)

        revision = kwargs.pop("revision", None)
        cache_dir = kwargs.pop("cache_dir", None)
        token = kwargs.pop("token", None)
        local_files_only = kwargs.pop("local_files_only", False)

        # Accepted by many HF call sites, but not needed by this loader.
        kwargs.pop("low_cpu_mem_usage", None)
        kwargs.pop("trust_remote_code", None)
        kwargs.pop("weights_only", None)
        kwargs.pop("use_safetensors", None)

        if dtype is not None and torch_dtype is None:
            torch_dtype = dtype

        path = str(pretrained_model_name_or_path)

        # Hub repo ID support: resolve repo into a local snapshot first.
        if not os.path.isdir(path):
            from huggingface_hub import snapshot_download

            snapshot_kwargs = {
                "repo_id": path,
                "local_files_only": local_files_only,
                "allow_patterns": [
                    "config.json",
                    "generation_config.json",
                    "model.safetensors",
                    "pytorch_model.bin",
                    "tokenizer.json",
                    "tokenizer_config.json",
                    "special_tokens_map.json",
                    "chat_template.jinja",
                ],
            }
            if revision is not None:
                snapshot_kwargs["revision"] = revision
            if cache_dir is not None:
                snapshot_kwargs["cache_dir"] = cache_dir
            if token is not None:
                snapshot_kwargs["token"] = token

            path = snapshot_download(**snapshot_kwargs)

        if config is None:
            config = SLMConfig.from_pretrained(path)

        model = cls(config, *model_args)

        safetensors_path = Path(path) / "model.safetensors"
        bin_path = Path(path) / "pytorch_model.bin"

        if safetensors_path.exists():
            state_dict = safetensors.torch.load_file(str(safetensors_path), device="cpu")
        elif bin_path.exists():
            state_dict = torch.load(str(bin_path), map_location="cpu")
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
            raise RuntimeError(
                f"Missing keys while loading {path}: {unexpected_missing}"
            )

        if unexpected_keys:
            raise RuntimeError(
                f"Unexpected keys while loading {path}: {sorted(unexpected_keys)}"
            )

        if getattr(config, "tie_word_embeddings", False):
            model.tie_weights()

        # Normalize dtype passed by HF / lm-eval / CLI tools.
        if isinstance(torch_dtype, str):
            original_torch_dtype = torch_dtype

            if torch_dtype == "auto":
                cfg_dtype = getattr(config, "torch_dtype", None)
                if isinstance(cfg_dtype, str):
                    torch_dtype = getattr(torch, cfg_dtype, None)
                else:
                    torch_dtype = cfg_dtype
            else:
                torch_dtype = {
                    "float16": torch.float16,
                    "fp16": torch.float16,
                    "bfloat16": torch.bfloat16,
                    "bf16": torch.bfloat16,
                    "float32": torch.float32,
                    "fp32": torch.float32,
                }.get(torch_dtype, getattr(torch, torch_dtype, None))

            if torch_dtype is None:
                raise ValueError(
                    f"Unknown torch_dtype string: {original_torch_dtype!r}. "
                    "Expected a torch.dtype or one of: "
                    "'bfloat16', 'bf16', 'float16', 'fp16', 'float32', "
                    "'fp32', 'auto'."
                )

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
                "missing_keys": sorted(missing_keys),
                "unexpected_keys": sorted(unexpected_keys),
                "mismatched_keys": [],
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

        Direct assignment keeps tied embeddings explicit and stable across the
        pinned Transformers stack.
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
