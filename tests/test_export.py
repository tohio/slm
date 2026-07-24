"""Native Hugging Face export contract tests."""

import json

import torch
from transformers import AutoConfig, AutoModelForCausalLM

from export.export import _convert_to_native_llama, _remote_code_artifacts
from model import SLMConfig, SLMForCausalLM


class _TokenizerContract:
    def __init__(self, vocab_size: int):
        self.vocab_size = vocab_size
        self.pad_token_id = 0
        self.bos_token_id = 1
        self.eos_token_id = 2

    def __len__(self) -> int:
        return self.vocab_size


def _tiny_source_model() -> tuple[SLMForCausalLM, _TokenizerContract]:
    torch.manual_seed(7)
    config = SLMConfig(
        vocab_size=128,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        rope_theta=500_000.0,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    model = SLMForCausalLM(config).eval()
    return model, _TokenizerContract(config.vocab_size)


def test_native_llama_conversion_preserves_logits():
    source, tokenizer = _tiny_source_model()
    native = _convert_to_native_llama(source, tokenizer, torch.float32)

    input_ids = torch.tensor([[1, 11, 12, 13, 14]], dtype=torch.long)
    attention_mask = torch.ones_like(input_ids)
    with torch.no_grad():
        source_logits = source(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits
        native_logits = native(
            input_ids=input_ids,
            attention_mask=attention_mask,
            use_cache=False,
        ).logits

    torch.testing.assert_close(
        native_logits,
        source_logits,
        rtol=1e-5,
        atol=1e-5,
    )
    assert native.config.model_type == "llama"
    assert native.config.num_hidden_layers == source.config.num_hidden_layers
    assert native.config.num_key_value_heads == source.config.num_key_value_heads
    assert native.config.tie_word_embeddings is True
    assert native.lm_head.weight is native.model.embed_tokens.weight


def test_native_package_loads_without_remote_code(tmp_path):
    source, tokenizer = _tiny_source_model()
    native = _convert_to_native_llama(source, tokenizer, torch.float32)
    native.save_pretrained(tmp_path, safe_serialization=True)

    config_json = json.loads((tmp_path / "config.json").read_text(encoding="utf-8"))
    assert config_json["model_type"] == "llama"
    assert config_json["architectures"] == ["LlamaForCausalLM"]
    assert "auto_map" not in config_json
    assert not list(tmp_path.glob("*.py"))

    config = AutoConfig.from_pretrained(
        tmp_path,
        trust_remote_code=False,
        local_files_only=True,
    )
    loaded = AutoModelForCausalLM.from_pretrained(
        tmp_path,
        trust_remote_code=False,
        local_files_only=True,
    )

    assert config.model_type == "llama"
    assert loaded.config.model_type == "llama"
    assert next(loaded.parameters()).dtype == torch.float32
    assert loaded.lm_head.weight is loaded.model.embed_tokens.weight


def test_remote_code_artifact_detection_is_recursive():
    files = [
        "README.md",
        "config.json",
        "model.safetensors",
        "modeling_slm.py",
        "slm_remote/configuration_slm.py",
        "tools/example.py",
    ]

    assert _remote_code_artifacts(files) == [
        "modeling_slm.py",
        "slm_remote/configuration_slm.py",
        "tools/example.py",
    ]
