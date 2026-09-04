"""
tests/model/test_model.py
--------------------------
Unit tests for the SLM model architecture.

Tests forward pass shapes, causal masking, parameter counts, weight tying,
RMSNorm, SwiGLU, and GQA. No GPU required — runs on CPU.
"""

import json

import pytest
import torch

from tests.conftest import make_mini_config


def _make_tiny_config():
    from model.config import SLMConfig

    return SLMConfig(
        vocab_size=256,
        hidden_size=64,
        intermediate_size=128,
        num_hidden_layers=2,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
    )


def _fill_parameters_with_sentinels(model) -> None:
    """Make checkpoint tensors unmistakably different from random init."""
    with torch.no_grad():
        for index, parameter in enumerate(model.parameters(), start=1):
            parameter.fill_(index / 100.0)


def _assert_state_dicts_equal(expected, actual) -> None:
    assert list(expected) == list(actual)
    for name, expected_tensor in expected.items():
        assert torch.equal(expected_tensor, actual[name]), (
            f"Checkpoint tensor changed during native loading: {name}"
        )


def _save_pytorch_checkpoint(model, output_dir, *, sharded: bool) -> str:
    """Write a real legacy PyTorch checkpoint for native-loader coverage."""
    output_dir.mkdir(parents=True, exist_ok=True)
    model.config.save_pretrained(output_dir)
    state_dict = {
        name: tensor.detach().cpu().clone()
        for name, tensor in model.state_dict().items()
    }

    if not sharded:
        artifact = "pytorch_model.bin"
        torch.save(state_dict, output_dir / artifact)
        return artifact

    keys = list(state_dict)
    midpoint = max(1, len(keys) // 2)
    shard_keys = (keys[:midpoint], keys[midpoint:])
    shard_names = (
        "pytorch_model-00001-of-00002.bin",
        "pytorch_model-00002-of-00002.bin",
    )
    weight_map = {}
    for names, shard_name in zip(shard_keys, shard_names, strict=True):
        shard = {name: state_dict[name] for name in names}
        torch.save(shard, output_dir / shard_name)
        weight_map.update({name: shard_name for name in names})

    index = {
        "metadata": {
            "total_size": sum(
                tensor.numel() * tensor.element_size()
                for tensor in state_dict.values()
            )
        },
        "weight_map": weight_map,
    }
    artifact = "pytorch_model.bin.index.json"
    (output_dir / artifact).write_text(
        json.dumps(index, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    return artifact


# ── RMSNorm ────────────────────────────────────────────────────────────────────

class TestRMSNorm:
    def test_output_shape(self):
        from model.norm import RMSNorm
        norm = RMSNorm(384)
        x = torch.randn(2, 16, 384)
        out = norm(x)
        assert out.shape == x.shape

    def test_output_dtype_preserved(self):
        from model.norm import RMSNorm
        norm = RMSNorm(384)
        x = torch.randn(2, 16, 384).to(torch.float16)
        out = norm(x)
        assert out.dtype == torch.float16

    def test_normalized_rms_close_to_one(self):
        """After RMSNorm with weight=1, RMS of output should be ~1."""
        from model.norm import RMSNorm
        norm = RMSNorm(384)
        torch.nn.init.ones_(norm.weight)
        x = torch.randn(2, 16, 384)
        out = norm(x)
        rms = out.pow(2).mean(dim=-1).sqrt()
        assert torch.allclose(rms, torch.ones_like(rms), atol=1e-4)

    def test_no_bias(self):
        from model.norm import RMSNorm
        norm = RMSNorm(384)
        assert not hasattr(norm, "bias") or norm.bias is None

    def test_matches_native_llama_bfloat16_order(self):
        from transformers.models.llama.modeling_llama import LlamaRMSNorm

        from model.norm import RMSNorm

        hidden_size = 64
        eps = 1e-5
        slm_norm = RMSNorm(hidden_size, eps=eps).to(dtype=torch.bfloat16)
        llama_norm = LlamaRMSNorm(hidden_size, eps=eps).to(dtype=torch.bfloat16)

        with torch.no_grad():
            weight = torch.linspace(0.5, 1.5, hidden_size, dtype=torch.bfloat16)
            slm_norm.weight.copy_(weight)
            llama_norm.weight.copy_(weight)

        generator = torch.Generator().manual_seed(17)
        hidden_states = torch.randn(
            2,
            7,
            hidden_size,
            generator=generator,
            dtype=torch.float32,
        ).to(torch.bfloat16)

        assert torch.equal(slm_norm(hidden_states), llama_norm(hidden_states))


# ── SwiGLU MLP ─────────────────────────────────────────────────────────────────

class TestSwiGLUMLP:
    def test_output_shape(self):
        from model.mlp import SwiGLUMLP
        config = make_mini_config()
        mlp = SwiGLUMLP(config)
        x = torch.randn(2, 16, config.hidden_size)
        out = mlp(x)
        assert out.shape == x.shape

    def test_no_bias(self):
        from model.mlp import SwiGLUMLP
        config = make_mini_config()
        mlp = SwiGLUMLP(config)
        for name, param in mlp.named_parameters():
            assert "bias" not in name, f"Found bias parameter: {name}"

    def test_three_projections(self):
        from model.mlp import SwiGLUMLP
        config = make_mini_config()
        mlp = SwiGLUMLP(config)
        assert hasattr(mlp, "gate_proj")
        assert hasattr(mlp, "up_proj")
        assert hasattr(mlp, "down_proj")


# ── GQA Attention ──────────────────────────────────────────────────────────────

class TestGroupedQueryAttention:
    def test_output_shape(self):
        from model.attention import GroupedQueryAttention, RotaryEmbedding
        config = make_mini_config()
        attn = GroupedQueryAttention(config, layer_idx=0)
        x = torch.randn(2, 16, config.hidden_size)
        position_ids = torch.arange(16).unsqueeze(0)
        position_embeddings = RotaryEmbedding(config)(x, position_ids)
        out = attn(x, position_embeddings=position_embeddings)
        assert out.shape == x.shape

    def test_kv_heads_fewer_than_q_heads(self):
        from model.attention import GroupedQueryAttention
        config = make_mini_config()
        attn = GroupedQueryAttention(config, layer_idx=0)
        assert attn.num_kv_heads < attn.num_heads
        assert attn.num_heads % attn.num_kv_heads == 0

    def test_kv_cache_shape(self):
        from transformers.cache_utils import DynamicCache

        from model.attention import GroupedQueryAttention, RotaryEmbedding
        config = make_mini_config()
        attn = GroupedQueryAttention(config, layer_idx=0)
        x = torch.randn(2, 16, config.hidden_size)
        position_ids = torch.arange(16).unsqueeze(0)
        position_embeddings = RotaryEmbedding(config)(x, position_ids)
        cache = DynamicCache(config=config)
        attn(
            x,
            position_embeddings=position_embeddings,
            past_key_values=cache,
            use_cache=True,
        )
        k = cache.layers[0].keys
        v = cache.layers[0].values
        assert k.shape[1] == config.num_key_value_heads
        assert k.shape[2] == 16  # seq_len
        assert k.shape[3] == config.head_dim
        assert v.shape == k.shape

    def test_no_bias(self):
        from model.attention import GroupedQueryAttention
        config = make_mini_config()
        attn = GroupedQueryAttention(config, layer_idx=0)
        bias_params = [
            name for name, _ in attn.named_parameters() if "bias" in name
        ]
        assert not bias_params, f"Found bias parameters: {bias_params}"


# ── Decoder Block ──────────────────────────────────────────────────────────────

class TestDecoderBlock:
    def test_output_shape(self):
        from model.attention import RotaryEmbedding
        from model.block import SLMDecoderBlock
        config = make_mini_config()
        block = SLMDecoderBlock(config, layer_idx=0)
        x = torch.randn(2, 16, config.hidden_size)
        position_ids = torch.arange(16).unsqueeze(0)
        position_embeddings = RotaryEmbedding(config)(x, position_ids)
        out = block(x, position_embeddings=position_embeddings)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from input (residual adds attention/MLP)."""
        from model.attention import RotaryEmbedding
        from model.block import SLMDecoderBlock
        config = make_mini_config()
        block = SLMDecoderBlock(config, layer_idx=0)
        x = torch.randn(2, 16, config.hidden_size)
        position_ids = torch.arange(16).unsqueeze(0)
        position_embeddings = RotaryEmbedding(config)(x, position_ids)
        out = block(x, position_embeddings=position_embeddings)
        assert not torch.allclose(out, x)


# ── SLMModel ───────────────────────────────────────────────────────────────────

class TestSLMModel:
    def test_forward_output_shape(self):
        from model.model import SLMModel
        config = make_mini_config()
        model = SLMModel(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        out = model(input_ids)
        assert out.last_hidden_state.shape == (2, 16, config.hidden_size)

    def test_num_layers(self):
        from model.model import SLMModel
        config = make_mini_config()
        model = SLMModel(config)
        assert len(model.layers) == config.num_hidden_layers

    def test_init_respects_initializer_range(self):
        """The HF wrapper should apply configured initializer_range."""
        from model.model import SLMForCausalLM
        config = make_mini_config()
        model = SLMForCausalLM(config)

        # The configured std is 0.02. After init, the embedding weight's
        # standard deviation should be in a plausible range around that.
        # PyTorch's default Embedding init is N(0, 1) — detection threshold
        # is any std > 0.1, which indicates post_init() did not run.
        embed_std = model.get_input_embeddings().weight.std().item()
        assert embed_std < 0.05, (
            f"Embedding std {embed_std:.4f} suggests post_init() did not run "
            f"(expected ~{config.initializer_range})"
        )


# ── SLMForCausalLM ─────────────────────────────────────────────────────────────

class TestSLMForCausalLM:
    def test_forward_logits_shape(self):
        from model.model import SLMForCausalLM
        config = make_mini_config()
        model = SLMForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        out = model(input_ids)
        assert out.logits.shape == (2, 16, config.vocab_size)

    def test_forward_with_labels_returns_loss(self):
        from model.model import SLMForCausalLM
        config = make_mini_config()
        model = SLMForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 16))
        labels = torch.randint(0, config.vocab_size, (2, 16))
        out = model(input_ids, labels=labels)
        assert out.loss is not None
        assert out.loss.item() > 0
        assert torch.isfinite(out.loss)

    def test_loss_is_finite(self):
        from model.model import SLMForCausalLM
        config = make_mini_config()
        model = SLMForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 32))
        labels = input_ids.clone()
        out = model(input_ids, labels=labels)
        assert torch.isfinite(out.loss), f"Loss is not finite: {out.loss}"

    def test_labels_disable_cache_by_default(self):
        from model.model import SLMForCausalLM

        config = _make_tiny_config()
        model = SLMForCausalLM(config).train()
        input_ids = torch.randint(0, config.vocab_size, (2, 16))

        out = model(input_ids, labels=input_ids)

        assert out.past_key_values is None

    def test_shared_causal_loss_honors_num_items_in_batch(self):
        from model.model import SLMForCausalLM

        config = _make_tiny_config()
        model = SLMForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))
        labels = input_ids.clone()
        labels[0, :4] = -100
        num_items = (labels[..., 1:] != -100).sum()

        out = model(
            input_ids,
            labels=labels,
            num_items_in_batch=num_items,
        )
        expected = torch.nn.functional.cross_entropy(
            out.logits[..., :-1, :].float().reshape(-1, config.vocab_size),
            labels[..., 1:].reshape(-1),
            ignore_index=-100,
            reduction="sum",
        ) / num_items

        torch.testing.assert_close(out.loss, expected)

    def test_logits_to_keep_limits_generation_projection(self):
        from model.model import SLMForCausalLM

        config = _make_tiny_config()
        model = SLMForCausalLM(config).eval()
        input_ids = torch.randint(0, config.vocab_size, (2, 16))

        out = model(input_ids, logits_to_keep=1)

        assert out.logits.shape == (2, 1, config.vocab_size)

    def test_return_dict_false_returns_tuple(self):
        from model.model import SLMForCausalLM

        config = _make_tiny_config()
        model = SLMForCausalLM(config)
        input_ids = torch.randint(0, config.vocab_size, (2, 8))

        out = model(input_ids, labels=input_ids, return_dict=False)

        assert isinstance(out, tuple)
        assert out[0].ndim == 0
        assert out[1].shape == (2, 8, config.vocab_size)

    @pytest.mark.parametrize(
        "input_ids,inputs_embeds",
        [
            (None, None),
            (
                torch.ones((1, 4), dtype=torch.long),
                torch.zeros((1, 4, 64)),
            ),
        ],
    )
    def test_requires_exactly_one_model_input(self, input_ids, inputs_embeds):
        from model.model import SLMForCausalLM

        model = SLMForCausalLM(_make_tiny_config())

        with pytest.raises(ValueError, match="exactly one"):
            model(input_ids=input_ids, inputs_embeds=inputs_embeds)

    def test_gradient_checkpointing_backward(self):
        """The Transformers checkpointing hook must reach the plain SLMModel."""
        from model.config import SLMConfig
        from model.model import SLMForCausalLM

        config = SLMConfig(
            vocab_size=256,
            hidden_size=64,
            intermediate_size=128,
            num_hidden_layers=2,
            num_attention_heads=4,
            num_key_value_heads=2,
            max_position_embeddings=64,
        )
        model = SLMForCausalLM(config)
        model.gradient_checkpointing_enable()
        input_ids = torch.randint(0, config.vocab_size, (2, 16))

        loss = model(input_ids, labels=input_ids).loss
        loss.backward()

        assert model.model.gradient_checkpointing is True
        assert any(
            parameter.grad is not None
            for parameter in model.parameters()
        )

    def test_weight_tying(self):
        from model.model import SLMForCausalLM
        config = make_mini_config()
        model = SLMForCausalLM(config)
        assert model.lm_head.weight is model.model.embed_tokens.weight, (
            "LM head weight is not tied to embedding weight"
        )

    def test_parameter_count_approximately_22m(self):
        """
        Mini model should be approximately 22M unique parameters.

        The mini config is deterministic (6 layers × 384 hidden × 32k vocab
        × tied), so parameter count should be tightly predictable. A ±2%
        band catches real drift (e.g. a change to _default_intermediate_size,
        an accidental untie, vocab size change).
        """
        from model.model import SLMForCausalLM
        config = make_mini_config()
        model = SLMForCausalLM(config)
        n_params = sum(p.numel() for p in model.parameters())
        expected = 22_000_000
        drift = abs(n_params - expected) / expected
        assert drift < 0.02, (
            f"Parameter count {n_params:,} drifted {drift:.1%} from "
            f"expected ~{expected:,} (tolerance 2%)"
        )

    def test_causal_mask_lower_triangular(self):
        """
        Verify the model is causal — token at position i should not
        attend to tokens at positions > i. We check this by verifying
        that permuting future tokens does not change the logits at
        earlier positions.
        """
        from model.model import SLMForCausalLM
        config = make_mini_config()
        model = SLMForCausalLM(config)
        model.eval()

        with torch.no_grad():
            seq = torch.randint(0, config.vocab_size, (1, 8))
            out1 = model(seq).logits

            # Perturb last token — should not affect first token's logits
            seq2 = seq.clone()
            seq2[0, -1] = (seq2[0, -1] + 1) % config.vocab_size
            out2 = model(seq2).logits

        assert torch.allclose(out1[0, 0], out2[0, 0], atol=1e-5), (
            "First token logits changed when last token was perturbed — "
            "causal mask may not be working"
        )

    def test_no_bias_parameters(self):
        """
        No parameter anywhere in the model should be named '*bias*'.

        The previous version of this test excluded 'norm' and 'rotary' from
        the check. Neither RMSNorm nor RotaryEmbedding has bias parameters,
        so the exclusions were defensive against cases that don't exist.
        Removing them means if anyone ever adds a bias to RMSNorm (or any
        other module) it gets caught here.
        """
        from model.model import SLMForCausalLM
        config = make_mini_config()
        model = SLMForCausalLM(config)
        bias_params = [
            name for name, _ in model.named_parameters() if "bias" in name
        ]
        assert len(bias_params) == 0, f"Found bias parameters: {bias_params}"

    @pytest.mark.parametrize(
        ("checkpoint_format", "sharded"),
        [
            ("safetensors", False),
            ("safetensors", True),
            ("pytorch", False),
            ("pytorch", True),
        ],
    )
    def test_native_from_pretrained_preserves_checkpoint_tensors(
        self,
        tmp_path,
        checkpoint_format,
        sharded,
    ):
        """Regression for Transformers 5 reinitializing loaded custom weights."""
        from transformers import PreTrainedModel

        from model.model import SLMForCausalLM

        assert "from_pretrained" not in SLMForCausalLM.__dict__, (
            "SLM must use the native PreTrainedModel.from_pretrained implementation"
        )
        assert "tie_weights" not in SLMForCausalLM.__dict__, (
            "SLM must use the native tied-weight mapping during checkpoint loading"
        )

        config = _make_tiny_config()
        model = SLMForCausalLM(config).eval()
        _fill_parameters_with_sentinels(model)
        expected_state = {
            name: tensor.detach().cpu().clone()
            for name, tensor in model.state_dict().items()
        }
        input_ids = torch.arange(8, dtype=torch.long).unsqueeze(0)
        with torch.no_grad():
            expected_logits = model(input_ids).logits.detach().cpu()

        if checkpoint_format == "safetensors":
            save_kwargs = {}
            if sharded:
                save_kwargs["max_shard_size"] = "10KB"
            model.save_pretrained(str(tmp_path), **save_kwargs)
            expected_artifact = (
                "model.safetensors.index.json"
                if sharded
                else "model.safetensors"
            )
            use_safetensors = True
        else:
            expected_artifact = _save_pytorch_checkpoint(
                model,
                tmp_path,
                sharded=sharded,
            )
            use_safetensors = False

        saved_config = json.loads(
            (tmp_path / "config.json").read_text(encoding="utf-8")
        )
        assert "auto_map" not in saved_config
        assert not list(tmp_path.glob("*.py"))
        assert (tmp_path / expected_artifact).is_file()

        # A different seed makes accidental post-load reinitialization obvious.
        torch.manual_seed(991)
        loaded, loading_info = PreTrainedModel.from_pretrained.__func__(
            SLMForCausalLM,
            str(tmp_path),
            output_loading_info=True,
            use_safetensors=use_safetensors,
            weights_only=True,
        )

        for key in (
            "missing_keys",
            "unexpected_keys",
            "mismatched_keys",
            "error_msgs",
        ):
            assert not loading_info[key], (
                f"Native loading reported {key}: {loading_info[key]}"
            )
        assert loaded.training is False
        _assert_state_dicts_equal(expected_state, loaded.state_dict())
        assert (
            loaded.lm_head.weight.data_ptr()
            == loaded.model.embed_tokens.weight.data_ptr()
        )

        with torch.no_grad():
            actual_logits = loaded(input_ids).logits.detach().cpu()
        assert torch.equal(expected_logits, actual_logits)



class TestSLMConfigValidation:
    def test_legacy_remote_code_mapping_is_not_propagated(self):
        from model.config import SLMConfig

        config = SLMConfig(
            auto_map={
                "AutoConfig": "config.SLMConfig",
                "AutoModelForCausalLM": "model.SLMForCausalLM",
            }
        )
        assert "auto_map" not in config.to_dict()

    @pytest.mark.parametrize(
        "override",
        [
            {"vocab_size": 0},
            {"hidden_size": 0},
            {"intermediate_size": 0},
            {"num_hidden_layers": 0},
            {"num_attention_heads": 0},
            {"num_key_value_heads": 0},
            {"max_position_embeddings": 0},
            {"rope_theta": 0},
            {"rms_norm_eps": 0},
            {"initializer_range": 0},
            {"attention_dropout": -0.1},
            {"attention_dropout": 1.0},
        ],
    )
    def test_invalid_numeric_config_is_rejected(self, override):
        from model.config import SLMConfig

        values = {
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
        }
        values.update(override)

        with pytest.raises(ValueError):
            SLMConfig(**values)

    def test_odd_rope_head_dimension_is_rejected(self):
        from model.config import SLMConfig

        with pytest.raises(ValueError, match="must be even"):
            SLMConfig(
                vocab_size=256,
                hidden_size=60,
                intermediate_size=128,
                num_hidden_layers=2,
                num_attention_heads=4,
                num_key_value_heads=2,
                max_position_embeddings=64,
            )

    def test_unimplemented_rope_scaling_is_rejected(self):
        from model.config import SLMConfig

        common = {
            "vocab_size": 256,
            "hidden_size": 64,
            "intermediate_size": 128,
            "num_hidden_layers": 2,
            "num_attention_heads": 4,
            "num_key_value_heads": 2,
            "max_position_embeddings": 64,
        }

        with pytest.raises(ValueError, match="not implemented"):
            SLMConfig(
                **common,
                rope_scaling={"rope_type": "linear", "factor": 2.0},
            )

        with pytest.raises(ValueError, match="not implemented"):
            SLMConfig(
                **common,
                rope_parameters={"rope_type": "linear", "factor": 2.0},
            )
