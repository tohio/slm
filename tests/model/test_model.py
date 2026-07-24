"""
tests/model/test_model.py
--------------------------
Unit tests for the SLM model architecture.

Tests forward pass shapes, causal masking, parameter counts, weight tying,
RMSNorm, SwiGLU, and GQA. No GPU required — runs on CPU.
"""

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

    def test_save_and_load(self, tmp_path):
        from model.model import SLMForCausalLM
        config = make_mini_config()
        model = SLMForCausalLM(config)
        model.eval()

        input_ids = torch.randint(0, config.vocab_size, (1, 8))
        with torch.no_grad():
            logits_before = model(input_ids).logits

        model.save_pretrained(str(tmp_path))
        loaded = SLMForCausalLM.from_pretrained(str(tmp_path))
        loaded.eval()

        with torch.no_grad():
            logits_after = loaded(input_ids).logits

        assert torch.allclose(logits_before, logits_after, atol=1e-5), (
            "Logits differ after save/load — weight tying or serialisation issue"
        )


class TestSLMConfigValidation:
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
