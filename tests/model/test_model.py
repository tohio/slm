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
        from model.attention import GroupedQueryAttention
        config = make_mini_config()
        attn = GroupedQueryAttention(config, layer_idx=0)
        x = torch.randn(2, 16, config.hidden_size)
        out, _ = attn(x)
        assert out.shape == x.shape

    def test_kv_heads_fewer_than_q_heads(self):
        from model.attention import GroupedQueryAttention
        config = make_mini_config()
        attn = GroupedQueryAttention(config, layer_idx=0)
        assert attn.num_kv_heads < attn.num_heads
        assert attn.num_heads % attn.num_kv_heads == 0

    def test_kv_cache_shape(self):
        from model.attention import GroupedQueryAttention
        config = make_mini_config()
        attn = GroupedQueryAttention(config, layer_idx=0)
        x = torch.randn(2, 16, config.hidden_size)
        _, cache = attn(x, use_cache=True)
        assert cache is not None
        k, v = cache
        assert k.shape[1] == config.num_key_value_heads
        assert k.shape[2] == 16  # seq_len
        assert k.shape[3] == config.head_dim

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
        from model.block import SLMDecoderBlock
        config = make_mini_config()
        block = SLMDecoderBlock(config, layer_idx=0)
        x = torch.randn(2, 16, config.hidden_size)
        out, _ = block(x)
        assert out.shape == x.shape

    def test_residual_connection(self):
        """Output should differ from input (residual adds attention/MLP)."""
        from model.block import SLMDecoderBlock
        config = make_mini_config()
        block = SLMDecoderBlock(config, layer_idx=0)
        x = torch.randn(2, 16, config.hidden_size)
        out, _ = block(x)
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
        """Standalone SLMModel should apply configured initializer_range."""
        from model.model import SLMModel
        config = make_mini_config()
        model = SLMModel(config)

        # The configured std is 0.02. After init, the embedding weight's
        # standard deviation should be in a plausible range around that.
        # PyTorch's default Embedding init is N(0, 1) — detection threshold
        # is any std > 0.1, which indicates post_init() did not run.
        embed_std = model.embed_tokens.weight.std().item()
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

    def test_weight_tying(self):
        from model.model import SLMForCausalLM
        config = make_mini_config()
        model = SLMForCausalLM(config)
        assert model.lm_head.weight is model.model.embed_tokens.weight, (
            "LM head weight is not tied to embedding weight"
        )

    def test_parameter_count_approximately_25m(self):
        """
        Mini model should be approximately 25M parameters.

        The mini config is deterministic (6 layers × 384 hidden × 32k vocab
        × tied), so parameter count should be tightly predictable. A ±5%
        band catches real drift (e.g. a change to _default_intermediate_size,
        an accidental untie, vocab size change).
        """
        from model.model import SLMForCausalLM
        config = make_mini_config()
        model = SLMForCausalLM(config)
        n_params = sum(p.numel() for p in model.parameters())
        expected = 25_000_000
        drift = abs(n_params - expected) / expected
        assert drift < 0.05, (
            f"Parameter count {n_params:,} drifted {drift:.1%} from "
            f"expected ~{expected:,} (tolerance 5%)"
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