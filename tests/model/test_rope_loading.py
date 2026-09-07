"""Native Transformers loading regression checks, requiring the training stack."""
import pytest
import torch

pytest.importorskip("transformers", reason="Run with requirements-training.txt")
from model import SLMConfig, SLMForCausalLM
from model.attention import RotaryEmbedding


def tiny_config():
    return SLMConfig(vocab_size=32, hidden_size=32, intermediate_size=64,
        num_hidden_layers=2, num_attention_heads=4, num_key_value_heads=2,
        max_position_embeddings=32, rope_theta=500000.0)


@pytest.mark.parametrize("dtype", [torch.float32, torch.bfloat16])
def test_native_reload_restores_only_derived_rope(tmp_path, dtype):
    torch.manual_seed(17)
    original = SLMForCausalLM(tiny_config()).eval()
    before = {name: value.detach().clone() for name, value in original.state_dict().items()}
    assert not any("inv_freq" in name for name in before)
    original.save_pretrained(tmp_path)
    loaded, info = SLMForCausalLM.from_pretrained(tmp_path, dtype=dtype, output_loading_info=True)
    loaded.eval()
    rope = loaded.model.rotary_emb
    assert rope.inv_freq.dtype == torch.float32
    assert torch.isfinite(rope.inv_freq).all()
    assert torch.equal(rope.inv_freq, RotaryEmbedding(loaded.config).inv_freq)
    assert not info.get("mismatched_keys")
    for name, value in loaded.state_dict().items():
        assert torch.equal(value, before[name].to(value.dtype)), name
    tokens = torch.tensor([[2, 8, 9, 10, 11]])
    with torch.no_grad():
        logits = loaded(tokens, use_cache=False).logits
        assert torch.isfinite(logits).all()
        sequences = []
        for cache in (False, True):
            sequences.append(loaded.generate(tokens, do_sample=False, max_new_tokens=4, use_cache=cache))
        assert torch.equal(sequences[0], sequences[1])
        if dtype == torch.float32:
            torch.testing.assert_close(logits, original(tokens, use_cache=False).logits, rtol=0, atol=0)


def test_scratch_training_and_dtype_conversion_keep_rope_derived():
    model = SLMForCausalLM(tiny_config())
    tokens = torch.tensor([[2, 8, 9, 10, 11]])
    loss = model(tokens, labels=tokens).loss
    assert torch.isfinite(loss)
    loss.backward()
    gradients = [p.grad for p in model.parameters() if p.grad is not None]
    assert gradients and all(torch.isfinite(g).all() for g in gradients)
    expected = model.model.rotary_emb.inv_freq.clone()
    model.bfloat16()
    assert model.model.rotary_emb.inv_freq.dtype == torch.float32
    assert torch.equal(model.model.rotary_emb.inv_freq, expected)


def test_to_empty_and_state_dict_loading_reconstruct_buffers():
    rope = RotaryEmbedding(tiny_config())
    expected = rope.inv_freq.clone()
    rope.to_empty(device="cpu")
    assert torch.equal(rope.inv_freq, expected)
    rope.inv_freq.fill_(float("nan"))
    rope.load_state_dict({}, strict=True)
    assert torch.equal(rope.inv_freq, expected)
