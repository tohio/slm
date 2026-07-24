#!/usr/bin/env python3
"""One-shot CUDA acceptance test with no dataset downloads."""

from __future__ import annotations

import math
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import torch

from config.runtime import configure_torch_runtime
from infra.verify_environment import verify_cuda, verify_versions
from model import SLMConfig, SLMForCausalLM


def _tiny_model() -> SLMForCausalLM:
    config = SLMConfig(
        vocab_size=256,
        hidden_size=128,
        intermediate_size=256,
        num_hidden_layers=2,
        num_attention_heads=8,
        num_key_value_heads=2,
        max_position_embeddings=128,
    )
    return SLMForCausalLM(config).to(device="cuda", dtype=torch.bfloat16)


def _training_step(model, input_ids: torch.Tensor, label: str) -> float:
    model.train()
    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=1.0e-4,
        fused=True,
    )
    optimizer.zero_grad(set_to_none=True)
    output = model(input_ids, labels=input_ids, use_cache=False)
    loss = output.loss
    if loss is None or not torch.isfinite(loss):
        raise RuntimeError(f"{label} produced non-finite loss: {loss}")
    loss.backward()
    for parameter in model.parameters():
        if parameter.grad is not None and not torch.isfinite(parameter.grad).all():
            raise RuntimeError(f"{label} produced a non-finite gradient")
    optimizer.step()
    return float(loss.detach())


def _verify_generation(model, prompt: torch.Tensor) -> None:
    model.eval()
    with torch.no_grad():
        cached = model.generate(
            prompt,
            max_new_tokens=4,
            do_sample=False,
            use_cache=True,
            pad_token_id=model.config.pad_token_id,
        )
        uncached = model.generate(
            prompt,
            max_new_tokens=4,
            do_sample=False,
            use_cache=False,
            pad_token_id=model.config.pad_token_id,
        )
    if not torch.equal(cached, uncached):
        raise RuntimeError(
            "Cached and uncached greedy generation diverged:\n"
            f"cached={cached.tolist()}\nuncached={uncached.tolist()}"
        )


def main() -> None:
    verify_versions()
    verify_cuda()
    configure_torch_runtime()
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.cuda.reset_peak_memory_stats()

    input_ids = torch.randint(4, 256, (2, 32), device="cuda")
    model = _tiny_model()

    started = time.perf_counter()
    eager_loss = _training_step(model, input_ids, "eager step")

    compiled_model = torch.compile(
        model,
        backend="inductor",
        mode="default",
    )
    compiled_loss = _training_step(compiled_model, input_ids, "compiled step")
    _verify_generation(model, input_ids[:1, :8])
    torch.cuda.synchronize()

    elapsed = time.perf_counter() - started
    peak_gb = torch.cuda.max_memory_allocated() / 1e9
    if not math.isfinite(peak_gb):
        raise RuntimeError(f"Invalid peak-memory measurement: {peak_gb}")

    print(f"eager_loss={eager_loss:.6f}")
    print(f"compiled_loss={compiled_loss:.6f}")
    print(f"peak_allocated_gb={peak_gb:.3f}")
    print(f"elapsed_seconds={elapsed:.2f}")
    print("GPU smoke test passed.")


if __name__ == "__main__":
    main()
