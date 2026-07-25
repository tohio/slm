"""Bounded offline vLLM smoke test for a native SLM export."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Load a native SLM export with vLLM and generate one response",
    )
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--max-model-len", type=int, default=256)
    parser.add_argument("--gpu-memory-utilization", type=float, default=0.5)
    args = parser.parse_args()

    if not args.model.is_dir():
        parser.error(f"model directory does not exist: {args.model}")
    if not (args.model / "config.json").is_file():
        parser.error(f"native config.json is missing: {args.model}")

    try:
        from transformers import AutoTokenizer
        from vllm import LLM, SamplingParams
    except ImportError as exc:
        raise RuntimeError(
            "vLLM is not installed in this environment. Run this smoke test "
            "inside the supported vLLM serving environment."
        ) from exc

    tokenizer = AutoTokenizer.from_pretrained(
        args.model,
        trust_remote_code=False,
        local_files_only=True,
    )
    prompt = tokenizer.apply_chat_template(
        [
            {"role": "system", "content": "Answer clearly and concisely."},
            {"role": "user", "content": "What is the capital of France?"},
        ],
        tokenize=False,
        add_generation_prompt=True,
    )

    engine = LLM(
        model=str(args.model),
        trust_remote_code=False,
        dtype="bfloat16",
        max_model_len=args.max_model_len,
        gpu_memory_utilization=args.gpu_memory_utilization,
        generation_config="vllm",
    )
    outputs = engine.generate(
        [prompt],
        SamplingParams(
            temperature=0.0,
            max_tokens=16,
        ),
    )
    if len(outputs) != 1 or not outputs[0].outputs:
        raise RuntimeError("vLLM returned no generation output")

    completion = outputs[0].outputs[0]
    if not completion.token_ids:
        raise RuntimeError("vLLM returned an empty token sequence")

    print(
        "vLLM smoke passed: "
        f"{len(completion.token_ids)} generated token(s); "
        f"text={completion.text.strip()!r}"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
