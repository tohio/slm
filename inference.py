"""
inference.py
------------
Interactive and batch inference against any SLM checkpoint.

Supports all pipeline stages — load pretrain, SFT, or DPO checkpoints
and compare outputs directly. Useful for:
  - Verifying model behavior after each training stage
  - Generating proof-of-deployment samples for documentation
  - Side-by-side comparison across checkpoints
  - Batch generation from a prompt file

Usage:
  # Interactive mode (default — talks to the model)
  python inference.py --checkpoint /results/slm_dpo/checkpoints/last.nemo

  # Single prompt
  python inference.py --checkpoint /results/slm_dpo/checkpoints/last.nemo \\
      --prompt "Explain recursion to a 10-year-old."

  # Compare two checkpoints on the same prompt
  python inference.py \\
      --checkpoint /results/slm_dpo/checkpoints/last.nemo \\
      --compare   /results/slm_sft_code/checkpoints/last.nemo \\
      --prompt "Write a Python function to check if a number is prime."

  # Batch from file (one prompt per line)
  python inference.py --checkpoint /results/slm_dpo/checkpoints/last.nemo \\
      --prompt-file prompts.txt --output results.jsonl

  # Adjust generation params
  python inference.py --checkpoint /results/slm_dpo/checkpoints/last.nemo \\
      --temperature 0.9 --max-tokens 512 --top-p 0.95
"""

import argparse
import json
import logging
import sys
import time
from pathlib import Path

import torch

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("inference")

# Default generation params
DEFAULT_MAX_TOKENS   = 512
DEFAULT_TEMPERATURE  = 0.7
DEFAULT_TOP_P        = 0.9
DEFAULT_TOP_K        = 50

# Chat template — wraps raw prompts in the conversation format
# Must match the special tokens baked into the tokenizer
SYSTEM_PROMPT = "You are a helpful, honest, and harmless AI assistant."

STOP_TOKENS = ["<|endofturn|>", "<|user|>", "<|system|>"]


def format_prompt(user_input: str, system: str = SYSTEM_PROMPT) -> str:
    """Wrap a raw user message in the chat template."""
    return f"<|system|>{system}<|user|>{user_input}<|assistant|>"


def load_model(checkpoint: str, device: str = "cuda"):
    """Load a NeMo GPT model from a .nemo checkpoint."""
    from nemo.collections.nlp.models.language_modeling.megatron_gpt_model import MegatronGPTModel

    logger.info(f"Loading checkpoint: {checkpoint}")
    t0 = time.time()

    model = MegatronGPTModel.restore_from(
        restore_path=checkpoint,
        map_location=torch.device(device),
    )
    model.eval()
    model.to(device)

    elapsed = time.time() - t0
    param_count = sum(p.numel() for p in model.parameters()) / 1e6
    logger.info(f"Model loaded in {elapsed:.1f}s — {param_count:.0f}M parameters")
    return model


def generate(
    model,
    prompt: str,
    max_new_tokens: int = DEFAULT_MAX_TOKENS,
    temperature: float = DEFAULT_TEMPERATURE,
    top_p: float = DEFAULT_TOP_P,
    top_k: int = DEFAULT_TOP_K,
    device: str = "cuda",
) -> tuple[str, dict]:
    """
    Generate a response from a prompt.
    Returns (response_text, metadata).
    """
    tokenizer = model.tokenizer
    input_ids = tokenizer.text_to_ids(prompt)
    n_prompt_tokens = len(input_ids)

    input_tensor = torch.tensor([input_ids], dtype=torch.long, device=device)

    t0 = time.time()
    with torch.no_grad():
        output_ids = model.generate(
            input_ids=input_tensor,
            max_length=n_prompt_tokens + max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            top_k=top_k,
            do_sample=temperature > 0,
            pad_token_id=tokenizer.pad_id,
            eos_token_id=tokenizer.eos_id,
        )
    elapsed = time.time() - t0

    generated_ids = output_ids[0][n_prompt_tokens:].tolist()
    response = tokenizer.ids_to_text(generated_ids)

    # Trim at stop tokens
    for stop in STOP_TOKENS:
        if stop in response:
            response = response[:response.index(stop)]
    response = response.strip()

    n_generated = len(generated_ids)
    tokens_per_sec = n_generated / elapsed if elapsed > 0 else 0

    meta = {
        "prompt_tokens":    n_prompt_tokens,
        "generated_tokens": n_generated,
        "elapsed_sec":      round(elapsed, 2),
        "tokens_per_sec":   round(tokens_per_sec, 1),
        "temperature":      temperature,
        "top_p":            top_p,
    }
    return response, meta


def interactive_mode(model, args, device: str):
    """REPL — type prompts, get responses."""
    print("\n" + "=" * 60)
    print("  SLM Interactive Inference")
    print(f"  Checkpoint: {args.checkpoint}")
    print(f"  Temperature: {args.temperature}  Max tokens: {args.max_tokens}")
    print("  Type 'quit' or Ctrl+C to exit")
    print("  Prefix with '/raw ' to skip chat template")
    print("=" * 60 + "\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (KeyboardInterrupt, EOFError):
            print("\nExiting.")
            break

        if not user_input:
            continue
        if user_input.lower() in ("quit", "exit", "q"):
            break

        # Allow raw prompts for debugging
        if user_input.startswith("/raw "):
            prompt = user_input[5:]
        else:
            prompt = format_prompt(user_input)

        print("Assistant: ", end="", flush=True)
        response, meta = generate(
            model, prompt,
            max_new_tokens=args.max_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            device=device,
        )
        print(response)
        print(
            f"\n  [{meta['generated_tokens']} tokens · "
            f"{meta['tokens_per_sec']:.1f} tok/s · "
            f"{meta['elapsed_sec']}s]\n"
        )


def single_prompt_mode(model, args, device: str):
    """Generate a single response and print it."""
    prompt = format_prompt(args.prompt)
    response, meta = generate(
        model, prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        top_k=args.top_k,
        device=device,
    )
    print(f"\nPrompt:   {args.prompt}")
    print(f"Response: {response}")
    print(
        f"\n[{meta['generated_tokens']} tokens · "
        f"{meta['tokens_per_sec']:.1f} tok/s · "
        f"{meta['elapsed_sec']}s]"
    )
    return response, meta


def compare_mode(model_a, model_b, args, device: str):
    """Generate responses from two checkpoints on the same prompt."""
    prompt = format_prompt(args.prompt)

    print(f"\nPrompt: {args.prompt}\n")
    print("─" * 60)

    print(f"[A] {Path(args.checkpoint).parent.parent.name}")
    response_a, meta_a = generate(
        model_a, prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    print(response_a)
    print(f"    [{meta_a['generated_tokens']} tokens · {meta_a['tokens_per_sec']:.1f} tok/s]\n")

    print("─" * 60)
    print(f"[B] {Path(args.compare).parent.parent.name}")
    response_b, meta_b = generate(
        model_b, prompt,
        max_new_tokens=args.max_tokens,
        temperature=args.temperature,
        top_p=args.top_p,
        device=device,
    )
    print(response_b)
    print(f"    [{meta_b['generated_tokens']} tokens · {meta_b['tokens_per_sec']:.1f} tok/s]")


def batch_mode(model, args, device: str):
    """Run inference on all prompts in a file, write JSONL output."""
    prompt_file = Path(args.prompt_file)
    if not prompt_file.exists():
        logger.error(f"Prompt file not found: {prompt_file}")
        sys.exit(1)

    prompts = [l.strip() for l in prompt_file.read_text().splitlines() if l.strip()]
    logger.info(f"Running batch inference on {len(prompts)} prompts → {args.output}")

    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with open(output_path, "w", encoding="utf-8") as fout:
        for i, user_input in enumerate(prompts):
            prompt = format_prompt(user_input)
            response, meta = generate(
                model, prompt,
                max_new_tokens=args.max_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
                top_k=args.top_k,
                device=device,
            )
            result = {
                "id":       i,
                "prompt":   user_input,
                "response": response,
                **meta,
            }
            fout.write(json.dumps(result, ensure_ascii=False) + "\n")

            logger.info(
                f"[{i+1}/{len(prompts)}] {user_input[:60]}{'...' if len(user_input) > 60 else ''}"
            )
            logger.info(f"  → {response[:100]}{'...' if len(response) > 100 else ''}")

    logger.info(f"Batch complete. Results written to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="SLM Inference")
    parser.add_argument("--checkpoint",   required=True,
                        help="Path to .nemo checkpoint (pretrain, SFT, or DPO)")
    parser.add_argument("--compare",      default=None,
                        help="Second checkpoint to compare against (enables compare mode)")
    parser.add_argument("--prompt",       default=None,
                        help="Single prompt string (skips interactive mode)")
    parser.add_argument("--prompt-file",  default=None,
                        help="File of prompts, one per line (batch mode)")
    parser.add_argument("--output",       default="/results/inference/batch_results.jsonl",
                        help="Output file for batch mode")
    parser.add_argument("--system",       default=SYSTEM_PROMPT,
                        help="System prompt override")
    parser.add_argument("--temperature",  type=float, default=DEFAULT_TEMPERATURE)
    parser.add_argument("--max-tokens",   type=int,   default=DEFAULT_MAX_TOKENS)
    parser.add_argument("--top-p",        type=float, default=DEFAULT_TOP_P)
    parser.add_argument("--top-k",        type=int,   default=DEFAULT_TOP_K)
    parser.add_argument("--device",       default="cuda")
    parser.add_argument("--no-chat-template", action="store_true",
                        help="Pass prompts raw without wrapping in chat template")
    args = parser.parse_args()

    if not Path(args.checkpoint).exists():
        logger.error(f"Checkpoint not found: {args.checkpoint}")
        sys.exit(1)

    # Override format_prompt if --no-chat-template
    if args.no_chat_template:
        global format_prompt
        format_prompt = lambda text, system=None: text

    # Override system prompt
    if args.system != SYSTEM_PROMPT:
        _system = args.system
        _orig = format_prompt
        format_prompt = lambda text, system=_system: _orig(text, system)

    # Load primary model
    model = load_model(args.checkpoint, device=args.device)

    # Route to the right mode
    if args.prompt_file:
        batch_mode(model, args, args.device)

    elif args.compare:
        if not Path(args.compare).exists():
            logger.error(f"Compare checkpoint not found: {args.compare}")
            sys.exit(1)
        model_b = load_model(args.compare, device=args.device)
        compare_mode(model, model_b, args, args.device)

    elif args.prompt:
        single_prompt_mode(model, args, args.device)

    else:
        interactive_mode(model, args, args.device)


if __name__ == "__main__":
    main()