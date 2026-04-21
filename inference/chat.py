"""
inference/chat.py
------------------
Interactive chat CLI for SLM.

Maintains conversation history across turns and formats messages
using the SLM chat template via tokenizer.apply_chat_template().
This is the same code path used by SFTTrainer and DPOTrainer during
training — inference must use it too for the model to respond correctly.

Special-token IDs (PAD, EOS, ENDOFTURN) are read from the loaded tokenizer
via inference.utils.resolve_special_token_ids() — never hardcoded here.

Usage:
    python inference/chat.py --model results/slm-125m-dpo/final
    python inference/chat.py --model tohio/slm-125m
    python inference/chat.py --model results/slm-125m-dpo/final --system "You are a coding assistant."
"""

import argparse
import logging
import sys
from pathlib import Path

logging.basicConfig(level=logging.WARNING)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from inference.utils import load_model_and_tokenizer

DEFAULT_SYSTEM = "You are a helpful, harmless, and honest assistant."

COMMANDS = {
    "/reset":   "Clear conversation history",
    "/system":  "Set a new system prompt (/system <prompt>)",
    "/history": "Show conversation history",
    "/quit":    "Exit",
    "/help":    "Show this help",
}


def generate_response(
    model,
    tokenizer,
    special_ids,
    messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """
    Generate a single assistant response.

    Uses tokenizer.apply_chat_template() to format the conversation —
    the same code path used during SFT and DPO training.

    Passes an explicit attention_mask even though there's no padding
    (single-prompt, single-batch) — avoids the transformers warning
    and keeps the call-site robust if we ever extend to batched chat.
    """
    import torch

    input_ids = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    attention_mask = torch.ones_like(input_ids)
    in_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=special_ids.pad,
            eos_token_id=special_ids.eos_list,
        )

    new_tokens = outputs[0][in_len:].tolist()
    # Strip trailing stop tokens if included in output
    for stop_id in special_ids.eos_list:
        if stop_id in new_tokens:
            new_tokens = new_tokens[: new_tokens.index(stop_id)]

    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


def print_help():
    print("\nCommands:")
    for cmd, desc in COMMANDS.items():
        print(f"  {cmd:<12} {desc}")
    print()


def _context_usage(tokenizer, messages: list[dict], model) -> tuple[int, int]:
    """
    Return (current_tokens, max_context_tokens) for the given conversation.
    Used for the "conversation getting long" warning, measured against actual
    model context rather than a hardcoded character count.
    """
    token_ids = tokenizer.apply_chat_template(
        messages, tokenize=True, add_generation_prompt=False,
    )
    current = len(token_ids)
    max_ctx = getattr(model.config, "max_position_embeddings", 2048)
    return current, max_ctx


def chat_loop(model, tokenizer, special_ids, system_prompt: str, args):
    """Main interactive chat loop."""
    messages = [{"role": "system", "content": system_prompt}]

    print(f"\n{'='*55}")
    print("  SLM Chat")
    print(f"{'='*55}")
    print(f"  Model:  {args.model}")
    print(f"  System: {system_prompt[:60]}{'...' if len(system_prompt) > 60 else ''}")
    print(f"  Type /help for commands, /quit to exit")
    print(f"{'='*55}\n")

    while True:
        try:
            user_input = input("You: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\nGoodbye.")
            break

        if not user_input:
            continue

        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()

            if cmd == "/quit":
                print("Goodbye.")
                break
            elif cmd == "/reset":
                messages = [{"role": "system", "content": system_prompt}]
                print("Conversation reset.\n")
            elif cmd == "/system":
                new_system = user_input[len("/system"):].strip()
                if new_system:
                    system_prompt = new_system
                    messages = [{"role": "system", "content": system_prompt}]
                    print(f"System prompt updated. Conversation reset.\n")
                else:
                    print(f"Current system: {system_prompt}\n")
            elif cmd == "/history":
                print("\nConversation history:")
                for msg in messages:
                    content = msg["content"][:100]
                    print(f"  [{msg['role'].upper()}]: {content}"
                          f"{'...' if len(msg['content']) > 100 else ''}")
                print()
            elif cmd == "/help":
                print_help()
            else:
                print(f"Unknown command: {cmd}. Type /help for commands.\n")
            continue

        messages.append({"role": "user", "content": user_input})

        try:
            print("Assistant: ", end="", flush=True)
            response = generate_response(
                model, tokenizer, special_ids, messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(response)
            print()
        except Exception as e:
            print(f"\n[Error: {e}]")
            messages.pop()
            continue

        messages.append({"role": "assistant", "content": response})

        # Warn at 75% of context window, measured in actual tokens.
        current, max_ctx = _context_usage(tokenizer, messages, model)
        if current > 0.75 * max_ctx:
            print(
                f"[Note: conversation uses {current}/{max_ctx} tokens "
                f"({100 * current / max_ctx:.0f}%) — consider /reset]\n"
            )


def main():
    parser = argparse.ArgumentParser(description="SLM interactive chat")
    parser.add_argument("--model",          type=str,   required=True, help="Model path or Hub ID")
    parser.add_argument("--system",         type=str,   default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--max-new-tokens", type=int,   default=512)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--top-p",          type=float, default=0.9)
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model precision (default: bfloat16)",
    )
    args = parser.parse_args()

    model, tokenizer, special_ids = load_model_and_tokenizer(args.model, dtype=args.dtype)
    chat_loop(model, tokenizer, special_ids, args.system, args)


if __name__ == "__main__":
    main()