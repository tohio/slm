"""
inference/chat.py
------------------
Interactive chat CLI for SLM.

Maintains conversation history across turns and formats messages
using the SLM chat template. Supports streaming output.

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

DEFAULT_SYSTEM = "You are a helpful, harmless, and honest assistant."

COMMANDS = {
    "/reset":  "Clear conversation history",
    "/system": "Set a new system prompt (/system <prompt>)",
    "/history": "Show conversation history",
    "/quit":   "Exit",
    "/help":   "Show this help",
}


def format_prompt(messages: list[dict]) -> str:
    """Format conversation history into SLM chat template."""
    parts = []
    for msg in messages:
        role = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<|system|>{content}<|endofturn|>")
        elif role == "user":
            parts.append(f"<|user|>{content}<|endofturn|>")
        elif role == "assistant":
            parts.append(f"<|assistant|>{content}<|endofturn|>")
    parts.append("<|assistant|>")
    return "".join(parts)


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast

    local_path = Path(model_path)
    if local_path.exists():
        from model import SLMConfig, SLMForCausalLM
        AutoConfig.register("slm", SLMConfig)
        AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        torch_dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    tokenizer_path = local_path / "tokenizer" if (local_path / "tokenizer").exists() else model_path
    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))
    tokenizer.pad_token_id = 0
    tokenizer.eos_token_id = 3

    return model, tokenizer


def generate_response(
    model,
    tokenizer,
    messages: list[dict],
    max_new_tokens: int = 512,
    temperature: float = 0.7,
    top_p: float = 0.9,
    repetition_penalty: float = 1.1,
) -> str:
    """Generate a single assistant response."""
    import torch

    prompt = format_prompt(messages)
    inputs = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048).to(model.device)
    input_len = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=0,
            eos_token_id=3,
        )

    new_tokens = outputs[0][input_len:]
    response = tokenizer.decode(new_tokens, skip_special_tokens=True).strip()
    return response


def print_help():
    print("\nCommands:")
    for cmd, desc in COMMANDS.items():
        print(f"  {cmd:<12} {desc}")
    print()


def chat_loop(model, tokenizer, system_prompt: str, args):
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

        # Handle commands
        if user_input.startswith("/"):
            cmd = user_input.split()[0].lower()

            if cmd == "/quit":
                print("Goodbye.")
                break

            elif cmd == "/reset":
                messages = [{"role": "system", "content": system_prompt}]
                print("Conversation reset.\n")
                continue

            elif cmd == "/system":
                new_system = user_input[len("/system"):].strip()
                if new_system:
                    system_prompt = new_system
                    messages = [{"role": "system", "content": system_prompt}]
                    print(f"System prompt updated. Conversation reset.\n")
                else:
                    print(f"Current system: {system_prompt}\n")
                continue

            elif cmd == "/history":
                print("\nConversation history:")
                for msg in messages:
                    role = msg["role"].upper()
                    content = msg["content"][:100]
                    print(f"  [{role}]: {content}{'...' if len(msg['content']) > 100 else ''}")
                print()
                continue

            elif cmd == "/help":
                print_help()
                continue

            else:
                print(f"Unknown command: {cmd}. Type /help for commands.\n")
                continue

        # Add user message
        messages.append({"role": "user", "content": user_input})

        # Generate response
        try:
            print("Assistant: ", end="", flush=True)
            response = generate_response(
                model, tokenizer, messages,
                max_new_tokens=args.max_new_tokens,
                temperature=args.temperature,
                top_p=args.top_p,
            )
            print(response)
            print()
        except Exception as e:
            print(f"\n[Error: {e}]")
            messages.pop()  # Remove failed user message
            continue

        # Add assistant response to history
        messages.append({"role": "assistant", "content": response})

        # Warn if context is getting long
        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars > 6000:
            print("[Note: conversation is getting long — consider /reset to start fresh]\n")


def main():
    parser = argparse.ArgumentParser(description="SLM interactive chat")
    parser.add_argument("--model", type=str, required=True, help="Model path or Hub ID")
    parser.add_argument("--system", type=str, default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--max-new-tokens", type=int, default=512)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--top-p", type=float, default=0.9)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)
    chat_loop(model, tokenizer, args.system, args)


if __name__ == "__main__":
    main()