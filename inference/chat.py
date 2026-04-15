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

# Token IDs for stop tokens — must match tokenizer special tokens
EOS_TOKEN_ID      = 3   # <EOS>
ENDOFTURN_TOKEN_ID = 7  # <|endofturn|> — used by model to end assistant turns
PAD_TOKEN_ID      = 0   # <PAD>

COMMANDS = {
    "/reset":   "Clear conversation history",
    "/system":  "Set a new system prompt (/system <prompt>)",
    "/history": "Show conversation history",
    "/quit":    "Exit",
    "/help":    "Show this help",
}


def format_prompt(messages: list[dict]) -> str:
    """Format conversation history into SLM chat template."""
    parts = []
    for msg in messages:
        role    = msg["role"]
        content = msg["content"]
        if role == "system":
            parts.append(f"<|system|>{content}<|endofturn|>")
        elif role == "user":
            parts.append(f"<|user|>{content}<|endofturn|>")
        elif role == "assistant":
            parts.append(f"<|assistant|>{content}<|endofturn|>")
    parts.append("<|assistant|>")
    return "".join(parts)


def _load_tokenizer(model_path: str):
    """
    Load tokenizer from local path or Hub, bypassing TokenizersBackend.

    PreTrainedTokenizerFast.from_pretrained() fails when tokenizer_config.json
    references TokenizersBackend, an unknown class in transformers.
    Load directly from tokenizer.json via the tokenizers library instead.
    """
    from tokenizers import Tokenizer as HFTokenizer
    from transformers import PreTrainedTokenizerFast

    local_path = Path(model_path)

    if local_path.exists():
        candidates = [
            local_path / "tokenizer" / "tokenizer.json",
            local_path / "tokenizer.json",
        ]
        tokenizer_file = next((p for p in candidates if p.exists()), None)
        if tokenizer_file is None:
            raise FileNotFoundError(f"tokenizer.json not found in {local_path}")
        _tok = HFTokenizer.from_file(str(tokenizer_file))
    else:
        from huggingface_hub import hf_hub_download
        tokenizer_file = hf_hub_download(repo_id=model_path, filename="tokenizer.json")
        _tok = HFTokenizer.from_file(tokenizer_file)

    tokenizer = PreTrainedTokenizerFast(tokenizer_object=_tok)
    tokenizer.pad_token_id  = PAD_TOKEN_ID
    tokenizer.eos_token_id  = ENDOFTURN_TOKEN_ID  # <|endofturn|> ends assistant turns
    tokenizer.bos_token_id  = 2
    return tokenizer


def load_model_and_tokenizer(model_path: str):
    """
    Load model and tokenizer.

    Always registers SLMConfig and SLMForCausalLM with AutoConfig and
    AutoModelForCausalLM — required whether loading from a local path or
    the HuggingFace Hub, since the custom model_type 'slm' is not known
    to transformers out of the box.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
    from model import SLMConfig, SLMForCausalLM

    # Register unconditionally — required for both local and Hub loading
    AutoConfig.register("slm", SLMConfig)
    AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)

    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    tokenizer = _load_tokenizer(model_path)
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

    prompt  = format_prompt(messages)
    inputs  = tokenizer(prompt, return_tensors="pt", truncation=True, max_length=2048, add_special_tokens=False).to(model.device)
    in_len  = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=[EOS_TOKEN_ID, ENDOFTURN_TOKEN_ID],  # stop on <EOS> or <|endofturn|>
        )

    new_tokens = outputs[0][in_len:]
    return tokenizer.decode(new_tokens, skip_special_tokens=True).strip()


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
                    print(f"  [{msg['role'].upper()}]: {content}{'...' if len(msg['content']) > 100 else ''}")
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
                model, tokenizer, messages,
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

        total_chars = sum(len(m["content"]) for m in messages)
        if total_chars > 6000:
            print("[Note: conversation is getting long — consider /reset to start fresh]\n")


def main():
    parser = argparse.ArgumentParser(description="SLM interactive chat")
    parser.add_argument("--model",          type=str,  required=True, help="Model path or Hub ID")
    parser.add_argument("--system",         type=str,  default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--max-new-tokens", type=int,  default=512)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--top-p",          type=float, default=0.9)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)
    chat_loop(model, tokenizer, args.system, args)


if __name__ == "__main__":
    main()