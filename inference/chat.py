"""
inference/chat.py
------------------
Interactive chat CLI for SLM.

Maintains conversation history across turns and formats messages
using the SLM chat template via tokenizer.apply_chat_template().
This is the same code path used by SFTTrainer and DPOTrainer during
training — inference must use it too for the model to respond correctly.

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

# Token IDs for stop tokens — read from tokenizer config, but also defined
# here as fallback constants for the eos_token_id list passed to generate().
EOS_TOKEN_ID       = 3   # <EOS>
ENDOFTURN_TOKEN_ID = 7   # <|endofturn|> — model uses this to end assistant turns
PAD_TOKEN_ID       = 0   # <PAD>

COMMANDS = {
    "/reset":   "Clear conversation history",
    "/system":  "Set a new system prompt (/system <prompt>)",
    "/history": "Show conversation history",
    "/quit":    "Exit",
    "/help":    "Show this help",
}


def load_model_and_tokenizer(model_path: str):
    """
    Load model and tokenizer from local path or Hub.

    Loads the tokenizer via PreTrainedTokenizerFast.from_pretrained() so
    the baked-in chat_template from train_tokenizer.py is available for
    apply_chat_template(). Do not reconstruct from tokenizer.json directly
    as that bypasses tokenizer_config.json and loses the chat template.

    Always registers SLMConfig and SLMForCausalLM with AutoConfig and
    AutoModelForCausalLM — required whether loading from a local path or
    the HuggingFace Hub, since the custom model_type 'slm' is not known
    to transformers out of the box.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
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

    # Resolve tokenizer path — check for a tokenizer/ subdirectory first
    # (local checkpoints saved by train_sft.py), then fall back to the
    # model directory itself (Hub checkpoints and exported models).
    local_path = Path(model_path)
    if local_path.exists():
        tokenizer_path = local_path / "tokenizer"
        if not (tokenizer_path / "tokenizer_config.json").exists():
            tokenizer_path = local_path
    else:
        tokenizer_path = model_path  # Hub ID — from_pretrained handles it

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    if not getattr(tokenizer, "chat_template", None):
        raise ValueError(
            f"Tokenizer at {tokenizer_path} has no chat_template. "
            f"Retrain the tokenizer: python tokenizer/train_tokenizer.py"
        )

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
    """
    Generate a single assistant response.

    Uses tokenizer.apply_chat_template() to format the conversation —
    the same code path used during SFT and DPO training.
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

    in_len = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids,
            max_new_tokens=max_new_tokens,
            temperature=temperature,
            top_p=top_p,
            do_sample=True,
            repetition_penalty=repetition_penalty,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=[EOS_TOKEN_ID, ENDOFTURN_TOKEN_ID],
        )

    new_tokens = outputs[0][in_len:].tolist()
    # Strip trailing stop tokens if included in output
    for stop_id in [ENDOFTURN_TOKEN_ID, EOS_TOKEN_ID]:
        if stop_id in new_tokens:
            new_tokens = new_tokens[:new_tokens.index(stop_id)]

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
    parser.add_argument("--model",          type=str,   required=True, help="Model path or Hub ID")
    parser.add_argument("--system",         type=str,   default=DEFAULT_SYSTEM, help="System prompt")
    parser.add_argument("--max-new-tokens", type=int,   default=512)
    parser.add_argument("--temperature",    type=float, default=0.7)
    parser.add_argument("--top-p",          type=float, default=0.9)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)
    chat_loop(model, tokenizer, args.system, args)


if __name__ == "__main__":
    main()