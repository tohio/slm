"""
inference/generate.py
----------------------
Batch text generation from a trained SLM checkpoint.

Supports greedy, top-p, and top-k sampling. Reads prompts from
stdin or a file and writes completions to stdout or a file.

Usage:
    # Single prompt
    echo "The history of AI" | python inference/generate.py --model results/slm-125m-dpo/final

    # From file
    python inference/generate.py \
        --model results/slm-125m-dpo/final \
        --input prompts.txt \
        --output completions.jsonl

    # From Hub
    python inference/generate.py --model tohio/slm-125m

    # Chat format — wrap prompts in the chat template
    python inference/generate.py \
        --model results/slm-125m-dpo/final \
        --chat \
        --input prompts.txt
"""

import argparse
import json
import logging
import sys
from pathlib import Path

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

# Token IDs for stop tokens — read from tokenizer config at load time,
# but also defined here as fallback constants for eos_token_id.
EOS_TOKEN_ID       = 3   # <EOS>
ENDOFTURN_TOKEN_ID = 7   # <|endofturn|> — model uses this to end turns
PAD_TOKEN_ID       = 0   # <PAD>


def load_model_and_tokenizer(model_path: str):
    """
    Load model and tokenizer from local path or Hub.

    Loads the tokenizer via PreTrainedTokenizerFast.from_pretrained() so
    special tokens and chat_template are read from the saved
    tokenizer_config.json rather than hardcoded constants. Do not
    reconstruct from tokenizer.json directly — that bypasses the config.
    """
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM, PreTrainedTokenizerFast
    from model import SLMConfig, SLMForCausalLM

    # Register unconditionally — required for both local and Hub loading
    AutoConfig.register("slm", SLMConfig)
    AutoModelForCausalLM.register(SLMConfig, SLMForCausalLM)

    log.info(f"Loading model from {model_path}...")
    model = AutoModelForCausalLM.from_pretrained(
        model_path,
        dtype=torch.bfloat16,
        device_map="auto",
    )
    model.eval()

    # Resolve tokenizer path — check for a tokenizer/ subdirectory first
    # (local checkpoints), then fall back to the model directory itself
    # (Hub checkpoints and exported models).
    local_path = Path(model_path)
    if local_path.exists():
        tokenizer_path = local_path / "tokenizer"
        if not (tokenizer_path / "tokenizer_config.json").exists():
            tokenizer_path = local_path
    else:
        tokenizer_path = model_path  # Hub ID

    tokenizer = PreTrainedTokenizerFast.from_pretrained(str(tokenizer_path))

    n_params = sum(p.numel() for p in model.parameters())
    log.info(f"Model loaded — {n_params / 1e6:.1f}M parameters")

    return model, tokenizer


def generate(
    model,
    tokenizer,
    prompts: list[str],
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
    chat: bool = False,
) -> list[str]:
    """
    Generate completions for a list of prompts.

    Args:
        model: Loaded model.
        tokenizer: Loaded tokenizer.
        prompts: List of input prompt strings.
        max_new_tokens: Maximum tokens to generate per prompt.
        temperature: Sampling temperature. 0 = greedy.
        top_p: Nucleus sampling probability.
        top_k: Top-k sampling.
        do_sample: Whether to sample. False = greedy.
        repetition_penalty: Penalize repeated tokens.
        chat: If True, wrap each prompt in the chat template as a user
              message before generating. Uses apply_chat_template() so
              the format matches what the model was trained on.

    Returns:
        List of generated completion strings (prompt stripped).
    """
    import torch

    if chat:
        # Wrap each prompt as a user message and apply the chat template.
        # This produces the same format used during SFT training.
        input_ids_list = [
            tokenizer.apply_chat_template(
                [{"role": "user", "content": p}],
                tokenize=True,
                add_generation_prompt=True,
                truncation=True,
                max_length=2048,
            )
            for p in prompts
        ]
        # Pad to same length for batch processing
        max_len = max(len(ids) for ids in input_ids_list)
        padded = [
            [PAD_TOKEN_ID] * (max_len - len(ids)) + ids
            for ids in input_ids_list
        ]
        import torch
        input_ids = torch.tensor(padded, dtype=torch.long).to(model.device)
        attention_mask = (input_ids != PAD_TOKEN_ID).long()
        inputs = {"input_ids": input_ids, "attention_mask": attention_mask}
    else:
        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=2048,
            add_special_tokens=False,
        ).to(model.device)

    input_length = inputs["input_ids"].shape[1]

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=PAD_TOKEN_ID,
            eos_token_id=[EOS_TOKEN_ID, ENDOFTURN_TOKEN_ID],
        )

    completions = []
    for output in outputs:
        new_tokens = output[input_length:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(completion.strip())

    return completions


def main():
    parser = argparse.ArgumentParser(description="SLM text generation")
    parser.add_argument("--model",          type=str,   required=True, help="Model path or Hub ID")
    parser.add_argument("--input",          type=Path,  default=None,  help="Input prompts file (one per line)")
    parser.add_argument("--output",         type=Path,  default=None,  help="Output JSONL file")
    parser.add_argument("--max-new-tokens", type=int,   default=256)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--top-p",          type=float, default=0.95)
    parser.add_argument("--top-k",          type=int,   default=50)
    parser.add_argument("--greedy",         action="store_true", help="Use greedy decoding")
    parser.add_argument("--batch-size",     type=int,   default=4)
    parser.add_argument(
        "--chat",
        action="store_true",
        help="Wrap prompts in the chat template as user messages (use for chat/instruct models)",
    )
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)

    if args.input:
        prompts = [l.strip() for l in open(args.input) if l.strip()]
    else:
        prompts = [l.strip() for l in sys.stdin if l.strip()]

    if not prompts:
        log.error("No prompts provided")
        sys.exit(1)

    log.info(f"Generating {len(prompts)} completions (chat={args.chat}, greedy={args.greedy})...")

    out_file = open(args.output, "w") if args.output else None

    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        completions = generate(
            model, tokenizer, batch,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=not args.greedy,
            chat=args.chat,
        )

        for prompt, completion in zip(batch, completions):
            record = {"prompt": prompt, "completion": completion}
            line = json.dumps(record, ensure_ascii=False)
            if out_file:
                out_file.write(line + "\n")
            else:
                print(f"\nPrompt:     {prompt}")
                print(f"Completion: {completion}")
                print()

    if out_file:
        out_file.close()
        log.info(f"Completions written to {args.output}")


if __name__ == "__main__":
    main()