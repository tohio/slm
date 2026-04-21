"""
inference/generate.py
----------------------
Batch text generation from a trained SLM checkpoint.

Supports greedy, top-p, and top-k sampling. Reads prompts from
stdin or a file and writes completions to stdout or a file.

Special-token IDs and model+tokenizer loading are delegated to
inference.utils — IDs come from the loaded tokenizer, not hardcoded.

For batched chat, left-padding is performed by tokenizer.pad() rather than
a hand-rolled loop; this guarantees correct attention masks and interacts
properly with the model's positional-encoding path.

Usage:
    # Single prompt (base model completion)
    echo "The history of AI" | python inference/generate.py --model results/slm-125m/final

    # From file
    python inference/generate.py \\
        --model results/slm-125m-dpo/final \\
        --input prompts.txt \\
        --output completions.jsonl \\
        --chat

    # From Hub
    python inference/generate.py --model tohio/slm-125m

    # Raw completion without a BOS prefix
    python inference/generate.py --model results/slm-125m/final --no-bos
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

from inference.utils import load_model_and_tokenizer


def _prepare_batch(tokenizer, prompts: list[str], *, chat: bool, add_bos: bool):
    """
    Tokenize and left-pad a batch of prompts for generation.

    Left-padding via tokenizer.pad is the correct, battle-tested path for
    causal-LM batch generation: attention masks are produced automatically
    and pad tokens are placed where the model won't attend to them, so
    content positions start at consistent logical offsets in each row.

    chat=True wraps each prompt as a single user message through
    apply_chat_template — the format the model was trained on.
    chat=False tokenizes the prompt raw, with BOS prepended by default
    (the base model saw BOS at sequence start during pretraining).
    """
    if chat:
        encoded = [
            {
                "input_ids": tokenizer.apply_chat_template(
                    [{"role": "user", "content": p}],
                    tokenize=True,
                    add_generation_prompt=True,
                    truncation=True,
                    max_length=2048,
                )
            }
            for p in prompts
        ]
    else:
        encoded = [
            {
                "input_ids": tokenizer(
                    p,
                    truncation=True,
                    max_length=2048,
                    add_special_tokens=add_bos,
                )["input_ids"]
            }
            for p in prompts
        ]

    # Left-pad via the tokenizer so attention_mask is produced correctly.
    # Save and restore padding_side in case the caller set it elsewhere.
    previous_side = tokenizer.padding_side
    tokenizer.padding_side = "left"
    try:
        batch = tokenizer.pad(
            encoded,
            padding=True,
            return_tensors="pt",
        )
    finally:
        tokenizer.padding_side = previous_side

    return batch


def generate(
    model,
    tokenizer,
    special_ids,
    prompts: list[str],
    *,
    max_new_tokens: int = 256,
    temperature: float = 0.8,
    top_p: float = 0.95,
    top_k: int = 50,
    do_sample: bool = True,
    repetition_penalty: float = 1.1,
    chat: bool = False,
    add_bos: bool = True,
) -> list[str]:
    """
    Generate completions for a list of prompts.

    Returns completions with input prompts stripped. See module docstring
    for parameter notes.
    """
    import torch

    batch = _prepare_batch(tokenizer, prompts, chat=chat, add_bos=add_bos)
    input_ids      = batch["input_ids"].to(model.device)
    attention_mask = batch["attention_mask"].to(model.device)
    input_length   = input_ids.shape[1]

    with torch.no_grad():
        outputs = model.generate(
            input_ids=input_ids,
            attention_mask=attention_mask,
            max_new_tokens=max_new_tokens,
            temperature=temperature if do_sample else 1.0,
            top_p=top_p if do_sample else 1.0,
            top_k=top_k if do_sample else 0,
            do_sample=do_sample,
            repetition_penalty=repetition_penalty,
            pad_token_id=special_ids.pad,
            eos_token_id=special_ids.eos_list,
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
    parser.add_argument(
        "--no-bos",
        action="store_true",
        help="(raw mode only) Do not prepend BOS. Default prepends BOS, matching pretraining.",
    )
    parser.add_argument(
        "--dtype",
        type=str,
        default="bfloat16",
        choices=["bfloat16", "float16", "float32"],
        help="Model precision (default: bfloat16)",
    )
    args = parser.parse_args()

    # For chat mode, require_chat_template=True catches missing templates.
    # For raw mode, the template is irrelevant — still required to be present
    # so downstream users of the checkpoint don't get silently-broken chat.
    model, tokenizer, special_ids = load_model_and_tokenizer(args.model, dtype=args.dtype)

    if args.input:
        prompts = [l.strip() for l in open(args.input) if l.strip()]
    else:
        prompts = [l.strip() for l in sys.stdin if l.strip()]

    if not prompts:
        log.error("No prompts provided")
        sys.exit(1)

    log.info(
        f"Generating {len(prompts)} completions "
        f"(chat={args.chat}, greedy={args.greedy}, add_bos={not args.no_bos})..."
    )

    out_file = open(args.output, "w") if args.output else None

    for i in range(0, len(prompts), args.batch_size):
        batch = prompts[i : i + args.batch_size]
        completions = generate(
            model, tokenizer, special_ids, batch,
            max_new_tokens=args.max_new_tokens,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=args.top_k,
            do_sample=not args.greedy,
            chat=args.chat,
            add_bos=not args.no_bos,
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