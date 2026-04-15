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

# Token IDs — must match tokenizer special tokens
EOS_TOKEN_ID       = 3   # <EOS>
ENDOFTURN_TOKEN_ID = 7   # <|endofturn|> — model uses this to end turns
PAD_TOKEN_ID       = 0   # <PAD>


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
    tokenizer.pad_token_id = PAD_TOKEN_ID
    tokenizer.eos_token_id = ENDOFTURN_TOKEN_ID
    tokenizer.bos_token_id = 2
    return tokenizer


def load_model_and_tokenizer(model_path: str):
    """Load model and tokenizer from local path or Hub."""
    import torch
    from transformers import AutoConfig, AutoModelForCausalLM
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

    tokenizer = _load_tokenizer(model_path)

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

    Returns:
        List of generated completion strings (prompt stripped).
    """
    import torch

    inputs = tokenizer(
        prompts,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=2048,
        add_special_tokens=False,
    ).to(model.device)

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
            eos_token_id=[EOS_TOKEN_ID, ENDOFTURN_TOKEN_ID],  # stop on <EOS> or <|endofturn|>
        )

    input_lengths = inputs["input_ids"].shape[1]
    completions = []
    for output in outputs:
        new_tokens = output[input_lengths:]
        completion = tokenizer.decode(new_tokens, skip_special_tokens=True)
        completions.append(completion.strip())

    return completions


def main():
    parser = argparse.ArgumentParser(description="SLM text generation")
    parser.add_argument("--model",          type=str,  required=True, help="Model path or Hub ID")
    parser.add_argument("--input",          type=Path, default=None,  help="Input prompts file (one per line)")
    parser.add_argument("--output",         type=Path, default=None,  help="Output JSONL file")
    parser.add_argument("--max-new-tokens", type=int,  default=256)
    parser.add_argument("--temperature",    type=float, default=0.8)
    parser.add_argument("--top-p",          type=float, default=0.95)
    parser.add_argument("--top-k",          type=int,  default=50)
    parser.add_argument("--greedy",         action="store_true", help="Use greedy decoding")
    parser.add_argument("--batch-size",     type=int,  default=4)
    args = parser.parse_args()

    model, tokenizer = load_model_and_tokenizer(args.model)

    if args.input:
        prompts = [l.strip() for l in open(args.input) if l.strip()]
    else:
        prompts = [l.strip() for l in sys.stdin if l.strip()]

    if not prompts:
        log.error("No prompts provided")
        sys.exit(1)

    log.info(f"Generating {len(prompts)} completions...")

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