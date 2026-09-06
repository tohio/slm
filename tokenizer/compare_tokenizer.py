"""
tokenizer/compare_tokenizer.py
------------------------------
Compare the size-specific SLM tokenizer with a Hugging Face reference tokenizer.

This is a diagnostic tool, not a pass/fail gate. It uses five fixed cases to
show token counts, token pieces, roundtrip behavior, and aggregate differences.

Usage:
    python tokenizer/compare_tokenizer.py --size mini
    python tokenizer/compare_tokenizer.py \
        --size mini \
        --reference HuggingFaceTB/SmolLM2-135M
"""

import argparse
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from config.paths import tokenizer_dir


DEFAULT_REFERENCE = "HuggingFaceTB/SmolLM2-135M"

TEST_CASES = [
    (
        "prose",
        "The quick brown fox jumps over the lazy dog while the sun sets behind the hills.",
    ),
    (
        "code",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n - 1) + fibonacci(n - 2)",
    ),
    (
        "numbers/url",
        "Visit https://example.com/report?id=42 on 2026-09-06; total cost: $1,249.95.",
    ),
    (
        "unicode",
        "Café résumé naïve façade — こんにちは世界 — مرحبا بالعالم",
    ),
    (
        "technical",
        "A transformer language model maps token embeddings through repeated attention and feed-forward blocks, then predicts the next token from the final hidden state.",
    ),
]


def load_slm_tokenizer(path: Path):
    from tokenizers import Tokenizer

    tokenizer_file = path / "slm_tokenizer.json"
    if not tokenizer_file.exists():
        raise FileNotFoundError(f"SLM tokenizer not found at {tokenizer_file}")
    return Tokenizer.from_file(str(tokenizer_file))


def load_reference_tokenizer(model_name: str):
    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(model_name)


def _preview(values, limit: int = 24) -> str:
    values = list(values)
    if len(values) <= limit:
        return repr(values)
    return repr(values[:limit])[:-1] + f", ...] ({len(values)} total)"


def encode_slm(tokenizer, text: str) -> dict:
    encoded = tokenizer.encode(text)
    decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
    return {
        "ids": encoded.ids,
        "tokens": encoded.tokens,
        "decoded": decoded,
        "roundtrip": decoded == text,
    }


def encode_reference(tokenizer, text: str) -> dict:
    ids = tokenizer.encode(text, add_special_tokens=False)
    tokens = tokenizer.convert_ids_to_tokens(ids)
    decoded = tokenizer.decode(
        ids,
        skip_special_tokens=True,
        clean_up_tokenization_spaces=False,
    )
    return {
        "ids": ids,
        "tokens": tokens,
        "decoded": decoded,
        "roundtrip": decoded == text,
    }


def print_case(label: str, text: str, slm: dict, reference: dict) -> None:
    slm_n = len(slm["ids"])
    ref_n = len(reference["ids"])
    delta = slm_n - ref_n
    delta_pct = (delta / ref_n * 100.0) if ref_n else 0.0

    print(f"\n=== {label} ===")
    print(f"Input: {text!r}")
    print()
    print(f"SLM ({slm_n} tokens, roundtrip={'yes' if slm['roundtrip'] else 'no'})")
    print(f"  pieces: {_preview(slm['tokens'])}")
    print(f"  ids:    {_preview(slm['ids'])}")
    print()
    print(
        f"Reference ({ref_n} tokens, roundtrip={'yes' if reference['roundtrip'] else 'no'})"
    )
    print(f"  pieces: {_preview(reference['tokens'])}")
    print(f"  ids:    {_preview(reference['ids'])}")
    print()
    print(f"Delta: {delta:+d} tokens ({delta_pct:+.1f}% vs reference)")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Compare the SLM tokenizer with a Hugging Face reference tokenizer"
    )
    parser.add_argument(
        "--size",
        choices=("smoke", "mini", "125m", "350m", "1b"),
        default=os.environ.get("SIZE"),
        help="SLM tokenizer size",
    )
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=None,
        help="Explicit SLM tokenizer directory; overrides --size path resolution",
    )
    parser.add_argument(
        "--reference",
        default=DEFAULT_REFERENCE,
        help=f"Hugging Face reference tokenizer (default: {DEFAULT_REFERENCE})",
    )
    args = parser.parse_args()

    if args.tokenizer is None and args.size is None:
        parser.error("--size is required unless --tokenizer is provided")

    slm_path = args.tokenizer or tokenizer_dir(args.size)
    slm_tokenizer = load_slm_tokenizer(slm_path)
    reference_tokenizer = load_reference_tokenizer(args.reference)

    print("Tokenizer comparison")
    print(f"  SLM:       {slm_path}")
    print(f"  Reference: {args.reference}")
    print(f"  Cases:     {len(TEST_CASES)}")
    print()
    print("This report is diagnostic only; lower token count is not an automatic pass/fail.")

    rows = []
    total_slm = 0
    total_ref = 0

    for label, text in TEST_CASES:
        slm_result = encode_slm(slm_tokenizer, text)
        ref_result = encode_reference(reference_tokenizer, text)
        print_case(label, text, slm_result, ref_result)

        slm_n = len(slm_result["ids"])
        ref_n = len(ref_result["ids"])
        total_slm += slm_n
        total_ref += ref_n
        rows.append((label, slm_n, ref_n, slm_n - ref_n))

    print("\n=== Summary ===")
    print(f"{'Case':<14} {'SLM':>7} {'Reference':>11} {'Delta':>8}")
    print("-" * 43)
    for label, slm_n, ref_n, delta in rows:
        print(f"{label:<14} {slm_n:>7} {ref_n:>11} {delta:>+8}")
    print("-" * 43)
    print(f"{'total':<14} {total_slm:>7} {total_ref:>11} {total_slm - total_ref:>+8}")
    print(
        f"{'average':<14} {total_slm / len(rows):>7.1f} "
        f"{total_ref / len(rows):>11.1f} "
        f"{(total_slm - total_ref) / len(rows):>+8.1f}"
    )


if __name__ == "__main__":
    main()
