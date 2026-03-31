"""
tokenizer/test_tokenizer.py
-----------------------------
Validate the trained tokenizer — special tokens, encoding/decoding
roundtrip, fertility (tokens per word), and chat template formatting.

Usage:
    python tokenizer/test_tokenizer.py
    python tokenizer/test_tokenizer.py --tokenizer data/tokenizer
"""

import argparse
import json
import logging
import os
import sys
from pathlib import Path

from dotenv import load_dotenv

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))

from tokenizer.train_tokenizer import SPECIAL_TOKENS, BOS_ID, EOS_ID, PAD_ID, UNK_ID


def load_tokenizer(tokenizer_dir: Path):
    """Load the trained tokenizer."""
    from tokenizers import Tokenizer
    path = tokenizer_dir / "slm_tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {path}")
    return Tokenizer.from_file(str(path))


def test_special_tokens(tokenizer) -> bool:
    """Verify all special tokens exist with correct IDs."""
    log.info("=== Special Token Verification ===")
    all_passed = True
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        actual_id = tokenizer.token_to_id(token)
        status = "✓" if actual_id == expected_id else "✗"
        if actual_id != expected_id:
            all_passed = False
        print(f"  {status} {token:<25} expected={expected_id:>2}, actual={actual_id}")
    return all_passed


def test_roundtrip(tokenizer) -> bool:
    """Test encode/decode roundtrip on sample texts."""
    log.info("=== Encode/Decode Roundtrip ===")
    test_cases = [
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "Machine learning is a subset of artificial intelligence.",
        "こんにちは世界",  # Japanese — should still encode via byte fallback
        "1 + 1 = 2",
        "   ",  # whitespace
    ]

    all_passed = True
    for text in test_cases:
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
        passed = text.strip() == decoded.strip()
        status = "✓" if passed else "✗"
        if not passed:
            all_passed = False
        n_tokens = len(encoded.ids)
        print(f"  {status} [{n_tokens:>3} tokens] {repr(text[:50])}")
        if not passed:
            print(f"      Expected: {repr(text[:80])}")
            print(f"      Got:      {repr(decoded[:80])}")

    return all_passed


def test_fertility(tokenizer, sample_texts: list[str]) -> float:
    """
    Compute tokenizer fertility — tokens per word.

    Lower fertility = more efficient tokenizer.
    GPT-2 (BPE, 50k vocab): ~1.3 tokens/word on English
    SentencePiece (32k):    ~1.4 tokens/word on English
    Target: < 1.5 tokens/word
    """
    log.info("=== Fertility (tokens per word) ===")
    total_tokens = 0
    total_words = 0

    for text in sample_texts[:1000]:
        words = text.split()
        if not words:
            continue
        tokens = tokenizer.encode(text).ids
        total_tokens += len(tokens)
        total_words += len(words)

    fertility = total_tokens / max(total_words, 1)
    print(f"  Total words:   {total_words:,}")
    print(f"  Total tokens:  {total_tokens:,}")
    print(f"  Fertility:     {fertility:.3f} tokens/word")

    if fertility < 1.5:
        print(f"  ✓ Good fertility (< 1.5)")
    else:
        print(f"  ✗ High fertility (> 1.5) — consider larger vocab")

    return fertility


def test_chat_template(tokenizer) -> None:
    """Test chat template formatting with special tokens."""
    log.info("=== Chat Template ===")

    def format_chat(messages: list[dict]) -> str:
        """Format a list of messages into a chat string."""
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
        parts.append("<|assistant|>")  # prompt model to respond
        return "".join(parts)

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Germany?"},
    ]

    chat_string = format_chat(messages)
    encoded = tokenizer.encode(chat_string)

    print(f"  Chat string:\n{chat_string}")
    print(f"\n  Tokens: {len(encoded.ids)}")
    print(f"  Token IDs (first 20): {encoded.ids[:20]}")

    # Verify special tokens appear correctly
    special_in_output = [
        tokenizer.id_to_token(tid)
        for tid in encoded.ids
        if tokenizer.id_to_token(tid) in SPECIAL_TOKENS
    ]
    print(f"  Special tokens found: {special_in_output}")


def test_vocab_coverage(tokenizer) -> None:
    """Print vocabulary statistics."""
    log.info("=== Vocabulary Stats ===")
    vocab = tokenizer.get_vocab()

    print(f"  Vocab size:         {len(vocab):,}")
    print(f"  Special tokens:     {len(SPECIAL_TOKENS)}")
    print(f"  Regular tokens:     {len(vocab) - len(SPECIAL_TOKENS):,}")

    # Sample some tokens
    items = list(vocab.items())
    items.sort(key=lambda x: x[1])
    print(f"\n  First 20 tokens (by ID):")
    for token, tid in items[:20]:
        print(f"    {tid:>5}: {repr(token)}")


def main():
    parser = argparse.ArgumentParser(description="Test SLM tokenizer")
    parser.add_argument(
        "--tokenizer",
        type=Path,
        default=DATA_DIR / "tokenizer",
        help="Tokenizer directory",
    )
    parser.add_argument(
        "--sample-data",
        type=Path,
        default=DATA_DIR / "validated" / "train.jsonl",
        help="Sample data for fertility test",
    )
    args = parser.parse_args()

    tokenizer = load_tokenizer(args.tokenizer)
    log.info(f"Loaded tokenizer from {args.tokenizer}")

    # Run tests
    special_ok = test_special_tokens(tokenizer)
    roundtrip_ok = test_roundtrip(tokenizer)

    # Load sample texts for fertility
    sample_texts = []
    if args.sample_data.exists():
        with open(args.sample_data) as f:
            for i, line in enumerate(f):
                if i >= 1000:
                    break
                record = json.loads(line)
                if record.get("source") != "code":  # test on natural language
                    sample_texts.append(record.get("text", ""))
    else:
        sample_texts = ["The quick brown fox jumps over the lazy dog."] * 100

    fertility = test_fertility(tokenizer, sample_texts)
    test_chat_template(tokenizer)
    test_vocab_coverage(tokenizer)

    # Summary
    print("\n=== Summary ===")
    print(f"  Special tokens: {'✓ PASS' if special_ok else '✗ FAIL'}")
    print(f"  Roundtrip:      {'✓ PASS' if roundtrip_ok else '✗ FAIL'}")
    print(f"  Fertility:      {fertility:.3f} tokens/word {'✓' if fertility < 1.5 else '✗'}")


if __name__ == "__main__":
    main()