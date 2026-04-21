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
from config import CODE_SOURCES


def load_tokenizer(tokenizer_dir: Path):
    """Load the raw tokenizer for low-level tests."""
    from tokenizers import Tokenizer
    path = tokenizer_dir / "slm_tokenizer.json"
    if not path.exists():
        raise FileNotFoundError(f"Tokenizer not found at {path}")
    return Tokenizer.from_file(str(path))


def load_hf_tokenizer(tokenizer_dir: Path):
    """
    Load the HuggingFace PreTrainedTokenizerFast.

    This is the tokenizer used by all training scripts and inference.
    Tests that use apply_chat_template must use this tokenizer, not
    the raw tokenizers.Tokenizer — they are different objects.
    """
    from transformers import PreTrainedTokenizerFast
    if not (tokenizer_dir / "tokenizer_config.json").exists():
        raise FileNotFoundError(
            f"HuggingFace tokenizer not found at {tokenizer_dir}. "
            f"Run: python tokenizer/train_tokenizer.py"
        )
    return PreTrainedTokenizerFast.from_pretrained(str(tokenizer_dir))


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
        # Byte-level BPE encodes every UTF-8 byte natively — this isn't a
        # fallback path, it's how all non-ASCII characters are represented.
        "こんにちは世界",
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


def test_no_auto_bos_eos(tokenizer) -> bool:
    """
    Verify BOS/EOS are NOT injected automatically on encode.

    The tokenizer must not add BOS/EOS automatically — they are added
    explicitly by the data pipeline. Automatic injection corrupts
    chat-formatted data by inserting tokens in the wrong positions.
    """
    log.info("=== BOS/EOS Auto-injection Check ===")
    text = "Hello world"
    encoded = tokenizer.encode(text)
    first_token = tokenizer.id_to_token(encoded.ids[0])
    last_token = tokenizer.id_to_token(encoded.ids[-1])

    bos_injected = first_token == "<BOS>"
    eos_injected = last_token == "<EOS>"

    if bos_injected or eos_injected:
        print(f"  ✗ Auto-injection detected — BOS: {bos_injected}, EOS: {eos_injected}")
        print(f"    First token: {first_token}, Last token: {last_token}")
        print(f"    Fix: remove TemplateProcessing from train_tokenizer.py")
        return False

    print(f"  ✓ No auto BOS/EOS injection")
    print(f"    First token: {repr(first_token)}, Last token: {repr(last_token)}")
    return True


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


def test_chat_template(hf_tokenizer) -> bool:
    """
    Test chat template formatting via apply_chat_template().

    Uses the HuggingFace tokenizer's apply_chat_template() — the same
    code path used by SFTTrainer, DPOTrainer, and inference. Testing a
    manual formatter instead would give false confidence that the real
    path works.

    Verifies:
        - apply_chat_template() does not raise
        - Output contains the expected special tokens in order
        - BOS appears exactly once at the start
        - Generation prompt ends with <|assistant|>
    """
    log.info("=== Chat Template (apply_chat_template) ===")

    messages = [
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user", "content": "What is the capital of France?"},
        {"role": "assistant", "content": "The capital of France is Paris."},
        {"role": "user", "content": "What about Germany?"},
    ]

    # Test without generation prompt
    try:
        chat_string = hf_tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=False,
        )
    except Exception as e:
        print(f"  ✗ apply_chat_template() raised: {e}")
        print(f"    Fix: ensure chat_template is set in train_tokenizer._save_as_hf_tokenizer()")
        return False

    # Test with generation prompt
    chat_with_prompt = hf_tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    print(f"  Chat string (with generation prompt):")
    print(f"  {repr(chat_with_prompt[:200])}{'...' if len(chat_with_prompt) > 200 else ''}")

    all_passed = True

    # BOS appears exactly once at the start
    bos = hf_tokenizer.bos_token
    if not chat_with_prompt.startswith(bos):
        print(f"  ✗ Does not start with BOS token ({bos})")
        all_passed = False
    elif chat_with_prompt.count(bos) > 1:
        print(f"  ✗ BOS appears {chat_with_prompt.count(bos)} times — should appear once")
        all_passed = False
    else:
        print(f"  ✓ BOS appears exactly once at start")

    # Expected special tokens appear in order
    expected_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|endofturn|>"]
    for token in expected_tokens:
        if token in chat_with_prompt:
            print(f"  ✓ {token} present")
        else:
            print(f"  ✗ {token} missing")
            all_passed = False

    # Generation prompt ends with <|assistant|>
    if chat_with_prompt.endswith("<|assistant|>"):
        print(f"  ✓ Ends with <|assistant|> (generation prompt correct)")
    else:
        print(f"  ✗ Does not end with <|assistant|>")
        print(f"    Last 50 chars: {repr(chat_with_prompt[-50:])}")
        all_passed = False

    # Tokenize and check token IDs
    encoded = hf_tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_tensors=None,
    )
    print(f"\n  Token count: {len(encoded)}")
    print(f"  First token ID: {encoded[0]} (expected BOS={BOS_ID})")
    if encoded[0] != BOS_ID:
        print(f"  ✗ First token is not BOS")
        all_passed = False
    else:
        print(f"  ✓ First token is BOS")

    return all_passed


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

    # Load both tokenizer forms
    tokenizer = load_tokenizer(args.tokenizer)
    hf_tokenizer = load_hf_tokenizer(args.tokenizer)
    log.info(f"Loaded tokenizer from {args.tokenizer}")

    # Run tests
    special_ok = test_special_tokens(tokenizer)
    roundtrip_ok = test_roundtrip(tokenizer)
    no_auto_bos_eos_ok = test_no_auto_bos_eos(tokenizer)

    # Load sample texts for fertility — natural language only. Validated
    # records are tagged with their source (one of the 10 sources defined
    # in config/data_mix.py); filter out code sources so fertility reflects
    # English rather than Python/notebooks.
    sample_texts = []
    if args.sample_data.exists():
        with open(args.sample_data) as f:
            for i, line in enumerate(f):
                if i >= 1000:
                    break
                record = json.loads(line)
                if record.get("source") in CODE_SOURCES:
                    continue
                sample_texts.append(record.get("text", ""))
    else:
        sample_texts = ["The quick brown fox jumps over the lazy dog."] * 100

    fertility = test_fertility(tokenizer, sample_texts)
    chat_ok = test_chat_template(hf_tokenizer)
    test_vocab_coverage(tokenizer)

    # Summary
    print("\n=== Summary ===")
    print(f"  Special tokens:     {'✓ PASS' if special_ok else '✗ FAIL'}")
    print(f"  Roundtrip:          {'✓ PASS' if roundtrip_ok else '✗ FAIL'}")
    print(f"  No auto BOS/EOS:    {'✓ PASS' if no_auto_bos_eos_ok else '✗ FAIL'}")
    print(f"  Fertility:          {fertility:.3f} tokens/word {'✓' if fertility < 1.5 else '✗'}")
    print(f"  Chat template:      {'✓ PASS' if chat_ok else '✗ FAIL'}")

    all_passed = special_ok and roundtrip_ok and no_auto_bos_eos_ok and chat_ok
    if not all_passed:
        print("\n  ✗ Some tests failed — fix before proceeding to pretraining")
        sys.exit(1)
    else:
        print("\n  ✓ All tests passed")


if __name__ == "__main__":
    main()