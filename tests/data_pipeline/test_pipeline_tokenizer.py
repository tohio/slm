"""
tests/data_pipeline/test_pipeline_tokenizer.py
-----------------------------------------------
Validates real outputs from 'make tokenizer'.

Run after: make tokenizer
Command:   make test-tokenizer

Checks:
    - Tokenizer files exist (slm_tokenizer.json, tokenizer_config.json)
    - All 16 special tokens exist with correct IDs
    - Encode/decode roundtrip on sample texts
    - No auto BOS/EOS injection
    - Fertility < 1.5 tokens/word on validated data
    - Chat template works via apply_chat_template()
    - BOS appears exactly once at start of chat-formatted output
"""

import json
from pathlib import Path

import pytest

from tests.conftest import requires_stage, read_jsonl, pipeline_path

# Import special tokens and ID constants from the training module, not
# hand-copies. train_tokenizer.py runs an import-time consistency assertion
# between SPECIAL_TOKENS ordering and the ID constants, so the single
# import below is safe — drift fails at import, not silently at assert time.
from tokenizer.train_tokenizer import SPECIAL_TOKENS, BOS_ID, EOS_ID, PAD_ID
# CODE_SOURCES is the authoritative "which sources are code" set — imported
# from config/data_mix.py where it lives alongside DATA_MIX. Tokenizer prose
# fertility tests also exclude symbol-heavy generated sources that are not
# representative natural-language prose.
from config import CODE_SOURCES

NON_PROSE_FERTILITY_SOURCES = set(CODE_SOURCES) | {"synthetic_arithmetic"}


pytestmark = requires_stage("tokenizer")


def load_raw_tokenizer():
    from tokenizers import Tokenizer
    path = pipeline_path("tokenizer", "slm_tokenizer.json")
    return Tokenizer.from_file(str(path))


def load_hf_tokenizer():
    from transformers import PreTrainedTokenizerFast
    return PreTrainedTokenizerFast.from_pretrained(str(pipeline_path("tokenizer")))


class TestTokenizerFiles:
    def test_tokenizer_dir_exists(self):
        assert pipeline_path("tokenizer").exists()

    def test_slm_tokenizer_json_exists(self):
        assert pipeline_path("tokenizer", "slm_tokenizer.json").exists()

    def test_tokenizer_config_json_exists(self):
        assert pipeline_path("tokenizer", "tokenizer_config.json").exists()

    def test_vocab_json_exists(self):
        assert pipeline_path("tokenizer", "vocab.json").exists()

    def test_special_tokens_json_exists(self):
        assert pipeline_path("tokenizer", "special_tokens.json").exists()


class TestSpecialTokens:
    def test_all_special_tokens_present_with_correct_ids(self):
        tokenizer = load_raw_tokenizer()
        failures = []
        for expected_id, token in enumerate(SPECIAL_TOKENS):
            actual_id = tokenizer.token_to_id(token)
            if actual_id != expected_id:
                failures.append(f"{token}: expected {expected_id}, got {actual_id}")
        assert len(failures) == 0, "Special token ID mismatches:\n" + "\n".join(failures)

    def test_vocab_size_is_32000(self):
        tokenizer = load_raw_tokenizer()
        assert len(tokenizer.get_vocab()) == 32000


class TestRoundtrip:
    @pytest.mark.parametrize("text", [
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n):\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "Machine learning is a subset of artificial intelligence.",
        "1 + 1 = 2",
    ])
    def test_roundtrip(self, text):
        tokenizer = load_raw_tokenizer()
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
        assert text.strip() == decoded.strip(), (
            f"Roundtrip failed:\n  input:  {repr(text)}\n  output: {repr(decoded)}"
        )


class TestNoBosEosAutoInjection:
    def test_no_auto_bos_injection(self):
        tokenizer = load_raw_tokenizer()
        encoded = tokenizer.encode("Hello world")
        first_token = tokenizer.id_to_token(encoded.ids[0])
        assert first_token != "<BOS>", (
            "Tokenizer is auto-injecting BOS — "
            "remove TemplateProcessing from train_tokenizer.py"
        )

    def test_no_auto_eos_injection(self):
        tokenizer = load_raw_tokenizer()
        encoded = tokenizer.encode("Hello world")
        last_token = tokenizer.id_to_token(encoded.ids[-1])
        assert last_token != "<EOS>", (
            "Tokenizer is auto-injecting EOS — "
            "remove TemplateProcessing from train_tokenizer.py"
        )


class TestFertility:
    def test_fertility_below_threshold(self):
        """Fertility should be < 1.5 tokens/word on natural language text."""
        tokenizer = load_raw_tokenizer()

        # Load sample from validated data
        validated_path = pipeline_path("validated", "train.jsonl")
        if not validated_path.exists():
            pytest.skip("validated/train.jsonl not found — run make validate first")

        # Filter out code and symbol-heavy generated sources. Fertility here
        # measures natural-language prose compression, not code/math density.
        sample_texts = []
        with open(validated_path) as f:
            for i, line in enumerate(f):
                if i >= 500:
                    break
                doc = json.loads(line)
                if doc.get("source") in NON_PROSE_FERTILITY_SOURCES:
                    continue
                sample_texts.append(doc.get("text", ""))

        if not sample_texts:
            pytest.skip("No natural language samples found in validated data")

        total_tokens = sum(len(tokenizer.encode(t).ids) for t in sample_texts)
        total_words = sum(len(t.split()) for t in sample_texts)
        fertility = total_tokens / max(total_words, 1)

        assert fertility < 1.5, (
            f"Tokenizer fertility too high: {fertility:.3f} tokens/word "
            f"(threshold: 1.5). Consider increasing vocab_size."
        )


class TestChatTemplate:
    def test_apply_chat_template_runs(self):
        hf_tokenizer = load_hf_tokenizer()
        messages = [
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "What is 2 + 2?"},
            {"role": "assistant", "content": "2 + 2 = 4."},
        ]
        result = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        assert isinstance(result, str)
        assert len(result) > 0

    def test_bos_appears_exactly_once_at_start(self):
        hf_tokenizer = load_hf_tokenizer()
        messages = [
            {"role": "user", "content": "Hello"},
            {"role": "assistant", "content": "Hi there!"},
        ]
        result = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        bos = hf_tokenizer.bos_token
        assert result.startswith(bos), f"Chat output does not start with BOS token ({bos})"
        assert result.count(bos) == 1, (
            f"BOS appears {result.count(bos)} times — expected exactly 1"
        )

    def test_generation_prompt_ends_with_assistant_token(self):
        hf_tokenizer = load_hf_tokenizer()
        messages = [{"role": "user", "content": "Hello"}]
        result = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        assert result.endswith("<|assistant|>"), (
            f"Generation prompt does not end with <|assistant|>. "
            f"Last 50 chars: {repr(result[-50:])}"
        )

    def test_chat_special_tokens_present(self):
        hf_tokenizer = load_hf_tokenizer()
        messages = [
            {"role": "system", "content": "You are helpful."},
            {"role": "user", "content": "Hi"},
            {"role": "assistant", "content": "Hello!"},
        ]
        result = hf_tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=False
        )
        for token in ["<|system|>", "<|user|>", "<|assistant|>", "<|endofturn|>"]:
            assert token in result, f"Expected token {token} not found in chat output"

    def test_first_tokenized_id_is_bos(self):
        hf_tokenizer = load_hf_tokenizer()
        messages = [{"role": "user", "content": "Hello"}]
        result = hf_tokenizer.apply_chat_template(
            messages, tokenize=True, add_generation_prompt=False
        )
        # Normalise to a flat list of ints — apply_chat_template can return:
        #   - list[int]           (standard)
        #   - list[Encoding]      (some tokenizer versions)
        #   - Encoding            (rare)
        if hasattr(result, "ids"):
            token_ids = result.ids
        elif result and hasattr(result[0], "ids"):
            token_ids = result[0].ids
        else:
            token_ids = list(result)
        assert token_ids[0] == BOS_ID, (
            f"First token ID is {token_ids[0]}, expected BOS_ID={BOS_ID}"
        )
class TestSpecialTokensDistinct:
    def test_bos_eos_pad_have_distinct_ids(self):
        assert BOS_ID != EOS_ID, f"BOS_ID and EOS_ID are both {BOS_ID}"
        assert BOS_ID != PAD_ID, f"BOS_ID and PAD_ID are both {BOS_ID}"
        assert EOS_ID != PAD_ID, f"EOS_ID and PAD_ID are both {EOS_ID}"


class TestRoundtripBreadth:
    """
    Domain-breadth roundtrip. The TestRoundtrip class above covers common
    cases; these catch silent information loss on edge cases that matter
    for training data quality.
    """
    @pytest.mark.parametrize("text,label", [
        # Punctuation-heavy
        ('Dr. Smith said, "It\'s 3:14 p.m.—let\'s go!"', "punctuation"),
        # URL + number + date (common in web text)
        ("Visit https://example.com on 2024-03-15 for $49.99.", "url_number_date"),
        # Whitespace — leading, trailing, repeated
        ("  leading spaces\n\n\ttabs and\n    newlines  ", "whitespace"),
        # Unicode / multilingual
        ("Café résumé naïve façade — 你好 — مرحبا", "unicode"),
        # Code with common operators
        ("x = [i**2 for i in range(10) if i % 2 == 0]", "code_operators"),
        # Markdown-ish formatting common in training data
        ("## Section\n\n- **bold** item\n- _italic_ item\n\n```python\ncode\n```", "markdown"),
    ])
    def test_roundtrip_breadth(self, text, label):
        tokenizer = load_raw_tokenizer()
        encoded = tokenizer.encode(text)
        decoded = tokenizer.decode(encoded.ids, skip_special_tokens=True)
        # Whitespace-only differences are expected (decoders often normalize);
        # the test is that content survives, not exact byte equality.
        assert text.strip() == decoded.strip(), (
            f"Roundtrip failed on {label}:\n"
            f"  input:  {repr(text)}\n"
            f"  output: {repr(decoded)}"
        )


class TestTokenizedBinIntegrity:
    """
    After tokenize_data.py runs, sanity-check that the resulting bin files
    contain valid IDs and decode to real text. Catches bugs where the
    tokenizer is fine but the tokenization pipeline (batching, special
    token insertion, EOS separators) is broken.
    """
    def test_tokenized_bin_ids_in_vocab_range(self):
        tokenized_dir = pipeline_path("tokenized")
        if not (tokenized_dir / "train.bin").exists():
            pytest.skip("train.bin not found — run make tokenize first")
        
        import numpy as np
        train = np.memmap(tokenized_dir / "train.bin", dtype=np.uint16, mode="r")
        # Sample 10k random positions rather than scanning the whole file
        import random
        rng = random.Random(42)
        positions = rng.sample(range(len(train)), min(10_000, len(train)))
        sample = train[positions]
        
        assert sample.max() < 32000, f"Token ID {sample.max()} exceeds vocab_size 32000"
        assert sample.min() >= 0, f"Negative token ID found: {sample.min()}"

    def test_tokenized_bin_decodes_to_real_text(self):
        tokenized_dir = pipeline_path("tokenized")
        if not (tokenized_dir / "train.bin").exists():
            pytest.skip("train.bin not found — run make tokenize first")
        
        import numpy as np
        tokenizer = load_hf_tokenizer()
        train = np.memmap(tokenized_dir / "train.bin", dtype=np.uint16, mode="r")
        
        # Decode three 200-token windows from different positions
        positions = [0, len(train) // 2, max(0, len(train) - 200)]
        for pos in positions:
            window = train[pos:pos + 200].tolist()
            decoded = tokenizer.decode(window, skip_special_tokens=True)
            # Real text has letters. Garbage bytes interpreted as IDs would
            # decode to mostly control chars or single-byte pieces.
            alpha_ratio = sum(c.isalpha() for c in decoded) / max(len(decoded), 1)
            assert alpha_ratio > 0.3, (
                f"Decoded window at position {pos} is {alpha_ratio:.1%} alphabetic — "
                f"expected >30%. Sample: {repr(decoded[:100])}"
            )


class TestFertilityBaseline:
    """
    Compare compression against a reference tokenizer on the same text.
    Flags if your tokenizer is significantly less efficient than a
    well-trained public tokenizer — which would mean more training tokens
    needed to learn the same patterns.
    """
    def test_compression_within_range_of_reference(self):
        validated_path = pipeline_path("validated", "train.jsonl")
        if not validated_path.exists():
            pytest.skip("validated/train.jsonl not found")
        
        try:
            from transformers import AutoTokenizer
            # Mistral's tokenizer is vocab=32000 (same as ours) and widely
            # validated. If you don't have network access in CI, skip.
            ref_tok = AutoTokenizer.from_pretrained(
                "mistralai/Mistral-7B-v0.1", use_fast=True
            )
        except Exception as e:
            pytest.skip(f"Reference tokenizer unavailable: {e}")
        
        our_tok = load_raw_tokenizer()
        
        # Collect 200 prose samples
        samples = []
        with open(validated_path) as f:
            for i, line in enumerate(f):
                if len(samples) >= 200:
                    break
                doc = json.loads(line)
                if doc.get("source") in NON_PROSE_FERTILITY_SOURCES:
                    continue
                samples.append(doc.get("text", ""))
        
        total_chars = sum(len(s) for s in samples)
        our_tokens = sum(len(our_tok.encode(s).ids) for s in samples)
        ref_tokens = sum(len(ref_tok.encode(s, add_special_tokens=False)) for s in samples)
        
        our_cpt = total_chars / our_tokens
        ref_cpt = total_chars / ref_tokens
        ratio = our_cpt / ref_cpt  # >1 means ours is more efficient, <1 less
        
        # Fail if we're more than 20% less efficient than reference.
        # Some gap is expected (Mistral was trained on way more data),
        # but >20% suggests a real problem.
        assert ratio > 0.80, (
            f"Tokenizer efficiency vs Mistral:\n"
            f"  ours:      {our_cpt:.2f} chars/token\n"
            f"  reference: {ref_cpt:.2f} chars/token\n"
            f"  ratio:     {ratio:.2f} (threshold: > 0.80)\n"
            f"Your tokenizer is significantly less efficient than the reference. "
            f"Consider retraining on more data or inspecting the training corpus."
        )