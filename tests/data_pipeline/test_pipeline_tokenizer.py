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

from tests.conftest import DATA_DIR, requires_stage, read_jsonl, pipeline_path

# Import special tokens and ID constants from the training module, not
# hand-copies. train_tokenizer.py runs an import-time consistency assertion
# between SPECIAL_TOKENS ordering and the ID constants, so the single
# import below is safe — drift fails at import, not silently at assert time.
from tokenizer.train_tokenizer import SPECIAL_TOKENS, BOS_ID, EOS_ID, PAD_ID
# CODE_SOURCES is the authoritative "which sources are code" set — imported
# from config/data_mix.py where it lives alongside DATA_MIX. Previously we
# hand-maintained a `source != "code"` filter, which was dead code under
# the 10-source mix (no source is literally named "code").
from config import CODE_SOURCES


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

        # Filter out code sources using the authoritative set from config.
        # The previous filter was `source != "code"` which is dead — no
        # document has source="code" under the 10-source mix.
        sample_texts = []
        with open(validated_path) as f:
            for i, line in enumerate(f):
                if i >= 500:
                    break
                doc = json.loads(line)
                if doc.get("source") in CODE_SOURCES:
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