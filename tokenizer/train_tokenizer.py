"""
tokenizer/train_tokenizer.py
-----------------------------
Train a BPE tokenizer using HuggingFace tokenizers library.

Trains on the validated dataset to ensure the tokenizer vocabulary
reflects the actual cleaned data distribution. A domain-specific
tokenizer encodes the training text more efficiently than a generic
tokenizer (e.g. GPT-2's), reducing the number of tokens per document
and improving training efficiency.

Special tokens are baked in at training time — they cannot be added
later without retraining the tokenizer and resizing model embeddings.

Special tokens:
    Structural:   <PAD>, <UNK>, <BOS>, <EOS>
    Chat:         <|system|>, <|user|>, <|assistant|>, <|endofturn|>
    Code:         <|code|>, <|endofcode|>
    Tool use:     <|tool|>, <|endoftool|>
    Reasoning:    <|reasoning|>, <|endofreasoning|>
    RAG context:  <|context|>, <|endofcontext|>

Vocab size: 32,000 (sufficient for English + code at 125M–1B scale)

BOS/EOS handling:
    BOS and EOS are NOT added automatically by the tokenizer post-processor.
    They must be added explicitly by the training data pipeline where needed.
    Automatic injection via TemplateProcessing corrupts chat-formatted data
    by inserting tokens at positions the model does not expect.

Chat template:
    The Jinja2 chat template is baked into the HuggingFace tokenizer at
    save time so tokenizer.apply_chat_template() works correctly in SFT,
    DPO, and inference — matching the special token structure exactly.

    The template includes {% generation %} and {% endgeneration %} tags
    around assistant responses, required by trl's assistant_only_loss=True
    for answer-only loss masking in SFTTrainer.

Output:
    data/tokenizer/slm_tokenizer.json   — full tokenizer (HF format)
    data/tokenizer/vocab.json           — vocabulary
    data/tokenizer/merges.txt           — BPE merge rules
    data/tokenizer/special_tokens.json  — special token definitions
    data/tokenizer/tokenizer.json       — HuggingFace PreTrainedTokenizerFast
    data/tokenizer/tokenizer_config.json — includes chat_template

Usage:
    python tokenizer/train_tokenizer.py
    python tokenizer/train_tokenizer.py --vocab-size 32000
    python tokenizer/train_tokenizer.py --input data/validated/train.jsonl
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
TOKENIZER_DIR = DATA_DIR / "tokenizer"

# ── Special tokens ─────────────────────────────────────────────────────────────

SPECIAL_TOKENS = [
    # Structural — must be first for correct IDs
    "<PAD>",            # 0
    "<UNK>",            # 1
    "<BOS>",            # 2
    "<EOS>",            # 3
    # Chat
    "<|system|>",       # 4
    "<|user|>",         # 5
    "<|assistant|>",    # 6
    "<|endofturn|>",    # 7
    # Code
    "<|code|>",         # 8
    "<|endofcode|>",    # 9
    # Tool use
    "<|tool|>",         # 10
    "<|endoftool|>",    # 11
    # Reasoning
    "<|reasoning|>",    # 12
    "<|endofreasoning|>", # 13
    # RAG context
    "<|context|>",      # 14
    "<|endofcontext|>", # 15
]

# Token ID constants for use in training scripts
PAD_ID = 0
UNK_ID = 1
BOS_ID = 2
EOS_ID = 3
SYSTEM_ID = 4
USER_ID = 5
ASSISTANT_ID = 6
ENDOFTURN_ID = 7
CODE_ID = 8
ENDOFCODE_ID = 9
TOOL_ID = 10
ENDOFTOOL_ID = 11
REASONING_ID = 12
ENDOFREASONING_ID = 13
CONTEXT_ID = 14
ENDOFCONTEXT_ID = 15

# ── Chat template ──────────────────────────────────────────────────────────────
# Jinja2 template baked into the HuggingFace tokenizer so that
# tokenizer.apply_chat_template() produces the correct format in SFT,
# DPO, and inference.
#
# Format:
#   <BOS><|system|>{content}<|endofturn|>
#   <|user|>{content}<|endofturn|>
#   <|assistant|>{content}<EOS><|endofturn|>
#   <|assistant|>   ← generation prompt (add_generation_prompt=True)
#
# BOS is prepended once at the start of the full sequence.
# EOS is appended at the end of each assistant turn.
# No BOS/EOS on system or user turns — the role tokens are the delimiters.
#
# {% generation %} / {% endgeneration %} tags wrap the assistant response
# content. These are required by trl's assistant_only_loss=True for
# answer-only loss masking in SFTTrainer — trl uses them to identify
# which tokens to compute loss on.

CHAT_TEMPLATE = (
    "{{ bos_token }}"
    "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
            "<|system|>{{ message['content'] }}<|endofturn|>"
        "{% elif message['role'] == 'user' %}"
            "<|user|>{{ message['content'] }}<|endofturn|>"
        "{% elif message['role'] == 'assistant' %}"
            "<|assistant|>"
            "{% generation %}"
            "{{ message['content'] }}{{ eos_token }}<|endofturn|>"
            "{% endgeneration %}"
        "{% endif %}"
    "{% endfor %}"
    "{% if add_generation_prompt %}<|assistant|>{% endif %}"
)


# ── Text iterator ──────────────────────────────────────────────────────────────

def text_iterator(input_path: Path, batch_size: int = 1000):
    """
    Yield batches of text strings from a JSONL file.

    The HuggingFace tokenizers trainer expects an iterator of
    strings or lists of strings. We yield batches for efficiency.

    Args:
        input_path: Path to JSONL file with "text" field.
        batch_size: Number of texts per batch.
    """
    batch = []
    with open(input_path, encoding="utf-8") as f:
        for line in f:
            record = json.loads(line)
            text = record.get("text", "").strip()
            if text:
                batch.append(text)
            if len(batch) >= batch_size:
                yield batch
                batch = []
    if batch:
        yield batch


# ── Train ──────────────────────────────────────────────────────────────────────

def train_tokenizer(
    input_path: Path,
    output_dir: Path,
    vocab_size: int = 32_000,
    min_frequency: int = 2,
) -> None:
    """
    Train a BPE tokenizer on the validated dataset.

    Args:
        input_path: Path to validated JSONL file.
        output_dir: Directory to save tokenizer files.
        vocab_size: Target vocabulary size. Default: 32,000.
        min_frequency: Minimum token frequency to include in vocab.
    """
    from tokenizers import Tokenizer
    from tokenizers.models import BPE
    from tokenizers.trainers import BpeTrainer
    from tokenizers.pre_tokenizers import ByteLevel
    from tokenizers.decoders import ByteLevel as ByteLevelDecoder
    from tokenizers.normalizers import NFC

    output_dir.mkdir(parents=True, exist_ok=True)

    log.info(f"Training BPE tokenizer on {input_path}")
    log.info(f"Vocab size: {vocab_size:,}")
    log.info(f"Special tokens: {len(SPECIAL_TOKENS)}")

    # Build tokenizer
    tokenizer = Tokenizer(BPE(unk_token="<UNK>"))

    # NFC normalization — handles unicode composed/decomposed forms
    tokenizer.normalizer = NFC()

    # Byte-level pre-tokenizer — handles any unicode without UNK tokens.
    # Same as GPT-2's approach — every byte is representable.
    tokenizer.pre_tokenizer = ByteLevel(add_prefix_space=False)
    tokenizer.decoder = ByteLevelDecoder()

    # No post-processor — BOS/EOS are NOT added automatically.
    # They must be added explicitly by the data pipeline where needed.
    # Automatic injection via TemplateProcessing corrupts chat-formatted
    # data by inserting tokens at positions the model does not expect.

    # Trainer
    trainer = BpeTrainer(
        vocab_size=vocab_size,
        min_frequency=min_frequency,
        special_tokens=SPECIAL_TOKENS,
        show_progress=True,
        initial_alphabet=ByteLevel.alphabet(),
    )

    # Train
    log.info("Training...")
    tokenizer.train_from_iterator(
        text_iterator(input_path),
        trainer=trainer,
        length=None,
    )

    # Verify special token IDs are correct
    for expected_id, token in enumerate(SPECIAL_TOKENS):
        actual_id = tokenizer.token_to_id(token)
        if actual_id != expected_id:
            log.warning(
                f"Special token ID mismatch: {token} "
                f"expected={expected_id}, actual={actual_id}"
            )

    # Save raw tokenizer
    tokenizer_path = output_dir / "slm_tokenizer.json"
    tokenizer.save(str(tokenizer_path))
    log.info(f"Tokenizer saved to {tokenizer_path}")

    # Save vocab and merges separately for inspection
    tokenizer.model.save(str(output_dir))
    log.info(f"Vocab and merges saved to {output_dir}")

    # Save special token definitions
    special_tokens_path = output_dir / "special_tokens.json"
    with open(special_tokens_path, "w") as f:
        json.dump(
            {token: i for i, token in enumerate(SPECIAL_TOKENS)},
            f, indent=2
        )
    log.info(f"Special tokens saved to {special_tokens_path}")

    # Save as HuggingFace PreTrainedTokenizerFast with chat template
    _save_as_hf_tokenizer(tokenizer, output_dir)

    # Print vocab stats
    vocab = tokenizer.get_vocab()
    log.info(f"Final vocab size: {len(vocab):,}")
    log.info(f"Special tokens verified: {len(SPECIAL_TOKENS)}")


def _save_as_hf_tokenizer(tokenizer, output_dir: Path) -> None:
    """
    Save the tokenizer as a HuggingFace PreTrainedTokenizerFast.

    Bakes in the Jinja2 chat template including {% generation %} /
    {% endgeneration %} tags required by trl's assistant_only_loss=True.

    Args:
        tokenizer: Trained tokenizers.Tokenizer instance.
        output_dir: Directory to save the HuggingFace tokenizer.
    """
    try:
        from transformers import PreTrainedTokenizerFast

        hf_tokenizer = PreTrainedTokenizerFast(
            tokenizer_object=tokenizer,
            bos_token="<BOS>",
            eos_token="<EOS>",
            unk_token="<UNK>",
            pad_token="<PAD>",
            additional_special_tokens=SPECIAL_TOKENS[4:],
        )

        hf_tokenizer.chat_template = CHAT_TEMPLATE

        hf_tokenizer.save_pretrained(str(output_dir))
        log.info(f"HuggingFace tokenizer saved to {output_dir}")
        log.info(f"Chat template baked in — apply_chat_template() ready")
        log.info(f"Generation tags included — assistant_only_loss=True supported")

    except Exception as e:
        log.error(f"Could not save HF tokenizer: {e}")
        raise


# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Train SLM BPE tokenizer")
    parser.add_argument(
        "--input",
        type=Path,
        default=DATA_DIR / "validated" / "train.jsonl",
        help="Input JSONL file (default: data/validated/train.jsonl)",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=TOKENIZER_DIR,
        help="Output directory (default: data/tokenizer)",
    )
    parser.add_argument(
        "--vocab-size",
        type=int,
        default=32_000,
        help="Vocabulary size (default: 32000)",
    )
    parser.add_argument(
        "--min-frequency",
        type=int,
        default=2,
        help="Minimum token frequency (default: 2)",
    )
    args = parser.parse_args()

    if not args.input.exists():
        log.error(f"Input file not found: {args.input}")
        log.error("Run: python validation/scripts/validate.py")
        sys.exit(1)

    train_tokenizer(
        input_path=args.input,
        output_dir=args.output,
        vocab_size=args.vocab_size,
        min_frequency=args.min_frequency,
    )

    log.info("Tokenizer training complete.")
    log.info(f"Next step: python tokenizer/test_tokenizer.py")


if __name__ == "__main__":
    main()