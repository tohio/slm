"""
train_tokenizer.py
------------------
Trains a custom BPE tokenizer using SentencePiece on a sample
of the curated dataset.

Why train a custom tokenizer?
  - Better token efficiency for your specific data mix (general + code)
  - Bakes in domain-specific special tokens from the start
  - Pre-training and inference use the same vocabulary

Workflow:
  1. Sample text from curated JSONL files (no need to use all data)
  2. Train SentencePiece BPE model
  3. Validate tokenizer on sample sentences
  4. Save model + vocab

Usage:
    python train_tokenizer.py --config ../configs/tokenizer.yaml
                              --input-dir /data/curated/stages/pii
                              --output-dir /data/tokenizer
                              [--sample-size 10000000]
"""

import argparse
import logging
import json
import os
import random
from pathlib import Path

import yaml
import sentencepiece as spm

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("tokenizer.train")


def get_num_threads(config_value: int | None = None) -> int:
    """
    Determine number of threads to use for tokenizer training.
    Uses all available CPUs by default — tokenizer training is
    embarrassingly parallel and benefits from full CPU utilization.
    Config value is used as an override if explicitly set to a non-None value.
    """
    cpu_count = os.cpu_count() or 1
    if config_value is not None and config_value > 0:
        threads = min(config_value, cpu_count)
        if config_value > cpu_count:
            logger.warning(
                f"num_threads={config_value} in config exceeds available CPUs ({cpu_count}). "
                f"Using {cpu_count}."
            )
    else:
        threads = cpu_count
    logger.info(f"Using {threads} threads for tokenizer training (available CPUs: {cpu_count})")
    return threads


def sample_text_from_jsonl(
    input_dir: Path,
    output_file: Path,
    sample_size: int,
    seed: int = 42,
) -> int:
    """
    Sample up to `sample_size` sentences from JSONL files.
    Writes one sentence per line to output_file for SentencePiece training.
    Returns actual number of lines written.
    """
    random.seed(seed)
    output_file.parent.mkdir(parents=True, exist_ok=True)

    all_docs = []
    for jsonl_file in sorted(input_dir.glob("*.jsonl")):
        with open(jsonl_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if line:
                    all_docs.append(line)

    logger.info(f"Found {len(all_docs):,} documents total")

    # Shuffle for representative sample
    random.shuffle(all_docs)

    lines_written = 0
    with open(output_file, "w", encoding="utf-8") as fout:
        for doc_line in all_docs:
            doc = json.loads(doc_line)
            text = doc.get("text", "").strip()
            if not text:
                continue

            # Write sentences (split on newlines for variety)
            sentences = [s.strip() for s in text.split("\n") if len(s.strip()) > 20]
            for sentence in sentences:
                fout.write(sentence + "\n")
                lines_written += 1
                if lines_written >= sample_size:
                    break

            if lines_written >= sample_size:
                break

    logger.info(f"Wrote {lines_written:,} sentences to {output_file}")
    return lines_written


def train_tokenizer(config: dict, input_text_file: str, output_dir: Path):
    """Train SentencePiece BPE model using config parameters."""
    cfg = config["tokenizer"]
    output_dir.mkdir(parents=True, exist_ok=True)

    model_prefix = str(output_dir / "slm_tokenizer")

    # Resolve num_threads dynamically — use all available CPUs
    # unless the config explicitly overrides with a specific value
    num_threads = get_num_threads(cfg.get("num_threads", None))

    # Build SentencePiece training args
    train_args = {
        "input": input_text_file,
        "model_prefix": model_prefix,
        "model_type": cfg.get("model_type", "bpe"),
        "vocab_size": cfg.get("vocab_size", 32000),
        "character_coverage": cfg.get("character_coverage", 0.9995),
        "pad_id": cfg.get("pad_id", 0),
        "unk_id": cfg.get("unk_id", 1),
        "bos_id": cfg.get("bos_id", 2),
        "eos_id": cfg.get("eos_id", 3),
        "normalization_rule_name": cfg.get("normalization_rule_name", "nmt_nfkc_cf"),
        "remove_extra_whitespaces": cfg.get("remove_extra_whitespaces", True),
        "add_dummy_prefix": cfg.get("add_dummy_prefix", True),
        "input_sentence_size": cfg.get("input_sentence_size", 10_000_000),
        "shuffle_input_sentence": cfg.get("shuffle_input_sentence", True),
        "num_threads": num_threads,
        "max_sentence_length": cfg.get("max_sentence_length", 4096),
        "byte_fallback": cfg.get("byte_fallback", True),
        "train_extremely_large_corpus": cfg.get("train_extremely_large_corpus", False),
    }

    # Add user-defined special tokens
    user_symbols = cfg.get("user_defined_symbols", [])
    if user_symbols:
        train_args["user_defined_symbols"] = ",".join(user_symbols)

    logger.info(f"Training SentencePiece BPE tokenizer")
    logger.info(f"  Vocab size:  {train_args['vocab_size']:,}")
    logger.info(f"  Model type:  {train_args['model_type']}")
    logger.info(f"  Threads:     {num_threads}")
    logger.info(f"  Output:      {model_prefix}.{{model,vocab}}")

    spm.SentencePieceTrainer.Train(**train_args)

    logger.info("Tokenizer training complete")
    return model_prefix


def validate_tokenizer(model_path: str):
    """Quick sanity checks on the trained tokenizer."""
    sp = spm.SentencePieceProcessor()
    sp.Load(f"{model_path}.model")

    logger.info(f"Validating tokenizer ({sp.GetPieceSize():,} vocab)")

    test_cases = [
        "The quick brown fox jumps over the lazy dog.",
        "def fibonacci(n: int) -> int:\n    if n <= 1:\n        return n\n    return fibonacci(n-1) + fibonacci(n-2)",
        "Hello! How can I help you today?",
        "<|user|> What is Python? <|assistant|> Python is a high-level programming language.",
    ]

    for text in test_cases:
        tokens = sp.EncodeAsIds(text)
        decoded = sp.DecodeIds(tokens)
        logger.info(f"  Input:   {text[:60]}{'...' if len(text) > 60 else ''}")
        logger.info(f"  Tokens:  {len(tokens)} | First 10: {tokens[:10]}")
        logger.info(f"  Decoded: {decoded[:60]}{'...' if len(decoded) > 60 else ''}")
        logger.info("")

    # Check special tokens are present
    special_tokens = ["<|system|>", "<|user|>", "<|assistant|>", "<|code|>"]
    for tok in special_tokens:
        tok_id = sp.PieceToId(tok)
        logger.info(f"  Special token '{tok}' → id={tok_id}")


def main():
    parser = argparse.ArgumentParser(description="Train SLM BPE Tokenizer")
    parser.add_argument("--config", required=True, help="Path to tokenizer.yaml")
    parser.add_argument("--input-dir", required=True, help="Directory of curated JSONL files")
    parser.add_argument("--output-dir", required=True, help="Output directory for tokenizer files")
    parser.add_argument("--sample-size", type=int, default=10_000_000, help="Number of sentences to sample")
    parser.add_argument("--skip-sampling", action="store_true", help="Skip sampling if training_sample.txt exists")
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.safe_load(f)

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)
    sample_file = output_dir / "training_sample.txt"
    complete_marker = output_dir / ".complete"

    # Step 1: Sample text
    if args.skip_sampling and sample_file.exists():
        logger.info(f"Skipping sampling — using existing {sample_file}")
    else:
        logger.info("Sampling text from curated documents...")
        sample_text_from_jsonl(input_dir, sample_file, args.sample_size)

    # Step 2: Train tokenizer
    model_prefix = train_tokenizer(config, str(sample_file), output_dir)

    # Step 3: Validate
    validate_tokenizer(model_prefix)

    logger.info(f"Tokenizer saved to {output_dir}/")
    logger.info(f"  Model:  slm_tokenizer.model")
    logger.info(f"  Vocab:  slm_tokenizer.vocab")

    # Step 4: Write completion marker
    complete_marker.touch()
    logger.info(f"✓ Completion marker written: {complete_marker}")


if __name__ == "__main__":
    main()