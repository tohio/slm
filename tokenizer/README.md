# Tokenizer

A custom BPE tokenizer trained on our specific data mix, rather than reusing an existing tokenizer (GPT-2, LLaMA). This decision affects every downstream stage — the tokenizer's vocabulary efficiency directly impacts how many tokens are needed to represent a given amount of information, and therefore how much the model learns per training step.

## Why Train a Custom Tokenizer?

Reusing an existing tokenizer is pragmatic and often the right call. We train a custom one for two reasons:

**Token efficiency.** A tokenizer trained on your data mix assigns shorter token sequences to patterns that are common in your corpus. For a model that mixes general English text with Python code, a tokenizer trained on that same mix will encode Python keywords, indentation patterns, and common identifiers more efficiently than GPT-2's tokenizer (trained on English web text) or LLaMA's (optimized for multilingual text).

**Special tokens.** Our conversation format uses structural markers (`<|user|>`, `<|assistant|>`, `<|code|>`) that need to be atomic tokens — not split across subwords. Building these in from the start is cleaner than retrofitting them onto an existing vocabulary.

## Vocabulary

32,000 tokens — consistent with LLaMA and Mistral. This is large enough for good coverage of English and Python, small enough that the embedding table doesn't dominate memory at 125M parameters.

**Special tokens:**

| Token | ID | Purpose |
|---|---|---|
| `<PAD>` | 0 | Padding |
| `<UNK>` | 1 | Unknown (rarely used — byte fallback handles most cases) |
| `<BOS>` | 2 | Beginning of sequence |
| `<EOS>` | 3 | End of sequence |
| `<\|system\|>` | 4 | System prompt marker |
| `<\|user\|>` | 5 | User turn marker |
| `<\|assistant\|>` | 6 | Assistant turn marker |
| `<\|endofturn\|>` | 7 | End of conversational turn |
| `<\|code\|>` | 8 | Code block start |
| `<\|endofcode\|>` | 9 | Code block end |

**Byte fallback** is enabled — any character that falls outside the vocabulary is encoded as individual UTF-8 bytes rather than `<UNK>`. This means the tokenizer is lossless for any Unicode input.

## Training

Trained on a 10M sentence sample from the curated dataset (post-PII redaction stage). SentencePiece BPE with NFC unicode normalization. 10M sentences is sufficient for stable BPE merge statistics — using more doesn't meaningfully improve vocabulary quality.

## Usage

```bash
# Train tokenizer (run after curation, before pre-training)
make tokenizer

# Or directly
python tokenizer/train_tokenizer.py \
    --config tokenizer/configs/tokenizer.yaml \
    --input-dir /data/curated/stages/pii \
    --output-dir /data/tokenizer

# Outputs:
#   /data/tokenizer/slm_tokenizer.model   ← used by NeMo training configs
#   /data/tokenizer/slm_tokenizer.vocab   ← human-readable vocabulary
```

The training script validates the tokenizer on a small set of test sentences after training — check that special tokens round-trip correctly and that Python code is tokenized sensibly before moving to pre-training.
