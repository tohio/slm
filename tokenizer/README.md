# tokenizer

Trains a domain-specific BPE tokenizer on the validated dataset using HuggingFace `tokenizers`. Special tokens are baked in at training time — they cannot be added later without retraining the tokenizer and resizing model embeddings.

---

## Special Tokens

| ID | Token | Purpose |
|---|---|---|
| 0 | `<PAD>` | Padding |
| 1 | `<UNK>` | Unknown (fallback) |
| 2 | `<BOS>` | Beginning of sequence |
| 3 | `<EOS>` | End of sequence |
| 4 | `<\|system\|>` | System prompt start |
| 5 | `<\|user\|>` | User turn start |
| 6 | `<\|assistant\|>` | Assistant turn start |
| 7 | `<\|endofturn\|>` | End of any turn |
| 8 | `<\|code\|>` | Code block start |
| 9 | `<\|endofcode\|>` | Code block end |
| 10 | `<\|tool\|>` | Tool call start |
| 11 | `<\|endoftool\|>` | Tool call end |
| 12 | `<\|reasoning\|>` | Reasoning block start |
| 13 | `<\|endofreasoning\|>` | Reasoning block end |
| 14 | `<\|context\|>` | RAG context start |
| 15 | `<\|endofcontext\|>` | RAG context end |

---

## Usage

**Train**

```bash
make tokenizer

# Upload to S3 for use on GPU instance
make tokenizer-upload

# Or directly
python tokenizer/train_tokenizer.py
python tokenizer/train_tokenizer.py --vocab-size 32000
python tokenizer/train_tokenizer.py --input data/validated/train.jsonl
```

**Test**

```bash
make tokenizer-test

# Or directly
python tokenizer/test_tokenizer.py
```

**Download on GPU instance**

```bash
make tokenizer-download
```

**Load in Python**

```python
from tokenizers import Tokenizer

tokenizer = Tokenizer.from_file("data/tokenizer/slm_tokenizer.json")
encoded = tokenizer.encode("Hello world")
print(encoded.ids)    # token IDs
print(encoded.tokens) # token strings

# Decode
text = tokenizer.decode(encoded.ids)
```

**Load as HuggingFace tokenizer**

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("data/tokenizer")
inputs = tokenizer("Hello world", return_tensors="pt")
```

---

## Chat Template

```python
def format_chat(messages):
    parts = []
    for msg in messages:
        role, content = msg["role"], msg["content"]
        if role == "system":
            parts.append(f"<|system|>{content}<|endofturn|>")
        elif role == "user":
            parts.append(f"<|user|>{content}<|endofturn|>")
        elif role == "assistant":
            parts.append(f"<|assistant|>{content}<|endofturn|>")
    parts.append("<|assistant|>")
    return "".join(parts)
```

---

## Output Files

```
data/tokenizer/
├── slm_tokenizer.json      full tokenizer (HF tokenizers format)
├── tokenizer.json          HuggingFace PreTrainedTokenizerFast
├── tokenizer_config.json   HF tokenizer config
├── vocab.json              vocabulary (token → ID)
├── merges.txt              BPE merge rules
└── special_tokens.json     special token ID mapping
```

---

## Design Decisions

**Why BPE over Unigram/WordPiece?** BPE is the standard for modern LLMs — GPT-2, LLaMA, Mistral, Qwen all use BPE. Unigram (SentencePiece) is common but BPE via HuggingFace `tokenizers` integrates more naturally with the rest of the HF stack.

**Why byte-level BPE?** Byte-level BPE (same as GPT-2) represents every Unicode character as a sequence of bytes. No UNK tokens for out-of-vocabulary characters — every possible input is encodable. Essential for handling multilingual text and special characters in code.

**Why 32k vocab?** At 125M–1B parameters, a 32k vocab gives a good tradeoff — small enough that the embedding table doesn't dominate the parameter budget, large enough for good token efficiency (~1.3 tokens/word on English). Larger models could use 64k–128k.

**Why train on validated data?** The tokenizer vocabulary reflects the frequency distribution of the training corpus. Training on validated (higher quality) data ensures the vocabulary is optimized for the actual pretraining distribution, not the noisier raw web data.

**Why bake in special tokens now?** Adding special tokens after training requires resizing the embedding matrix and the LM head — the new token embeddings start randomly initialized and the model must relearn their semantics. Baking them in at tokenizer training time means they're part of the vocabulary from day one.