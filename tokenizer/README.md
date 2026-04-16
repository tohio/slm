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

The test suite validates special token IDs, encode/decode roundtrip, BOS/EOS auto-injection (must be absent), fertility, and chat template via `apply_chat_template()`. It exits with code 1 if any test fails — the pipeline will not proceed with a broken tokenizer.

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

The Jinja2 chat template is baked into the HuggingFace tokenizer at training time. Use `apply_chat_template()` — do not format chat strings manually. Manual formatters bypass the template and will produce output that does not match what the model was trained on.

```python
from transformers import PreTrainedTokenizerFast

tokenizer = PreTrainedTokenizerFast.from_pretrained("data/tokenizer")

messages = [
    {"role": "system", "content": "You are a helpful assistant."},
    {"role": "user", "content": "What is the capital of France?"},
]

# Correct — uses the baked-in template
prompt = tokenizer.apply_chat_template(
    messages,
    tokenize=False,
    add_generation_prompt=True,
)
# Output:
# <BOS><|system|>You are a helpful assistant.<|endofturn|><|user|>What is the capital of France?<|endofturn|><|assistant|>

# Tokenized directly
input_ids = tokenizer.apply_chat_template(
    messages,
    tokenize=True,
    add_generation_prompt=True,
    return_tensors="pt",
)
```

### Format

```
<BOS>
<|system|>{system content}<|endofturn|>
<|user|>{user content}<|endofturn|>
<|assistant|>{assistant content}<EOS><|endofturn|>
<|user|>{user content}<|endofturn|>
<|assistant|>   ← generation prompt (add_generation_prompt=True)
```

**BOS** appears once at the very start of the full sequence.
**EOS** appears after each assistant response, signalling end of generation.
Role tokens (`<|system|>`, `<|user|>`, `<|assistant|>`) act as turn delimiters — no BOS/EOS on system or user turns.

---

## BOS/EOS Handling

BOS and EOS are **not** added automatically by the tokenizer. They must be added explicitly by the data pipeline:

- **Pretraining** — the tokenization script prepends BOS and appends EOS to each document.
- **SFT/DPO** — `apply_chat_template()` handles BOS/EOS placement automatically via the baked-in template.
- **Inference** — `apply_chat_template(add_generation_prompt=True)` prepends BOS correctly.

Automatic injection via `TemplateProcessing` was intentionally removed. It corrupted chat-formatted data by inserting BOS/EOS at positions the model does not expect.

---

## Output Files

```
data/tokenizer/
├── slm_tokenizer.json        full tokenizer (HF tokenizers format)
├── tokenizer.json            HuggingFace PreTrainedTokenizerFast
├── tokenizer_config.json     HF tokenizer config — includes chat_template
├── vocab.json                vocabulary (token → ID)
├── merges.txt                BPE merge rules
└── special_tokens.json       special token ID mapping
```

---

## Design Decisions

**Why BPE over Unigram/WordPiece?** BPE is the standard for modern LLMs — GPT-2, LLaMA, Mistral, Qwen all use BPE. Unigram (SentencePiece) is common but BPE via HuggingFace `tokenizers` integrates more naturally with the rest of the HF stack.

**Why byte-level BPE?** Byte-level BPE (same as GPT-2) represents every Unicode character as a sequence of bytes. No UNK tokens for out-of-vocabulary characters — every possible input is encodable. Essential for handling multilingual text and special characters in code.

**Why 32k vocab?** At 125M–1B parameters, a 32k vocab gives a good tradeoff — small enough that the embedding table doesn't dominate the parameter budget, large enough for good token efficiency (~1.3 tokens/word on English). Larger models could use 64k–128k.

**Why no automatic BOS/EOS injection?** Automatic injection via `TemplateProcessing` runs on every `encode()` call regardless of context. During pretraining this doubles BOS/EOS tokens if the data pipeline also adds them. During SFT and DPO it inserts tokens at positions the chat template does not expect, corrupting the turn structure the model needs to learn. Explicit placement by the data pipeline is unambiguous and correct in all contexts.

**Why bake in the chat template?** `apply_chat_template()` is the standard HuggingFace interface used by `SFTTrainer`, `DPOTrainer`, and inference scripts. Without a baked-in template it falls back to a generic format that does not match the model's special tokens, causing the model to echo role tokens instead of generating responses. Baking it in at tokenizer training time means every downstream consumer gets the correct format automatically.

**Why train on validated data?** The tokenizer vocabulary reflects the frequency distribution of the training corpus. Training on validated (higher quality) data ensures the vocabulary is optimized for the actual pretraining distribution, not the noisier raw web data.

**Why bake in special tokens now?** Adding special tokens after training requires resizing the embedding matrix and the LM head — the new token embeddings start randomly initialized and the model must relearn their semantics. Baking them in at tokenizer training time means they're part of the vocabulary from day one.