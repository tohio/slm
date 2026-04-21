# inference

Local inference scripts for SLM — batch generation and interactive chat CLI.

Special-token IDs (PAD, BOS, EOS, `<|endofturn|>`) are resolved from the
loaded tokenizer at runtime, never hardcoded. This means a tokenizer retrain
that reorders or adds specials doesn't silently break generation.

---

## Files

```
inference/
├── utils.py        shared model + tokenizer loader, special-token ID resolver
├── chat.py         interactive multi-turn chat CLI
├── generate.py     batch text generation from prompts
└── README.md
```

Both `chat.py` and `generate.py` use `inference.utils.load_model_and_tokenizer()`, which handles:
- SLMConfig / SLMForCausalLM registration with AutoConfig / AutoModel
- Tokenizer resolution (checkpoint's `tokenizer/` subdir → checkpoint root → Hub ID)
- Chat-template presence check (fails loudly if missing)
- Resolving `<PAD>`, `<BOS>`, `<EOS>`, `<|endofturn|>` to their actual IDs via `convert_tokens_to_ids`

---

## Chat CLI

Uses `tokenizer.apply_chat_template()` — the same code path as SFT and DPO training.

```bash
# Local checkpoint
python inference/chat.py --model results/slm-125m-dpo/final

# From HuggingFace Hub
python inference/chat.py --model tohio/slm-125m

# Custom system prompt
python inference/chat.py \
    --model results/slm-125m-dpo/final \
    --system "You are an expert Python programmer."

# Override precision
python inference/chat.py --model results/slm-125m-dpo/final --dtype float16
```

**In-chat commands:**

| Command | Action |
|---|---|
| `/reset` | Clear conversation history |
| `/system <prompt>` | Update system prompt and reset |
| `/history` | Show conversation history |
| `/help` | Show all commands |
| `/quit` | Exit |

**Generation parameters (chat.py defaults):**

| Parameter | Default | Notes |
|---|---|---|
| `--max-new-tokens` | 512 | Tokens to generate per turn |
| `--temperature` | 0.7 | Higher = more random |
| `--top-p` | 0.9 | Nucleus sampling |
| `--dtype` | `bfloat16` | Matches training precision |

When the conversation fills more than 75% of the model's context window (measured in real tokens, not characters), a note is printed suggesting `/reset`.

---

## Batch Generation

```bash
# Raw prompt completion (base model) — BOS is prepended by default
echo "The history of AI" | python inference/generate.py \
    --model results/slm-125m/final

# Chat format — wraps prompt as a user message (instruct/chat models)
echo "What is the capital of France?" | python inference/generate.py \
    --model results/slm-125m-dpo/final \
    --chat

# From file
python inference/generate.py \
    --model results/slm-125m-dpo/final \
    --input prompts.txt \
    --output completions.jsonl \
    --chat

# Greedy decoding
python inference/generate.py \
    --model results/slm-125m-dpo/final \
    --greedy \
    --max-new-tokens 100

# Raw completion from mid-sentence (no BOS)
python inference/generate.py \
    --model results/slm-125m/final \
    --no-bos \
    --input sentence_continuations.txt
```

**Generation parameters (generate.py defaults):**

| Parameter | Default | Notes |
|---|---|---|
| `--max-new-tokens` | 256 | Tokens to generate |
| `--temperature` | 0.8 | Higher = more random |
| `--top-p` | 0.95 | Nucleus sampling |
| `--top-k` | 50 | Top-k sampling |
| `--greedy` | False | Disable sampling (overrides temperature/top-p/top-k) |
| `--batch-size` | 4 | Prompts per batch |
| `--chat` | False | Wrap prompts in chat template — use for instruct/chat models |
| `--no-bos` | False | (raw mode only) Skip BOS prefix. Default matches pretraining. |
| `--dtype` | `bfloat16` | Model precision |

---

## `--chat` flag

Use `--chat` when running instruct or chat model variants. It wraps each prompt as a user message and applies `tokenizer.apply_chat_template()`, producing the same format the model was trained on during SFT:

```
<BOS><|user|>What is the capital of France?<|endofturn|><|assistant|>
```

Without `--chat`, prompts are passed directly as raw text — correct for base-model completion but will produce poor results from instruct/chat variants that expect the chat template format.

---

## `--no-bos` flag

In raw (non-`--chat`) mode, `generate.py` prepends the BOS token by default. This matches what the base model saw at sequence start during pretraining and is the right default for open-ended completion from a fresh prompt.

Use `--no-bos` when continuing a mid-sentence fragment, concatenating outputs, or reproducing specific benchmarks that tokenize without special tokens. It has no effect in `--chat` mode, where the chat template controls token placement.

---

## Batched chat generation

When `--chat` is combined with `--batch-size > 1`, prompts of different lengths are left-padded via `tokenizer.pad(..., padding_side="left")`. Left-padding is the correct approach for causal-LM batch generation: it keeps content aligned to the right so generation starts at consistent logical positions in every row, and the attention mask produced by `tokenizer.pad` automatically masks out the padding positions.

---

## Production Serving

For production use, serve with vLLM via the `serve/` directory. vLLM provides 10–50× higher throughput through PagedAttention and continuous batching.