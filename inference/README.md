# inference

Local inference scripts for SLM — batch generation and interactive chat CLI.

---

## Files

```
inference/
├── generate.py     batch text generation from prompts
├── chat.py         interactive multi-turn chat CLI
└── README.md
```

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

---

## Batch Generation

```bash
# Raw prompt completion (base model)
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

---

## `--chat` flag

Use `--chat` when running instruct or chat model variants. It wraps each
prompt as a user message and applies `tokenizer.apply_chat_template()`,
producing the same format the model was trained on during SFT:

```
<BOS><|user|>What is the capital of France?<|endofturn|><|assistant|>
```

Without `--chat`, prompts are passed directly as raw text — correct for
base model completion but will produce poor results from instruct/chat
variants that expect the chat template format.

---

## Production Serving

For production use, serve with vLLM via the `serve/` directory. vLLM provides
10–50× higher throughput through PagedAttention and continuous batching.