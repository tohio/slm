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

---

## Batch Generation

```bash
# From stdin
echo "The history of AI" | python inference/generate.py \
    --model results/slm-125m-dpo/final

# From file
python inference/generate.py \
    --model results/slm-125m-dpo/final \
    --input prompts.txt \
    --output completions.jsonl

# Greedy decoding
python inference/generate.py \
    --model results/slm-125m-dpo/final \
    --greedy \
    --max-new-tokens 100
```

---

## Generation Parameters

| Parameter | Default | Notes |
|---|---|---|
| `--max-new-tokens` | 256 / 512 | Tokens to generate |
| `--temperature` | 0.8 / 0.7 | Higher = more random |
| `--top-p` | 0.95 / 0.9 | Nucleus sampling |
| `--top-k` | 50 | Top-k sampling |
| `--greedy` | False | Disable sampling |

---

## Production Serving

For production use, serve with vLLM via the `serve/` directory. vLLM provides 10–50× higher throughput than the inference scripts here through PagedAttention and continuous batching.