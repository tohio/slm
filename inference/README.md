# inference

Local inference utilities for SLM checkpoints and exported Hub models.

---

## Owns

- `inference/chat.py` — interactive chat CLI
- `inference/generate.py` — batch/raw prompt generation
- `inference/utils.py` — model/tokenizer loading and special-token resolution

Serving through vLLM lives in `serve/`.

---

## Model references

Local checkpoints:

```text
results/runs/<size>/pretrain/final
results/runs/<size>/sft_instruct/final
results/runs/<size>/dpo_chat/final
results/runs/<size>/sft_code/final
```

Hub models:

```text
tohio/slm-<size>
tohio/slm-<size>-instruct
tohio/slm-<size>-chat
tohio/slm-<size>-code
```

---

## Interactive chat

Local checkpoint:

```bash
python inference/chat.py --model results/runs/125m/dpo_chat/final
```

Hub model:

```bash
python inference/chat.py --model tohio/slm-125m-chat
```

Custom generation settings:

```bash
python inference/chat.py   --model tohio/slm-125m-chat   --system "You are a concise assistant."   --max-new-tokens 256   --temperature 0.7   --top-p 0.9
```

Interactive commands:

```text
/help
/reset
/system <prompt>
/history
/quit
```

---

## Batch generation

Raw base-model completion:

```bash
echo "The capital of France is" | python inference/generate.py   --model results/runs/125m/pretrain/final   --max-new-tokens 30   --greedy
```

Chat-formatted generation:

```bash
echo "Explain attention in one sentence." | python inference/generate.py   --model results/runs/125m/dpo_chat/final   --chat   --max-new-tokens 80
```

File input/output:

```bash
python inference/generate.py   --model tohio/slm-125m-chat   --input prompts.txt   --output completions.jsonl   --chat
```

Common options:

```text
--max-new-tokens
--temperature
--top-p
--top-k
--greedy
--batch-size
--chat
--no-bos
--trust-remote-code
--device
```

---

## Token behavior

- Runtime code resolves token IDs from the loaded tokenizer.
- Raw completion prepends BOS by default.
- `--no-bos` disables BOS for continuation-style generation.
- `--chat` wraps prompts as user messages with the tokenizer chat template.
- Chat, instruct, and code variants should normally use `--chat` or `chat.py`.

---

## Serving

Use `serve/` for an OpenAI-compatible vLLM server.
