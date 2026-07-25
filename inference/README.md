# Inference

This directory provides interactive chat and batch text generation for local
training checkpoints, local exports, and published SLM models.

## Contents

| File | Purpose |
|---|---|
| `chat.py` | Stateful terminal chat with reset, system-prompt, and history controls |
| `generate.py` | Raw or chat-formatted generation from standard input or a text file |
| `utils.py` | Model/tokenizer loading and runtime special-token resolution |

For a local training checkpoint, the loader prefers its `tokenizer/`
subdirectory. For a native export or Hub model, tokenizer files are loaded
from the model root. Models are placed with Transformers `device_map="auto"`.

## Model references

```text
$RESULTS_DIR/runs/<size>/pretrain/final
$RESULTS_DIR/runs/<size>/sft_instruct/final
$RESULTS_DIR/runs/<size>/dpo_chat/final
$RESULTS_DIR/runs/<size>/sft_code/final

$EXPORTS_DIR/<size>/<variant>

tohio/slm-<size>
tohio/slm-<size>-instruct
tohio/slm-<size>-chat
tohio/slm-<size>-code
```

## Interactive chat

Start a local chat checkpoint:

```bash
python inference/chat.py \
  --model results/runs/125m/dpo_chat/final
```

Start a published model with custom generation settings:

```bash
python inference/chat.py \
  --model tohio/slm-125m-chat \
  --system "You are a concise assistant." \
  --max-new-tokens 256 \
  --temperature 0.7 \
  --top-p 0.9 \
  --dtype bfloat16
```

Interactive commands:

| Command | Effect |
|---|---|
| `/help` | Show commands |
| `/reset` | Clear conversation history |
| `/system <prompt>` | Replace the system prompt and reset history |
| `/history` | Print the current conversation |
| `/quit` | Exit |

## Batch and raw generation

Greedy base-model continuation from standard input:

```bash
printf '%s\n' "The capital of France is" |
  python inference/generate.py \
    --model results/runs/125m/pretrain/final \
    --max-new-tokens 30 \
    --greedy
```

Chat-formatted generation:

```bash
printf '%s\n' "Explain attention in one sentence." |
  python inference/generate.py \
    --model results/runs/125m/dpo_chat/final \
    --chat \
    --max-new-tokens 80
```

Read one prompt per line and write JSONL completions:

```bash
python inference/generate.py \
  --model tohio/slm-125m-chat \
  --input prompts.txt \
  --output completions.jsonl \
  --chat \
  --batch-size 4
```

Generation options include `--max-new-tokens`, `--temperature`, `--top-p`,
`--top-k`, `--greedy`, `--batch-size`, `--dtype`, and
`--repetition-penalty`.

## Token and prompt behavior

- Raw mode prepends BOS by default; `--no-bos` disables it for continuation
  experiments.
- `--chat` renders each input as a user message with the tokenizer's saved
  chat template. It ignores `--no-bos` because the template owns special-token
  placement.
- `chat.py` and chat-formatted generation require a tokenizer with a chat
  template.
- Runtime stop IDs are resolved from the loaded tokenizer and include EOS and
  end-of-turn.
- Use raw mode for the base model. Use `--chat` or `chat.py` for instruct,
  chat, and code-instruction variants.

For multi-user HTTP serving, use the vLLM assets in `serve/`.
