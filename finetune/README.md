# Supervised Fine-Tuning (SFT)

Adapts the pre-trained base model to follow instructions and produce well-formed responses. SFT runs in two sequential stages — general chat first, then coding — each building on the previous checkpoint.

Uses **NeMo-Aligner 0.7.0** inside `nvcr.io/nvidia/nemo:25.02`. Input is the `mcore_gpt.nemo` file produced by `make convert-pretrain`.

## Why Sequential SFT?

**Debuggability.** Each stage produces an independently evaluable checkpoint. If the code SFT degrades general conversation quality, it's immediately visible by comparing the chat and code SFT checkpoints.

**Specialization control.** The code SFT uses a lower learning rate (5e-6 vs 1e-5) specifically to reduce catastrophic forgetting of chat capabilities from stage 1.

## Stage 1 — General Chat

**Dataset:** OpenAssistant OASST1 + Dolly 15k (~25k examples combined)
**Goal:** Instruction following, multi-turn conversation, appropriate response length and tone.
**Key config:** `answer_only_loss: true` — cross-entropy loss is computed only on assistant response tokens, not the human prompt. Without it the model learns to predict user messages, which wastes capacity and dilutes the gradient signal.

**Format:**
```json
{
  "conversations": [
    {"from": "system",    "value": "You are a helpful assistant."},
    {"from": "human",     "value": "What is Python?"},
    {"from": "assistant", "value": "Python is a high-level..."}
  ]
}
```

## Stage 2 — Coding

**Dataset:** CodeSearchNet Python (~100k examples, two formats)
**Goal:** Code generation, code explanation, debugging assistance.
**Lower learning rate (5e-6)** reduces catastrophic forgetting of the chat capabilities from stage 1.

## Design Decisions

**Full fine-tuning over LoRA**
At 125M parameters, the model fits comfortably in GPU memory for full fine-tuning. LoRA adds complexity without meaningful benefit at this scale. LoRA becomes the right choice at 7B+ where full fine-tuning VRAM requirements become prohibitive.

## Prerequisites

```bash
# 1. Pretrain checkpoint must be converted to mcore_gpt.nemo format
make pretrain
make convert-pretrain     # produces /results/slm_gpt_125m/mcore_gpt.nemo

# 2. SFT datasets must be prepared
make prepare-sft-data     # downloads from HuggingFace to /data/sft/
```

## Usage

```bash
# Run both stages sequentially (default)
make sft

# Or run individually
bash finetune/scripts/train_sft.sh --stage chat
bash finetune/scripts/train_sft.sh --stage code

# Evaluate
make eval-sft
```

## What to Watch

After chat SFT, generation samples should show:
- Responses that stop at the right place
- Appropriate response length
- No repetition loops

After code SFT:
- Code blocks are fenced correctly (` ```python `)
- Generated functions are syntactically valid Python
- General conversation quality is not visibly degraded vs chat checkpoint