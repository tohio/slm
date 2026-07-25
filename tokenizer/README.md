# tokenizer

## Purpose

BPE tokenizer training and tokenizer validation for SLM.

- tokenizer training in `train_tokenizer.py`
- project special-token definitions
- Hugging Face tokenizer config output
- tokenizer validation checks in `test_tokenizer.py`

Corpus tokenization for pretraining lives in `pretrain/data/tokenize_data.py`.

## How It Fits In

The tokenizer is trained from validated data and reused unchanged by
pretraining, post-training, evaluation, export, inference, and serving; see
[Architecture](../docs/ARCHITECTURE.md).

## Key files

```text
tokenizer/
├── train_tokenizer.py
├── test_tokenizer.py
└── README.md
```

## Inputs

Default training input:

```text
data/runs/<size>/validated/train.jsonl
```

Optional direct input:

```bash
python tokenizer/train_tokenizer.py --size 125m --input data/runs/125m/validated/train.jsonl
```

## Outputs

Default output directory:

```text
data/runs/<size>/tokenizer/
```

Expected files:

```text
slm_tokenizer.json
tokenizer.json
tokenizer_config.json
vocab.json
merges.txt
special_tokens.json
special_tokens_map.json
chat_template.jinja
```

Export copies the required tokenizer files into the checkpoint root before pushing to Hugging Face.

## Special tokens

The four structural tokens are fixed first:

```text
<PAD>
<UNK>
<BOS>
<EOS>
```

Additional special tokens:

```text
<|system|>
<|user|>
<|assistant|>
<|endofturn|>
<|code|>
<|endofcode|>
<|tool|>
<|endoftool|>
<|reasoning|>
<|endofreasoning|>
<|context|>
<|endofcontext|>
```

Training code can assert the special-token layout. Runtime code resolves token IDs from the loaded tokenizer instead of importing hard-coded IDs.

## Commands

Train tokenizer:

```bash
make tokenizer SIZE=125m
```

Validate tokenizer:

```bash
make tokenizer-test SIZE=125m
make test-tokenizer SIZE=125m
```

Upload tokenizer artifacts:

```bash
make artifacts-upload SIZE=125m ARTIFACT_STAGES="tokenizer,metadata"
```

Direct calls:

```bash
python tokenizer/train_tokenizer.py --size 125m
python tokenizer/train_tokenizer.py --size 125m --vocab-size 32000
python tokenizer/train_tokenizer.py --size 125m --min-frequency 2
python tokenizer/test_tokenizer.py --size 125m
```

## Chat template

The tokenizer saves the chat template used by SFT, DPO, inference, and serving.

The template includes assistant-generation markers required by TRL answer-only loss masking. Any chat-template change requires tokenizer, SFT, DPO, inference, export, and serving compatibility review.

## BOS/EOS policy

The tokenizer does not automatically inject BOS/EOS through a post-processor. Training and inference code add those tokens explicitly where appropriate.

This avoids corrupting chat-formatted examples with unexpected special tokens.

## Pretraining tokenization

After tokenizer training, pretraining tokenization is run from the pretrain folder:

```bash
make tokenize SIZE=125m
```

Direct call:

```bash
python pretrain/data/tokenize_data.py --size 125m --chunk-size 256 --verify
```

Tokenization writes:

```text
data/runs/<size>/tokenized/train.bin
data/runs/<size>/tokenized/val.bin
```
