# tokenizer

Tokenizer training and tokenizer validation for SLM.

---

## Responsibility

`tokenizer/` owns:

- BPE tokenizer training
- project special token definitions
- tokenizer config output
- tokenizer validation tests

Tokenization of the training corpus lives in `pretrain/data/tokenize_data.py`.

---

## Files

```text
tokenizer/
├── train_tokenizer.py
├── test_tokenizer.py
└── README.md
```

---

## Inputs

```text
data/runs/<size>/validated/train.jsonl
data/runs/<size>/validated/val.jsonl
```

---

## Outputs

```text
data/runs/<size>/tokenizer/tokenizer.json
data/runs/<size>/tokenizer/tokenizer_config.json
```

A compatibility copy may also be restored to:

```text
data/tokenizer/
```

---

## Special tokens

Special tokens are defined in `train_tokenizer.py` and asserted during tokenizer training. They include:

```text
<PAD>
<UNK>
<BOS>
<EOS>
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

---

## Commands

Train tokenizer:

```bash
make tokenizer SIZE=125m
```

Run tokenizer validation:

```bash
make tokenizer-test SIZE=125m
make test-tokenizer SIZE=125m
```

Upload tokenizer artifacts through the RUN_ID artifact flow:

```bash
make artifacts-upload SIZE=125m ARTIFACT_STAGES="tokenizer,metadata"
```

Direct calls:

```bash
python tokenizer/train_tokenizer.py
python tokenizer/test_tokenizer.py
```

---

## Runtime ID resolution

Training code can use tokenizer constants from `train_tokenizer.py`. Runtime code resolves token IDs from the loaded tokenizer via `inference/utils.py` so exported checkpoints cannot silently mis-map special tokens.

---

## Chat template

The tokenizer owns the chat template used by SFT, DPO, inference, and serving. Any change to the chat format requires tokenizer and downstream checkpoint compatibility review.
