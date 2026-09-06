# Tokenizer

This directory trains and validates the size-specific tokenizer used by
pretraining, SFT, DPO, evaluation, export, and inference.

## Contents

| File | Purpose |
|---|---|
| `train_tokenizer.py` | Train the byte-level BPE tokenizer and write its manifest |
| `test_tokenizer.py` | Validate vocabulary, special tokens, round trips, fertility, and chat rendering |
| `compare_tokenizer.py` | Compare the SLM tokenizer with a Hugging Face reference on five fixed diagnostic cases |

Corpus tokenization is a separate pretraining-data stage implemented by
`pretrain/data/tokenize_data.py`.

## Input and output

The default input is the validated training split:

```text
$DATA_DIR/runs/<size>/validated/train.jsonl
```

The tokenizer and its completion manifest are written to:

```text
$DATA_DIR/runs/<size>/tokenizer/
```

The directory contains the raw BPE tokenizer, Hugging Face tokenizer files,
vocabulary and merge files, the chat template, special-token metadata, and the
stage manifest. A matching manifest allows an unchanged tokenizer build to be
reused; changed input or tokenizer settings produce a new build.

## Tokenizer contract

The tokenizer uses:

- NFC normalization;
- byte-level BPE pre-tokenization with `add_prefix_space=False`;
- a default vocabulary size of 32,000;
- a default minimum token frequency of 2;
- no automatic BOS/EOS post-processor.

The first four vocabulary entries are fixed:

| ID | Token |
|---:|---|
| 0 | `<PAD>` |
| 1 | `<UNK>` |
| 2 | `<BOS>` |
| 3 | `<EOS>` |

IDs 4–15 are the chat, code, tool, reasoning, and context delimiters defined in
`train_tokenizer.py`. Runtime code resolves these tokens by string; changing
their spelling or order is a pipeline-wide compatibility change.

The saved chat template adds one BOS at the start of a conversation, role and
end-of-turn delimiters around each message, and EOS on assistant turns.
Assistant content is wrapped in Jinja generation markers so TRL can construct
assistant-only loss masks during SFT.

## Usage

Train and validate a tokenizer:

```bash
make tokenizer SIZE=125m
make tokenizer-test SIZE=125m
```

Equivalent direct commands:

```bash
python tokenizer/train_tokenizer.py \
  --size 125m \
  --vocab-size 32000 \
  --min-frequency 2

python tokenizer/test_tokenizer.py --size 125m
```

Compare the SLM tokenizer with a Hugging Face reference tokenizer:

```bash
python tokenizer/compare_tokenizer.py --size 125m
```

The comparison uses five fixed cases: English prose, code, numbers/URL text,
Unicode/multilingual text, and longer technical prose. For each case it shows
token counts, token pieces, token IDs, roundtrip behavior, and the token-count
delta. The default reference is `HuggingFaceTB/SmolLM2-135M`.

The comparison is diagnostic only. A lower token count is not an automatic
quality pass/fail criterion.

Use another Hugging Face reference when needed:

```bash
python tokenizer/compare_tokenizer.py \
  --size 125m \
  --reference HuggingFaceTB/SmolLM2-135M
```

Use explicit paths when testing a nonstandard corpus or output directory:

```bash
python tokenizer/train_tokenizer.py \
  --size 125m \
  --input /data/slm/data/runs/125m/validated/train.jsonl \
  --output /tmp/slm-tokenizer

python tokenizer/test_tokenizer.py \
  --size 125m \
  --tokenizer /tmp/slm-tokenizer
```

After training the tokenizer, encode the validated corpus for pretraining:

```bash
make tokenize SIZE=125m
```

When data preparation and training use different hosts, transfer the tokenizer
together with tokenized data and metadata:

```bash
make artifacts-upload \
  SIZE=125m \
  ARTIFACT_STAGES="tokenized,tokenizer,metadata"
```

## Compatibility rules

- Do not substitute a tokenizer from another size or data run.
- Keep the tokenizer fingerprint recorded by corpus tokenization with the
  corresponding `.bin` files.
- Synchronize the selected size's tokenizer into the repository runtime path
  with `make restore-size-tokenizer SIZE=<size>` before model training.
- Review SFT, DPO, inference, export, and serving whenever the chat template or
  special-token set changes.
