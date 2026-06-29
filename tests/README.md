# Tests

The test suite validates pipeline artifacts and unit-level invariants.

Data pipeline tests run on the CPU/data instance. GPU pipeline tests run on the training instance.

---

## Structure

```text
tests/
├── conftest.py
├── data_pipeline/
│   ├── helpers.py
│   ├── test_pipeline_curator.py
│   ├── test_pipeline_validate.py
│   └── test_pipeline_tokenizer.py
├── gpu_pipeline/
│   ├── test_pipeline_training.py
│   ├── test_pipeline_sft.py
│   └── test_pipeline_dpo.py
├── model/
│   ├── test_model.py
│   └── test_cache_and_mask.py
├── test_config_gen.py
└── test_accel_gen.py
```

---

## Size behavior

Test targets use `TEST_SIZE`.

Default behavior:

```text
make test-*        -> TEST_SIZE=mini
make test-* SIZE=125m -> TEST_SIZE=125m
```

This keeps normal development on `mini` and makes full-artifact checks explicit.

---

## Data pipeline tests

```bash
make test-curator
make test-validate
make test-tokenizer
make test-data-pipeline
```

Typical mini sequence:

```bash
make curate-mini
make validate SIZE=mini
make tokenizer SIZE=mini
make tokenize SIZE=mini
make test-data-pipeline
```

These tests validate curated shards, validation outputs, tokenizer files, and tokenized binaries.

---

## GPU pipeline tests

```bash
make test-training
make test-sft-instruct
make test-sft-code
make test-dpo-chat
make test-gpu-pipeline
```

Typical mini sequence:

```bash
make pretrain-mini SIZE=mini GPUS=1
make sft-instruct-mini SIZE=mini GPUS=1
make sft-code-mini SIZE=mini GPUS=1
make dpo-chat-mini SIZE=mini GPUS=1
make test-gpu-pipeline
```

Full-run checks:

```bash
make test-training SIZE=125m
make test-sft-instruct SIZE=125m
make test-sft-code SIZE=125m
make test-dpo-chat SIZE=125m
```

Compatibility aliases:

```bash
make test-sft-chat SIZE=125m
make test-dpo SIZE=125m
```

---

## What each target validates

| Target | Validates |
|---|---|
| `test-curator` | curation output structure and source stats |
| `test-validate` | validation outputs and retained document quality |
| `test-tokenizer` | tokenizer assets, special tokens, tokenized binaries |
| `test-training` | base checkpoint loading, finite loss, dataset indexing |
| `test-sft-instruct` | instruct checkpoint loading, chat template preservation, generation |
| `test-sft-code` | code checkpoint loading and code-oriented generation |
| `test-dpo-chat` | preference data shape, chat DPO checkpoint loading, generation |
| `test-model` | architecture construction, masks, cache behavior |
| `test-config-gen` | generated training config invariants |
| `test-accel-gen` | Accelerate config rendering |

---

## Unit tests

```bash
make test-model
make test-config-gen
make test-accel-gen
make test-unit
```

Unit tests do not require pipeline artifacts.
