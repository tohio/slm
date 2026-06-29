# Tests

The test suite validates pipeline outputs and unit-level invariants. Data pipeline tests run on the CPU curation instance. GPU pipeline tests run on the training instance.

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

## CPU pipeline tests

```bash
make curate-mini   && make test-curator
make validate      && make test-validate
make tokenize      && make test-tokenizer

make test-data-pipeline
```

These tests validate curated shards, validation outputs, tokenizer files, and tokenized binaries.

---

## GPU pipeline tests

GPU tests default to `mini` unless `SIZE` is supplied explicitly.

```bash
make pretrain-mini       GPUS=1 && make test-training
make sft-instruct-mini   GPUS=1 && make test-sft-instruct
make sft-code-mini       GPUS=1 && make test-sft-code
make dpo-chat-mini       GPUS=1 && make test-dpo-chat

make test-gpu-pipeline
```

Full-run checks:

```bash
make test-training     SIZE=125m
make test-sft-instruct SIZE=125m
make test-sft-code     SIZE=125m
make test-dpo-chat     SIZE=125m
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
| `test-curator` | raw shards, source stats, curation output structure |
| `test-validate` | validation output structure and retained document quality |
| `test-tokenizer` | tokenizer assets, special tokens, tokenized binaries |
| `test-training` | base checkpoint loading, finite loss, dataset indexing |
| `test-sft-instruct` | instruct checkpoint loading, chat template preservation, generation |
| `test-sft-code` | code checkpoint loading and code-token behavior |
| `test-dpo-chat` | preference data shape, chat DPO checkpoint loading, generation |
| `test-model` | architecture unit tests |
| `test-config-gen` | config generation invariants |
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

---

## Size behavior

The Makefile default pipeline size is `125m`, but GPU pipeline tests intentionally default to `mini` for normal development. Passing `SIZE=125m`, `350m`, or `1b` opts into full-artifact checks.
