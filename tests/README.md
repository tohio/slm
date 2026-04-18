# Tests

The test suite validates pipeline outputs at each stage. Tests are scoped to the instance and stage that produced the outputs — data pipeline tests run on the CPU curation instance, GPU pipeline tests run on the training instance.

---

## Structure

```
tests/
├── conftest.py                         Shared fixtures, DATA_DIR resolution, fasttext mock
│
├── data_pipeline/                      CPU curation instance — run after each data stage
│   ├── test_pipeline_curator.py        Validates make curate-mini outputs
│   ├── test_pipeline_validate.py       Validates make validate outputs
│   └── test_pipeline_tokenizer.py      Validates make tokenizer outputs
│
├── model/                              No pipeline outputs needed — runs anywhere
│   ├── test_model.py                   Model architecture unit tests
│   └── test_cache_and_mask.py          KV cache and attention mask correctness
│
└── gpu_pipeline/                       GPU training instance — run after each training stage
    ├── test_pipeline_training.py       Validates make pretrain-mini outputs
    ├── test_pipeline_sft.py            Validates make sft-mini and sft-code-mini outputs
    └── test_pipeline_dpo.py            Validates make dpo-mini outputs
```

---

## Workflow

Tests are run immediately after the stage they validate. Each test target is paired with its make stage:

**CPU curation instance:**

```bash
make curate-mini   && make test-curator
make validate      && make test-validate
make tokenizer     && make test-tokenizer

# Or run all data pipeline tests at once
make test-data-pipeline
```

**GPU training instance:**

```bash
make pretrain-mini  GPUS=1  && make test-training
make sft-mini       GPUS=1  && make test-sft-chat
make sft-code-mini  GPUS=1  && make test-sft-code
make dpo-mini       GPUS=1  && make test-dpo

# Or run all GPU pipeline tests at once
make test-gpu-pipeline
```

**Model unit tests — no pipeline outputs needed:**

```bash
make test-model
```

---

## What Each Test Validates

### `test-curator` — after `make curate-mini`

| Check | What it catches |
|---|---|
| Raw shards exist for all 3 sources | Download stage failed or was skipped |
| Filtered docs pass quality checks | Filter stage did not run or has a bug |
| No docs below 500 chars in filtered output | Min length filter not applied |
| Deduped dirs exist and are non-empty | Dedup stage failed |
| No exact duplicates in deduped output | Exact dedup not working |
| `train.jsonl` exists and is non-empty | Blend stage failed |
| `train.jsonl` contains all 3 sources | Source mix broken |
| No short docs in `train.jsonl` | Filtered data not used as blend input |
| No exact duplicates in `train.jsonl` | Deduped data not used as blend input |
| `blend_stats.json` exists and is correct | Stats not written or incomplete |
| `blend_stats.json` doc count matches file | Stats don't match actual output |

### `test-validate` — after `make validate`

| Check | What it catches |
|---|---|
| `validated/train.jsonl` exists | Validation stage failed |
| Validated output is a subset of curated | Validation adding docs (wrong) |
| All validated docs pass quality filters | Validator accepting bad docs |
| Retention rate ≥ 70% | KenLM threshold too aggressive |
| `validation_stats.json` is consistent | Stats not written or wrong |

### `test-tokenizer` — after `make tokenizer`

| Check | What it catches |
|---|---|
| All tokenizer files exist | Training incomplete |
| All 16 special tokens have correct IDs | Special token ordering wrong |
| Vocab size is 32,000 | Wrong vocab size trained |
| Encode/decode roundtrip correct | BPE merges or decoder broken |
| No auto BOS/EOS injection | TemplateProcessing incorrectly set |
| Fertility < 1.5 tokens/word | Tokenizer encoding inefficiently |
| Chat template works via `apply_chat_template` | Template not baked in |
| BOS appears exactly once at start | Template formatting wrong |
| Generation prompt ends with `<\|assistant\|>` | Template generation prompt wrong |

### `test-training` — after `make pretrain-mini`

| Check | What it catches |
|---|---|
| `results/slm-mini/final/` exists | Training didn't complete |
| Model files present | Checkpoint not saved |
| Tokenizer saved alongside model | `shutil.copytree` failed |
| Config matches `gpt_mini.yaml` | Wrong config used |
| Forward pass loss is finite | NaN/Inf during training |
| Loss < 6.0 on training data | Model not learning (collapse) |
| Dataset indexing correct | `PretrainingDataset` bug |
| Labels = input_ids shifted left | Dataset shift bug |

### `test-sft-chat` — after `make sft-mini`

| Check | What it catches |
|---|---|
| SFT data files exist with correct format | `prepare_sft` not run or broken |
| All examples end with assistant turn | Data preparation bug |
| Chat model loads | Checkpoint not saved correctly |
| Tokenizer has chat template | Template lost during SFT |
| Forward pass loss finite | Training instability |
| Generation doesn't crash | `prepare_inputs_for_generation` broken |

### `test-sft-code` — after `make sft-code-mini`

| Check | What it catches |
|---|---|
| Code model loads | Checkpoint not saved |
| Forward pass loss finite | Training instability |
| Code special tokens in vocab | Tokenizer not copied correctly |

### `test-dpo` — after `make dpo-mini`

| Check | What it catches |
|---|---|
| DPO data has prompt/chosen/rejected fields | `prepare_dpo` format wrong |
| Chosen ≠ rejected | Identical pairs in dataset |
| Prompt is list of message dicts | Wrong format for trl DPOTrainer |
| DPO stats consistent | Stats not written or wrong |
| DPO model loads | Checkpoint not saved |
| Forward pass loss finite | Training instability |
| Generation doesn't crash | Model broken after DPO |

### `test-model` — no pipeline outputs needed

Two files cover the model layer, run together by `make test-model`:

**`test_model.py` — architecture unit tests:**

| Check | What it catches |
|---|---|
| RMSNorm output shape and dtype | Implementation bug |
| SwiGLU output shape, 3 projections, no bias | Architecture deviation |
| GQA output shape, KV heads < Q heads | GQA not implemented correctly |
| KV cache correct shape | Cache implementation bug |
| Forward pass logits shape | Model forward pass broken |
| Loss finite with labels | Loss computation broken |
| Weight tying | `tie_weights()` not working |
| Parameter count ~25M (mini config) | Architecture mismatch vs config |
| Causal mask — future tokens don't affect past | Causal mask broken (no-cache path) |
| No bias parameters | Architecture deviation |
| Save/load roundtrip | Tied weight serialisation bug |

**`test_cache_and_mask.py` — KV cache and mask correctness:**

| Check | What it catches |
|---|---|
| Prefill + continue matches full forward | Causal mask offset wrong when `q_len < kv_len` (multi-token prefill with cache) |
| Token-by-token generation matches full forward | Bug in the `q_len == 1` cache path |
| Batched forward respects padding mask | `attention_mask` ignored during eval (contaminates batched generation with padded prompts) |
| Parameter counts within 10% for 125m / 350m / 1b | Config drift in any production tier |

---

## Requirements

**No downloads required for any tests.** The fasttext model (`lid.176.ftz`) is mocked in `conftest.py`. KenLM is not used directly in tests. GPU pipeline tests require CUDA but only run if the relevant checkpoint exists — they skip automatically if the model isn't there.

**`DATA_DIR` must be set** for data pipeline tests. It is set automatically by `setup.sh` and written to `~/.bashrc`. If tests are skipping unexpectedly, check:

```bash
echo $DATA_DIR
```

If empty, run:

```bash
source ~/.bashrc
# or
export DATA_DIR=/data/slm/data
```