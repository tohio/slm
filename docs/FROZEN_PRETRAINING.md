# Frozen pretraining data, reuse, and migration

This guide covers the consolidated frozen-test, artifact, cross-size, diagnostic,
hardware-configuration, and RoPE-loading changes. It does not change the existing
English-language tokenizer/model scope, add another trainer, or add tool calling.
Mini is for plumbing validation; 125M, 350M, and potentially 1B are capability
training targets. Smoke remains a separate bounded execution path.

## 1. Data roles and frozen membership

`train` supplies gradient updates. `val` supplies training-time loss/perplexity,
tuning, and checkpoint selection. `test` is opened as a model evaluation dataset
only after optimization/checkpoint selection has ended, or by the explicit final
evaluation command. Integrity checks may read test bytes/checksums before training;
they do not evaluate the model on test or use test metrics for training decisions.

`make freeze-test SIZE=<size>` upgrades a complete existing curated blend. It
keeps the existing `val.jsonl` **byte-for-byte unchanged** and selects test from
training using seeded SHA-256 ranks of the existing normalized exact hashes.
The default is approximately 0.5% of the original total **document count**, not a
guaranteed exact token percentage. With the existing validation fraction this
targets roughly 99% train / 0.5% validation / 0.5% test. Actual document and token
counts are reported; document lengths and removals can change the proportions.

The existing exact-hash and DataTrove MinHash policy are reused for all three
relationships. Protection priority is existing validation, then test, then
training: remediate test against validation, training against validation, and
training against test. No second MinHash threshold or independent policy is
introduced. MinHash remains the existing probabilistic LSH policy, not a proof
that no semantically similar documents exist. Exact overlap is checked again
before promotion. Reported removals must match physical document/character
removals. Work is staged beside the original curated directory and promoted only
when all gates pass.

`test_contract.json` contains full split SHA-256 hashes/counts, selection policy,
original blend identity, and pair-audit provenance. `test_membership.jsonl` is a
text-free membership index retained with curation provenance. Repeated freezing
verifies and reuses the same membership; a changed seed/policy or a tampered
holdout fails. Use a **new dataset root/run** for a different corpus, rather than
silently replacing a frozen test. Never delete an interrupted transaction's
backup without first recovering a complete original or promoted directory.

Validation applies the same existing source-aware filtering policy to all three
splits and records per-split rejection/accounting statistics. It does not rewrite
retained text. The validated frozen contract records which curated records
survived. Once that validated contract exists, changing its filtering policy is
not an in-place operation on the same frozen run.

Tokenization produces and verifies all three binaries and metadata sidecars:

```text
$DATA_DIR/runs/<dataset-size>/
  curated/train.jsonl, val.jsonl, test.jsonl
  curated/test_contract.json, test_membership.jsonl, _SUCCESS.json
  validated/train.jsonl, val.jsonl, test.jsonl
  validated/test_contract.json, validation_stats.json, _SUCCESS.json
  tokenizer/slm_tokenizer.json, tokenizer_config.json, ...
  tokenized/train.bin, val.bin, test.bin
  tokenized/train.json, val.json, test.json
  tokenized/test_contract.json, token_mixture.json, _SUCCESS.json
  metadata/pipeline_manifest.json, provenance/...
  RUN_ID
```

Every binary is checked for exact byte/token counts, vocabulary range, matching
BOS/EOS document counts and boundaries, full binary SHA-256, tokenizer identity,
and frozen input identity. The mixture report includes all three splits. Mini
fails closed below **1.4 billion usable selected training tokens**, both after
Mini tokenization and at Mini preflight/training. A short corpus is not silently
padded, and borrowing a larger dataset does not waive this floor.

## 2. Migrate an existing Mini run safely

First preserve the old model **with its exact old tokenizer and validation
artifacts**. Use a separate backup/root, not a symlink to directories about to be
replaced. The RoPE fix alone does not require retraining the old base model: it
reconstructs parameter-free state on load. It does not reinitialize embeddings or
any other learned weights.

The reported old Mini validation baseline can be checked **before replacing its
matched data**, using the existing evaluation entry point:

```bash
.venv/bin/python eval/eval.py \
  --mode pretraining-validation \
  --model "$RESULTS_DIR/runs/mini/pretrain/final" \
  --size mini --dataset-size mini --data-dir "$DATA_DIR" \
  --dtype float32 --batch-size 1 \
  --expected-validation-loss 2.660014985683328 \
  --validation-loss-tolerance 0.001 \
  --json-out "$RESULTS_DIR/mini-rope-validation.json"
```

The tolerance is an explicit comparison setting, not a measured guarantee. Run
BF16 separately as well. This command validates the old tokenizer/validation
binary pairing and **does not evaluate test**. The ~2.660 result is a supplied
baseline to reproduce on the real checkpoint, not an implementation-time result.

For the data upgrade, start from the existing complete curated blend; do not
recall all source downloads or rerun blending first:

```bash
make regenerate-mini-frozen DATA_DIR=/data/slm/data WORKERS=14
```

This runs freeze, validation, tokenizer training, tokenization, and the existing
data-pipeline gates for Mini. Stage manifests reuse what remains valid; only
invalidated stages rebuild. It does not run pretraining/SFT, delete model
checkpoints, or upload anything. Check `validation_stats.json`, all three
`tokenized/*.json` split records, and the usable-training-token floor before
choosing what to retain. The tokenizer may change because its training input
changed. Do not pair a newly trained tokenizer with the old model or alter old
embeddings to make it fit.

**The newly carved test was part of the old model's training pool. It is not an
unseen test for that old model.** Final frozen-test reporting therefore requires
an audit-version-2 checkpoint trained with this frozen exclusion already in
place. The old checkpoint remains usable for its original validation and
qualitative diagnostics; it is not silently relabeled as a valid final-test run.
New frozen-data training must use a fresh model output root. Existing version-1
training audits are not silently upgraded for resume.

Record/index the complete local bundle before cross-size use, without uploading:

```bash
make artifacts-index SIZE=mini DATASET_SIZE=mini
```

The command prints its source dataset `RUN_ID`. Keep this ID with the complete
frozen bundle. Upload only after reviewing the counts and retention choice.

## 3. Model size versus dataset size

`SIZE` chooses the model architecture/training recipe and model output paths.
`DATASET_SIZE` chooses the complete dataset source and defaults to `$(SIZE)`.
`DATASET_RUN_ID` names that source run and defaults to `$(RUN_ID)` for existing
commands. Source layouts remain separate; curation is not collapsed into one
mandatory shared dataset.

For example, after curating/tokenizing a 350M-source dataset and indexing or
restoring its complete artifact set:

```bash
make artifacts-index SIZE=350m
# Substitute the printed source ID for the placeholder below.
make pretrain SIZE=mini DATASET_SIZE=350m DATASET_RUN_ID=350m-YYYYMMDD-abcdef
make pretrain SIZE=125m DATASET_SIZE=350m DATASET_RUN_ID=350m-YYYYMMDD-abcdef
make pretrain SIZE=350m DATASET_SIZE=350m DATASET_RUN_ID=350m-YYYYMMDD-abcdef
```

Generate the corresponding model config for the intended GPU count before these
commands (see section 7). They are alternative runs, not a command to launch all
three jobs simultaneously.

Each consumer resolves the source tokenizer, train/val/test binaries, sidecars,
frozen contract, and provenance together. There is no fallback to a global or
Mini tokenizer when consuming a 350M dataset. Full binary checksums and the
source pipeline metadata are checked before training. This reads the artifact
bytes and can be storage-I/O intensive, but prevents a same-size middle-of-file
corruption from escaping a head/tail-only check. `SIZE`, `DATASET_SIZE`, source
`RUN_ID`, fingerprints, selected tokens, and schedule are recorded in the audit.
Outputs remain `$RESULTS_DIR/runs/<model-size>/pretrain`, not the dataset size.

`training.cross_size_max_train_tokens` in the **model** YAML caps the unique
training prefix when source and model sizes differ. An explicit
`training.max_train_tokens` overrides that cap, including for same-size use.
Same-size default behavior still uses the realized corpus. The existing epoch
contract resolves consumed tokens/steps from the selected prefix, with optimizer
step rounding reported. Holdouts are not sliced to the training budget.
`selected_unique_tokens` counts distinct token **positions** in selected windows,
not distinct vocabulary IDs, and is not a claim that a deliberately shortened
smoke/benchmark run has visited the entire prefix.

Implement and verify reuse before additional production curation. A 350M corpus
can be curated directly and reused for Mini/125M; there is no need to curate
125M merely as an intermediate step. Defer 1B curation until a concrete 1B
training plan exists.

## 4. Artifact backends and retention

The six existing stages remain `raw`, `curated`, `validated`, `tokenized`,
`tokenizer`, and `metadata`. **There is no separate test stage.** Every backend
uses `<size>/<run_id>/<stage>` under its configured prefix.

`ARTIFACT_BACKEND=s3` is the default. `ARTIFACT_BACKEND=hf` uses **HF Storage
Buckets**, not published Dataset repositories. Only the selected backend is
called. S3 keeps the SDK credential chain, including instance roles; static AWS
keys are not required by the environment gate. Its connection pool scales with
file-worker concurrency multiplied by per-file multipart concurrency.

For HF, create an appropriate Storage Bucket and grant the token access. Add
these operational settings to `.env` (never commit tokens):

```dotenv
ARTIFACT_BACKEND=hf
HF_ARTIFACT_BUCKET=your-namespace/your-artifact-bucket
HF_ARTIFACT_PREFIX=slm/data
```

The curation stack's pinned Hub version remains unchanged. Install the optional
isolated transfer environment with `make install-artifacts-hf`. Make selects
`.venv-artifacts/bin/python` when present; otherwise it uses the current Python.
Override `ARTIFACT_HF_PYTHON` explicitly when needed. A training environment with
the pinned Hub-1.x bucket APIs may use its own Python instead. The helper process
inherits `HF_TOKEN`; credentials are not serialized into transfer jobs.

Retention profiles:

| Profile | Transferred artifacts |
|---|---|
| `training-ready` (default) | Validated val/test text and frozen metadata; all three tokenized binaries/sidecars; tokenizer; metadata/provenance |
| `full` | All six complete stages; validated training text included |

Training-ready omits the large validated training text. Its validated completion
manifest explicitly records that projection in `_RETENTION.json` and retains
the original full manifest. It must not masquerade as a full validation output.
Tokenized training data stays with its exact tokenizer and metadata for fast
retraining. Curated frozen membership/manifests are copied into metadata
provenance even when curated text is not retained.

```bash
make artifacts-upload SIZE=mini DATASET_SIZE=350m \
  DATASET_RUN_ID=350m-YYYYMMDD-abcdef \
  ARTIFACT_BACKEND=hf ARTIFACT_RETENTION=training-ready

make artifacts-download SIZE=mini DATASET_SIZE=350m \
  DATASET_RUN_ID=350m-YYYYMMDD-abcdef \
  DATA_DIR=/data/slm/restored ARTIFACT_BACKEND=hf
```

Use `ARTIFACT_BACKEND=s3` for the same workflow on S3. Explicit
`ARTIFACT_STAGES` remains available; raw/curated selection requires `full`.
Profiles choose what to transfer; they do **not** silently delete older remote
objects or any local corpus stage. `curate-all` uploads the training-ready set.
Raw downloads, curation scratch, and redundant training text are deletion
candidates only after the frozen retained bundle is independently verified.
`make clean-data` refuses a frozen local run unless `ALLOW_FROZEN_DELETE=1` is
explicitly supplied; that override is destructive and is not a retention tool.

Each bundle has an inventory of full file SHA-256 hashes and sizes. The
completion descriptor is published **last**. The same RUN_ID cannot be reused to
overwrite different established test/tokenizer/binary identities. Restore
validates the selected inventory and stage contracts before promoting any stage.
A different existing local RUN_ID requires a fresh `DATA_DIR` or explicit
`ARTIFACT_OVERWRITE=1`. A power-interrupted restore leaves
`_RESTORE_PENDING.json`; training/upload refuse it. Inspect its recorded backup
and staging paths and recover a complete bundle before clearing the marker.
Ordinary failures roll back promoted stages.

Final models and evaluation artifacts are high-value retention items but can be
large. Include them explicitly, under the existing metadata stage:

```bash
make artifacts-upload SIZE=mini DATASET_SIZE=350m \
  DATASET_RUN_ID=350m-YYYYMMDD-abcdef ARTIFACT_INCLUDE_RESULTS=1
# Restore model results only into a fresh RESULTS_DIR/runs/mini:
make artifacts-download SIZE=mini DATASET_SIZE=350m \
  DATASET_RUN_ID=350m-YYYYMMDD-abcdef \
  DATA_DIR=/data/slm/restored RESULTS_DIR=/data/slm/restored-results \
  ARTIFACT_RESTORE_RESULTS=1
```

This archives final checkpoints/associated reports, not every intermediate
checkpoint. Results retain their **model-size** layout. Ordinary artifact restore
does not overwrite active training outputs. Keep raw test/validation text,
matched tokenized holdouts, tokenizer, manifests, and final model/evaluation
records for as long as those results need to remain reproducible.

## 5. Separate dependency stacks

`requirements.txt` now contains genuinely shared utilities. Install
`requirements-curation.txt` on CPU curation hosts, and the existing
`requirements-gpu.txt` / `requirements-training.txt` path on training hosts.
Curation retains DataTrove, KenLM/FastText, and its compatible Hub stack. Training
does not install those packages. Do not install both stacks into one environment.

`make install`, `install-uv`, `install-conda`, and CPU setup use the curation
requirements. GPU setup uses training requirements only. The optional
`make install-evaluation` installs lm-eval for benchmark tasks. The pretraining
loss/probe modes do not need lm-eval. HF artifact transfer can use its separate
`requirements-artifacts-hf.txt` environment as described above.

## 6. Evaluation and fixed qualitative probes

The existing pretraining loop saves train metrics and its final model before
final-test reporting. Training-time validation remains the Trainer's selection
metric. After training, `final_pretraining_eval.json` records final validation
loss/perplexity, test loss/perplexity, token accounting, selected-prefix
positions-per-parameter, and separate qualitative categories. A diagnostic
failure does not discard the already saved checkpoint. Qualitative probe
failures are recorded, not used as a training pass/fail gate.

Generation settings are YAML-driven:

```yaml
generation_probes:
  enabled: true
  every_steps: 5000
  steps: []
  at_final: true
  do_sample: false
  max_new_tokens: 64
final_evaluation:
  prefix_count: 5
```

The fixed prompts concern computing history, prime numbers, Python functions,
the water cycle, and neural-network learning. They are raw pretraining prompts,
not chat instructions. Logs contain checkpoint step, prompt, continuation, and
decoding settings. Callbacks preserve model training mode and random-generator
states. The prompt list is the same across sizes; there are no quality thresholds
and no automatic training decisions based on generated text.

Do not restart an existing Mini job to add probes. Use saved checkpoints:

```bash
make pretrain-probes SIZE=mini \
  PROBE_MODEL=/data/slm/results/runs/mini/pretrain/final
```

For an intermediate `checkpoint-<step>` lacking a bundled tokenizer, probes
read its parent run audit and verify the source/final tokenizer's fingerprint.
`PROBE_TOKENIZER=/path/to/original/tokenizer` selects an explicitly restored
copy and still requires that audit match. No global tokenizer is substituted.

For an audit-version-2 final checkpoint trained with the frozen exclusion:

```bash
make eval-pretrain-final SIZE=mini DATASET_SIZE=350m \
  DATASET_RUN_ID=350m-YYYYMMDD-abcdef
```

This extends `eval/eval.py` and reuses `SLMTrainer`, the memmap dataset, and
`inference.generate`; it is not a second evaluation framework. Ordinary
`eval-*` benchmark targets remain size-aware; the explicit `eval-mini` alias
uses Mini paths rather than the default 125M paths. Smoke is kept separate.

Categories are `held_out_test_prefix`, `generic_generalization`, and optional
`corpus_supported_qa`. Generic failure must not be interpreted as the same thing
as failure on unseen in-distribution test text. Corpus QA is not invented from
unsupported facts: `CORPUS_QA=/path/qa.jsonl` supplies up to 32 records with
`prompt`, `document_sha256` (SHA-256 of UTF-8 document text), verbatim `evidence`,
and `reference_answer`. The document must occur in frozen test and contain the
evidence. Reports retain source/line/hash provenance and reference answers for
review, rather than asserting an unvalidated automatic answer-quality score.
Without a supplied file, QA is explicitly reported as `not_provided`.

## 7. GPU count and controlled utilization experiments

In examples, replace `N` with the number of GPUs you choose to use. No GPU model
or count is mandated by these changes. Mini joins the existing configuration
generator and dynamic DDP flow; no new distributed implementation is introduced.

```bash
make config-gen SIZE=mini GPUS=N
make pretrain-preflight SIZE=mini GPUS=N
make pretrain SIZE=mini GPUS=N
```

`make pretrain-mini GPUS=N` runs that same config-generation step before launch.
It can replace the checked-in Mini YAML; inspect it first, or generate/review an
explicit config and use `make pretrain PRETRAIN_CONFIG=...`. Smoke continues to
use its bounded recipe separately. Existing unsupported-hardware/environment
gates are not weakened, and this work adds no extra topology checks.

Benchmark only when it will not interfere with an active training job:

```bash
make pretrain-benchmark SIZE=mini GPUS=N \
  BENCH_MICRO_BATCHES=2,4,8,16 BENCH_WARMUP=10 BENCH_STEPS=30 \
  BENCH_OUTPUT=/data/slm/results/benchmarks/mini-experiment-001
```

Candidates must divide the current global batch at the selected GPU count;
invalid choices fail rather than alter the batch. Each candidate changes only
micro-batch and its compensating gradient accumulation. Sequence length, model,
selected token budget, full-run scheduler, optimizer, and intended global batch
stay fixed. Floating-point grouping can still differ; this is not a bitwise
identity guarantee. Each trial starts separately with a reproducible seed and
fresh scratch output, stops by callback after warmup/measured steps, and never
rewrites the production YAML or checkpoint. No best candidate is automatically
adopted. The fixed BF16 launch matches the existing pretraining Make launcher.

Reports include global tokens/second, optimizer steps/second, per-rank sampled
GPU utilization, peak allocated/reserved VRAM, and validation throughput. Missing
`nvidia-smi` sampling is reported as unavailable, not inferred from memory usage.
There is no test evaluation during these comparisons. No hardware speedup is
claimed until these experiments are actually run on the intended server.

## 8. Required verification on the appropriate hosts

On the training stack, run `make test-frozen-contract` and:

```bash
.venv/bin/python -m pytest tests/model/test_rope_loading.py \
  tests/test_training_args.py tests/test_config_gen.py -v
make test-gpu-gate
```

The RoPE tests cover fresh and checkpoint-loaded models, exact learned-weight
preservation, FP32/BF16 finite logits, cached/non-cached greedy generation,
parameter-free frequency reconstruction, dtype/device transitions, and scratch
backpropagation. They do not substitute for reproducing the real Mini baseline.

On the curation stack, run the real existing MinHash integration and then the
artifact gates against rebuilt data:

```bash
.venv/bin/python -m pytest tests/test_frozen_split_integration.py -v
make test-data-pipeline SIZE=mini
```

Finally, perform a live selected-backend upload/restore into a fresh data root,
verify the actual train/val/test token counts and Mini floor, and test the
hardware configurations to be used. Mocked transport tests validate bundle
logic; they do not establish credentials, permissions, network reliability, or
live service behavior. Missing optional stacks cause the corresponding standalone
new integration test modules to skip, not silently stand in for a successful
server check.

API references for the transfer adapters:
[HF Storage Buckets](https://huggingface.co/docs/huggingface_hub/guides/buckets),
[HF bucket API](https://huggingface.co/docs/huggingface_hub/package_reference/hf_api),
[Boto3 transfer configuration](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/s3.html#concurrent-transfer-operations).
