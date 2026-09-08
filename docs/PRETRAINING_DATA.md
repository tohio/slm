# Pretraining data and artifact reuse

Reference for corpus builders and training operators: split roles, matched
artifacts, cross-size budgets, storage routing, and final evaluation.

## Train, validation, and test

`train` supplies gradient updates. `val` supplies training-time loss/perplexity,
tuning, and checkpoint selection. `test` is reserved for final model evaluation.
Integrity checks may read test bytes before training; they do not evaluate the
model on test or use test metrics to make training decisions.

Normal curation establishes the test split at the end of blending. It preserves
the existing `val.jsonl` byte-for-byte and selects test documents from training
using seeded ranks of normalized exact hashes. The default test target is
approximately 0.5% of the original **document count**, not an exact token
percentage. Together with validation this targets roughly 99% train / 0.5% val /
0.5% test. Actual counts depend on lengths and overlap removals.

All three split pairs use the existing exact-hash and DataTrove MinHash policy.
Validation has priority over test, then training. Reported remediation counts
must match physical removals before staged data is promoted. MinHash remains a
probabilistic near-duplicate detector, not a guarantee of semantic disjointness.

`test_contract.json` records split hashes/counts, policy, original blend identity,
and pair-audit provenance. `test_membership.jsonl` records text-free membership.
Repeated blending verifies this identity and compares current source signatures,
budgets, mix, seed, and implementation with the saved blend manifest rather than
reshuffling test or silently accepting a stale corpus. A changed
corpus/policy requires an explicit new data run; do not discard provenance to
force reuse. Validation applies the same source-aware filtering policy to all
three splits and records rejection statistics. Tokenizer training uses only
validated **train** text.

```text
$DATA_DIR/runs/<dataset-size>/
  curated/{train,val,test}.jsonl
  curated/test_contract.json, test_membership.jsonl, _SUCCESS.json
  validated/{train,val,test}.jsonl
  validated/test_contract.json, validation_stats.json, _SUCCESS.json
  tokenizer/slm_tokenizer.json, tokenizer_config.json, ...
  tokenized/{train,val,test}.bin
  tokenized/{train,val,test}.json
  tokenized/test_contract.json, token_mixture.json, _SUCCESS.json
  metadata/pipeline_manifest.json, provenance/...
  RUN_ID
```

All binaries receive the same token-count, byte-count, token-range, BOS/EOS,
checksum, and tokenizer-fingerprint checks. Mini requires at least **1.4B usable
selected train tokens** after sequence-window truncation. A short corpus is not
silently padded; cross-size reuse does not waive that floor.

## Model size versus dataset size

`SIZE` selects architecture, training recipe, and model output paths.
`DATASET_SIZE` selects the complete dataset source and defaults to `SIZE`.
`DATASET_RUN_ID` identifies that source run and defaults to `RUN_ID`. Separate
size-scoped dataset directories remain supported.

After preparing and indexing or restoring a complete 350M dataset run:

```bash
make config-gen SIZE=mini GPUS=1
make pretrain SIZE=mini DATASET_SIZE=350m DATASET_RUN_ID=350m-YYYYMMDD-abcdef
```

Substitute the actual recorded source ID. The same pattern supports 125M and
350M consumers. A consumer resolves the dataset tokenizer, all three binaries,
sidecars, test contract, and metadata together; it never falls back to a Mini
or global tokenizer. Outputs stay under `$RESULTS_DIR/runs/<model-size>/`.
The training audit records both sizes, source run ID, fingerprints, selected
tokens, and schedule.

`training.cross_size_max_train_tokens` in the **model YAML** limits the unique
training prefix when sizes differ. `training.max_train_tokens` explicitly
overrides that cap, including for same-size use. Same-size defaults use the
realized corpus. Existing epoch and optimizer-step rounding rules determine the
consumed training budget; holdouts are not sliced to that budget.

The pretraining final checkpoint carries that dataset tokenizer forward.
Instruct/code SFT, DPO preparation/training, and export use the actual input
checkpoint bundle, never `runs/<model-size>/tokenizer` as a fallback. DPO includes
the active rendering files in its preparation fingerprint. See
[post-training identity](TRAIN.md#post-training-identity-and-resume).

## User-selected artifact transfer

The existing stages are `raw`, `curated`, `validated`, `tokenized`, `tokenizer`,
and `metadata`. There is **no separate test stage**. Selecting `validated`
includes its train/val/test text; selecting `tokenized` includes all three
binaries and sidecars. Metadata includes test fingerprints/counts.

There are no retention profiles, automatic train-file omission, or automatic
local cleanup. You select stages with `ARTIFACT_STAGES`; the default artifact
upload/download selection remains `raw,tokenized,tokenizer,metadata`.

```bash
make artifacts-upload SIZE=mini ARTIFACT_STAGES=validated,tokenized,tokenizer,metadata
make artifacts-download SIZE=mini RUN_ID=mini-YYYYMMDD-abcdef \
  ARTIFACT_STAGES=validated,tokenized,tokenizer,metadata
```

`ARTIFACT_OVERWRITE=1` is explicit replacement: selected artifact-stage uploads
also mirror the selected remote stage, removing stale remote objects. Normal
uploads skip existing keys. As before, selected `metadata` is the mutable run
index and is refreshed after data. No unselected stage is uploaded implicitly.
Avoid partial `--glob` selection for stages intended for verified restoration.

Downloads preserve the existing size-based skip/explicit-overwrite behavior,
then verify the requested stages. An interrupted/failed restore leaves a marker
that blocks training; repeat with the same run/stages, using explicit overwrite
to replace corrupt same-size files. Use a fresh data root when switching source
runs rather than mixing datasets.

### S3 and Hugging Face routing

`ARTIFACT_BACKEND=s3` retains S3 object storage and its SDK credential chain.
`ARTIFACT_BACKEND=hf` routes **only the selected artifacts**:

| Selected content | HF destination |
|---|---|
| `curated`, `validated` JSONL stages and their stage manifests | Dataset repository |
| `raw`, `tokenized`, `tokenizer`, top-level `metadata` | Storage Bucket |
| Arbitrary operational objects, checkpoints, evaluation files | Storage Bucket through generic object transfer |
| Completed model and tokenizer explicitly published through export | Model repository |

Both artifact destinations preserve `<prefix>/<size>/<run_id>/<stage>`.
Dataset card configurations expose only the real train/validation/test JSONL,
not manifests or the membership index. A missing Dataset repository is created
**private**; existing repository visibility is preserved. A restore pins one
Dataset commit for all its dataset stages.

Curation does not need a model. The pretraining tokenizer belongs in the Bucket
with its exact dataset run. Model publication happens only through the existing
post-training export commands; curation never creates a Model repository.

For Dataset transfers configure `HF_DATASET_REPO=namespace/dataset-name` and
`HF_TOKEN`. For Bucket objects, create an HF Bucket and HF S3 credentials, then
configure:

```text
HF_ARTIFACT_BUCKET=namespace/bucket-name
HF_ARTIFACT_PREFIX=slm/data
HF_S3_ACCESS_KEY_ID=<HF-generated access key>
HF_S3_SECRET_ACCESS_KEY=<HF-generated secret>
```

HF Bucket credentials are **not AWS credentials and not HF_TOKEN**. The adapter
uses HF's S3-compatible gateway through the role's existing boto3 installation;
no separate transfer environment or Hub-version upgrade is required. Dataset-only
pushes do not require Bucket credentials. Object-only pushes do not require a
Dataset repository.

```bash
make artifacts-upload SIZE=mini ARTIFACT_BACKEND=hf ARTIFACT_STAGES=curated
make artifacts-upload SIZE=mini ARTIFACT_BACKEND=hf ARTIFACT_STAGES=tokenizer,tokenized,metadata
make artifacts-download SIZE=mini DATASET_SIZE=350m DATASET_RUN_ID=350m-YYYYMMDD-abcdef \
  ARTIFACT_BACKEND=hf ARTIFACT_STAGES=validated,tokenized,tokenizer,metadata
```

Generic Bucket objects use the same transfer CLI as S3; they are not forced into
Dataset repositories:

```bash
.venv/bin/python curator/scripts/upload_s3.py upload --backend hf \
  --src ./operational-files --dst mini/mini-YYYYMMDD-abcdef/checkpoints
.venv/bin/python curator/scripts/upload_s3.py list --backend hf --prefix mini/
.venv/bin/python curator/scripts/upload_s3.py download --backend hf \
  --src mini/mini-YYYYMMDD-abcdef/checkpoints --dst ./restored-checkpoints
```

Only the selected backend is called. Connection pools reflect outer upload
workers and multipart concurrency. Bucket transfer is an object-storage workflow,
not a claim that every S3 API is implemented by HF.

## Final evaluation and qualitative probes

Pretraining reports training loss and validation loss/perplexity. Final test
loss/perplexity, held-out test-prefix completions, generic prompts, and supplied
corpus-supported QA remain separate categories. Failure on a generic prompt is
not equivalent to failure on held-out in-distribution documents.

```bash
make eval-pretrain-final SIZE=mini DATASET_SIZE=350m DATASET_RUN_ID=350m-YYYYMMDD-abcdef
make pretrain-probes SIZE=mini
```

Test-document completions require the matched validated JSONL in addition to
binaries. Include `validated` when restoring that evaluation input.
`generation_probes` in the model YAML controls sparse step cadence and final
completion; defaults are greedy decoding and 64 new tokens. Probes log step,
prompt, continuation, and settings and are never pass/fail gates.

Normal model loading reconstructs derived RoPE state inside the model without
changing learned weights. Existing checkpoints remain paired with their original
tokenizer/data. A test selected from data an older checkpoint already trained on
is not an unseen test for that checkpoint. Final-test evaluation requires matching
training provenance; old-checkpoint validation is a separate evaluation mode.

## See Also

- [Curation](CURATION.md), [Training](TRAIN.md), [Command reference](COMMANDS.md)
- [Infrastructure setup](../infra/README.md), [Model loading](../model/README.md)
- [HF S3 gateway](https://huggingface.co/docs/hub/storage-buckets-s3)
- [HF Dataset configurations](https://huggingface.co/docs/hub/datasets-manual-configuration)
