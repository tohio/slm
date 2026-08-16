# Troubleshooting

This guide covers common environment, data-access, curation-resume, storage,
and artifact-transfer failures.

## Environment configuration

Confirm that `.env` contains no blank values or placeholders:

```bash
grep -nE '=\.\.\.|^[A-Z][A-Z0-9_]*=[[:space:]]*(#.*)?$' .env
```

No output means the check passed. Verify the installed package contract:

```bash
.venv/bin/python infra/verify_environment.py --profile curation
```

## Hugging Face dataset access

If a source returns `401`, `403`, a gated-repository error, or a
repository-not-found error, confirm access with the same account represented
by `HF_TOKEN`:

```bash
.venv/bin/python - <<'PY'
import os

from dotenv import load_dotenv
from huggingface_hub import HfApi

load_dotenv()
api = HfApi(token=os.environ["HF_TOKEN"])

for repo_id in (
    "bigcode/the-stack-dedup",
    "bigcode/the-stack-smol",
    "nvidia/Nemotron-CC-Math-v1",
):
    info = api.dataset_info(repo_id)
    print(repo_id, info.sha)
PY
```

If the check fails, sign in to that Hugging Face account, accept the dataset's
terms, and rerun it. Creating another token does not grant access to an account
that has not accepted the terms.

## Storage and resource failures

Check capacity, inodes, memory, and the open-file limit:

```bash
df -h /data
df -ih /data
free -h
ulimit -n
```

Do not delete stage manifests to recover space while expecting the stage to
remain resumable. The disk setup procedure is in
[`DISK_SETUP.md`](DISK_SETUP.md).

## Resume interrupted curation

Rerun the same command with the same profile and configuration:

```bash
make curate SIZE=125m WORKERS=62
```

Manifest-complete stages with matching input, implementation, configuration,
and output signatures are reused. Use `FORCE=1` only for a specifically
diagnosed stale or invalid stage; it is not part of normal resume.

Inspect a source without rebuilding it:

```bash
python curator/scripts/sample_source.py \
  --size 125m \
  --stage raw \
  --source wikipedia \
  --limit 10
```

## Stage validation failures

Run the gate for the stage that just completed:

```bash
make test-curator SIZE=125m
make test-validate SIZE=125m
make tokenizer-test SIZE=125m
make test-data-pipeline SIZE=125m
```

These commands inspect existing outputs. See [`TESTING.md`](TESTING.md) for
artifact requirements and the complete test policy.

## Artifact upload failures

Load `.env`, verify the AWS identity, and confirm access to the configured
prefix:

```bash
set -a
source .env
set +a

aws sts get-caller-identity
aws s3 ls "s3://$S3_BUCKET/$S3_PREFIX/"
```

Resume the upload with the same stage set:

```bash
make artifacts-upload \
  SIZE=125m \
  ARTIFACT_STAGES="tokenized,tokenizer,metadata"
```

The upload reuses the current day's local `RUN_ID`. For a later upload to the
same artifact run, pass the recorded value explicitly:

```bash
make artifacts-upload \
  SIZE=125m \
  RUN_ID=125m-YYYYMMDD-abcdef \
  ARTIFACT_STAGES="tokenized,tokenizer,metadata"
```

## GPU environment failures

On a training host, require the CUDA contract and then run the dataset-free
acceptance gate:

```bash
.venv/bin/python infra/verify_environment.py --require-cuda
make test-gpu-gate
```

See [`HARDWARE.md`](HARDWARE.md) for the supported hardware contract.

## See Also

- [Testing](TESTING.md)
- [Disk setup](DISK_SETUP.md)
- [Command reference](COMMANDS.md)
- [Curation component guide](../curator/README.md)
