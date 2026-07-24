# validation

Post-curation quality validation for train/val JSONL splits.

---

## Owns

- validation filtering after curation
- one canonical source-aware validation path
- KenLM perplexity filtering
- validated train/val outputs
- RUN_ID artifact integration

The curator does heuristic filtering first. This stage catches lower-quality prose that still passes curation.

---

## Key files

```text
validation/
└── scripts/
    └── validate.py
```

---

## Inputs

Default inputs:

```text
data/runs/<size>/curated/train.jsonl
data/runs/<size>/curated/val.jsonl
```

Optional model inputs:

```text
data/models/en.arpa.bin
data/models/lid.176.ftz
```

`en.arpa.bin` is used for KenLM perplexity filtering. `lid.176.ftz` is used for language detection.

---

## Outputs

```text
data/runs/<size>/validated/train.jsonl
data/runs/<size>/validated/val.jsonl
data/runs/<size>/metadata/
```

---

## Commands

Install validation prerequisites:

```bash
make install-kenlm
make download-fasttext-model DATA_DIR=/data/slm/data
make download-kenlm-model DATA_DIR=/data/slm/data
```

Run validation:

```bash
make validate SIZE=125m
```

Upload through the RUN_ID artifact flow:

```bash
make artifacts-upload SIZE=125m ARTIFACT_STAGES="validated,metadata"
```

Convenience alias for the same RUN_ID artifact flow:

```bash
make validate-upload SIZE=125m
```

Direct calls:

```bash
python validation/scripts/validate.py --size 125m
python validation/scripts/validate.py --size 125m --perplexity-threshold 800
python validation/scripts/validate.py --size 125m --no-perplexity
```

Override input/output paths:

```bash
python validation/scripts/validate.py   --train data/runs/125m/curated/train.jsonl   --val data/runs/125m/curated/val.jsonl   --train-output data/runs/125m/validated/train.jsonl   --val-output data/runs/125m/validated/val.jsonl
```

---

## Filters

Validation applies source-aware prose structure and repetition checks, plus
KenLM perplexity filtering. Language identification is enforced earlier by the
curator's quality stage.

| Filter | Purpose |
|---|---|
| terminal punctuation / prose structure | catches truncated or malformed prose |
| repeated lines | catches boilerplate and repeated text |
| KenLM perplexity | removes gibberish, spam, and unnatural text |

Code, synthetic/template-like, math, and specialized sources bypass prose-only heuristics where those heuristics would incorrectly reject useful non-prose records.

---

## Perplexity threshold

If `--perplexity-threshold` is not provided, validation samples train records and computes an automatic threshold. The same threshold is reused for val so train and val remain comparable.

KenLM is required by default. `--no-perplexity` is an explicit alternate
contract for debugging and is recorded in the completion manifest; missing
KenLM dependencies never silently disable filtering.

---

## Tests

```bash
make test-validate SIZE=mini
make test-validate SIZE=125m
```
