# validation

Additional quality validation on curated train/val JSONL. The main validation signal is KenLM perplexity filtering, with optional Datatrove mode.

---

## Responsibility

`validation/` owns:

- validation filtering after curation
- train/val validation outputs
- validation reports and stats
- optional Datatrove validation path

---

## Files

```text
validation/
└── scripts/
    ├── validate.py
    └── upload_validated.py
```

---

## Inputs

```text
data/runs/<size>/curated/train.jsonl
data/runs/<size>/curated/val.jsonl
data/models/en.arpa.bin
data/models/lid.176.ftz
```

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
make download-kenlm-model    DATA_DIR=/data/slm/data
```

Run validation:

```bash
make validate SIZE=125m
make validate-datatrove SIZE=125m
```

Legacy direct upload:

```bash
make validate-upload SIZE=125m
```

Preferred reusable artifact upload:

```bash
make artifacts-upload SIZE=125m ARTIFACT_STAGES="validated,metadata"
```

Direct calls:

```bash
python validation/scripts/validate.py
python validation/scripts/validate.py --use-datatrove
python validation/scripts/validate.py --perplexity-threshold 800
python validation/scripts/validate.py --no-perplexity
```

---

## Filters

Validation catches content that passes heuristic curation filters:

| Filter | What it catches |
|---|---|
| terminal punctuation | incomplete/truncated content |
| repeated n-grams | boilerplate and templated text |
| language detection | non-English content |
| KenLM perplexity | gibberish, spam, malformed text |

---

## Tests

```bash
make test-validate SIZE=mini
make test-validate SIZE=125m
```
