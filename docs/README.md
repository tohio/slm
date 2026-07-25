# Documentation

Project-level references:

- [Architecture](ARCHITECTURE.md) — component ownership, model lineage, and
  artifact flow.
- [System diagram](architecture.svg) — pipeline and model-family overview.
- [Command Reference](COMMANDS.md) — Make targets, variables, and stage commands.
- [Curation](CURATION.md) — new-host setup through uploaded training artifacts.
- [Training](TRAIN.md) — artifact restore, pretraining, SFT, DPO, and export.
- [Disk Setup](DISK_SETUP.md) — persistent storage layout and instance setup.
- [Hardware Guide](HARDWARE.md) — GPU sizing and training-time estimates.
- [Testing](TESTING.md) — test layers, stage gates, and artifact requirements.
- [Troubleshooting](TROUBLESHOOTING.md) — setup, data access, resume, storage,
  and artifact-transfer diagnostics.

Component guides:

| Area | Guide |
|---|---|
| Shared configuration | [`config/README.md`](../config/README.md) |
| Hardware-aware config generation | [`config_gen/README.md`](../config_gen/README.md) |
| Curation | [`curator/README.md`](../curator/README.md) |
| Validation | [`validation/README.md`](../validation/README.md) |
| Tokenizer | [`tokenizer/README.md`](../tokenizer/README.md) |
| Pretraining | [`pretrain/README.md`](../pretrain/README.md) |
| SFT | [`finetune/README.md`](../finetune/README.md) |
| DPO | [`alignment/README.md`](../alignment/README.md) |
| Evaluation | [`eval/README.md`](../eval/README.md) |
| Export | [`export/README.md`](../export/README.md) |
| Inference | [`inference/README.md`](../inference/README.md) |
| Serving | [`serve/README.md`](../serve/README.md) |
| Infrastructure | [`infra/README.md`](../infra/README.md) |
| Tests | [`tests/README.md`](../tests/README.md) |
| Utilities | [`scripts/README.md`](../scripts/README.md) |
| Notebooks | [`notebooks/README.md`](../notebooks/README.md) |
