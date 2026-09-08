# Notebooks

These notebooks provide interactive inspection of each pipeline stage. They
are exploratory companions to the command-line workflow, not the source of
truth for production runs.

| Notebook | Focus |
|---|---|
| `01_model_exploration.ipynb` | Model construction, shapes, and parameter counts |
| `02_data_exploration.ipynb` | Source and curated-data inspection |
| `03_validation_exploration.ipynb` | Validation rules and rejection statistics |
| `04_tokenizer_exploration.ipynb` | Vocabulary, special tokens, and fertility |
| `05_pretrain_exploration.ipynb` | Base-training artifacts and diagnostics |
| `06_sft_exploration.ipynb` | SFT datasets and checkpoints |
| `07_dpo_exploration.ipynb` | Preference data and DPO outputs |
| `08_eval_exploration.ipynb` | Benchmark result analysis |
| `09_inference_exploration.ipynb` | Generation behavior |

The curation stack supplies notebook tools for data/validation/tokenizer
inspection (02–04). Register that kernel on the curation host:

```bash
make setup-curate
source .venv/bin/activate
python -m ipykernel install --user --name slm-curate --display-name "SLM curation"
jupyter lab
```

Model-facing notebooks (01 and 05–09) require the pinned training environment,
not the curation Transformers version. The training installer does not include
Jupyter/kernel tooling; install that optional tooling in the training environment
before registering a separate kernel. Do not merge the two requirement stacks.

Set `DATA_DIR`, `RESULTS_DIR`, and `EXPORTS_DIR` before launching Jupyter so
notebooks resolve the same run-scoped artifacts as the CLI.

Do not use notebook cells to bypass stage manifests, overwrite final
checkpoints, or mutate prepared datasets. Re-run production work through the
documented Make or Python entry points, then use notebooks to inspect the
result.
