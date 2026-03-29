# Notebooks

Interactive exploration and analysis across all pipeline stages.
Each notebook is self-contained and documents its prerequisites at the top.

---

## Overview

| Notebook | Purpose | Requires GPU | Run after |
|---|---|---|---|
| `data_exploration.ipynb` | Inspect curated output — retention funnel, length distributions, token counts | No | `make docker-curate` |
| `tokenizer_analysis.ipynb` | Vocab coverage, fertility vs GPT-2, special token validation | No | `make tokenizer` |
| `dataset_blend.ipynb` | Token counts, Chinchilla scaling analysis, WARC estimates | No | `make tokenizer` |
| `training_run.ipynb` | Launch any training stage interactively, monitor output, browse checkpoints | Yes | `make setup-instance` |
| `training_curves.ipynb` | Loss curves from W&B or log files, LR schedule visualization | No | Any training stage |
| `inference.ipynb` | Load any checkpoint, interactive generation, temperature sweep, batch inference | Yes | Any training stage |
| `model_comparison.ipynb` | Side-by-side responses across checkpoints on the same prompts | Yes | At least 2 checkpoints |
| `eval_analysis.ipynb` | Perplexity chart, MMLU by subject, win rate breakdown, generation inspector | No | `make eval-dpo` |

**Three notebooks run fine on CPU** (no GPU required): `data_exploration`, `tokenizer_analysis`, `dataset_blend`. Run these on the curation instance immediately after curation completes.

---

## Launching Jupyter

All notebooks assume the standard path layout (`/data`, `/results`, `/workspace/slm`).
Run Jupyter inside the Docker container so paths resolve correctly.

```bash
# CPU instance — data exploration, tokenizer analysis
make docker-shell-cpu
# inside the container:
pip install jupyter --quiet
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --notebook-dir=/workspace/slm/notebooks

# GPU instance — training, inference, model comparison
make docker-shell-gpu
# inside the container:
pip install jupyter --quiet
jupyter notebook --ip=0.0.0.0 --port=8888 --no-browser --allow-root \
    --notebook-dir=/workspace/slm/notebooks
```

Then open the URL printed in the terminal. If running on a remote instance, forward the port:

```bash
# On your local machine
ssh -L 8888:localhost:8888 <your-instance>
# then open http://localhost:8888 in your browser
```

---

## Execution Order

Not all notebooks are independent. The recommended order follows the pipeline:

```
make docker-curate
    └── data_exploration.ipynb       ← inspect curation output

make tokenizer
    └── tokenizer_analysis.ipynb     ← validate tokenizer
    └── dataset_blend.ipynb          ← token counts + Chinchilla analysis

make pretrain / sft / dpo
    └── training_run.ipynb           ← launch and monitor training
    └── training_curves.ipynb        ← visualize loss curves

make eval-pretrain / eval-sft / eval-dpo
    └── eval_analysis.ipynb          ← deep-dive into metrics

(any checkpoint)
    └── inference.ipynb              ← interact with the model
    └── model_comparison.ipynb       ← compare checkpoints side-by-side
```

---

## W&B Integration

`training_curves.ipynb` supports two data sources:

- **W&B** — set `USE_WANDB = True` and `WANDB_PROJECT = "slm"` at the top of the notebook. Requires a W&B account and `--wandb` flag during training.
- **Log files** — falls back to parsing NeMo log files from `/results/pretrain_logs/`, `/results/sft_logs/`, `/results/dpo_logs/` automatically if W&B is disabled.

To enable W&B logging during training:

```bash
make pretrain  # add --wandb flag in train.sh, or
bash pretrain/scripts/train.sh --wandb
```

---

## Notes

- **`model_comparison.ipynb`** loads two models simultaneously. Two 125M checkpoints fit comfortably on a single A6000 (48GB). If VRAM is tight, run them sequentially by re-running the load cell with a different checkpoint.
- **`inference.ipynb`** includes a `--no-chat-template` mode for testing pretrain checkpoints — raw text continuation rather than instruction following.
- **`dataset_blend.ipynb`** reads token counts from the `.idx` files directly — no model load required. The Chinchilla analysis updates automatically based on actual token counts from your curation run.
- All notebooks save output charts to `/results/eval/` for use in the README screenshots section.