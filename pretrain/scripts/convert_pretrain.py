"""
pretrain/scripts/convert_pretrain.py
-------------------------------------
Converts a NeMo 2.x distributed checkpoint to mcore_gpt.nemo format
for NeMo-Aligner SFT/DPO.

NeMo 2.x checkpoint structure:
  <ckpt_dir>/
  ├── context/
  │   ├── model.yaml          ← NeMo 2.x config (GPTConfig dataclass)
  │   ├── io.json
  │   └── slm_tokenizer.model
  └── weights/
      ├── metadata.json
      ├── common.pt
      └── (sharded weight files)

mcore_gpt.nemo structure (tar file):
  ├── model_config.yaml       ← flat NeMo 1.x model config (no nested model: key)
  ├── model_weights/          ← same distributed checkpoint files
  │   ├── metadata.json
  │   ├── common.pt
  │   └── (sharded weight files)
  └── slm_tokenizer.model     ← tokenizer

Usage:
  python pretrain/scripts/convert_pretrain.py \
      --input /results/slm_gpt_125m \
      --output /results/slm_gpt_125m/mcore_gpt.nemo \
      --tokenizer /data/tokenizer/slm_tokenizer.model
"""

import argparse
import shutil
import tarfile
import tempfile
from pathlib import Path

import yaml


def parse_args():
    parser = argparse.ArgumentParser(description="Convert NeMo 2.x checkpoint to mcore_gpt.nemo")
    parser.add_argument("--input",     required=True, help="Path to NeMo 2.x checkpoint directory or results dir")
    parser.add_argument("--output",    required=True, help="Output path for mcore_gpt.nemo")
    parser.add_argument("--tokenizer", required=True, help="Path to SentencePiece tokenizer model")
    return parser.parse_args()


def load_nemo2_config(context_dir: Path) -> dict:
    model_yaml = context_dir / "model.yaml"
    if not model_yaml.exists():
        raise FileNotFoundError(f"model.yaml not found in {context_dir}")
    with open(model_yaml) as f:
        return yaml.safe_load(f)


def build_nemo1_config(nemo2_cfg: dict) -> dict:
    """
    Translate NeMo 2.x GPTConfig to flat NeMo 1.x model_config.yaml.

    CRITICAL: model_config.yaml must be FLAT — all fields at top level,
    NOT nested under a 'model:' key. NeMo-Aligner reads this directly
    as the model config dict via OmegaConf.
    """
    cfg = nemo2_cfg.get("config", nemo2_cfg)

    num_layers           = cfg.get("num_layers", 12)
    hidden_size          = cfg.get("hidden_size", 768)
    ffn_hidden_size      = cfg.get("ffn_hidden_size", 3072)
    num_attention_heads  = cfg.get("num_attention_heads", 12)
    seq_length           = cfg.get("seq_length", 2048)
    hidden_dropout       = cfg.get("hidden_dropout", 0.1)
    attention_dropout    = cfg.get("attention_dropout", 0.1)
    layernorm_epsilon    = cfg.get("layernorm_epsilon", 1e-5)
    init_method_std      = cfg.get("init_method_std", 0.02)
    normalization        = cfg.get("normalization", "LayerNorm")
    position_embedding   = cfg.get("position_embedding_type", "learned_absolute")
    share_embeddings     = cfg.get("share_embeddings_and_output_weights", False)
    make_vocab_divisible = cfg.get("make_vocab_size_divisible_by", 128)
    tp_size              = cfg.get("tensor_model_parallel_size", 1)
    pp_size              = cfg.get("pipeline_model_parallel_size", 1)
    bias_dropout_fusion  = cfg.get("bias_dropout_fusion", False)
    masked_softmax_fusion = cfg.get("masked_softmax_fusion", True)

    norm_map = {"LayerNorm": "layernorm", "RMSNorm": "rmsnorm"}
    norm = norm_map.get(normalization, normalization.lower())

    return {
        "mcore_gpt": True,
        "megatron_amp_O2": True,
        "num_layers": num_layers,
        "hidden_size": hidden_size,
        "ffn_hidden_size": ffn_hidden_size,
        "num_attention_heads": num_attention_heads,
        "seq_length": seq_length,
        "max_position_embeddings": seq_length,
        "encoder_seq_length": seq_length,
        "kv_channels": None,
        "num_query_groups": None,
        "hidden_dropout": hidden_dropout,
        "attention_dropout": attention_dropout,
        "ffn_dropout": 0.0,
        "normalization": norm,
        "layernorm_epsilon": layernorm_epsilon,
        "init_method_std": init_method_std,
        "use_scaled_init_method": True,
        "apply_query_key_layer_scaling": False,
        "attention_softmax_in_fp32": False,
        "position_embedding_type": position_embedding,
        "share_embeddings_and_output_weights": share_embeddings,
        "make_vocab_size_divisible_by": make_vocab_divisible,
        "bias": True,
        "bias_gelu_fusion": True,
        "masked_softmax_fusion": masked_softmax_fusion,
        "bias_dropout_add_fusion": bias_dropout_fusion,
        "gradient_as_bucket_view": True,
        "activation": "gelu",
        "tensor_model_parallel_size": tp_size,
        "pipeline_model_parallel_size": pp_size,
        "virtual_pipeline_model_parallel_size": None,
        "sequence_parallel": False,
        "native_amp_init_scale": 4294967296,
        "native_amp_growth_interval": 1000,
        "fp16_lm_cross_entropy": False,
        "activations_checkpoint_granularity": None,
        "activations_checkpoint_method": None,
        "activations_checkpoint_num_layers": None,
        "tokenizer": {
            "library": "sentencepiece",
            "type": None,
            "model": "slm_tokenizer.model",
            "vocab_file": None,
            "merge_file": None,
        },
        "data": {
            "data_impl": "mmap",
            "splits_string": "99,1,0",
            "seq_length": seq_length,
            "skip_warmup": True,
            "num_workers": 0,
            "dataloader_type": "single",
            "eod_mask_loss": False,
            "data_prefix": [],
        },
        "optim": {
            "name": "distributed_fused_adam",
            "lr": 1e-5,
            "weight_decay": 0.01,
            "betas": [0.9, 0.98],
            "sched": {
                "name": "CosineAnnealing",
                "warmup_steps": 50,
                "constant_steps": 0,
                "min_lr": 0.0,
            },
        },
    }


def build_nemo_tarball(ckpt_dir: Path, output_path: Path, tokenizer_path: Path):
    context_dir = ckpt_dir / "context"
    weights_dir = ckpt_dir / "weights"

    if not context_dir.exists():
        raise FileNotFoundError(f"context/ not found in {ckpt_dir}")
    if not weights_dir.exists():
        raise FileNotFoundError(f"weights/ not found in {ckpt_dir}")

    print(f"Loading NeMo 2.x config from {context_dir / 'model.yaml'}")
    nemo2_cfg = load_nemo2_config(context_dir)

    print("Translating config to flat NeMo 1.x format...")
    nemo1_cfg = build_nemo1_config(nemo2_cfg)

    with tempfile.TemporaryDirectory() as tmpdir:
        tmp = Path(tmpdir)

        config_path = tmp / "model_config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(nemo1_cfg, f, default_flow_style=False, sort_keys=True)
        print(f"  ✓ model_config.yaml written (flat, {len(nemo1_cfg)} top-level keys)")

        tokenizer_dest = tmp / "slm_tokenizer.model"
        shutil.copy2(tokenizer_path, tokenizer_dest)
        print(f"  ✓ tokenizer copied")

        weights_dest = tmp / "model_weights"
        shutil.copytree(weights_dir, weights_dest)
        file_count = sum(1 for _ in weights_dest.rglob("*") if _.is_file())
        print(f"  ✓ weights copied ({file_count} files)")

        output_path.parent.mkdir(parents=True, exist_ok=True)
        print(f"Building tarball: {output_path}")
        with tarfile.open(output_path, "w") as tar:
            tar.add(config_path,    arcname="model_config.yaml")
            tar.add(tokenizer_dest, arcname="slm_tokenizer.model")
            tar.add(weights_dest,   arcname="model_weights")

    size_gb = output_path.stat().st_size / 1024**3
    print(f"✓ mcore_gpt.nemo created: {output_path} ({size_gb:.2f} GB)")


def find_best_checkpoint(results_dir: Path) -> Path:
    candidates = list(results_dir.glob("*/*/checkpoints/*-last"))
    valid = [c for c in candidates if (c / "context").exists() and (c / "weights").exists()]
    if not valid:
        raise FileNotFoundError(f"No valid NeMo 2.x checkpoints found under {results_dir}")

    def extract_val_loss(p: Path) -> float:
        try:
            return float(p.name.split("val_loss=")[1].split("-")[0])
        except (IndexError, ValueError):
            return float("inf")

    best = sorted(valid, key=extract_val_loss)[0]
    print(f"Best checkpoint: {best.name} (val_loss={extract_val_loss(best):.4f})")
    return best


def main():
    args = parse_args()
    input_path     = Path(args.input)
    output_path    = Path(args.output)
    tokenizer_path = Path(args.tokenizer)

    if not tokenizer_path.exists():
        raise FileNotFoundError(f"Tokenizer not found: {tokenizer_path}")

    if not (input_path / "context").exists():
        print(f"Searching for best checkpoint in {input_path}...")
        ckpt_dir = find_best_checkpoint(input_path)
    else:
        ckpt_dir = input_path

    print(f"\n=== NeMo 2.x → mcore_gpt.nemo Conversion ===")
    print(f"Input:     {ckpt_dir}")
    print(f"Output:    {output_path}")
    print(f"Tokenizer: {tokenizer_path}")
    print()

    build_nemo_tarball(ckpt_dir, output_path, tokenizer_path)

    print("\n✓ Conversion complete. Ready for NeMo-Aligner SFT/DPO.")
    print(f"  Use: model.restore_from_path={output_path}")


if __name__ == "__main__":
    main()