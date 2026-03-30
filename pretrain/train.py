"""
pretrain/train.py
-----------------
NeMo 2.x GPT pre-training entry point.
Called by pretrain/scripts/train.sh via torchrun or direct python.

Uses the NeMo 2.x LLM collection API:
  - nemo.collections.llm.GPTModel + GPTConfig
  - nemo.collections.llm.PreTrainingDataModule (mmap .bin/.idx)
  - nemo.lightning.Trainer + MegatronStrategy
  - megatron.core.optimizer.OptimizerConfig

All hyperparameters are passed via CLI arguments parsed by argparse.
No Hydra/YAML config required — configuration is pure Python.

Base container: nvcr.io/nvidia/nemo:25.02
"""

import argparse
import os
import torch

from nemo import lightning as nl
from nemo.collections import llm
from nemo.collections.common.tokenizers import SentencePieceTokenizer
from nemo.lightning.pytorch.callbacks import ModelCheckpoint
from megatron.core.optimizer import OptimizerConfig


def parse_args():
    parser = argparse.ArgumentParser(description="SLM GPT Pre-Training (NeMo 2.x)")

    # ── Model architecture ────────────────────────────────────────────────────
    parser.add_argument("--num-layers",          type=int,   default=12)
    parser.add_argument("--hidden-size",         type=int,   default=768)
    parser.add_argument("--ffn-hidden-size",     type=int,   default=3072)
    parser.add_argument("--num-attention-heads", type=int,   default=12)
    parser.add_argument("--seq-length",          type=int,   default=2048)

    # ── Parallelism ───────────────────────────────────────────────────────────
    parser.add_argument("--tensor-model-parallel-size",   type=int, default=1)
    parser.add_argument("--pipeline-model-parallel-size", type=int, default=1)
    parser.add_argument("--gpus",                         type=int, default=1)

    # ── Training ──────────────────────────────────────────────────────────────
    parser.add_argument("--max-steps",         type=int,   default=200000)
    parser.add_argument("--micro-batch-size",  type=int,   default=4)
    parser.add_argument("--global-batch-size", type=int,   default=32)
    parser.add_argument("--precision",         type=str,   default="bf16-mixed")
    parser.add_argument("--gradient-clip-val", type=float, default=1.0)
    parser.add_argument("--val-check-interval",type=int,   default=1000)
    parser.add_argument("--limit-val-batches", type=int,   default=50)
    parser.add_argument("--log-every-n-steps", type=int,   default=10)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    parser.add_argument("--lr",              type=float, default=3e-4)
    parser.add_argument("--min-lr",         type=float, default=3e-5)
    parser.add_argument("--weight-decay",   type=float, default=0.1)
    parser.add_argument("--warmup-steps",   type=int,   default=2000)

    # ── Data ──────────────────────────────────────────────────────────────────
    # Paths to mmap dataset prefixes (without .bin/.idx extension)
    # Format: --data-paths 0.7 /data/curated/tokenized/text_document
    #                       0.3 /data/curated/tokenized/text_document
    parser.add_argument("--data-paths",    type=str, nargs="+",
                        default=["1.0", "/data/curated/tokenized/text_document"])
    parser.add_argument("--split",         type=str, default="99,1,0")
    parser.add_argument("--num-workers",   type=int, default=4)

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    parser.add_argument("--tokenizer-model", type=str,
                        default="/data/tokenizer/slm_tokenizer.model")

    # ── Checkpointing ─────────────────────────────────────────────────────────
    parser.add_argument("--results-dir", type=str, default="/results/slm_gpt_125m")
    parser.add_argument("--wandb",       action="store_true")
    parser.add_argument("--resume",      action="store_true",
                        help="Resume from latest checkpoint if available")

    return parser.parse_args()


def main():
    args = parse_args()

    # ── Tokenizer ─────────────────────────────────────────────────────────────
    tokenizer = SentencePieceTokenizer(model_path=args.tokenizer_model)

    # ── Data ──────────────────────────────────────────────────────────────────
    # PreTrainingDataModule reads NeMo-compatible mmap .bin/.idx files.
    # paths accepts a flat list of [weight, prefix, weight, prefix, ...]
    data = llm.PreTrainingDataModule(
        paths=args.data_paths,
        seq_length=args.seq_length,
        micro_batch_size=args.micro_batch_size,
        global_batch_size=args.global_batch_size,
        split=args.split,
        num_workers=args.num_workers,
        tokenizer=tokenizer,
    )

    # ── Model config ──────────────────────────────────────────────────────────
    gpt_config = llm.GPTConfig(
        num_layers=args.num_layers,
        hidden_size=args.hidden_size,
        ffn_hidden_size=args.ffn_hidden_size,
        num_attention_heads=args.num_attention_heads,
        seq_length=args.seq_length,
        init_method_std=0.02,
        hidden_dropout=0.1,
        attention_dropout=0.1,
        layernorm_epsilon=1e-5,
        make_vocab_size_divisible_by=128,
        apply_query_key_layer_scaling=True,
        bias_activation_fusion=True,
        bias_dropout_fusion
        masked_softmax_fusion=True,
        activation_func=torch.nn.functional.gelu,
        normalization="LayerNorm",
        position_embedding_type="learned_absolute",
        share_embeddings_and_output_weights=False,
    )

    model = llm.GPTModel(gpt_config, tokenizer=tokenizer)

    # ── Optimizer ─────────────────────────────────────────────────────────────
    opt_config = OptimizerConfig(
        optimizer="adam",
        lr=args.lr,
        min_lr=args.min_lr,
        weight_decay=args.weight_decay,
        adam_beta1=0.9,
        adam_beta2=0.95,
        adam_eps=1e-8,
        bf16=True,
        use_distributed_optimizer=True,
    )

    opt = nl.MegatronOptimizerModule(
        config=opt_config,
        lr_scheduler=nl.lr_scheduler.CosineAnnealingScheduler(
            warmup_steps=args.warmup_steps,
            constant_steps=0,
            min_lr=args.min_lr,
        ),
    )

    # ── Strategy ──────────────────────────────────────────────────────────────
    strategy = nl.MegatronStrategy(
        tensor_model_parallel_size=args.tensor_model_parallel_size,
        pipeline_model_parallel_size=args.pipeline_model_parallel_size,
        pipeline_dtype=torch.bfloat16,
        gradient_as_bucket_view=True,
    )

    # ── Callbacks ─────────────────────────────────────────────────────────────
    checkpoint_callback = ModelCheckpoint(
        monitor="val_loss",
        save_top_k=3,
        mode="min",
        save_last=True,
        filename="slm_gpt--{val_loss:.2f}-{step}",
        every_n_train_steps=1000,
    )

    # ── Logger ────────────────────────────────────────────────────────────────
    logger = nl.NeMoLogger(
        log_dir=args.results_dir,
        name="slm_gpt",
        wandb=nl.WandbLogger(project="slm", name="gpt_pretrain") if args.wandb else None,
    )

    # ── Resume ────────────────────────────────────────────────────────────────
    resume = nl.AutoResume(
        resume_if_exists=args.resume,
        resume_ignore_no_checkpoint=True,
    )

    # ── Trainer ───────────────────────────────────────────────────────────────
    trainer = nl.Trainer(
        devices=args.gpus,
        num_nodes=1,
        accelerator="gpu",
        strategy=strategy,
        plugins=nl.MegatronMixedPrecision(precision=args.precision),
        max_steps=args.max_steps,
        log_every_n_steps=args.log_every_n_steps,
        val_check_interval=args.val_check_interval,
        limit_val_batches=args.limit_val_batches,
        gradient_clip_val=args.gradient_clip_val,
        callbacks=[checkpoint_callback],
    )

    # ── Train ─────────────────────────────────────────────────────────────────
    llm.pretrain(
        model=model,
        data=data,
        trainer=trainer,
        log=logger,
        optim=opt,
        resume=resume,
    )


if __name__ == "__main__":
    main()