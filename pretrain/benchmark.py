"""Bounded micro-batch experiments using the existing Trainer / DDP launcher.

Each candidate runs in an isolated subprocess/output directory. The original
YAML, production checkpoints, full-run scheduler and token budget are unchanged.
"""
from __future__ import annotations
import argparse
import copy
import json
import os
import subprocess
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from curator.state import atomic_write_json


def candidate_configs(config: dict, micro_batches: list[int], gpus: int) -> list[dict]:
    if gpus < 1:
        raise ValueError("gpus must be positive")
    training = config["training"]
    global_batch = int(training["micro_batch_size"]) * int(training["gradient_accumulation_steps"]) * gpus
    candidates = []
    for micro in dict.fromkeys(micro_batches):
        if micro < 1 or global_batch % (micro * gpus):
            raise ValueError(f"micro_batch={micro} does not preserve global_batch={global_batch} with GPUS={gpus}")
        candidate = copy.deepcopy(config)
        candidate["training"]["micro_batch_size"] = micro
        candidate["training"]["gradient_accumulation_steps"] = global_batch // (micro * gpus)
        candidates.append(candidate)
    if not candidates:
        raise ValueError("At least one micro-batch candidate is required")
    return candidates


def make_measurement_callback(warmup_steps: int, measured_steps: int):
    import threading
    import time
    import torch
    from transformers import TrainerCallback
    if warmup_steps < 1 or measured_steps < 1:
        raise ValueError("Benchmark warmup and measured steps must be positive")

    class Measurement(TrainerCallback):
        started = None
        elapsed = None
        samples = None
        stop_event = None
        worker = None

        def sample_utilization(self):
            device = torch.cuda.current_device()
            properties = torch.cuda.get_device_properties(device)
            identity = getattr(properties, "uuid", None)
            visible = os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",")
            identity = str(identity) if identity else (visible[device] if visible != [""] and device < len(visible) else str(device))
            while not self.stop_event.wait(0.5):
                try:
                    result = subprocess.run(["nvidia-smi", "-i", identity,
                        "--query-gpu=utilization.gpu", "--format=csv,noheader,nounits"],
                        text=True, capture_output=True, timeout=2, check=True)
                    self.samples.append(float(result.stdout.strip()))
                except (OSError, ValueError, subprocess.SubprocessError):
                    # Utilization is unavailable rather than inferred from VRAM.
                    return

        def on_step_end(self, args, state, control, **kwargs):
            if state.global_step == warmup_steps:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
                    self.stop_event = threading.Event()
                    self.samples = []
                    self.worker = threading.Thread(target=self.sample_utilization, daemon=True)
                    self.worker.start()
                self.started = time.perf_counter()
            if state.global_step >= warmup_steps + measured_steps:
                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                self.elapsed = time.perf_counter() - self.started
                if self.stop_event:
                    self.stop_event.set()
                    self.worker.join(timeout=3)
                control.should_training_stop = True
                control.should_save = False
                control.should_evaluate = False
            return control

        def metrics(self):
            if self.elapsed is None:
                raise RuntimeError("Training ended before all benchmark steps completed")
            return {"seconds": self.elapsed, "measured_steps": measured_steps,
                "gpu_utilization_percent": sum(self.samples) / len(self.samples) if self.samples else None,
                "gpu_utilization_samples": len(self.samples or []),
                "vram_allocated_peak_bytes": torch.cuda.max_memory_allocated() if torch.cuda.is_available() else 0,
                "vram_reserved_peak_bytes": torch.cuda.max_memory_reserved() if torch.cuda.is_available() else 0,
                "device": torch.cuda.get_device_name() if torch.cuda.is_available() else "cpu"}
    return Measurement()


def finish_measurement(trainer, callback, cfg, run_contract, output: Path, seq_len: int):
    import torch
    local = callback.metrics()
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    ranks = [None] * world_size
    if torch.distributed.is_available() and torch.distributed.is_initialized():
        torch.distributed.all_gather_object(ranks, local)
    else:
        ranks = [local]
    # Only validation is used for throughput comparisons. Test is never opened.
    evaluation = trainer.evaluate(metric_key_prefix="benchmark_validation")
    global_batch = cfg["training"]["micro_batch_size"] * cfg["training"]["gradient_accumulation_steps"] * world_size
    seconds = max(rank["seconds"] for rank in ranks)
    payload = {"mode": "hardware_benchmark", "production_config_modified": False,
        "global_batch": global_batch, "sequence_length": seq_len,
        "micro_batch_size": cfg["training"]["micro_batch_size"],
        "gradient_accumulation_steps": cfg["training"]["gradient_accumulation_steps"],
        "tokens_per_second": local["measured_steps"] * global_batch * seq_len / seconds,
        "steps_per_second": local["measured_steps"] / seconds,
        "ranks": ranks, "validation_throughput": evaluation,
        "training_contract": run_contract}
    if int(os.environ.get("RANK", "0")) == 0:
        atomic_write_json(output, payload)
    return payload


def main():
    import yaml
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--micro-batches", required=True, help="Comma-separated divisors of the current global batch")
    parser.add_argument("--gpus", type=int, default=1)
    parser.add_argument("--accelerate", default=".venv/bin/accelerate")
    parser.add_argument("--warmup-steps", type=int, default=10)
    parser.add_argument("--steps", type=int, default=30)
    parser.add_argument("--dataset-size")
    parser.add_argument("--dataset-run-id")
    parser.add_argument("--data-dir", type=Path, default=Path(os.environ.get("DATA_DIR", "data")))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = yaml.safe_load(args.config.read_text())
    candidates = candidate_configs(cfg, [int(value) for value in args.micro_batches.split(",")], args.gpus)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    outcomes = []
    for candidate in candidates:
        micro = candidate["training"]["micro_batch_size"]
        destination = args.output_dir / f"micro-{micro}"
        destination.mkdir(exist_ok=False)
        config_path = destination / "config.yaml"
        config_path.write_text(yaml.safe_dump(candidate, sort_keys=False))
        report = destination / "measurement.json"
        command = [args.accelerate, "launch", "--num_processes", str(args.gpus),
            "--num_machines", "1", "--mixed_precision", "bf16", "--dynamo_backend", "no",
            "pretrain/train.py", "--config", str(config_path), "--data-dir", str(args.data_dir),
            "--benchmark-steps", str(args.steps), "--benchmark-warmup", str(args.warmup_steps),
            "--benchmark-output", str(report)]
        if args.dataset_size:
            command.extend(["--dataset-size", args.dataset_size])
        if args.dataset_run_id:
            command.extend(["--dataset-run-id", args.dataset_run_id])
        # Preserve the existing single-host launch mechanism, including DDP.
        result = subprocess.run(command, check=False, cwd=Path(__file__).resolve().parents[1])
        outcomes.append({"micro_batch": micro, "exit_code": result.returncode,
                         "report": str(report) if report.exists() else None})
        atomic_write_json(args.output_dir / "summary.json", {"candidates": outcomes})
    if not any(row["exit_code"] == 0 for row in outcomes):
        raise SystemExit("No candidate completed; inspect the subprocess errors")


if __name__ == "__main__":
    main()
