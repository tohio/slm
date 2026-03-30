"""
pretrain/train.py
-----------------
NeMo 1.x GPT pre-training entry point.

Thin wrapper around NeMo's built-in megatron_gpt_pretraining.py.
Passes our YAML config and CLI overrides through to the NeMo script.

All hyperparameters come from pretrain/configs/gpt_<size>.yaml.
Saves checkpoint in .nemo format directly — no conversion needed.

NeMo-Aligner SFT/DPO can load the .nemo checkpoint directly.

Base container: nvcr.io/nvidia/nemo:25.02
"""

import sys
import subprocess
from pathlib import Path


NEMO_PRETRAIN_SCRIPT = Path("/opt/NeMo/examples/nlp/language_modeling/megatron_gpt_pretraining.py")


def main():
    if not NEMO_PRETRAIN_SCRIPT.exists():
        print(f"ERROR: NeMo pretraining script not found: {NEMO_PRETRAIN_SCRIPT}")
        sys.exit(1)

    # Pass all args through to the NeMo script unchanged
    cmd = [sys.executable, str(NEMO_PRETRAIN_SCRIPT)] + sys.argv[1:]
    result = subprocess.run(cmd)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()