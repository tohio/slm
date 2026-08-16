#!/usr/bin/env python3
"""Verify the pinned training stack and, optionally, its CUDA runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess

TRAINING_EXPECTED = {
    "torch": "2.13.0",
    "transformers": "5.14.1",
    "accelerate": "1.14.0",
    "trl": "1.9.0",
    "datasets": "5.0.0",
}

CURATION_EXPECTED = {
    "transformers": "4.57.6",
    "datasets": "5.0.0",
    "tokenizers": "0.22.2",
    "huggingface-hub": "0.36.2",
    "fsspec": "2026.4.0",
    "datatrove": "0.9.0",
}

# Backwards-compatible training contract used by the GPU smoke test and
# existing imports.
EXPECTED = TRAINING_EXPECTED
MIN_CUDA_DRIVER = (580, 65, 6)


def _version_tuple(value: str) -> tuple[int, ...]:
    return tuple(int(part) for part in value.split("."))


def _driver_version() -> str:
    result = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=driver_version",
            "--format=csv,noheader",
        ],
        check=True,
        capture_output=True,
        text=True,
    )
    return result.stdout.splitlines()[0].strip()


def verify_versions(expected: dict[str, str] = TRAINING_EXPECTED) -> None:
    failures = []
    for package, expected_version in expected.items():
        installed = importlib.metadata.version(package)
        print(f"{package}={installed}")
        if installed.split("+", 1)[0] != expected_version:
            failures.append(
                f"{package}: expected {expected_version}, found {installed}"
            )
    if failures:
        raise SystemExit("Version mismatch:\n  " + "\n  ".join(failures))


def verify_cuda() -> None:
    import torch

    if not torch.cuda.is_available():
        raise SystemExit("CUDA is required but torch.cuda.is_available() is false")
    if torch.version.cuda != "13.0":
        raise SystemExit(f"Expected CUDA runtime 13.0, found {torch.version.cuda}")

    driver = _driver_version()
    if _version_tuple(driver) < MIN_CUDA_DRIVER:
        minimum = ".".join(str(part) for part in MIN_CUDA_DRIVER)
        raise SystemExit(f"Expected NVIDIA driver >= {minimum}, found {driver}")

    name = torch.cuda.get_device_name(0)
    capability = torch.cuda.get_device_capability(0)
    required_arch = f"sm_{capability[0]}{capability[1]}"
    compiled_arches = torch.cuda.get_arch_list()
    if required_arch not in compiled_arches:
        raise SystemExit(
            f"{name} requires {required_arch}, but torch contains {compiled_arches}"
        )
    if not torch.cuda.is_bf16_supported():
        raise SystemExit(f"{name} does not report BF16 support")

    print(f"cuda={torch.version.cuda}")
    print(f"driver={driver}")
    print(f"gpu={name}")
    print(f"capability={capability}")
    print(f"compiled_arches={compiled_arches}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--require-cuda",
        action="store_true",
        help="Also require CUDA 13.0, a compatible driver, native SM, and BF16",
    )
    parser.add_argument(
        "--profile",
        choices=("training", "curation"),
        default="training",
        help="Dependency contract to verify",
    )
    args = parser.parse_args()

    if args.require_cuda and args.profile != "training":
        parser.error("--require-cuda requires --profile training")

    expected = (
        TRAINING_EXPECTED
        if args.profile == "training"
        else CURATION_EXPECTED
    )
    verify_versions(expected)
    if args.require_cuda:
        verify_cuda()
    print("Environment verification passed.")


if __name__ == "__main__":
    main()
