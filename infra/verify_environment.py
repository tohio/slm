#!/usr/bin/env python3
"""Verify the pinned training stack and, optionally, its CUDA runtime."""

from __future__ import annotations

import argparse
import importlib.metadata
import subprocess

import torch


EXPECTED = {
    "torch": "2.13.0",
    "transformers": "5.14.1",
    "accelerate": "1.14.0",
    "trl": "1.9.0",
    "datasets": "5.0.0",
}
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


def verify_versions() -> None:
    failures = []
    for package, expected in EXPECTED.items():
        installed = importlib.metadata.version(package)
        print(f"{package}={installed}")
        if installed.split("+", 1)[0] != expected:
            failures.append(f"{package}: expected {expected}, found {installed}")
    if failures:
        raise SystemExit("Version mismatch:\n  " + "\n  ".join(failures))


def verify_cuda() -> None:
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
    args = parser.parse_args()

    verify_versions()
    if args.require_cuda:
        verify_cuda()
    print("Environment verification passed.")


if __name__ == "__main__":
    main()
