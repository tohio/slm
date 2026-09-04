from __future__ import annotations

import hashlib
import json
import logging
import random
from pathlib import Path
from typing import Any, Callable

import orjson
from curator.sources.hf import load_dataset

from config import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


class HFSyntheticSource:
    """Hugging Face backed synthetic source.

    Synthetic datasets are generated and validated outside this repository by
    `tohio/slm-synthetic-data`. The main SLM curator only consumes the published
    train split and writes normal raw JSONL shards for the existing filter,
    deduplication, and blend stages.
    """

    SOURCE_TAG = "synthetic"
    SHARD_PREFIX = "synthetic"
    HF_REPO = ""
    DEFAULT_SHARD_SIZE = 5_000

    def __init__(
        self,
        output_dir: Path,
        max_docs: int | None = None,
        max_chars: int | None = None,
        shard_size: int = DEFAULT_SHARD_SIZE,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.max_docs = max_docs
        self.max_chars = max_chars
        self.shard_size = shard_size
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        existing = sorted(self.output_dir.glob(f"{self.SHARD_PREFIX}_*.jsonl"))
        if existing:
            raise RuntimeError(
                f"{self.SOURCE_TAG} output directory is not empty. Use the "
                "canonical curator's manifest-aware restart/replacement flow."
            )

        if not self.HF_REPO:
            raise RuntimeError(f"{self.__class__.__name__} is missing HF_REPO")

        log.info(
            "%s: streaming %s split=train max_docs=%s max_chars=%s output=%s",
            self.SOURCE_TAG,
            self.HF_REPO,
            f"{self.max_docs:,}" if self.max_docs is not None else "None",
            f"{self.max_chars:,}" if self.max_chars is not None else "None",
            self.output_dir,
        )

        dataset = load_dataset(self.HF_REPO, split="train", streaming=True)

        output_files: list[Path] = []
        buffer: list[dict[str, Any]] = []
        written_docs = 0
        written_chars = 0

        for idx, row in enumerate(dataset):
            record = self._normalise_record(row=row, idx=idx)
            if record is None:
                continue

            buffer.append(record)
            written_docs += 1
            written_chars += len(record["text"])

            if len(buffer) >= self.shard_size:
                output_files.append(self._write_shard(buffer, len(output_files)))
                buffer = []

            if self.max_docs is not None and written_docs >= self.max_docs:
                break
            if self.max_chars is not None and written_chars >= self.max_chars:
                break

        if buffer:
            output_files.append(self._write_shard(buffer, len(output_files)))

        log.info(
            "%s complete — docs=%s chars=%s shards=%s repo=%s",
            self.SOURCE_TAG,
            f"{written_docs:,}",
            f"{written_chars:,}",
            len(output_files),
            self.HF_REPO,
        )
        return output_files

    def stats(self, output_files: list[Path] | None = None) -> dict[str, Any]:
        if output_files is None:
            output_files = sorted(self.output_dir.glob(f"{self.SHARD_PREFIX}_*.jsonl"))

        docs = 0
        chars = 0
        for path in output_files:
            with path.open("rb") as f:
                for line in f:
                    try:
                        row = orjson.loads(line)
                    except Exception:
                        continue
                    docs += 1
                    chars += len(row.get("text", ""))

        return {
            "source": self.SOURCE_TAG,
            "docs": docs,
            "chars": chars,
            "estimated_tokens": chars / CHARS_PER_TOKEN,
            "hf_repo": self.HF_REPO,
            "output_dir": str(self.output_dir),
        }

    def _write_shard(self, records: list[dict[str, Any]], shard_idx: int) -> Path:
        path = self.output_dir / f"{self.SHARD_PREFIX}_{shard_idx:05d}.jsonl"
        with path.open("wb") as f:
            for record in records:
                f.write(orjson.dumps(record))
                f.write(b"\n")
        return path

    def _normalise_record(self, row: dict[str, Any], idx: int) -> dict[str, Any] | None:
        text = self._format_text(row)
        if not text or len(text) < 20:
            return None

        metadata = self._metadata(row=row, idx=idx)
        stable_material = json.dumps(
            {
                "source": self.SOURCE_TAG,
                "idx": idx,
                "repo": self.HF_REPO,
                "text": text,
                "metadata": metadata,
            },
            sort_keys=True,
            default=str,
        )
        stable_id = hashlib.sha256(stable_material.encode("utf-8")).hexdigest()[:16]
        return {
            "id": stable_id,
            "text": text,
            "source": self.SOURCE_TAG,
            "generated": True,
            "metadata": metadata,
        }

    def _format_text(self, row: dict[str, Any]) -> str:
        raise NotImplementedError

    def _metadata(self, row: dict[str, Any], idx: int) -> dict[str, Any]:
        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        metadata = dict(metadata)
        metadata.update(
            {
                "category": self.SOURCE_TAG,
                "synthetic": True,
                "generated": True,
                "provider": "huggingface",
                "hf_repo": self.HF_REPO,
            }
        )
        return metadata

    @staticmethod
    def _clean_single_line(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        return " ".join(value.replace("\r", " ").replace("\n", " ").split()).strip()

    @staticmethod
    def _clean_multiline(value: Any) -> str:
        if not isinstance(value, str):
            return ""
        text = value.strip()
        if "\\n" in text and "\n" not in text:
            text = text.replace("\\n", "\n")
        text = text.replace("\\t", "\t").replace("\r", "")
        return text.strip()



class SyntheticPretrainSource(HFSyntheticSource):
    """Canonical multi-family synthetic pretraining dataset."""

    SOURCE_TAG = "synthetic_pretrain"
    SHARD_PREFIX = "synthetic_pretrain"
    HF_REPO = "tohio/slm-synthetic-pretrain"
    ALLOWED_SIGNALS = frozenset({
        "arithmetic",
        "task_code",
        "educational_qa_mcq_math",
        "educational_qa_mcq_general",
        "factual_restraint",
    })

    def __init__(
        self,
        output_dir: Path,
        max_docs: int | None = None,
        max_chars: int | None = None,
        shard_size: int = HFSyntheticSource.DEFAULT_SHARD_SIZE,
        seed: int = 42,
        family_quotas: dict[str, int] | None = None,
    ):
        super().__init__(
            output_dir=output_dir,
            max_docs=max_docs,
            max_chars=max_chars,
            shard_size=shard_size,
            seed=seed,
        )
        self.family_quotas = dict(family_quotas) if family_quotas else None
        if self.family_quotas is not None:
            unknown = set(self.family_quotas) - self.ALLOWED_SIGNALS
            if unknown:
                raise ValueError(f"Unknown synthetic pretrain signals: {sorted(unknown)}")
            if any(quota <= 0 for quota in self.family_quotas.values()):
                raise ValueError("Synthetic pretrain family quotas must all be > 0")
            if max_docs is not None and sum(self.family_quotas.values()) != max_docs:
                raise ValueError(
                    "Synthetic pretrain family quotas must sum to max_docs: "
                    f"quotas={sum(self.family_quotas.values())}, max_docs={max_docs}"
                )

    def download(self) -> list[Path]:
        if self.family_quotas is None:
            return super().download()

        existing = sorted(self.output_dir.glob(f"{self.SHARD_PREFIX}_*.jsonl"))
        if existing:
            raise RuntimeError(
                f"{self.SOURCE_TAG} output directory is not empty. Use the "
                "canonical curator's manifest-aware restart/replacement flow."
            )

        log.info(
            "%s: stratified streaming %s split=train quotas=%s output=%s",
            self.SOURCE_TAG,
            self.HF_REPO,
            self.family_quotas,
            self.output_dir,
        )
        dataset = load_dataset(self.HF_REPO, split="train", streaming=True)

        reservoirs: dict[str, list[dict[str, Any]]] = {
            signal: [] for signal in self.family_quotas
        }
        seen: dict[str, int] = {signal: 0 for signal in self.family_quotas}
        rngs = {
            signal: random.Random(f"{self.seed}:{signal}")
            for signal in self.family_quotas
        }

        for idx, row in enumerate(dataset):
            record = self._normalise_record(row=row, idx=idx)
            if record is None:
                continue
            signal = record["metadata"]["signal"]
            if signal not in self.family_quotas:
                continue

            seen[signal] += 1
            quota = self.family_quotas[signal]
            reservoir = reservoirs[signal]
            if len(reservoir) < quota:
                reservoir.append(record)
                continue

            replacement = rngs[signal].randrange(seen[signal])
            if replacement < quota:
                reservoir[replacement] = record

        underfilled = {
            signal: {"required": self.family_quotas[signal], "available": seen[signal]}
            for signal in self.family_quotas
            if seen[signal] < self.family_quotas[signal]
        }
        if underfilled:
            raise RuntimeError(
                "Synthetic pretrain dataset cannot satisfy the configured "
                f"family quotas: {underfilled}"
            )

        selected = [
            record
            for signal in self.family_quotas
            for record in reservoirs[signal]
        ]
        random.Random(self.seed).shuffle(selected)

        output_files: list[Path] = []
        for start in range(0, len(selected), self.shard_size):
            output_files.append(
                self._write_shard(
                    selected[start : start + self.shard_size],
                    len(output_files),
                )
            )

        log.info(
            "%s complete — docs=%s families=%s shards=%s repo=%s",
            self.SOURCE_TAG,
            f"{len(selected):,}",
            {signal: len(records) for signal, records in reservoirs.items()},
            len(output_files),
            self.HF_REPO,
        )
        return output_files

    def stats(self, output_files: list[Path] | None = None) -> dict[str, Any]:
        stats = super().stats(output_files)
        if output_files is None:
            output_files = sorted(self.output_dir.glob(f"{self.SHARD_PREFIX}_*.jsonl"))
        families = {signal: {"docs": 0, "chars": 0} for signal in self.ALLOWED_SIGNALS}
        for path in output_files:
            with path.open("rb") as f:
                for line in f:
                    try:
                        row = orjson.loads(line)
                    except Exception:
                        continue
                    metadata = row.get("metadata")
                    signal = metadata.get("signal") if isinstance(metadata, dict) else None
                    if signal in families:
                        families[signal]["docs"] += 1
                        families[signal]["chars"] += len(row.get("text", ""))
        stats["families"] = families
        return stats

    def _normalise_record(self, row: dict[str, Any], idx: int) -> dict[str, Any] | None:
        record = super()._normalise_record(row=row, idx=idx)
        if record is None:
            return None
        upstream_id = row.get("id")
        if isinstance(upstream_id, str) and upstream_id.strip():
            record["id"] = upstream_id.strip()
        return record

    def _format_text(self, row: dict[str, Any]) -> str:
        return self._clean_multiline(row.get("text"))

    def _metadata(self, row: dict[str, Any], idx: int) -> dict[str, Any]:
        metadata = super()._metadata(row=row, idx=idx)
        signal = metadata.get("signal")
        if signal not in self.ALLOWED_SIGNALS:
            raise ValueError(
                "Synthetic pretrain row has missing or unknown metadata.signal: "
                f"{signal!r}; expected one of {sorted(self.ALLOWED_SIGNALS)}"
            )
        return metadata
