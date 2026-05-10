
from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from datasets import load_dataset

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


def _stable_id(source: str, idx: int, text: str) -> str:
    return hashlib.sha256(f"{source}:{idx}:{text[:500]}".encode("utf-8")).hexdigest()[:16]


def _text_from_record(record: dict, fields: tuple[str, ...]) -> str:
    for field in fields:
        value = record.get(field)
        if isinstance(value, str) and value.strip():
            return value.strip()

    value = record.get("data")
    if isinstance(value, dict):
        for field in fields:
            nested = value.get(field)
            if isinstance(nested, str) and nested.strip():
                return nested.strip()

    return ""


def _load_streaming_dataset(dataset_name: str, split: str, config: str | None = None):
    if config:
        try:
            return load_dataset(dataset_name, config, split=split, streaming=True)
        except Exception as exc:
            log.warning(
                "%s/%s: configured streaming load failed (%s); trying default config",
                dataset_name,
                config,
                exc,
            )

    return load_dataset(dataset_name, split=split, streaming=True)


class _StreamingHFJsonlSource:
    SOURCE_TAG = "streaming_hf_source"
    SHARD_PREFIX = "streaming_hf_source"
    DATASET_NAME = ""
    CONFIG_NAME: str | None = None
    SPLIT = "train"
    TEXT_FIELDS: tuple[str, ...] = ("text", "content")
    KEEP_CATEGORIES: set[str] | None = None
    DEFAULT_DOCS = 100_000

    def __init__(
        self,
        output_dir: Path,
        max_docs: int | None = None,
        shard_size: int = 10_000,
    ):
        self.output_dir = Path(output_dir)
        self.max_docs = max_docs or self.DEFAULT_DOCS
        self.shard_size = shard_size
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self._docs_written = 0
        self._chars_written = 0
        self._docs_seen = 0
        self._docs_skipped = 0

    def _keep_record(self, record: dict) -> bool:
        if not self.KEEP_CATEGORIES:
            return True

        metadata = record.get("metadata")
        category = None
        if isinstance(metadata, dict):
            category = metadata.get("category")
        category = category or record.get("category") or record.get("subset")

        return category in self.KEEP_CATEGORIES

    def _metadata_for_record(self, record: dict) -> dict:
        metadata = record.get("metadata")
        if isinstance(metadata, dict):
            return metadata
        return {}

    def download(self) -> list[Path]:
        existing = sorted(self.output_dir.glob(f"{self.SHARD_PREFIX}_*.jsonl"))
        if existing:
            log.info(
                "%s: found %d existing shard(s); skipping generation",
                self.SOURCE_TAG,
                len(existing),
            )
            self._load_existing_stats(existing)
            return existing

        ds = _load_streaming_dataset(
            self.DATASET_NAME,
            split=self.SPLIT,
            config=self.CONFIG_NAME,
        )

        output_files: list[Path] = []
        records: list[dict] = []

        log.info(
            "%s: streaming %s%s split=%s max_docs=%s",
            self.SOURCE_TAG,
            self.DATASET_NAME,
            f"/{self.CONFIG_NAME}" if self.CONFIG_NAME else "",
            self.SPLIT,
            f"{self.max_docs:,}",
        )

        for record in ds:
            self._docs_seen += 1

            if not isinstance(record, dict) or not self._keep_record(record):
                self._docs_skipped += 1
                continue

            text = _text_from_record(record, self.TEXT_FIELDS)
            if not text:
                self._docs_skipped += 1
                continue

            rec = {
                "id": _stable_id(self.SOURCE_TAG, self._docs_written, text),
                "source": self.SOURCE_TAG,
                "text": text,
            }

            metadata = self._metadata_for_record(record)
            if metadata:
                rec["metadata"] = metadata

            for key in ("license", "language", "lang", "category", "subset", "path", "repo_name"):
                value = record.get(key)
                if value is not None:
                    rec[key] = value

            records.append(rec)
            self._docs_written += 1
            self._chars_written += len(text)

            if len(records) >= self.shard_size:
                output_files.append(self._write_shard(records, len(output_files)))
                records = []

            if self._docs_written >= self.max_docs:
                break

        if records:
            output_files.append(self._write_shard(records, len(output_files)))

        log.info(
            "%s: complete - docs=%s chars=%s skipped=%s shards=%s",
            self.SOURCE_TAG,
            f"{self._docs_written:,}",
            f"{self._chars_written:,}",
            f"{self._docs_skipped:,}",
            len(output_files),
        )

        return output_files

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        path = self.output_dir / f"{self.SHARD_PREFIX}_{shard_idx:04d}.jsonl"
        tmp_path = path.with_suffix(".jsonl.tmp")

        with tmp_path.open("w", encoding="utf-8") as f:
            for record in records:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")

        tmp_path.replace(path)
        log.info("%s: wrote %d docs -> %s", self.SOURCE_TAG, len(records), path)
        return path

    def _load_existing_stats(self, shards: list[Path]) -> None:
        docs = 0
        chars = 0
        for shard in shards:
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue
                    docs += 1
                    chars += len(str(rec.get("text", "")))
        self._docs_written = docs
        self._chars_written = chars

    def stats(self) -> dict:
        shards = sorted(self.output_dir.glob(f"{self.SHARD_PREFIX}_*.jsonl"))
        if shards and self._docs_written == 0:
            self._load_existing_stats(shards)

        return {
            "source": self.SOURCE_TAG,
            "dataset": self.DATASET_NAME,
            "config": self.CONFIG_NAME,
            "split": self.SPLIT,
            "shards": len(shards),
            "documents": self._docs_written,
            "total_chars": self._chars_written,
            "avg_chars_per_doc": self._chars_written // max(self._docs_written, 1),
            "estimated_tokens": self._chars_written // CHARS_PER_TOKEN,
            "docs_seen": self._docs_seen,
            "docs_skipped": self._docs_skipped,
            "output_dir": str(self.output_dir),
        }



class NemotronCCCodeSource(_StreamingHFJsonlSource):
    SOURCE_TAG = "nemotron_cc_code"
    SHARD_PREFIX = "nemotron_cc_code"
    DATASET_NAME = "nvidia/Nemotron-CC-Code-v1"
    CONFIG_NAME = None
    SPLIT = "train"
    TEXT_FIELDS = ("text", "content")
