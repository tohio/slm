from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path

from datasets import load_dataset

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


APPROVED_SPECIALIZED_CONFIGS = (
    "Nemotron-Pretraining-Code-Concepts",
    "Nemotron-Pretraining-Unconditional-Algorithmic",
    "Nemotron-Pretraining-Formal-Logic",
    "Nemotron-Pretraining-Economics",
)

EXCLUDED_SPECIALIZED_CONFIGS = (
    "Nemotron-Pretraining-Multiple-Choice",
)

# Raw overfetch buffer for capped runs. Filtering can remove a small fraction
# of specialized algorithmic/code records, so capped downloads fetch extra raw
# candidates while filter/dedup/blend enforce the final usable budget.
OVERFETCH_FACTOR = 1.20


def _stable_id(source: str, config: str, idx: int, text: str) -> str:
    return hashlib.sha256(
        f"{source}:{config}:{idx}:{text[:500]}".encode("utf-8")
    ).hexdigest()[:16]


class NemotronSpecializedSource:
    """
    NVIDIA Nemotron specialized synthetic/specialized pretraining source.

    The dataset requires an explicit config name. We stream approved configs
    sequentially and intentionally exclude the standalone Multiple-Choice config
    by default because it may introduce downstream DeepSeek-license obligations
    for distributed/hosted derivative models.

    Formal Logic and Economics contain MCQ-style text, but they are separate
    configs from the excluded Nemotron-Pretraining-Multiple-Choice config.
    """

    SOURCE_TAG = "nemotron_specialized"
    SHARD_PREFIX = "nemotron_specialized"
    DATASET_NAME = "nvidia/Nemotron-Pretraining-Specialized-v1.1"
    SPLIT = "train"
    DEFAULT_DOCS = 100_000
    TEXT_FIELDS = ("text", "content")

    def __init__(
        self,
        output_dir: Path,
        max_docs: int | None = None,
        shard_size: int = 10_000,
        configs: tuple[str, ...] = APPROVED_SPECIALIZED_CONFIGS,
    ):
        self.output_dir = Path(output_dir)
        self.requested_max_docs = max_docs
        self.raw_max_docs = (
            int(max_docs * OVERFETCH_FACTOR) if max_docs is not None else None
        )
        self.max_docs = self.raw_max_docs or self.DEFAULT_DOCS
        self.shard_size = shard_size
        self.configs = configs
        self.output_dir.mkdir(parents=True, exist_ok=True)

        if self.requested_max_docs is not None and self.raw_max_docs != self.requested_max_docs:
            log.info(
                "Nemotron Specialized overfetch enabled: requested=%s, raw_cap=%s, factor=%.2f",
                f"{self.requested_max_docs:,}",
                f"{self.raw_max_docs:,}",
                OVERFETCH_FACTOR,
            )

        self._docs_written = 0
        self._chars_written = 0
        self._docs_seen = 0
        self._docs_skipped = 0
        self._by_config: dict[str, int] = {}

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

        output_files: list[Path] = []
        records: list[dict] = []

        log.info(
            "%s: streaming %s configs=%s requested_max_docs=%s raw_max_docs=%s",
            self.SOURCE_TAG,
            self.DATASET_NAME,
            ",".join(self.configs),
            f"{self.requested_max_docs:,}" if self.requested_max_docs is not None else "default",
            f"{self.max_docs:,}",
        )

        iterators = []
        for config in self.configs:
            ds = load_dataset(
                self.DATASET_NAME,
                config,
                split=self.SPLIT,
                streaming=True,
            )
            iterators.append((config, iter(ds)))

        # Round-robin so mini runs sample all approved configs instead of
        # being monopolized by the first large config.
        while self._docs_written < self.max_docs and iterators:
            still_active = []

            for config, iterator in iterators:
                if self._docs_written >= self.max_docs:
                    still_active.append((config, iterator))
                    continue

                try:
                    row = next(iterator)
                except StopIteration:
                    log.info("%s: config exhausted: %s", self.SOURCE_TAG, config)
                    continue

                self._docs_seen += 1
                text = self._text_from_record(row)
                if not text:
                    self._docs_skipped += 1
                    still_active.append((config, iterator))
                    continue

                rec = {
                    "id": _stable_id(self.SOURCE_TAG, config, self._docs_written, text),
                    "source": self.SOURCE_TAG,
                    "text": text,
                    "nemotron_config": config,
                }

                for key in ("license", "metadata", "uuid"):
                    value = row.get(key)
                    if value is not None:
                        rec[key] = value

                records.append(rec)
                self._docs_written += 1
                self._chars_written += len(text)
                self._by_config[config] = self._by_config.get(config, 0) + 1

                if len(records) >= self.shard_size:
                    output_files.append(self._write_shard(records, len(output_files)))
                    records = []

                still_active.append((config, iterator))

            iterators = still_active

        if records:
            output_files.append(self._write_shard(records, len(output_files)))

        log.info(
            "%s: complete - docs=%s chars=%s skipped=%s shards=%s by_config=%s",
            self.SOURCE_TAG,
            f"{self._docs_written:,}",
            f"{self._chars_written:,}",
            f"{self._docs_skipped:,}",
            len(output_files),
            self._by_config,
        )

        return output_files

    def _text_from_record(self, record: dict) -> str:
        for field in self.TEXT_FIELDS:
            value = record.get(field)
            if isinstance(value, str) and value.strip():
                return value.strip()
        return ""

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
        by_config: dict[str, int] = {}

        for shard in shards:
            with shard.open("r", encoding="utf-8") as f:
                for line in f:
                    try:
                        rec = json.loads(line)
                    except Exception:
                        continue

                    docs += 1
                    chars += len(str(rec.get("text", "")))
                    config = rec.get("nemotron_config", "unknown")
                    by_config[config] = by_config.get(config, 0) + 1

        self._docs_written = docs
        self._chars_written = chars
        self._by_config = by_config

    def stats(self) -> dict:
        shards = sorted(self.output_dir.glob(f"{self.SHARD_PREFIX}_*.jsonl"))
        if shards and self._docs_written == 0:
            self._load_existing_stats(shards)

        return {
            "source": self.SOURCE_TAG,
            "dataset": self.DATASET_NAME,
            "configs": list(self.configs),
            "excluded_configs": list(EXCLUDED_SPECIALIZED_CONFIGS),
            "split": self.SPLIT,
            "shards": len(shards),
            "documents": self._docs_written,
            "total_chars": self._chars_written,
            "avg_chars_per_doc": self._chars_written // max(self._docs_written, 1),
            "estimated_tokens": self._chars_written // CHARS_PER_TOKEN,
            "docs_seen": self._docs_seen,
            "docs_skipped": self._docs_skipped,
            "requested_max_docs": self.requested_max_docs,
            "raw_max_docs": self.raw_max_docs,
            "effective_max_docs": self.max_docs,
            "overfetch_factor": OVERFETCH_FACTOR,
            "by_config": self._by_config,
            "output_dir": str(self.output_dir),
        }