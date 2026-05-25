from __future__ import annotations

import hashlib
import json
import logging
from pathlib import Path
from typing import Any, Callable

import orjson
from datasets import load_dataset

from curator.constants import CHARS_PER_TOKEN

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
            log.info(
                "%s: found %d existing shard(s) in %s; skipping HF download",
                self.SOURCE_TAG,
                len(existing),
                self.output_dir,
            )
            return existing

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


class SyntheticArithmeticSource(HFSyntheticSource):
    SOURCE_TAG = "synthetic_arithmetic"
    SHARD_PREFIX = "synthetic_arithmetic"
    HF_REPO = "tohio/slm-synthetic-arithmetic"

    def _format_text(self, row: dict[str, Any]) -> str:
        question = self._clean_single_line(row.get("question"))
        answer = self._clean_single_line(row.get("answer"))
        steps = row.get("steps") if isinstance(row.get("steps"), list) else []

        if not question or not answer:
            return ""

        parts = [f"Question: {question}"]
        if steps:
            parts.append("Steps:")
            for step in steps:
                step_text = self._clean_single_line(step)
                if step_text:
                    parts.append(f"- {step_text}")
        parts.append(f"Answer: {answer}")
        return "\n".join(parts)


class SyntheticTaskCodeSource(HFSyntheticSource):
    SOURCE_TAG = "synthetic_task_code"
    SHARD_PREFIX = "synthetic_task_code"
    HF_REPO = "tohio/slm-synthetic-task-code"

    def _format_text(self, row: dict[str, Any]) -> str:
        task = self._clean_single_line(row.get("task"))
        plan = row.get("plan") if isinstance(row.get("plan"), list) else []
        code = self._clean_multiline(row.get("code"))

        if not task or not code:
            return ""

        parts = [f"Task: {task}"]
        if plan:
            parts.append("Plan:")
            for step in plan:
                step_text = self._clean_single_line(step)
                if step_text:
                    parts.append(f"- {step_text}")
        parts.extend(["Solution:", code])
        return "\n".join(parts)

    def _metadata(self, row: dict[str, Any], idx: int) -> dict[str, Any]:
        metadata = super()._metadata(row=row, idx=idx)
        metadata.setdefault("language", "python")
        return metadata


class _EducationalQAMCQSource(HFSyntheticSource):
    """Shared formatter for externally curated multiple-choice sources."""

    INCLUDE_EVIDENCE = False

    def _format_text(self, row: dict[str, Any]) -> str:
        evidence = self._clean_multiline(row.get("evidence"))
        question = self._clean_single_line(row.get("question"))
        choices = row.get("choices") if isinstance(row.get("choices"), list) else []
        explanation = self._clean_single_line(row.get("explanation"))

        try:
            correct_index = int(row.get("correct_index"))
        except Exception:
            return ""

        if self.INCLUDE_EVIDENCE and not evidence:
            return ""
        if not question or len(choices) < 2 or not (0 <= correct_index < len(choices)):
            return ""

        choice_texts = [self._clean_single_line(choice) for choice in choices]
        if not all(choice_texts):
            return ""

        answer = choice_texts[correct_index]
        parts: list[str] = []
        if self.INCLUDE_EVIDENCE:
            parts.extend(["Evidence:", evidence])
        parts.extend([f"Question: {question}", "Choices:"])
        for i, choice_text in enumerate(choice_texts):
            marker = chr(ord("A") + i)
            parts.append(f"{marker}. {choice_text}")
        parts.append(f"Answer: {answer}")
        if explanation:
            parts.append(f"Explanation: {explanation}")
        return "\n".join(parts)


class EducationalQAMCQMathSource(_EducationalQAMCQSource):
    SOURCE_TAG = "educational_qa_mcq_math"
    SHARD_PREFIX = "educational_qa_mcq_math"
    HF_REPO = "tohio/slm-synthetic-educational-qa-mcq-math"


class EducationalQAMCQGeneralSource(_EducationalQAMCQSource):
    SOURCE_TAG = "educational_qa_mcq_general"
    SHARD_PREFIX = "educational_qa_mcq_general"
    HF_REPO = "tohio/slm-synthetic-educational-qa-mcq-general"
    INCLUDE_EVIDENCE = True


class FactualRestraintSource(HFSyntheticSource):
    SOURCE_TAG = "factual_restraint"
    SHARD_PREFIX = "factual_restraint"
    HF_REPO = "tohio/slm-synthetic-factual-restraint"

    def _format_text(self, row: dict[str, Any]) -> str:
        question = self._clean_single_line(row.get("question"))
        answer = self._clean_single_line(row.get("safe_answer") or row.get("answer"))
        if not question or not answer:
            return ""
        return f"Question: {question}\nAnswer: {answer}"
