"""
Shared Groq-backed synthetic generation implementation.

This module lives under curator/scripts because Groq generation is an active
data-generation step, not an upstream dataset definition. Synthetic source
adapters in curator/sources import this class and expose the normal Source
interface expected by curate.py.

Synthetic sources behave like downloaded sources: Stage 1 writes raw JSONL
shards to the target-scoped raw directory, then normal filtering, deduplication,
source stats, and blending run unchanged.

No shared cache is used. Existing shards in output_dir are skipped. Delete the
run-scoped raw/filtered/dedup directories to force fresh generation.
"""

from __future__ import annotations

import hashlib
import ast
import json
import logging
import os
import random
import re
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any

import orjson

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)

GROQ_API_URL = "https://api.groq.com/openai/v1/chat/completions"


def _load_dotenv_if_present() -> None:
    """
    Load repo-local .env without requiring the caller to export variables.

    This intentionally avoids adding a runtime dependency on python-dotenv.
    Values already present in the shell win over .env values.
    """
    repo_root = Path(__file__).resolve().parents[2]
    env_path = repo_root / ".env"
    if not env_path.exists():
        return

    for raw in env_path.read_text().splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[len("export "):].strip()
            if "=" not in line:
                continue
        key, value = line.split("=", 1)
        os.environ.setdefault(key.strip(), value.strip().strip('"').strip("'"))


_load_dotenv_if_present()

DEFAULT_GROQ_MODEL = os.environ.get("SYNTHETIC_GROQ_MODEL", "llama-3.1-8b-instant")
DEFAULT_TEMPERATURE = float(os.environ.get("SYNTHETIC_GROQ_TEMPERATURE", "0.8"))
DEFAULT_MAX_TOKENS = int(os.environ.get("SYNTHETIC_GROQ_MAX_TOKENS", "4096"))


def _env_model_for_source(source_tag: str) -> str:
    key = "GROQ_MODEL_" + source_tag.upper()
    return os.environ.get(key) or DEFAULT_GROQ_MODEL


class GroqSyntheticSource:
    """Base class for target-scoped Groq synthetic raw-shard generation."""

    SOURCE_TAG = "synthetic"
    SHARD_PREFIX = "synthetic"
    DEFAULT_DOCS = 100_000
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_SHARD_SIZE = 5_000
    MAX_CONSECUTIVE_FAILED_BATCHES = 3
    SYSTEM_PROMPT = (
        "You generate high-quality synthetic pretraining records. "
        "Return only valid JSON. Do not include markdown fences, assistant chatter, "
        "or explanatory text outside the JSON payload. Use real newlines inside text "
        "fields, not escaped literal \\n sequences. If generating code, the code and "
        "tests must be syntactically valid for the stated language."
    )

    def __init__(
        self,
        output_dir: Path,
        max_docs: int | None = None,
        max_chars: int | None = None,
        shard_size: int | None = None,
        seed: int = 42,
        model: str | None = None,
        batch_size: int | None = None,
        temperature: float | None = None,
        max_tokens: int | None = None,
    ):
        self.output_dir = Path(output_dir)
        self.max_docs = max_docs or self.DEFAULT_DOCS
        self.max_chars = max_chars
        self.shard_size = shard_size or self.DEFAULT_SHARD_SIZE
        self.seed = seed
        self.model = model or _env_model_for_source(self.SOURCE_TAG)
        self.batch_size = batch_size or self.DEFAULT_BATCH_SIZE
        self.temperature = DEFAULT_TEMPERATURE if temperature is None else temperature
        self.max_tokens = DEFAULT_MAX_TOKENS if max_tokens is None else max_tokens

        self.api_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.failed_batches = 0
        self.retries = 0
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        existing_shards = sorted(self.output_dir.glob(f"{self.SHARD_PREFIX}_*.jsonl"))
        if existing_shards:
            log.info(
                "%s: found %d existing shard(s) in %s; skipping Groq generation",
                self.SOURCE_TAG,
                len(existing_shards),
                self.output_dir,
            )
            return existing_shards

        if not os.environ.get("GROQ_API_KEY"):
            raise RuntimeError(
                "GROQ_API_KEY is required for Groq-backed synthetic source "
                f"{self.SOURCE_TAG}. Set it in the environment or .env."
            )

        rng = random.Random(self.seed)
        output_files: list[Path] = []
        buffer: list[dict[str, Any]] = []
        written_docs = 0
        written_chars = 0
        next_idx = 0
        consecutive_failed_batches = 0

        log.info(
            "%s: generating via Groq model=%s max_docs=%s max_chars=%s batch_size=%s output=%s",
            self.SOURCE_TAG,
            self.model,
            f"{self.max_docs:,}" if self.max_docs is not None else "None",
            f"{self.max_chars:,}" if self.max_chars is not None else "None",
            self.batch_size,
            self.output_dir,
        )

        while True:
            if self.max_docs is not None and written_docs >= self.max_docs:
                break
            if self.max_chars is not None and written_chars >= self.max_chars:
                break

            remaining_docs = self.max_docs - written_docs if self.max_docs is not None else self.batch_size
            batch_count = max(1, min(self.batch_size, remaining_docs))
            prompt = self._build_prompt(batch_count=batch_count, start_index=next_idx, rng=rng)
            rows = self._generate_batch(prompt=prompt, requested=batch_count)

            if not rows:
                self.failed_batches += 1
                consecutive_failed_batches += 1
                if consecutive_failed_batches >= self.MAX_CONSECUTIVE_FAILED_BATCHES:
                    raise RuntimeError(
                        f"{self.SOURCE_TAG}: Groq generation produced no records after "
                        f"{consecutive_failed_batches} consecutive failed batch(es). "
                        "Check GROQ_API_KEY, model access, account status, and network/IP restrictions."
                    )
                continue

            consecutive_failed_batches = 0

            for row in rows:
                record = self._normalise_record(row=row, idx=next_idx)
                next_idx += 1
                if record is None:
                    continue

                text = record["text"]
                buffer.append(record)
                written_docs += 1
                written_chars += len(text)

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
            "%s complete — docs=%s chars=%s shards=%s api_calls=%s prompt_tokens=%s completion_tokens=%s failed_batches=%s retries=%s",
            self.SOURCE_TAG,
            f"{written_docs:,}",
            f"{written_chars:,}",
            len(output_files),
            self.api_calls,
            self.prompt_tokens,
            self.completion_tokens,
            self.failed_batches,
            self.retries,
        )
        log.info("%s stats: %s", self.SOURCE_TAG, self.stats(output_files))
        return output_files

    def _generate_batch(self, prompt: str, requested: int) -> list[dict[str, Any]]:
        attempts = 0
        while attempts < 4:
            attempts += 1
            try:
                response = self._call_groq(prompt)
                content = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                rows = self._parse_records(content)
                if rows:
                    return rows[:requested]
                self.failed_batches += 1
            except Exception as exc:
                self.failed_batches += 1
                if attempts >= 4:
                    log.warning("%s: Groq generation failed after retries: %s", self.SOURCE_TAG, exc)
                    return []
                self.retries += 1
                time.sleep(min(2**attempts, 20))
        return []

    def _call_groq(self, prompt: str) -> dict[str, Any]:
        api_key = os.environ["GROQ_API_KEY"]
        payload = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }
        req = urllib.request.Request(
            GROQ_API_URL,
            data=json.dumps(payload).encode("utf-8"),
            headers={
                "Authorization": f"Bearer {api_key}",
                "Content-Type": "application/json",
                "User-Agent": "slm-curator/0.1 (+https://api.groq.com)",
            },
            method="POST",
        )

        try:
            with urllib.request.urlopen(req, timeout=120) as resp:
                body = resp.read()
        except urllib.error.HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="replace")
            raise RuntimeError(f"Groq HTTP {exc.code}: {detail[:1000]}") from exc

        self.api_calls += 1
        parsed = json.loads(body)
        usage = parsed.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        return parsed

    def _parse_records(self, content: str) -> list[dict[str, Any]]:
        text = content.strip()
        if not text:
            return []
        text = re.sub(r"^```(?:json|jsonl)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)

        try:
            obj = json.loads(text)
            if isinstance(obj, dict) and isinstance(obj.get("records"), list):
                return [r for r in obj["records"] if isinstance(r, dict)]
            if isinstance(obj, list):
                return [r for r in obj if isinstance(r, dict)]
        except Exception:
            pass

        array_match = re.search(r"\[[\s\S]*\]", text)
        if array_match:
            try:
                obj = json.loads(array_match.group(0))
                if isinstance(obj, list):
                    return [r for r in obj if isinstance(r, dict)]
            except Exception:
                pass

        rows: list[dict[str, Any]] = []
        for line in text.splitlines():
            line = line.strip().rstrip(",")
            if not line:
                continue
            try:
                obj = json.loads(line)
            except Exception:
                continue
            if isinstance(obj, dict):
                rows.append(obj)
        return rows

    def _normalise_record(self, row: dict[str, Any], idx: int) -> dict[str, Any] | None:
        raw_text = row.get("text")
        if not isinstance(raw_text, str):
            return None

        text = self._clean_generated_text(raw_text)
        if len(text) < 40:
            return None

        metadata = row.get("metadata") if isinstance(row.get("metadata"), dict) else {}
        metadata.update(self._record_metadata(row=row, idx=idx))
        metadata.update({"generator": "groq", "model": self.model, "synthetic": True})

        if not self._quality_ok(text=text, metadata=metadata):
            return None

        stable_material = json.dumps(
            {"source": self.SOURCE_TAG, "idx": idx, "text": text, "metadata": metadata},
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

    def _clean_generated_text(self, raw_text: str) -> str:
        """Normalize common LLM JSON escaping artifacts before filtering/dedup."""
        text = raw_text.strip()

        # Some model outputs place literal escaped newlines in the JSON string
        # instead of actual newlines. These hurt readability and create noisy
        # pretraining examples, so normalize them before writing raw shards.
        text = text.replace("\\r\\n", "\n")
        text = text.replace("\\n", "\n")
        text = text.replace("\\t", "\t")

        return text.strip()

    def _quality_ok(self, text: str, metadata: dict[str, Any]) -> bool:
        """Reject obvious synthetic-generation artifacts and malformed records."""
        lowered = text.lower()

        # Do not let assistant chatter or markdown fence artifacts enter the corpus.
        blocked_fragments = [
            "```",
            "here are the records",
            "here is the json",
            "as an ai language model",
            "i cannot generate",
        ]
        if any(fragment in lowered for fragment in blocked_fragments):
            return False

        if self.SOURCE_TAG == "synthetic_task_code":
            return self._task_code_quality_ok(text=text, metadata=metadata)

        if self.SOURCE_TAG == "educational_qa_mcq":
            return self._educational_qa_quality_ok(text=text, metadata=metadata)

        return True

    def _task_code_quality_ok(self, text: str, metadata: dict[str, Any]) -> bool:
        """Reject incomplete task-code records and keep only syntax-checked Python for now."""
        marker = "Solution:"
        if marker not in text:
            return False

        solution = text.split(marker, 1)[1].strip()
        if not solution:
            return False

        language = str(metadata.get("language", "")).lower()

        # Until we add language-specific validators for Go/Rust/Bash/etc.,
        # keep this Groq supplement to Python only. Bad non-Python synthetic
        # code is worse than a smaller but cleaner source.
        if language != "python":
            return False

        # Avoid plausible-but-wrong synthetic code in areas where syntax checks
        # are not enough to prove correctness. Keep this supplement focused on
        # simple utility functions, transformations, parsing, aggregation, and
        # tests rather than bug-fix/security/Unicode edge cases.
        risky_fragments = [
            "fix the bug",
            "fix this",
            "bug",
            "race condition",
            "unicode",
            "combining character",
            "email validation",
            "validate email",
            "security",
            "authentication",
            "password",
            "cryptographic",
            "encryption",
            "sanitize",
        ]
        lowered = text.lower()
        if any(fragment in lowered for fragment in risky_fragments):
            return False

        # The generated records often include comments and assert-based tests.
        # ast.parse handles those fine and catches broken parentheses/brackets,
        # such as: assert count_elements(["a", "a", "b"] == {"a": 2})
        try:
            ast.parse(solution)
        except SyntaxError:
            log.debug(
                "%s: rejected invalid Python synthetic task-code record: %r",
                self.SOURCE_TAG,
                text[:240],
            )
            return False

        return True

    def _educational_qa_quality_ok(self, text: str, metadata: dict[str, Any]) -> bool:
        """Reject trivial or needlessly ambiguous educational QA examples."""
        lowered = text.lower()

        # Keep these examples educational rather than generic life-routine filler.
        trivial_fragments = [
            "first thing you should do when you wake up",
            "get out of bed",
        ]
        if any(fragment in lowered for fragment in trivial_fragments):
            return False

        # Avoid controversial/measurement-dependent facts unless the prompt is
        # explicitly about ambiguity. This exact pattern appeared in inspection.
        ambiguous_fragments = [
            "which river is longer, the nile or the amazon",
            "answer: the nile",
        ]
        if all(fragment in lowered for fragment in ambiguous_fragments):
            return False

        if "Question:" not in text or "Answer:" not in text:
            return False

        return True

    def _write_shard(self, records: list[dict[str, Any]], shard_idx: int) -> Path:
        path = self.output_dir / f"{self.SHARD_PREFIX}_{shard_idx:05d}.jsonl"
        with path.open("wb") as f:
            for record in records:
                f.write(orjson.dumps(record))
                f.write(b"\n")
        return path

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
            "model": self.model,
            "api_calls": self.api_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "failed_batches": self.failed_batches,
            "retries": self.retries,
            "output_dir": str(self.output_dir),
        }

    def _record_metadata(self, row: dict[str, Any], idx: int) -> dict[str, Any]:
        return {}

    def _build_prompt(self, batch_count: int, start_index: int, rng: random.Random) -> str:
        raise NotImplementedError