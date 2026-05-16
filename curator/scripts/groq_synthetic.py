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

# Proactive pacing to avoid hammering Groq during long synthetic runs.
DEFAULT_MIN_REQUEST_INTERVAL_SECONDS = float(
    os.environ.get("GROQ_MIN_REQUEST_INTERVAL_SECONDS", "0.5")
)

# Retry behavior for HTTP 429 and transient HTTP failures.
DEFAULT_MAX_RETRIES = int(os.environ.get("GROQ_MAX_RETRIES", "6"))
DEFAULT_RETRY_BASE_SECONDS = float(os.environ.get("GROQ_RETRY_BASE_SECONDS", "2"))
DEFAULT_RETRY_MAX_SECONDS = float(os.environ.get("GROQ_RETRY_MAX_SECONDS", "60"))

# Prompt-only generation is the default for high-volume synthetic curation.
# API-enforced JSON Schema is too brittle at scale: one malformed item can make
# Groq reject an otherwise useful batch. Set GROQ_STRUCTURED_OUTPUTS=1 only for
# small diagnostics where hard schema rejection is desired.
DEFAULT_STRUCTURED_OUTPUTS = os.environ.get("GROQ_STRUCTURED_OUTPUTS", "0").lower() not in {
    "0",
    "false",
    "no",
}


def _env_model_for_source(source_tag: str) -> str:
    key = "GROQ_MODEL_" + source_tag.upper()
    return os.environ.get(key) or DEFAULT_GROQ_MODEL


class GroqHTTPError(RuntimeError):
    """HTTP error from Groq with status and optional Retry-After metadata."""

    def __init__(self, status: int, detail: str, retry_after: float | None = None):
        self.status = status
        self.detail = detail
        self.retry_after = retry_after
        super().__init__(f"Groq HTTP {status}: {detail[:1000]}")


class GroqSyntheticSource:
    """Base class for target-scoped Groq synthetic raw-shard generation."""

    SOURCE_TAG = "synthetic"
    SHARD_PREFIX = "synthetic"
    DEFAULT_DOCS = 100_000
    DEFAULT_BATCH_SIZE = 10
    DEFAULT_SHARD_SIZE = 5_000
    MAX_CONSECUTIVE_FAILED_BATCHES = 3
    PROGRESS_EVERY_DOCS = 10_000
    SYSTEM_PROMPT = (
        "You generate high-quality synthetic pretraining records. "
        "Return only the payload requested by the user. Prefer JSONL when the "
        "user asks for JSONL: one complete JSON object per line, no outer array, "
        "no records wrapper, no markdown fences, no assistant chatter. "
        "Do not place raw newline characters inside quoted JSON string values. "
        "If generating code, code and tests must be syntactically valid for the "
        "stated language."
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
        min_request_interval_seconds: float | None = None,
        max_retries: int | None = None,
        retry_base_seconds: float | None = None,
        retry_max_seconds: float | None = None,
        structured_outputs: bool | None = None,
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

        self.min_request_interval_seconds = (
            DEFAULT_MIN_REQUEST_INTERVAL_SECONDS
            if min_request_interval_seconds is None
            else min_request_interval_seconds
        )
        self.max_retries = DEFAULT_MAX_RETRIES if max_retries is None else max_retries
        self.retry_base_seconds = (
            DEFAULT_RETRY_BASE_SECONDS if retry_base_seconds is None else retry_base_seconds
        )
        self.retry_max_seconds = (
            DEFAULT_RETRY_MAX_SECONDS if retry_max_seconds is None else retry_max_seconds
        )
        self.structured_outputs = (
            DEFAULT_STRUCTURED_OUTPUTS if structured_outputs is None else structured_outputs
        )
        self._last_request_at = 0.0

        self.api_calls = 0
        self.prompt_tokens = 0
        self.completion_tokens = 0
        self.failed_batches = 0
        self.retries = 0
        self.rate_limit_retries = 0
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

        _load_dotenv_if_present()
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

        schema_enabled = bool(self.structured_outputs and self._structured_response_schema())

        log.info(
            "%s: generating via Groq model=%s max_docs=%s max_chars=%s "
            "batch_size=%s min_request_interval=%ss max_retries=%s structured_outputs=%s output=%s",
            self.SOURCE_TAG,
            self.model,
            f"{self.max_docs:,}" if self.max_docs is not None else "None",
            f"{self.max_chars:,}" if self.max_chars is not None else "None",
            self.batch_size,
            self.min_request_interval_seconds,
            self.max_retries,
            schema_enabled,
            self.output_dir,
        )

        while True:
            if self.max_docs is not None and written_docs >= self.max_docs:
                break
            if self.max_chars is not None and written_chars >= self.max_chars:
                break

            remaining_docs = (
                self.max_docs - written_docs
                if self.max_docs is not None
                else self.batch_size
            )
            batch_count = max(1, min(self.batch_size, remaining_docs))
            prompt = self._build_prompt(
                batch_count=batch_count,
                start_index=next_idx,
                rng=rng,
            )
            rows = self._generate_batch(prompt=prompt, requested=batch_count)

            if not rows:
                self.failed_batches += 1
                consecutive_failed_batches += 1
                if consecutive_failed_batches >= self.MAX_CONSECUTIVE_FAILED_BATCHES:
                    raise RuntimeError(
                        f"{self.SOURCE_TAG}: Groq generation produced no records after "
                        f"{consecutive_failed_batches} consecutive failed batch(es). "
                        "Check GROQ_API_KEY, model access, account status, network/IP "
                        "restrictions, throttling, prompt validity, and whether the "
                        "selected model supports structured outputs."
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

                if written_docs % self.PROGRESS_EVERY_DOCS == 0:
                    log.info(
                        "%s progress — docs=%s chars=%s api_calls=%s "
                        "failed_batches=%s retries=%s rate_limit_retries=%s",
                        self.SOURCE_TAG,
                        f"{written_docs:,}",
                        f"{written_chars:,}",
                        self.api_calls,
                        self.failed_batches,
                        self.retries,
                        self.rate_limit_retries,
                    )

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
            "%s complete — docs=%s chars=%s shards=%s api_calls=%s "
            "prompt_tokens=%s completion_tokens=%s failed_batches=%s "
            "retries=%s rate_limit_retries=%s",
            self.SOURCE_TAG,
            f"{written_docs:,}",
            f"{written_chars:,}",
            len(output_files),
            self.api_calls,
            self.prompt_tokens,
            self.completion_tokens,
            self.failed_batches,
            self.retries,
            self.rate_limit_retries,
        )
        log.info("%s stats: %s", self.SOURCE_TAG, self.stats(output_files))
        return output_files

    def _generate_batch(self, prompt: str, requested: int) -> list[dict[str, Any]]:
        attempts = 0
        last_error: Exception | None = None

        while attempts < self.max_retries:
            attempts += 1
            try:
                response = self._call_groq(prompt, requested=requested)
                content = (
                    response.get("choices", [{}])[0]
                    .get("message", {})
                    .get("content", "")
                )
                rows = self._parse_records(content)
                if rows:
                    if len(rows) < requested:
                        log.warning(
                            "%s: recovered %d/%d requested record(s) from Groq response",
                            self.SOURCE_TAG,
                            len(rows),
                            requested,
                        )
                    return rows[:requested]

                preview = content[:800].replace("\n", "\\n")
                log.warning(
                    "%s: unparseable Groq response preview: %s",
                    self.SOURCE_TAG,
                    preview,
                )
                last_error = RuntimeError("Groq response did not contain parseable records")

            except GroqHTTPError as exc:
                last_error = exc
                if not self._should_retry_http(exc.status):
                    if exc.status == 400 and self.structured_outputs and self._structured_response_schema():
                        log.error(
                            "%s: Groq rejected structured output request. "
                            "Use a model that supports response_format=json_schema, "
                            "or set GROQ_STRUCTURED_OUTPUTS=0 to fall back to prompt-only parsing. Error: %s",
                            self.SOURCE_TAG,
                            exc,
                        )
                    else:
                        log.warning(
                            "%s: non-retryable Groq HTTP error: %s",
                            self.SOURCE_TAG,
                            exc,
                        )
                    return []

                sleep_seconds = self._retry_sleep_seconds(
                    attempt=attempts,
                    retry_after=exc.retry_after,
                )
                if exc.status == 429:
                    self.rate_limit_retries += 1
                    log.warning(
                        "%s: Groq rate limited HTTP 429; sleeping %.1fs "
                        "(attempt %s/%s)",
                        self.SOURCE_TAG,
                        sleep_seconds,
                        attempts,
                        self.max_retries,
                    )
                else:
                    log.warning(
                        "%s: retryable Groq HTTP %s; sleeping %.1fs "
                        "(attempt %s/%s)",
                        self.SOURCE_TAG,
                        exc.status,
                        sleep_seconds,
                        attempts,
                        self.max_retries,
                    )

                if attempts >= self.max_retries:
                    break

                self.retries += 1
                time.sleep(sleep_seconds)
                continue

            except Exception as exc:
                last_error = exc

            if attempts >= self.max_retries:
                break

            self.retries += 1
            sleep_seconds = self._retry_sleep_seconds(attempt=attempts)
            log.warning(
                "%s: Groq generation retry after error/empty response: %s; "
                "sleeping %.1fs (attempt %s/%s)",
                self.SOURCE_TAG,
                last_error,
                sleep_seconds,
                attempts,
                self.max_retries,
            )
            time.sleep(sleep_seconds)

        log.warning(
            "%s: Groq generation failed after retries: %s",
            self.SOURCE_TAG,
            last_error,
        )
        return []

    def _call_groq(self, prompt: str, requested: int | None = None) -> dict[str, Any]:
        self._pace_request()

        api_key = os.environ["GROQ_API_KEY"]
        payload: dict[str, Any] = {
            "model": self.model,
            "temperature": self.temperature,
            "max_tokens": self.max_tokens,
            "messages": [
                {"role": "system", "content": self.SYSTEM_PROMPT},
                {"role": "user", "content": prompt},
            ],
        }

        schema = self._structured_response_schema(batch_count=requested)
        if self.structured_outputs and schema:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": self._structured_response_name(),
                    "schema": schema,
                    "strict": True,
                },
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
            retry_after = self._parse_retry_after(exc.headers.get("Retry-After"))
            raise GroqHTTPError(
                status=exc.code,
                detail=detail,
                retry_after=retry_after,
            ) from exc

        self.api_calls += 1
        parsed = json.loads(body)
        usage = parsed.get("usage") or {}
        self.prompt_tokens += int(usage.get("prompt_tokens") or 0)
        self.completion_tokens += int(usage.get("completion_tokens") or 0)
        return parsed

    def _pace_request(self) -> None:
        """Sleep before requests to keep a minimum interval between API calls."""
        interval = max(0.0, float(self.min_request_interval_seconds))
        if interval <= 0:
            self._last_request_at = time.monotonic()
            return

        now = time.monotonic()
        elapsed = now - self._last_request_at
        sleep_for = interval - elapsed
        if self._last_request_at > 0 and sleep_for > 0:
            time.sleep(sleep_for)

        self._last_request_at = time.monotonic()

    def _retry_sleep_seconds(
        self,
        attempt: int,
        retry_after: float | None = None,
    ) -> float:
        """Compute bounded retry sleep, honoring Retry-After when provided."""
        if retry_after is not None and retry_after > 0:
            return min(float(retry_after), self.retry_max_seconds)

        base = max(0.1, float(self.retry_base_seconds))
        sleep_seconds = min(base * (2 ** max(0, attempt - 1)), self.retry_max_seconds)
        jitter = random.uniform(0, min(1.0, sleep_seconds * 0.10))
        return min(sleep_seconds + jitter, self.retry_max_seconds)

    def _parse_retry_after(self, value: str | None) -> float | None:
        """Parse Retry-After seconds. Date-form Retry-After is ignored."""
        if not value:
            return None
        try:
            return max(0.0, float(value.strip()))
        except ValueError:
            return None

    def _should_retry_http(self, status: int) -> bool:
        """Return True for rate-limit/transient HTTP errors."""
        return status in {408, 409, 425, 429, 500, 502, 503, 504}

    def _parse_records(self, content: str) -> list[dict[str, Any]]:
        """
        Parse Groq output into a list of candidate record dictionaries.

        The parser is intentionally tolerant because LLM output is raw,
        untrusted data. It supports:
        - JSONL: one JSON object per line
        - a top-level object: {"records": [...]}
        - a top-level array: [...]
        - markdown-fenced payloads
        - string-encoded record objects inside arrays
        - partially malformed wrappers where individual object literals can
          still be recovered

        Bad rows are ignored here or later by source-specific normalizers.
        One malformed record should never poison a whole batch.
        """
        content = self._strip_output_wrapper(content or "")
        if not content:
            return []

        parsed_rows = self._parse_structured_payload(content)
        if parsed_rows:
            return parsed_rows

        jsonl_rows = self._parse_jsonl_payload(content)
        if jsonl_rows:
            return jsonl_rows

        records_array_rows = self._parse_records_array_lenient(content)
        if records_array_rows:
            return records_array_rows

        object_rows = self._extract_json_objects_lenient(content)
        if object_rows:
            return object_rows

        return []

    def _strip_output_wrapper(self, content: str) -> str:
        """Remove common markdown fences and assistant preambles."""
        text = content.strip()
        if not text:
            return ""

        if text.startswith("```"):
            text = re.sub(r"^```(?:json|jsonl)?\s*", "", text, flags=re.IGNORECASE)
            text = re.sub(r"\s*```$", "", text).strip()

        return text

    def _coerce_record(self, value: Any) -> dict[str, Any] | None:
        """
        Convert a candidate value into a record dict when possible.

        Some prompt-only model responses contain a JSON object encoded as a
        string inside a records array. Decode those strings and keep the row.
        """
        if isinstance(value, dict):
            return value

        if isinstance(value, str):
            stripped = value.strip()
            if not stripped:
                return None
            try:
                decoded = json.loads(stripped)
            except Exception:
                return None
            if isinstance(decoded, dict):
                return decoded

        return None

    def _coerce_records(self, value: Any) -> list[dict[str, Any]]:
        if isinstance(value, dict):
            records = value.get("records")
            if isinstance(records, list):
                return [
                    record
                    for item in records
                    if (record := self._coerce_record(item)) is not None
                ]
            record = self._coerce_record(value)
            return [record] if record is not None else []

        if isinstance(value, list):
            return [
                record
                for item in value
                if (record := self._coerce_record(item)) is not None
            ]

        return []

    def _parse_structured_payload(self, content: str) -> list[dict[str, Any]]:
        """
        Try parsing the whole payload or the largest obvious JSON object/array.
        """
        candidates: list[str] = [content]

        obj_start = content.find("{")
        obj_end = content.rfind("}")
        if obj_start != -1 and obj_end != -1 and obj_end > obj_start:
            candidates.append(content[obj_start : obj_end + 1])

        arr_start = content.find("[")
        arr_end = content.rfind("]")
        if arr_start != -1 and arr_end != -1 and arr_end > arr_start:
            candidates.append(content[arr_start : arr_end + 1])

        seen: set[str] = set()
        for candidate in candidates:
            candidate = candidate.strip()
            if not candidate or candidate in seen:
                continue
            seen.add(candidate)

            try:
                parsed = json.loads(candidate)
            except Exception:
                continue

            rows = self._coerce_records(parsed)
            if rows:
                return rows

        return []

    def _parse_jsonl_payload(self, content: str) -> list[dict[str, Any]]:
        """
        Parse one JSON object per line. This is the preferred prompt-only
        contract for synthetic generation.
        """
        rows: list[dict[str, Any]] = []
        for raw_line in content.splitlines():
            line = raw_line.strip().rstrip(",")
            if not line:
                continue
            if line.startswith("```") or line.lower() in {"json", "jsonl"}:
                continue

            try:
                parsed = json.loads(line)
            except Exception:
                continue

            rows.extend(self._coerce_records(parsed))

        return rows

    def _parse_records_array_lenient(self, content: str) -> list[dict[str, Any]]:
        """
        Recover individual array elements after a "records": [ prefix.

        This handles payloads like:
            {"records":[{...},"{\"task\": ...}",{...}
        where the wrapper may be truncated or mixed with string-encoded objects.
        """
        records_pos = content.find('"records"')
        if records_pos == -1:
            records_pos = content.find("'records'")
        if records_pos == -1:
            return []

        array_start = content.find("[", records_pos)
        if array_start == -1:
            return []

        decoder = json.JSONDecoder()
        idx = array_start + 1
        rows: list[dict[str, Any]] = []
        n = len(content)

        while idx < n:
            while idx < n and content[idx] in " \t\r\n,":
                idx += 1
            if idx >= n or content[idx] == "]":
                break

            try:
                item, end = decoder.raw_decode(content, idx)
            except Exception:
                next_object = content.find("{", idx + 1)
                next_string_object = content.find('"{', idx + 1)
                candidates = [pos for pos in (next_object, next_string_object) if pos != -1]
                if not candidates:
                    break
                idx = min(candidates)
                continue

            record = self._coerce_record(item)
            if record is not None:
                rows.append(record)
            idx = end

        return rows

    def _extract_json_objects_lenient(self, content: str) -> list[dict[str, Any]]:
        """
        Last-resort object extraction. It may also see nested metadata objects;
        source-specific normalization will drop irrelevant dicts.
        """
        decoder = json.JSONDecoder()
        rows: list[dict[str, Any]] = []
        idx = 0
        n = len(content)

        while idx < n:
            start = content.find("{", idx)
            if start == -1:
                break

            try:
                item, end = decoder.raw_decode(content, start)
            except Exception:
                idx = start + 1
                continue

            record = self._coerce_record(item)
            if record is not None:
                if isinstance(record.get("records"), list):
                    rows.extend(self._coerce_records(record))
                else:
                    rows.append(record)
            idx = end

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
        text = text.replace("\\r\\n", "\n")
        text = text.replace("\\n", "\n")
        text = text.replace("\\t", "\t")
        return text.strip()

    def _quality_ok(self, text: str, metadata: dict[str, Any]) -> bool:
        """Reject only generic LLM-generation artifacts.

        Source-specific validation belongs in the source adapters, not in this
        shared Groq transport/generation engine.
        """
        lowered = text.lower()

        blocked_fragments = [
            "```",
            "here are the records",
            "here is the json",
            "as an ai language model",
            "i cannot generate",
        ]
        if any(fragment in lowered for fragment in blocked_fragments):
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
            "rate_limit_retries": self.rate_limit_retries,
            "min_request_interval_seconds": self.min_request_interval_seconds,
            "structured_outputs": bool(self.structured_outputs and self._structured_response_schema()),
            "output_dir": str(self.output_dir),
        }

    def _record_metadata(self, row: dict[str, Any], idx: int) -> dict[str, Any]:
        return {}

    def _build_prompt(self, batch_count: int, start_index: int, rng: random.Random) -> str:
        raise NotImplementedError

    def _structured_response_name(self) -> str:
        safe = re.sub(r"[^a-zA-Z0-9_]+", "_", self.SOURCE_TAG).strip("_")
        return f"{safe}_records"

    def _structured_response_schema(self, batch_count: int | None = None) -> dict[str, Any] | None:
        return None
