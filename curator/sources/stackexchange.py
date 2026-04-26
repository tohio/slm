"""
curator/sources/stackexchange.py
---------------------------------
StackExchange data source.

Streams the `HuggingFaceH4/stack-exchange-preferences` dataset — a
multi-site StackExchange Q+A dump covering stackoverflow, math, english,
and dozens of other SE sites. Each record is formatted as a single Q+A
document using the highest-scored answer, following the convention used
by RedPajama and StarCoder's training data.

Format: each document is a Q+A pair formatted as plain text:
    Q: <question body, HTML stripped>

    A: <top-voted answer, HTML stripped>

Only answers with score >= min_answer_score (default 1) are kept, so
downvoted/wrong answers are excluded.

Schema note: HuggingFaceH4/stack-exchange-preferences stores each answer
with fields {answer_id, author, author_id, author_profile, pm_score,
selected, text}. The body text lives in `text`, NOT `answer_body` as
older versions of this file assumed — with the wrong key, every answer
came back as "" and every record was silently dropped. pm_score is
stored as a string (e.g. "2"), so we cast it to int defensively.

HTML stripping: question and answer bodies in the source dataset are
raw HTML (they start with <p> etc.). We strip tags before writing
because the downstream quality filter (curator/filters/quality.py)
treats angle brackets as symbols, which inflates the symbol-to-word
ratio past the rejection threshold and drops 99%+ of records. Earlier
versions of this file deferred stripping to the filter, but the filter
doesn't transform — it only accepts/rejects. Stripping here keeps the
filter's heuristics meaningful.

Output: JSONL with one Q+A per line:
    {
        "text": "Q: ...\\n\\nA: ...",
        "source": "stackexchange",
        "site": "stackoverflow | math | ...",
        "question_id": "..."
    }

Usage:
    from curator.sources.stackexchange import StackExchangeSource
    source = StackExchangeSource(output_dir=Path("data/raw/stackexchange"))
    source.download()
"""

import html
import logging
import re
from pathlib import Path

import orjson
from datasets import load_dataset
from tqdm import tqdm

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


# ── HTML stripping ────────────────────────────────────────────────────────────

_HTML_TAG_RE = re.compile(r"<[^>]+>")
_WHITESPACE_RE = re.compile(r"\n{3,}")
# Block-level closing tags carry paragraph-break semantics. Replace them
# with \n\n before the generic strip so structure survives even when the
# source HTML doesn't have blank lines between blocks. <br> and <br/>
# become a single newline.
_BLOCK_CLOSE_RE = re.compile(
    r"</(?:p|div|li|ul|ol|h[1-6]|blockquote|tr|table|pre)>",
    re.IGNORECASE,
)
_BR_RE = re.compile(r"<br\s*/?>", re.IGNORECASE)


def _strip_html(s: str) -> str:
    """
    Strip HTML tags and decode entities from a string.

    Designed for short Q+A bodies (typically <5KB), not full web pages.
    Uses stdlib only — no trafilatura/BeautifulSoup overhead.

    Steps:
        1. Remove HTML tags with a non-greedy regex
        2. Decode HTML entities (&amp; → &, &lt; → <, &nbsp; → space, etc.)
        3. Collapse runs of 3+ newlines (created by stripped <p>...</p>)

    Code blocks (<pre><code>...</code></pre>) lose their tags but keep
    their content with original whitespace. This matches what training
    data for code-aware models typically wants.

    Returns empty string for None or empty input.
    """
    if not s:
        return ""
    # Convert block-level closing tags to paragraph breaks before the
    # generic strip — this preserves paragraph structure even when the
    # source HTML has no whitespace between adjacent block tags.
    with_breaks = _BLOCK_CLOSE_RE.sub("\n\n", s)
    with_breaks = _BR_RE.sub("\n", with_breaks)
    # Remove remaining tags, then decode entities. Order matters: tag
    # attributes can contain entity-like sequences that shouldn't be
    # decoded before tags are gone.
    no_tags = _HTML_TAG_RE.sub("", with_breaks)
    decoded = html.unescape(no_tags)
    # Collapse runs of 3+ newlines (e.g. "</p>\n<p>" became "\n\n\n\n").
    return _WHITESPACE_RE.sub("\n\n", decoded).strip()


class StackExchangeSource:
    """
    Streams StackExchange Q+A dumps and writes sharded JSONL.

    Uses `HuggingFaceH4/stack-exchange-preferences` which provides a
    multi-site StackExchange dump with question metadata and all answers.
    We format each record as a single Q+A document using the
    highest-scored answer.

    Args:
        output_dir: Directory to write output JSONL files.
        min_length: Minimum combined Q+A character length. Below this, skipped.
        shard_size: Documents per output JSONL shard.
        max_docs: Maximum documents to write. None = no limit. Used for
            mini runs to validate the pipeline.
        min_answer_score: Minimum answer score to include. Filters out
            answers with negative or zero score (likely wrong/low-quality).
    """

    DATASET_NAME = "HuggingFaceH4/stack-exchange-preferences"
    SOURCE_TAG = "stackexchange"

    def __init__(
        self,
        output_dir: Path,
        min_length: int = 200,
        shard_size: int = 50_000,
        max_docs: int | None = None,
        min_answer_score: int = 1,
    ):
        self.output_dir = Path(output_dir)
        self.min_length = min_length
        self.shard_size = shard_size
        self.max_docs = max_docs
        self.min_answer_score = min_answer_score
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Stream StackExchange and write to sharded JSONL files."""
        existing_shards = sorted(self.output_dir.glob("stackexchange_*.jsonl"))
        shard_idx = len(existing_shards)
        skip_records = shard_idx * self.shard_size

        if skip_records > 0:
            log.info(
                f"StackExchange: found {shard_idx} existing shard(s) — "
                f"skipping first {skip_records:,} streamed records"
            )

        log.info(f"Streaming {self.DATASET_NAME} from HuggingFace...")
        stream = load_dataset(
            self.DATASET_NAME,
            split="train",
            streaming=True,
            trust_remote_code=True,
        )

        if self.max_docs:
            log.info(
                f"StackExchange: capped at {self.max_docs:,} documents (mini run)"
            )

        output_files: list[Path] = []
        buffer: list[dict] = []
        total_written = 0
        total_skipped_short = 0
        total_skipped_no_answer = 0
        total_stream_skipped = 0
        stop = False

        pbar = tqdm(desc="Streaming StackExchange", unit="doc")

        for idx, sample in enumerate(stream):
            # Resume: skip records belonging to already-written shards
            if idx < skip_records:
                total_stream_skipped += 1
                if total_stream_skipped % 100_000 == 0:
                    pbar.set_postfix_str(
                        f"skipping {total_stream_skipped:,}/{skip_records:,}"
                    )
                continue

            formatted = self._format(sample)
            if formatted is None:
                total_skipped_no_answer += 1
                continue

            if len(formatted["text"]) < self.min_length:
                total_skipped_short += 1
                continue

            buffer.append(formatted)

            if len(buffer) >= self.shard_size:
                path = self._write_shard(buffer, shard_idx)
                output_files.append(path)
                shard_idx += 1
                total_written += len(buffer)
                buffer = []
                pbar.update(self.shard_size)

            if self.max_docs is not None:
                if total_written + len(buffer) >= self.max_docs:
                    trim_to = max(0, self.max_docs - total_written)
                    buffer = buffer[:trim_to]
                    stop = True
                    break

        if buffer:
            path = self._write_shard(buffer, shard_idx)
            output_files.append(path)
            total_written += len(buffer)

        pbar.close()

        log.info(
            f"StackExchange complete — "
            f"written: {total_written:,}, "
            f"skipped short: {total_skipped_short:,} (< {self.min_length} chars), "
            f"skipped no answer: {total_skipped_no_answer:,}, "
            f"stream-skipped (resume): {total_stream_skipped:,}, "
            f"new shards: {len(output_files)}"
            f"{' (stopped at max_docs cap)' if stop else ''}"
        )
        return output_files

    def _format(self, sample: dict) -> dict | None:
        """
        Format a raw StackExchange sample into a Q+A document.

        HuggingFaceH4/stack-exchange-preferences schema:
            qid:      int
            question: str     — the question body (raw HTML)
            answers:  list    — each with {answer_id, author, author_id,
                                           author_profile, pm_score,
                                           selected, text}
            metadata: list    — [question_url, ...]

        We pick the highest-scoring answer above min_answer_score, strip
        HTML from both question and answer bodies, and concatenate.
        pm_score is stored as a string; cast defensively.
        """
        # Strip HTML at extraction time. The raw bodies start with <p>...
        # — leaving them in causes the downstream quality filter to
        # reject 99%+ of records via symbol-to-word ratio.
        question = _strip_html(sample.get("question") or "")
        if not question:
            return None

        answers = sample.get("answers") or []
        if not answers:
            return None

        # Pick best-scored answer above the min threshold
        best = None
        best_score = self.min_answer_score - 1
        for ans in answers:
            try:
                score = int(ans.get("pm_score", 0))
            except (TypeError, ValueError):
                score = 0
            if score > best_score:
                best = ans
                best_score = score

        if best is None:
            return None

        # The HF dataset stores the answer body in `text`. Older versions
        # of this file read `answer_body`, which returned "" for every
        # answer and silently dropped every record.
        answer_body = _strip_html(best.get("text") or "")
        if not answer_body:
            return None

        # Derive site from the question URL when present.
        # metadata[0] is usually the canonical question URL
        metadata = sample.get("metadata") or []
        site = ""
        qid = str(sample.get("qid", ""))
        if metadata and isinstance(metadata, list):
            first_url = str(metadata[0]) if metadata else ""
            # https://stackoverflow.com/questions/12345/... → stackoverflow
            if "://" in first_url:
                try:
                    host = first_url.split("://", 1)[1].split("/", 1)[0]
                    site = host.split(".")[0] if host else ""
                except Exception:
                    site = ""

        text = f"Q: {question}\n\nA: {answer_body}"

        return {
            "text": text,
            "source": self.SOURCE_TAG,
            "site": site,
            "question_id": qid,
        }

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        """Write records to a JSONL shard atomically via .tmp rename."""
        path = self.output_dir / f"stackexchange_{shard_idx:04d}.jsonl"
        tmp_path = path.with_suffix(".jsonl.tmp")
        try:
            with open(tmp_path, "wb") as f:
                for record in records:
                    f.write(orjson.dumps(record))
                    f.write(b"\n")
            tmp_path.replace(path)
        except Exception:
            tmp_path.unlink(missing_ok=True)
            raise
        log.debug(f"Wrote shard {shard_idx}: {len(records):,} docs → {path}")
        return path

    def stats(self) -> dict:
        """Return stats about already-downloaded shards."""
        shards = sorted(self.output_dir.glob("stackexchange_*.jsonl"))
        total_docs = 0
        total_chars = 0
        site_counts: dict[str, int] = {}

        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    total_docs += 1
                    total_chars += len(record.get("text", ""))
                    site = record.get("site", "unknown") or "unknown"
                    site_counts[site] = site_counts.get(site, 0) + 1

        return {
            "shards": len(shards),
            "documents": total_docs,
            "total_chars": total_chars,
            "avg_chars_per_doc": total_chars // max(total_docs, 1),
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
            "by_site": site_counts,
        }