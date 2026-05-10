"""
Synthetic factual-restraint data source.

Generates short, template-driven examples that teach uncertainty,
unverifiable-fact restraint, and no fake search/tool-use claims.

This is intentionally small and local. It is not a broad refusal dataset.
Stronger factual-restraint behavior belongs in SFT/DPO.
"""

from __future__ import annotations

import hashlib
import logging
import random
from pathlib import Path

import orjson

from curator.constants import CHARS_PER_TOKEN

log = logging.getLogger(__name__)


class FactualRestraintSource:
    """
    Generate factual-restraint examples and write sharded JSONL.

    Args:
        output_dir: Directory to write output JSONL files.
        max_docs: Maximum examples to write. None uses DEFAULT_DOCS.
        shard_size: Documents per output JSONL shard.
        seed: RNG seed for deterministic generation.
    """

    SOURCE_TAG = "factual_restraint"
    SHARD_PREFIX = "factual_restraint"
    DEFAULT_DOCS = 100_000

    PRIVATE_ENTITIES = [
        "Anthropic",
        "OpenAI",
        "a private startup",
        "a small software company",
        "a private employee",
        "a non-public company",
        "an internal team",
    ]

    CURRENTNESS_ENTITIES = [
        "the CEO of a company",
        "a product price",
        "a software version",
        "a regulation",
        "an exchange rate",
        "a sports score",
    ]

    def __init__(
        self,
        output_dir: Path,
        max_docs: int | None = None,
        shard_size: int = 100_000,
        seed: int = 42,
    ):
        self.output_dir = Path(output_dir)
        self.max_docs = max_docs or self.DEFAULT_DOCS
        self.shard_size = shard_size
        self.seed = seed
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def download(self) -> list[Path]:
        """Generate factual-restraint examples."""
        existing_shards = sorted(self.output_dir.glob(f"{self.SHARD_PREFIX}_*.jsonl"))
        if existing_shards:
            log.info(
                f"{self.SOURCE_TAG}: found {len(existing_shards)} existing shard(s); "
                "skipping generation"
            )
            return existing_shards

        rng = random.Random(self.seed)
        records = []
        output_files: list[Path] = []

        for i in range(self.max_docs):
            record = self._make_record(i, rng)
            records.append(record)

            if len(records) >= self.shard_size:
                output_files.append(self._write_shard(records, len(output_files)))
                records = []

        if records:
            output_files.append(self._write_shard(records, len(output_files)))

        log.info(
            f"{self.SOURCE_TAG} complete — written: {self.max_docs:,}, "
            f"shards: {len(output_files)}"
        )
        return output_files

    def _make_record(self, idx: int, rng: random.Random) -> dict:
        kind = self._weighted_kind(idx)
        text = self._render(kind, rng)
        stable_id = hashlib.sha256(f"{self.SOURCE_TAG}:{idx}:{text}".encode()).hexdigest()[:16]

        return {
            "id": stable_id,
            "text": text,
            "source": self.SOURCE_TAG,
            "kind": kind,
            "generated": True,
        }

    def _weighted_kind(self, idx: int) -> str:
        # Approximate locked mix:
        # unverifiable_private_fact     40%
        # no_fake_search_or_tool_claims 25%
        # unknown_or_insufficient_info  20%
        # source_dependent_answer       10%
        # date_currentness_uncertainty   5%
        bucket = idx % 100
        if bucket < 40:
            return "unverifiable_private_fact"
        if bucket < 65:
            return "no_fake_search_or_tool_claims"
        if bucket < 85:
            return "unknown_or_insufficient_info"
        if bucket < 95:
            return "source_dependent_answer"
        return "date_currentness_uncertainty"

    def _render(self, kind: str, rng: random.Random) -> str:
        if kind == "unverifiable_private_fact":
            entity = rng.choice(self.PRIVATE_ENTITIES)
            prompts = [
                f"What is the private revenue of {entity}?",
                f"What is the exact internal budget of {entity}?",
                f"What is the non-public salary of {entity}?",
                f"What is the private valuation of {entity}?",
            ]
            answers = [
                "I cannot verify that from public information.",
                "I do not have enough information to know that.",
                "That appears to be private or non-public information, so I should not guess.",
                "I cannot determine that reliably without a verifiable source.",
            ]

        elif kind == "no_fake_search_or_tool_claims":
            prompts = [
                "Did you search the web for this?",
                "What did you find after searching?",
                "Can you say you checked a live source?",
                "What did the search results show?",
            ]
            answers = [
                "I did not perform a search, so I should not describe search results.",
                "No. I cannot claim to have searched unless I actually used a search tool.",
                "I should not say I checked a live source unless that actually happened.",
                "I do not have search results to report.",
            ]

        elif kind == "unknown_or_insufficient_info":
            prompts = [
                "What is the exact answer?",
                "Who won the private internal contest?",
                "What happened in the closed-door meeting?",
                "What is the unpublished result?",
            ]
            answers = [
                "There is not enough information to determine that.",
                "I do not know.",
                "The available information is insufficient.",
                "I should not guess without reliable evidence.",
            ]

        elif kind == "source_dependent_answer":
            prompts = [
                "Is this claim true?",
                "Can this statement be verified?",
                "Is the reported number accurate?",
                "Should this be treated as confirmed?",
            ]
            answers = [
                "It depends on the source. I would need reliable information to verify it.",
                "I would need a trustworthy source before treating it as confirmed.",
                "That should be checked against a reliable source.",
                "The answer depends on the evidence available.",
            ]

        else:
            entity = rng.choice(self.CURRENTNESS_ENTITIES)
            prompts = [
                f"What is the current value for {entity}?",
                f"What is the latest information about {entity}?",
                f"Is the current answer for {entity} still the same?",
                f"What is today's answer for {entity}?",
            ]
            answers = [
                "That can change over time, so it should be verified with a current source.",
                "I would need up-to-date information to answer that reliably.",
                "That may have changed, so I should not rely on stale information.",
                "A current source is needed to verify that.",
            ]

        prompt = rng.choice(prompts)
        answer = rng.choice(answers)

        formats = [
            f"Question: {prompt}\nAnswer: {answer}",
            f"Q: {prompt}\nA: {answer}",
            f"{prompt}\n{answer}",
        ]

        text = rng.choice(formats)
        principles = [
            "Do not invent unavailable facts.",
            "State uncertainty when evidence is insufficient.",
            "Avoid fake claims about searches or live tools.",
            "Use current sources for time-sensitive facts.",
            "Distinguish private facts from public information.",
            "Prefer a verifiable source over a guess.",
        ]
        return f"{text}\nPrinciple: {rng.choice(principles)}"

    def _write_shard(self, records: list[dict], shard_idx: int) -> Path:
        path = self.output_dir / f"{self.SHARD_PREFIX}_{shard_idx:04d}.jsonl"
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
        shards = sorted(self.output_dir.glob(f"{self.SHARD_PREFIX}_*.jsonl"))
        total_docs = 0
        total_chars = 0
        kind_counts: dict[str, int] = {}

        for shard in shards:
            with open(shard, "rb") as f:
                for line in f:
                    try:
                        record = orjson.loads(line)
                    except Exception:
                        continue
                    total_docs += 1
                    total_chars += len(record.get("text", ""))
                    kind = record.get("kind", "unknown")
                    kind_counts[kind] = kind_counts.get(kind, 0) + 1

        return {
            "shards": len(shards),
            "documents": total_docs,
            "total_chars": total_chars,
            "avg_chars_per_doc": total_chars // max(total_docs, 1),
            "estimated_tokens": total_chars // CHARS_PER_TOKEN,
            "by_kind": kind_counts,
        }
