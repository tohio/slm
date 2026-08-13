"""Focused tests for version-locked benchmark-contamination gates."""

import hashlib

from config.benchmarks import (
    BENCHMARKS,
    LM_EVAL_DECONTAMINATION_NGRAM_SIZE,
    LM_EVAL_REVISION,
    LM_EVAL_VERSION,
    benchmark_decontamination_contract,
)
from curator.filters.benchmark_contamination import (
    BenchmarkContaminationAuditor,
    BenchmarkIndex,
    extract_benchmark_query,
    normalize_decontamination_text,
    word_ngrams,
)
from curator.filters.dedup import normalize


class _NaiveMatcher:
    def __init__(self, patterns):
        self.patterns = patterns

    def iter(self, text):
        for pattern, references in self.patterns.items():
            start = 0
            while True:
                position = text.find(pattern, start)
                if position < 0:
                    break
                yield position + len(pattern) - 1, references
                start = position + 1


def _index(query, benchmark="arc_easy"):
    normalized = normalize(query)
    reference = ({
        "benchmark": benchmark,
        "query_id": f"{benchmark}:0",
        "query_sha256": "query-hash",
    },)
    ngram_patterns = {}
    for ngram in word_ngrams(
        normalize_decontamination_text(query),
        LM_EVAL_DECONTAMINATION_NGRAM_SIZE,
    ):
        ngram_reference = ({
            **reference[0],
            "ngram_sha256": hashlib.sha256(ngram.encode()).hexdigest(),
        },)
        ngram_patterns[f" {ngram} "] = ngram_reference
    return BenchmarkIndex(
        matcher=_NaiveMatcher({f" {normalized} ": reference}),
        ngram_matcher=_NaiveMatcher(ngram_patterns),
        query_count=1,
        unique_pattern_count=1,
        ngram_size=LM_EVAL_DECONTAMINATION_NGRAM_SIZE,
        unique_ngram_count=len(ngram_patterns),
        queries_with_ngrams=int(bool(ngram_patterns)),
        task_query_counts={benchmark: 1},
        query_sha256="index-hash",
        contract={"test": True},
    )


def test_benchmark_contract_covers_evaluation_tasks_with_immutable_inputs():
    assert LM_EVAL_VERSION == "0.4.9"
    assert LM_EVAL_DECONTAMINATION_NGRAM_SIZE == 13
    assert len(LM_EVAL_REVISION) == 40
    assert set(BENCHMARKS) == {
        "hellaswag",
        "arc_easy",
        "arc_challenge",
        "mmlu",
        "truthfulqa",
        "humaneval",
    }
    for spec in BENCHMARKS.values():
        assert len(spec["dataset_revision"]) == 40
        assert spec["split"] in {"test", "validation"}
    assert set(benchmark_decontamination_contract()["benchmarks"]) == set(
        BENCHMARKS
    )


def test_query_extractors_match_canonical_task_inputs():
    assert extract_benchmark_query(
        "hellaswag_query_v1",
        {
            "activity_label": "Cooking",
            "ctx_a": "A person stirs soup.",
            "ctx_b": "they add salt",
        },
    ) == "Cooking: A person stirs soup. They add salt"
    assert extract_benchmark_query(
        "arc_query_v1", {"question": "What is an atom?"}
    ) == "Question: What is an atom?\nAnswer:"
    assert extract_benchmark_query(
        "mmlu_query_v1",
        {"question": "Choose one", "choices": ["one", "two", "three", "four"]},
    ) == "Choose one\nA. one\nB. two\nC. three\nD. four\nAnswer:"
    assert extract_benchmark_query(
        "truthfulqa_query_v1", {"question": "Why?"}
    ) == "Why?"
    assert extract_benchmark_query(
        "humaneval_query_v1", {"prompt": "def solve():\n"}
    ) == "def solve():\n"


def test_exact_benchmark_query_inside_document_fails_gate():
    query = "Question: What is an atom?\nAnswer:"
    auditor = BenchmarkContaminationAuditor(_index(query))

    auditor.observe(
        "train",
        7,
        {
            "text": "Introductory material. QUESTION: what is an atom? Answer! End.",
            "source": "fineweb",
        },
    )
    report = auditor.report({"train": 1, "validation": 0})

    assert report["passed"] is False
    assert report["matched_documents"] == 1
    assert report["matched_unique_queries"] == 1
    assert report["matched_documents_by_benchmark"] == {"arc_easy": 1}
    assert report["contaminated_documents"] == 1
    assert report["samples"][0]["source"] == "fineweb"
    assert "text" not in report["samples"][0]


def test_13_word_overlap_fails_when_full_query_does_not_match():
    query = (
        "alpha beta gamma delta epsilon zeta eta theta iota kappa lambda mu nu "
        "xi omicron"
    )
    auditor = BenchmarkContaminationAuditor(_index(query))
    auditor.observe(
        "train",
        3,
        {
            "text": (
                "unrelated prefix alpha, beta gamma delta epsilon zeta eta theta "
                "iota kappa lambda mu nu changed ending"
            ),
            "source": "common_crawl",
        },
    )

    report = auditor.report({"train": 1, "validation": 0})

    assert report["passed"] is False
    assert report["matched_documents"] == 0
    assert report["ngram_matched_documents"] == 1
    assert report["ngram_matched_unique_queries"] == 1
    assert report["contaminated_documents"] == 1
    assert report["samples"][0]["matches"] == []
    assert len(report["samples"][0]["ngram_matches"]) == 1


def test_disjoint_document_passes_benchmark_gate():
    auditor = BenchmarkContaminationAuditor(
        _index("Question: What is an atom?\nAnswer:")
    )
    auditor.observe(
        "validation",
        1,
        {"text": "A completely unrelated validation document.", "source": "pg19"},
    )

    report = auditor.report({"train": 0, "validation": 1})

    assert report["passed"] is True
    assert report["contaminated_documents"] == 0
    assert report["matched_documents"] == 0
    assert report["ngram_matched_documents"] == 0
    assert report["samples"] == []
