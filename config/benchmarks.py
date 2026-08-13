"""Version-locked evaluation and benchmark-decontamination contract."""

LM_EVAL_VERSION = "0.4.9"
LM_EVAL_REVISION = "452749513f817315042df9286241a61051392470"
LM_EVAL_DECONTAMINATION_NGRAM_SIZE = 13

# Dataset revisions were resolved from the Hugging Face repositories and are
# immutable inputs to curation. `query_extractor` names the local implementation
# of the corresponding lm-eval v0.4.9 task input/decontamination query.
BENCHMARKS = {
    "hellaswag": {
        "task": "hellaswag",
        "metric": "acc_norm",
        "num_fewshot": 10,
        "description": "Commonsense reasoning",
        "dataset_path": "hellaswag",
        "dataset_name": None,
        "dataset_revision": "218ec52e09a7e7462a5400043bb9a69a41d06b76",
        "split": "validation",
        "query_extractor": "hellaswag_query_v1",
    },
    "arc_easy": {
        "task": "arc_easy",
        "metric": "acc_norm",
        "num_fewshot": 25,
        "description": "Science QA (easy)",
        "dataset_path": "allenai/ai2_arc",
        "dataset_name": "ARC-Easy",
        "dataset_revision": "210d026faf9955653af8916fad021475a3f00453",
        "split": "test",
        "query_extractor": "arc_query_v1",
    },
    "arc_challenge": {
        "task": "arc_challenge",
        "metric": "acc_norm",
        "num_fewshot": 25,
        "description": "Science QA (challenge)",
        "dataset_path": "allenai/ai2_arc",
        "dataset_name": "ARC-Challenge",
        "dataset_revision": "210d026faf9955653af8916fad021475a3f00453",
        "split": "test",
        "query_extractor": "arc_query_v1",
    },
    "mmlu": {
        "task": "mmlu",
        "metric": "acc",
        "num_fewshot": 5,
        "description": "Broad knowledge (57 subjects)",
        "dataset_path": "cais/mmlu",
        "dataset_name": "all",
        "dataset_revision": "c30699e8356da336a370243923dbaf21066bb9fe",
        "split": "test",
        "query_extractor": "mmlu_query_v1",
    },
    "truthfulqa": {
        "task": "truthfulqa_mc2",
        "metric": "acc",
        "num_fewshot": 0,
        "description": "Factual accuracy",
        "dataset_path": "truthful_qa",
        "dataset_name": "multiple_choice",
        "dataset_revision": "741b8276f2d1982aa3d5b832d3ee81ed3b896490",
        "split": "validation",
        "query_extractor": "truthfulqa_query_v1",
    },
    "humaneval": {
        "task": "humaneval",
        "metric": "pass@1",
        "num_fewshot": 0,
        "description": "Code generation",
        "dataset_path": "openai/openai_humaneval",
        "dataset_name": "openai_humaneval",
        "dataset_revision": "7dce6050a7d6d172f3cc5c32aa97f52fa1a2e544",
        "split": "test",
        "query_extractor": "humaneval_query_v1",
    },
}


def benchmark_decontamination_contract() -> dict:
    """Return the immutable public-data inputs and extraction contract."""
    return {
        "lm_eval_version": LM_EVAL_VERSION,
        "lm_eval_revision": LM_EVAL_REVISION,
        "decontamination_ngram_size": LM_EVAL_DECONTAMINATION_NGRAM_SIZE,
        "benchmarks": {
            name: {
                key: spec[key]
                for key in (
                    "task",
                    "dataset_path",
                    "dataset_name",
                    "dataset_revision",
                    "split",
                    "query_extractor",
                )
            }
            for name, spec in BENCHMARKS.items()
        },
    }
