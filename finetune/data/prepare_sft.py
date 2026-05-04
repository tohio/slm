"""
Download and format SFT datasets for chat and code fine-tuning.

Formats all data into the SLM conversation format:
    [
        {"role": "system",    "content": "..."},
        {"role": "user",      "content": "..."},
        {"role": "assistant", "content": "..."}
    ]

Chat template application:
    No "text" field is produced. train_sft.py renames "conversations" →
    "messages" at load time; trl's SFTTrainer then auto-detects the
    conversational format and applies tokenizer.apply_chat_template()
    internally. Pre-formatting here would bypass the tokenizer's baked-in
    chat template (including {% generation %} tags) and break
    assistant_only_loss=True.

Stage 1 — Chat SFT:
    Dataset: teknium/OpenHermes-2.5
    Size:    ~1M examples
    Output:  data/sft/chat/train.jsonl + val.jsonl

Stage 2 — Code SFT:
    Dataset: ise-uiuc/Magicoder-OSS-Instruct-75K
    Size:    ~75k examples before filtering
    Output:  data/sft/code/train.jsonl + val.jsonl

    Code SFT keeps examples whose assistant responses contain actual code.
    Obvious prose-only / explanation-only examples are dropped so this stage
    teaches the model to emit code when code is requested.

Output format — one conversation per line:
    {
        "conversations": [
            {"role": "system",    "content": "..."},
            {"role": "user",      "content": "..."},
            {"role": "assistant", "content": "..."}
        ],
        "source": "openhermes",
        "sft_type": "general_assistant"
    }

Usage:
    python finetune/data/prepare_sft.py --stage both
    python finetune/data/prepare_sft.py --stage chat
    python finetune/data/prepare_sft.py --stage code
"""

import argparse
import json
import logging
import os
import random
import re
import sys
from collections import Counter
from pathlib import Path

from dotenv import load_dotenv

from finetune.data.response_control import build_response_control_records

load_dotenv()

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
)
log = logging.getLogger(__name__)

DATA_DIR = Path(os.environ.get("DATA_DIR", "data"))
SFT_DIR  = DATA_DIR / "sft"

# Default system prompts
DEFAULT_SYSTEM = "You are a helpful, harmless, and honest assistant."
CODE_SYSTEM = (
    "You are an expert programming assistant. "
    "When code is requested, write code directly and avoid unnecessary explanation. "
    "When explanation is requested, explain clearly in prose and do not rewrite the code."
)

# Handcrafted explanation examples are intentionally oversampled because the
# Magicoder code stage is dominated by code-output tasks. A small repeated
# prose-only explanation signal helps preserve task-mode distinction:
#   explain code -> explain in prose
#   write code   -> write code
HANDCRAFTED_CODE_EXPLANATION_REPEAT = 20


# Small response-control set for chat SFT. These examples target failure modes
# that broad instruction data often leaves weak in small models: direct factual
# answers, basic arithmetic, AI/ML concept grounding, factual restraint, and
# stopping after a complete answer. Keep this set small and explicit; it is a
# behavior anchor, not a replacement for OpenHermes.
HANDCRAFTED_RESPONSE_CONTROL_CHATS = [
    {
        "user": "What is 2 + 2?",
        "assistant": "4",
        "sft_type": "simple_factual",
    },
    {
        "user": "What is 7 + 5?",
        "assistant": "12",
        "sft_type": "simple_factual",
    },
    {
        "user": "What is 9 - 3?",
        "assistant": "6",
        "sft_type": "simple_factual",
    },
    {
        "user": "What is 3 * 4?",
        "assistant": "12",
        "sft_type": "simple_factual",
    },
    {
        "user": "What is 20 divided by 5?",
        "assistant": "4",
        "sft_type": "simple_factual",
    },
    {
        "user": "What is the capital of France?",
        "assistant": "Paris.",
        "sft_type": "simple_factual",
    },
    {
        "user": "What is the capital of Japan?",
        "assistant": "Tokyo.",
        "sft_type": "simple_factual",
    },
    {
        "user": "What is the capital of Italy?",
        "assistant": "Rome.",
        "sft_type": "simple_factual",
    },
    {
        "user": "What color is the sky on a clear day?",
        "assistant": "Blue.",
        "sft_type": "simple_factual",
    },
    {
        "user": "Give a short answer: What is the capital of Canada?",
        "assistant": "Ottawa.",
        "sft_type": "simple_factual",
    },
    {
        "user": "In AI, what is a transformer model?",
        "assistant": "A transformer is a neural network architecture that uses attention mechanisms to process sequences and learn relationships between tokens.",
        "sft_type": "ai_concept",
    },
    {
        "user": "Explain what a transformer is in artificial intelligence.",
        "assistant": "In AI, a transformer is a neural network architecture based on attention. It is widely used for language models because it can learn relationships between tokens in a sequence.",
        "sft_type": "ai_concept",
    },
    {
        "user": "What is self-attention in a transformer?",
        "assistant": "Self-attention lets each token compare itself with other tokens in the same sequence so the model can decide which context is most relevant.",
        "sft_type": "ai_concept",
    },
    {
        "user": "What is a neural network?",
        "assistant": "A neural network is a machine learning model made of connected layers that learn patterns from data.",
        "sft_type": "ai_concept",
    },
    {
        "user": "What is supervised learning?",
        "assistant": "Supervised learning trains a model on examples that include both inputs and correct target outputs.",
        "sft_type": "ai_concept",
    },
    {
        "user": "What is overfitting?",
        "assistant": "Overfitting happens when a model memorizes training data too closely and performs poorly on new data.",
        "sft_type": "ai_concept",
    },
    {
        "user": "What is an embedding in machine learning?",
        "assistant": "An embedding is a learned vector representation that maps items such as words, tokens, or users into a numeric space.",
        "sft_type": "ai_concept",
    },
    {
        "user": "What is the difference between training and inference?",
        "assistant": "Training updates model weights using data. Inference uses the trained model to produce outputs without updating the weights.",
        "sft_type": "ai_concept",
    },
    {
        "user": "What was Anthropic's private revenue last month?",
        "assistant": "I do not have enough verified public information to answer that.",
        "sft_type": "factual_restraint",
    },
    {
        "user": "What was OpenAI's exact internal revenue yesterday?",
        "assistant": "I do not have access to verified internal revenue figures, so I cannot answer that reliably.",
        "sft_type": "factual_restraint",
    },
    {
        "user": "What is the private salary of Anthropic's CEO this month?",
        "assistant": "I do not have verified public information about that private compensation detail.",
        "sft_type": "factual_restraint",
    },
    {
        "user": "What is Apple's current private unreleased product roadmap?",
        "assistant": "I do not have verified access to Apple's private unreleased roadmap.",
        "sft_type": "factual_restraint",
    },
    {
        "user": "Give me the exact current bank balance of Microsoft.",
        "assistant": "I cannot verify an exact current bank balance. Public financial filings may provide reported cash and equivalents for a specific reporting period.",
        "sft_type": "factual_restraint",
    },
    {
        "user": "Do you know the exact answer if the information is private and not publicly verified?",
        "assistant": "No. If information is private or not publicly verified, I should say that I cannot verify it.",
        "sft_type": "factual_restraint",
    },
    {
        "user": "Answer concisely: What is the capital of Spain?",
        "assistant": "Madrid.",
        "sft_type": "stop_behavior",
    },
    {
        "user": "Answer concisely: What is 10 + 1?",
        "assistant": "11",
        "sft_type": "stop_behavior",
    },
    {
        "user": "Answer in one sentence: What is a database?",
        "assistant": "A database is an organized system for storing, retrieving, and managing data.",
        "sft_type": "stop_behavior",
    },
    {
        "user": "Answer in one sentence: What is Python?",
        "assistant": "Python is a high-level programming language used for scripting, automation, data work, and application development.",
        "sft_type": "stop_behavior",
    },
    {
        "user": "What should you do when you are not sure about a factual claim?",
        "assistant": "I should say that I am not sure or cannot verify it, rather than inventing details.",
        "sft_type": "factual_restraint",
    },
    {
        "user": "What is a GPU?",
        "assistant": "A GPU is a processor designed for parallel computation, commonly used for graphics and machine learning workloads.",
        "sft_type": "ai_concept",
    },
]


def build_handcrafted_response_control_records() -> list[dict]:
    """Return generated response-control chat examples.

    Kept under the old function name so prepare_chat() does not need broader
    wiring changes. The source field is now "response_control".
    """
    return build_response_control_records(
        system=DEFAULT_SYSTEM,
        max_examples=2000,
    )

# Per-stage defaults for validation fraction. These are the sources of truth;
# CLI --val-fraction overrides them only when explicitly passed.
DEFAULT_VAL_FRACTION = {
    "chat": 0.02,
    "code": 0.05,
}


def write_jsonl(records: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        for record in records:
            f.write(json.dumps(record, ensure_ascii=False) + "\n")
    log.info(f"Wrote {len(records):,} records to {path}")


# ── Code SFT filtering helpers ────────────────────────────────────────────────

CODE_KEYWORDS = (
    "def ",
    "class ",
    "return ",
    "import ",
    "from ",
    "for ",
    "while ",
    "if ",
    "elif ",
    "else:",
    "try:",
    "except ",
    "with ",
    "lambda ",
    "async ",
    "await ",
)

PROSE_ONLY_STARTS = (
    "the given code",
    "the provided code",
    "this code",
    "this function",
    "this program",
    "to solve this problem",
    "to implement this",
    "here is an explanation",
    "here's an explanation",
    "in this solution",
    "we need to",
    "you can",
)

CODE_OUTPUT_TYPES = {"code_generation", "function_completion", "code_repair"}

CODE_BLOCK_RE = re.compile(
    r"```(?:[a-zA-Z0-9_+\-.#]+)?\s*\n(.*?)```",
    re.DOTALL,
)


def has_code_fence(text: str) -> bool:
    return "```" in text


def count_indented_code_lines(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped:
            continue
        if line.startswith(("    ", "	")) and not stripped.startswith(("-", "*")):
            count += 1
    return count


def count_code_keyword_lines(text: str) -> int:
    count = 0
    for line in text.splitlines():
        stripped = line.strip()
        if any(stripped.startswith(keyword) for keyword in CODE_KEYWORDS):
            count += 1
    return count


def has_programming_syntax(text: str) -> bool:
    patterns = [
        r"\w+\s*=\s*[^=]",
        r"\w+\([^)]*\)",
        r"[{};]",
        r"==|!=|<=|>=|->|=>",
        r"\[[^\]]+\]",
    ]
    return any(re.search(pattern, text) for pattern in patterns)


def looks_like_code(text: str) -> bool:
    """Return True when an assistant response contains substantial code.

    This intentionally uses broad signals because Magicoder contains multiple
    languages and formats. The goal is not perfect language detection; the goal
    is to drop obvious prose-only examples from the code SFT stage.
    """
    if not text or len(text.strip()) < 20:
        return False

    keyword_lines = count_code_keyword_lines(text)
    indented_lines = count_indented_code_lines(text)

    if has_code_fence(text) and (keyword_lines >= 1 or has_programming_syntax(text)):
        return True

    if keyword_lines >= 2:
        return True

    if indented_lines >= 2 and has_programming_syntax(text):
        return True

    if "def " in text and "return" in text:
        return True

    if "class " in text and ("def " in text or "return" in text):
        return True

    return False


def is_prose_heavy_without_code(text: str) -> bool:
    stripped = text.strip().lower()
    starts_like_explanation = stripped.startswith(PROSE_ONLY_STARTS)
    return starts_like_explanation and not looks_like_code(text)


STRICT_FUNCTION_COMPLETION_PHRASES = (
    "return only the function body",
    "return only function body",
    "return only the method body",
    "return only method body",
    "function body only",
    "method body only",
    "complete the function body",
    "complete this function body",
    "complete the method body",
    "complete this method body",
    "implement the function body",
    "implement this function body",
    "implement the method body",
    "implement this method body",
    "fill in the function body",
    "fill in this function body",
    "fill in the method body",
    "fill in this method body",
    "replace the pass statement",
    "replace pass with",
)


def is_strict_function_completion_prompt(prompt: str) -> bool:
    """Return True only for body/completion-style code prompts.

    Magicoder often says "complete the implementation" for broad tasks that
    expect a full function, class, script, or module. Those are useful code
    generation examples, but they are not HumanEval-style function-completion
    examples. Keep `function_completion` narrow so it teaches body-only /
    missing-code behavior instead of full-code generation.
    """
    prompt = prompt.lower()

    if any(phrase in prompt for phrase in STRICT_FUNCTION_COMPLETION_PHRASES):
        return True

    body_pattern = re.compile(
        r"\b(complete|implement|fill in|write)\b.{0,120}\b(function|method)\b.{0,120}\bbody\b",
        re.DOTALL,
    )
    if body_pattern.search(prompt):
        return True

    return_only_body_pattern = re.compile(
        r"\breturn only\b.{0,80}\b(function|method)\b.{0,40}\bbody\b",
        re.DOTALL,
    )
    if return_only_body_pattern.search(prompt):
        return True

    missing_code_pattern = re.compile(
        r"\b(fill in|complete)\b.{0,80}\b(missing code|todo|pass statement)\b",
        re.DOTALL,
    )
    if missing_code_pattern.search(prompt):
        return True

    return False


CODE_CREATION_PROMPT_RE = re.compile(
    r"\b(implement|write|create|complete|add|build|generate|finish|fill in)\b",
    re.DOTALL,
)

EXPLANATION_PROMPT_PATTERNS = (
    re.compile(r"\bexplain\s+(what|how|why|the|this|that|following)\b", re.DOTALL),
    re.compile(r"\bwhat\s+does\b", re.DOTALL),
    re.compile(r"\bwhat\s+will\s+(be\s+)?(printed|output)\b", re.DOTALL),
    re.compile(r"\bpredict\s+the\s+output\b", re.DOTALL),
    re.compile(r"\bunderstand\s+the\s+behavior\b", re.DOTALL),
)


def is_explanation_prompt(prompt: str) -> bool:
    """Return True for prompts that ask to explain/analyze existing code.

    Magicoder problem statements often contain words like "describe" while the
    actual task is still to implement code. Keep this narrow so implementation
    prompts do not get mislabeled as code_explanation.
    """
    prompt = prompt.lower()

    if not any(pattern.search(prompt) for pattern in EXPLANATION_PROMPT_PATTERNS):
        return False

    # If the prompt explicitly asks for new code, treat it as code generation
    # unless it is one of the output-prediction forms above.
    if (
        CODE_CREATION_PROMPT_RE.search(prompt)
        and "predict the output" not in prompt
        and "what will be printed" not in prompt
        and "what is the output" not in prompt
    ):
        return False

    return True


def looks_like_mostly_code(text: str) -> bool:
    """Return True when an assistant response is mostly code, not prose."""
    stripped = text.strip()
    if not stripped:
        return False

    if stripped.startswith("```"):
        return True

    lines = [line for line in stripped.splitlines() if line.strip()]
    if not lines:
        return False

    first = lines[0].strip().lower()
    code_starts = (
        "def ", "class ", "import ", "from ", "function ", "const ", "let ",
        "var ", "public ", "private ", "protected ", "#include", "using ",
        "package ", "func ", "fn ", "struct ", "enum ", "interface ",
        "#!/", "mv ", "cp ", "kubectl ", "minikube ",
    )
    if first.startswith(code_starts):
        return True

    fenced = extract_fenced_code(stripped)
    if fenced:
        fenced_lines = [line for line in fenced.splitlines() if line.strip()]
        if len(fenced_lines) >= max(3, len(lines) // 2):
            return True

    code_lines = 0
    prose_lines = 0
    for line in lines:
        s = line.strip()
        lower = s.lower()
        if (
            any(lower.startswith(prefix) for prefix in code_starts)
            or any(lower.startswith(keyword) for keyword in CODE_KEYWORDS)
            or re.search(r"[{};]", s)
            or re.search(r"\w+\s*=\s*[^=]", s)
        ):
            code_lines += 1
        else:
            prose_lines += 1

    return code_lines >= 3 and code_lines >= prose_lines


def classify_code_sft_type(instruction: str, solution: str) -> str:
    prompt = instruction.lower()
    answer = solution.lower()

    # Explanation and repair prompts are task-mode specific and should not be
    # swallowed by broad completion wording. Keep explanation detection narrow:
    # "implement this" prompts are code_generation even if the generated problem
    # statement contains explanatory words.
    if is_explanation_prompt(prompt):
        return "code_explanation"

    if "fix" in prompt or "debug" in prompt or "bug" in prompt:
        return "code_repair"

    # Keep function_completion strict. Broad prompts like "complete the
    # implementation of this class/method" usually expect full code and should
    # remain code_generation.
    if is_strict_function_completion_prompt(prompt):
        return "function_completion"

    if "```" in answer or looks_like_code(solution):
        return "code_generation"

    return "code_other"


def extract_fenced_code(text: str) -> str | None:
    """Return the largest fenced code block from a response, without fences."""
    blocks = [match.group(1).strip() for match in CODE_BLOCK_RE.finditer(text)]
    blocks = [block for block in blocks if block]
    if not blocks:
        return None
    return max(blocks, key=len).strip()


def normalize_code_solution(solution: str, sft_type: str) -> tuple[str, bool]:
    """Normalize assistant output for code-output SFT examples.

    Magicoder often emits a fenced code block followed by prose explanation.
    For code_generation, function_completion, and code_repair, keep only the
    largest fenced code block when present. This teaches code-output tasks to
    produce code directly. For code_explanation, keep the original prose.
    """
    solution = solution.strip()

    if sft_type not in CODE_OUTPUT_TYPES:
        return solution, False

    fenced = extract_fenced_code(solution)
    if fenced:
        return fenced, True

    return solution, False


# ── Handcrafted function-completion examples ─────────────────────────────────

HANDCRAFTED_FUNCTION_COMPLETIONS = [
    {
        "imports": "from typing import List",
        "signature": "def first_item(items: List[int]) -> int:",
        "docstring": "Return the first item in the list.",
        "body": "return items[0]",
    },
    {
        "imports": "from typing import List",
        "signature": "def last_item(items: List[int]) -> int:",
        "docstring": "Return the last item in the list.",
        "body": "return items[-1]",
    },
    {
        "imports": "from typing import List",
        "signature": "def sum_items(items: List[int]) -> int:",
        "docstring": "Return the sum of all integers in the list.",
        "body": "return sum(items)",
    },
    {
        "imports": "from typing import List",
        "signature": "def max_item(items: List[int]) -> int:",
        "docstring": "Return the largest integer in the list.",
        "body": "return max(items)",
    },
    {
        "imports": "from typing import List",
        "signature": "def min_item(items: List[int]) -> int:",
        "docstring": "Return the smallest integer in the list.",
        "body": "return min(items)",
    },
    {
        "imports": "from typing import List",
        "signature": "def count_positive(numbers: List[int]) -> int:",
        "docstring": "Return the number of positive integers.",
        "body": "return sum(1 for n in numbers if n > 0)",
    },
    {
        "imports": "from typing import List",
        "signature": "def filter_positive(numbers: List[int]) -> List[int]:",
        "docstring": "Return only the positive integers from the list.",
        "body": "return [n for n in numbers if n > 0]",
    },
    {
        "imports": "from typing import List",
        "signature": "def filter_even(numbers: List[int]) -> List[int]:",
        "docstring": "Return only the even integers from the list.",
        "body": "return [n for n in numbers if n % 2 == 0]",
    },
    {
        "imports": "from typing import List",
        "signature": "def double_values(numbers: List[int]) -> List[int]:",
        "docstring": "Return a new list with each value doubled.",
        "body": "return [n * 2 for n in numbers]",
    },
    {
        "imports": "from typing import List",
        "signature": "def contains_value(items: List[int], value: int) -> bool:",
        "docstring": "Return True if value is present in items.",
        "body": "return value in items",
    },
    {
        "imports": "from typing import List",
        "signature": "def average(numbers: List[float]) -> float:",
        "docstring": "Return the arithmetic mean of the numbers.",
        "body": "return sum(numbers) / len(numbers)",
    },
    {
        "imports": "from typing import List",
        "signature": "def safe_average(numbers: List[float]) -> float:",
        "docstring": "Return the average, or 0.0 for an empty list.",
        "body": "if not numbers:\n    return 0.0\nreturn sum(numbers) / len(numbers)",
    },
    {
        "imports": "from typing import List",
        "signature": "def has_close_elements(numbers: List[float], threshold: float) -> bool:",
        "docstring": "Return True if any two numbers are closer than threshold.",
        "body": "for i in range(len(numbers)):\n    for j in range(i + 1, len(numbers)):\n        if abs(numbers[i] - numbers[j]) < threshold:\n            return True\nreturn False",
    },
    {
        "imports": "from typing import List",
        "signature": "def all_unique(items: List[int]) -> bool:",
        "docstring": "Return True if all items are unique.",
        "body": "return len(items) == len(set(items))",
    },
    {
        "imports": "from typing import List",
        "signature": "def remove_duplicates(items: List[int]) -> List[int]:",
        "docstring": "Return items with duplicates removed while preserving order.",
        "body": "seen = set()\nresult = []\nfor item in items:\n    if item not in seen:\n        seen.add(item)\n        result.append(item)\nreturn result",
    },
    {
        "imports": "from typing import List",
        "signature": "def flatten(nested: List[List[int]]) -> List[int]:",
        "docstring": "Flatten a list of integer lists.",
        "body": "return [item for group in nested for item in group]",
    },
    {
        "imports": "from typing import List",
        "signature": "def chunk_list(items: List[int], size: int) -> List[List[int]]:",
        "docstring": "Split items into chunks of the given size.",
        "body": "return [items[i:i + size] for i in range(0, len(items), size)]",
    },
    {
        "imports": "from typing import Dict",
        "signature": "def get_with_default(data: Dict[str, int], key: str, default: int) -> int:",
        "docstring": "Return data[key] if present, otherwise default.",
        "body": "return data.get(key, default)",
    },
    {
        "imports": "from typing import Dict",
        "signature": "def invert_mapping(data: Dict[str, int]) -> Dict[int, str]:",
        "docstring": "Invert a dictionary from string-to-int into int-to-string.",
        "body": "return {value: key for key, value in data.items()}",
    },
    {
        "imports": "from typing import Dict",
        "signature": "def merge_counts(left: Dict[str, int], right: Dict[str, int]) -> Dict[str, int]:",
        "docstring": "Merge two count dictionaries by adding values for matching keys.",
        "body": "result = dict(left)\nfor key, value in right.items():\n    result[key] = result.get(key, 0) + value\nreturn result",
    },
    {
        "imports": "",
        "signature": "def reverse_string(text: str) -> str:",
        "docstring": "Return text reversed.",
        "body": "return text[::-1]",
    },
    {
        "imports": "",
        "signature": "def is_palindrome(text: str) -> bool:",
        "docstring": "Return True if text reads the same forward and backward.",
        "body": "cleaned = ''.join(ch.lower() for ch in text if ch.isalnum())\nreturn cleaned == cleaned[::-1]",
    },
    {
        "imports": "",
        "signature": "def count_vowels(text: str) -> int:",
        "docstring": "Return the number of vowels in text.",
        "body": "return sum(1 for ch in text.lower() if ch in 'aeiou')",
    },
    {
        "imports": "",
        "signature": "def title_case_words(text: str) -> str:",
        "docstring": "Return text with each word title-cased.",
        "body": "return ' '.join(word.capitalize() for word in text.split())",
    },
    {
        "imports": "",
        "signature": "def normalize_spaces(text: str) -> str:",
        "docstring": "Collapse repeated whitespace into single spaces.",
        "body": "return ' '.join(text.split())",
    },
    {
        "imports": "",
        "signature": "def starts_and_ends_with(text: str, prefix: str, suffix: str) -> bool:",
        "docstring": "Return True if text starts with prefix and ends with suffix.",
        "body": "return text.startswith(prefix) and text.endswith(suffix)",
    },
    {
        "imports": "",
        "signature": "def factorial(n: int) -> int:",
        "docstring": "Return n factorial.",
        "body": "result = 1\nfor value in range(2, n + 1):\n    result *= value\nreturn result",
    },
    {
        "imports": "",
        "signature": "def trailing_zeroes_in_factorial(num: int) -> int:",
        "docstring": "Return the number of trailing zeroes in num factorial.",
        "body": "count = 0\nwhile num >= 5:\n    num //= 5\n    count += num\nreturn count",
    },
    {
        "imports": "",
        "signature": "def clamp(value: int, low: int, high: int) -> int:",
        "docstring": "Clamp value to the inclusive range [low, high].",
        "body": "return max(low, min(value, high))",
    },
    {
        "imports": "",
        "signature": "def gcd(a: int, b: int) -> int:",
        "docstring": "Return the greatest common divisor of a and b.",
        "body": "while b:\n    a, b = b, a % b\nreturn abs(a)",
    },
    {
        "imports": "",
        "signature": "def is_prime(n: int) -> bool:",
        "docstring": "Return True if n is prime.",
        "body": "if n < 2:\n    return False\nfor value in range(2, int(n ** 0.5) + 1):\n    if n % value == 0:\n        return False\nreturn True",
    },
    {
        "imports": "from typing import List",
        "signature": "def binary_search(items: List[int], target: int) -> int:",
        "docstring": "Return the index of target in sorted items, or -1 if missing.",
        "body": "left, right = 0, len(items) - 1\nwhile left <= right:\n    mid = (left + right) // 2\n    if items[mid] == target:\n        return mid\n    if items[mid] < target:\n        left = mid + 1\n    else:\n        right = mid - 1\nreturn -1",
    },
    {
        "imports": "from typing import List",
        "signature": "def rotate_left(items: List[int], steps: int) -> List[int]:",
        "docstring": "Rotate items left by steps positions.",
        "body": "if not items:\n    return []\nsteps %= len(items)\nreturn items[steps:] + items[:steps]",
    },
    {
        "imports": "from typing import List",
        "signature": "def pair_sums(numbers: List[int], target: int) -> List[tuple[int, int]]:",
        "docstring": "Return pairs of numbers whose sum equals target.",
        "body": "pairs = []\nseen = set()\nfor number in numbers:\n    complement = target - number\n    if complement in seen:\n        pairs.append((complement, number))\n    seen.add(number)\nreturn pairs",
    },
    {
        "imports": "from typing import List",
        "signature": "def transpose(matrix: List[List[int]]) -> List[List[int]]:",
        "docstring": "Return the transpose of a rectangular matrix.",
        "body": "return [list(row) for row in zip(*matrix)]",
    },
    {
        "imports": "from typing import List",
        "signature": "def diagonal_sum(matrix: List[List[int]]) -> int:",
        "docstring": "Return the sum of the main diagonal.",
        "body": "return sum(matrix[i][i] for i in range(min(len(matrix), len(matrix[0]))))",
    },
    {
        "imports": "from typing import List",
        "signature": "def find_missing_number(numbers: List[int]) -> int:",
        "docstring": "Given numbers from 0..n with one missing, return the missing number.",
        "body": "n = len(numbers)\nreturn n * (n + 1) // 2 - sum(numbers)",
    },
    {
        "imports": "from typing import List",
        "signature": "def move_zeroes(numbers: List[int]) -> List[int]:",
        "docstring": "Return a list with all zeroes moved to the end.",
        "body": "nonzero = [n for n in numbers if n != 0]\nreturn nonzero + [0] * (len(numbers) - len(nonzero))",
    },
    {
        "imports": "from typing import List",
        "signature": "def running_total(numbers: List[int]) -> List[int]:",
        "docstring": "Return running totals for the input numbers.",
        "body": "total = 0\nresult = []\nfor number in numbers:\n    total += number\n    result.append(total)\nreturn result",
    },
    {
        "imports": "from typing import List",
        "signature": "def longest_word(words: List[str]) -> str:",
        "docstring": "Return the longest word, or an empty string for no words.",
        "body": "if not words:\n    return ''\nreturn max(words, key=len)",
    },
    {
        "imports": "from typing import List, Dict",
        "signature": "def group_by_first_letter(words: List[str]) -> Dict[str, List[str]]:",
        "docstring": "Group words by their first letter.",
        "body": "groups = {}\nfor word in words:\n    if not word:\n        continue\n    key = word[0].lower()\n    groups.setdefault(key, []).append(word)\nreturn groups",
    },
    {
        "imports": "from typing import List, Dict",
        "signature": "def word_counts(words: List[str]) -> Dict[str, int]:",
        "docstring": "Return a dictionary counting each word.",
        "body": "counts = {}\nfor word in words:\n    counts[word] = counts.get(word, 0) + 1\nreturn counts",
    },
    {
        "imports": "from typing import Optional, List",
        "signature": "def find_first_even(numbers: List[int]) -> Optional[int]:",
        "docstring": "Return the first even number, or None if there is none.",
        "body": "for number in numbers:\n    if number % 2 == 0:\n        return number\nreturn None",
    },
    {
        "imports": "from typing import Optional, List",
        "signature": "def find_first_match(words: List[str], prefix: str) -> Optional[str]:",
        "docstring": "Return the first word that starts with prefix, or None.",
        "body": "for word in words:\n    if word.startswith(prefix):\n        return word\nreturn None",
    },
    {
        "imports": "",
        "signature": "def parse_int(value: str, default: int = 0) -> int:",
        "docstring": "Parse value as an integer, returning default on failure.",
        "body": "try:\n    return int(value)\nexcept (TypeError, ValueError):\n    return default",
    },
    {
        "imports": "",
        "signature": "def safe_divide(a: float, b: float) -> float:",
        "docstring": "Return a divided by b, or 0.0 if b is zero.",
        "body": "if b == 0:\n    return 0.0\nreturn a / b",
    },
    {
        "imports": "from pathlib import Path",
        "signature": "def file_extension(path: str) -> str:",
        "docstring": "Return the lowercase file extension for path.",
        "body": "return Path(path).suffix.lower()",
    },
    {
        "imports": "from pathlib import Path",
        "signature": "def file_stem(path: str) -> str:",
        "docstring": "Return the filename without its extension.",
        "body": "return Path(path).stem",
    },
    {
        "imports": "import re",
        "signature": "def slugify(text: str) -> str:",
        "docstring": "Convert text to a lowercase URL slug.",
        "body": "slug = re.sub(r'[^a-zA-Z0-9]+', '-', text.lower()).strip('-')\nreturn slug",
    },
    {
        "imports": "import re",
        "signature": "def is_valid_email(email: str) -> bool:",
        "docstring": "Return True if email has a simple valid email shape.",
        "body": "return bool(re.match(r'^[^@\\s]+@[^@\\s]+\\.[^@\\s]+$', email))",
    },
    {
        "imports": "from datetime import datetime",
        "signature": "def format_date(value: datetime) -> str:",
        "docstring": "Return a date formatted as YYYY-MM-DD.",
        "body": "return value.strftime('%Y-%m-%d')",
    },
    {
        "imports": "from typing import List",
        "signature": "def join_nonempty(parts: List[str], sep: str = ', ') -> str:",
        "docstring": "Join non-empty strings with the separator.",
        "body": "return sep.join(part for part in parts if part)",
    },
    {
        "imports": "from typing import List",
        "signature": "def pad_to_length(items: List[int], length: int, fill: int = 0) -> List[int]:",
        "docstring": "Pad items with fill until it reaches length.",
        "body": "if len(items) >= length:\n    return items[:length]\nreturn items + [fill] * (length - len(items))",
    },
    {
        "imports": "from typing import List",
        "signature": "def split_even_odd(numbers: List[int]) -> tuple[List[int], List[int]]:",
        "docstring": "Return two lists: even numbers and odd numbers.",
        "body": "evens = [n for n in numbers if n % 2 == 0]\nodds = [n for n in numbers if n % 2 != 0]\nreturn evens, odds",
    },
    {
        "imports": "from typing import List",
        "signature": "def second_largest(numbers: List[int]) -> int:",
        "docstring": "Return the second-largest unique number.",
        "body": "unique = sorted(set(numbers))\nreturn unique[-2]",
    },
    {
        "imports": "from typing import List",
        "signature": "def remove_none(items: List[object]) -> List[object]:",
        "docstring": "Return a list with None values removed.",
        "body": "return [item for item in items if item is not None]",
    },
    {
        "imports": "from typing import List",
        "signature": "def every_other(items: List[int]) -> List[int]:",
        "docstring": "Return every other item from the list.",
        "body": "return items[::2]",
    },
]


def build_handcrafted_function_completion_records() -> list[dict]:
    """Return small body-only examples for HumanEval-style behavior."""
    records = []

    for example in HANDCRAFTED_FUNCTION_COMPLETIONS:
        imports = example["imports"].strip()
        signature = example["signature"].strip()
        docstring = example["docstring"].strip()
        body = example["body"].strip()

        snippet_parts = []
        if imports:
            snippet_parts.append(imports)
        snippet_parts.append(f'{signature}\n    """{docstring}"""')
        snippet = "\n\n".join(snippet_parts)

        prompt = (
            "Complete this Python function. Return only the function body.\n\n"
            f"{snippet}"
        )

        records.append({
            "conversations": [
                {"role": "system", "content": CODE_SYSTEM},
                {"role": "user", "content": prompt},
                {"role": "assistant", "content": body},
            ],
            "source": "handcrafted_function_completion",
            "sft_type": "function_completion",
            "normalized": False,
        })

    return records


# ── Handcrafted code-explanation examples ─────────────────────────────────────

HANDCRAFTED_CODE_EXPLANATIONS = [
    {
        "prompt": "Explain what this Python function does:\n\ndef square(x):\n    return x * x",
        "answer": "It returns the square of x by multiplying x by itself.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef is_even(n):\n    return n % 2 == 0",
        "answer": "It returns True when n is evenly divisible by 2, otherwise it returns False.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef first_item(items):\n    return items[0]",
        "answer": "It returns the first element from the items list.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef count_positive(numbers):\n    return sum(1 for n in numbers if n > 0)",
        "answer": "It counts how many numbers in the input are greater than zero.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef reverse_string(text):\n    return text[::-1]",
        "answer": "It returns a new string with the characters of text in reverse order.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef safe_get(mapping, key, default=None):\n    return mapping.get(key, default)",
        "answer": "It looks up key in the mapping and returns default if the key is missing.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef remove_duplicates(items):\n    return list(dict.fromkeys(items))",
        "answer": "It removes duplicate values while preserving the original order of the items.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef clamp(value, low, high):\n    return max(low, min(value, high))",
        "answer": "It restricts value to stay between low and high.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef has_close_elements(numbers, threshold):\n    for i in range(len(numbers)):\n        for j in range(i + 1, len(numbers)):\n            if abs(numbers[i] - numbers[j]) < threshold:\n                return True\n    return False",
        "answer": "It checks every pair of numbers and returns True if any pair is closer than the threshold.",
    },
    {
        "prompt": "Explain what this Python function does:\n\ndef trailing_zeroes_in_factorial(num):\n    count = 0\n    while num >= 5:\n        num //= 5\n        count += num\n    return count",
        "answer": "It counts how many trailing zeroes appear in num factorial by counting factors of 5.",
    },
    {
        "prompt": "Explain this JavaScript function:\n\nfunction add(a, b) {\n  return a + b;\n}",
        "answer": "It returns the sum of a and b.",
    },
    {
        "prompt": "Explain this JavaScript function:\n\nfunction isEmpty(arr) {\n  return arr.length === 0;\n}",
        "answer": "It returns true when the array has no elements.",
    },
    {
        "prompt": "Explain this JavaScript function:\n\nfunction getName(user) {\n  return user.name;\n}",
        "answer": "It returns the name property from the user object.",
    },
    {
        "prompt": "Explain this TypeScript reducer case:\n\ncase 'CUSTOMER_CLEAR_INFO':\n  return { ...state, info: {} };",
        "answer": "It returns a new state object with the info field reset to an empty object.",
    },
    {
        "prompt": "Explain this Rust function:\n\nfn double(x: i32) -> i32 {\n    x * 2\n}",
        "answer": "It returns x multiplied by 2.",
    },
    {
        "prompt": "Explain this Rust expression:\n\nnumbers.iter().filter(|n| **n > 0).count()",
        "answer": "It iterates over numbers, keeps only positive values, and counts how many there are.",
    },
    {
        "prompt": "Explain this SQL query:\n\nSELECT COUNT(*) FROM users WHERE active = true;",
        "answer": "It counts the number of active users.",
    },
    {
        "prompt": "Explain this Bash command:\n\ncp source.txt backup.txt",
        "answer": "It copies source.txt to a new file named backup.txt.",
    },
    {
        "prompt": "Explain this Python class method:\n\ndef available_seats(self):\n    return self.capacity - self.reserved",
        "answer": "It returns the number of seats that are still available by subtracting reserved seats from capacity.",
    },
    {
        "prompt": "Explain what this Python code does:\n\nresult = [x * x for x in numbers]",
        "answer": "It creates a new list containing the square of each value in numbers.",
    },
    {
        "prompt": "Explain what this Python code does:\n\nwith open(path) as f:\n    lines = f.readlines()",
        "answer": "It opens the file at path and reads all lines into a list.",
    },
    {
        "prompt": "Explain this Python conditional:\n\nif not items:\n    return []",
        "answer": "It checks whether items is empty and returns an empty list when it is.",
    },
    {
        "prompt": "Explain this Python loop:\n\nfor item in items:\n    print(item)",
        "answer": "It iterates through each item in items and prints it.",
    },
    {
        "prompt": "Explain this Python dictionary update:\n\ncounts[word] = counts.get(word, 0) + 1",
        "answer": "It increments the count for word, starting from 0 if the word is not already present.",
    },
    {
        "prompt": "Explain this Python statement:\n\nreturn sorted(items, key=lambda item: item.name)",
        "answer": "It returns items sorted by each item's name attribute.",
    },
    {
        "prompt": "Explain this C# property:\n\npublic string Name { get; set; }",
        "answer": "It declares a public string property named Name with automatic getter and setter methods.",
    },
    {
        "prompt": "Explain this C# condition:\n\nif (items.Count == 0) return false;",
        "answer": "It returns false when the items collection is empty.",
    },
    {
        "prompt": "Explain this Java method:\n\npublic int size() {\n    return items.size();\n}",
        "answer": "It returns the number of elements in the items collection.",
    },
    {
        "prompt": "Explain this JavaScript arrow function:\n\nconst double = x => x * 2;",
        "answer": "It defines a function that returns its input multiplied by 2.",
    },
    {
        "prompt": "Explain this Python exception handler:\n\ntry:\n    value = int(text)\nexcept ValueError:\n    value = 0",
        "answer": "It tries to convert text to an integer and falls back to 0 if conversion fails.",
    },
    {
        "prompt": "Explain this Python function:\n\ndef merge_dicts(a, b):\n    return {**a, **b}",
        "answer": "It returns a new dictionary containing keys from a and b, with b overriding duplicate keys.",
    },
    {
        "prompt": "Explain this Python function:\n\ndef get_extension(filename):\n    return filename.rsplit('.', 1)[-1]",
        "answer": "It returns the part of filename after the final dot.",
    },
    {
        "prompt": "Explain this Python function:\n\ndef fahrenheit_to_celsius(f):\n    return (f - 32) * 5 / 9",
        "answer": "It converts a Fahrenheit temperature to Celsius.",
    },
    {
        "prompt": "Explain this Python function:\n\ndef non_empty(strings):\n    return [s for s in strings if s]",
        "answer": "It returns only the strings that are not empty.",
    },
    {
        "prompt": "Explain this Python function:\n\ndef word_count(text):\n    return len(text.split())",
        "answer": "It returns the number of whitespace-separated words in text.",
    },
]


def build_handcrafted_code_explanation_records() -> list[dict]:
    """Return small prose-only examples for code-explanation behavior."""
    records = []

    for example in HANDCRAFTED_CODE_EXPLANATIONS:
        records.append({
            "conversations": [
                {"role": "system", "content": CODE_SYSTEM},
                {"role": "user", "content": example["prompt"].strip()},
                {"role": "assistant", "content": example["answer"].strip()},
            ],
            "source": "handcrafted_code_explanation",
            "sft_type": "code_explanation",
            "normalized": False,
        })

    return records


# ── Chat SFT — OpenHermes-2.5 ─────────────────────────────────────────────────

def prepare_chat(val_fraction: float) -> None:
    """
    Download and format OpenHermes-2.5 for chat SFT.

    OpenHermes-2.5 is a high-quality synthetic instruction dataset
    generated by GPT-4. Contains diverse instruction-following examples
    across coding, reasoning, creative writing, and general knowledge.

    Format: each example has a "conversations" field with role/value pairs.
    """
    from datasets import load_dataset

    out_dir    = SFT_DIR / "chat"
    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"

    if train_path.exists() and val_path.exists():
        log.info(f"Chat SFT data already exists at {out_dir}")
        return

    log.info("Loading OpenHermes-2.5...")
    dataset = load_dataset("teknium/OpenHermes-2.5", split="train")
    log.info(f"OpenHermes-2.5: {len(dataset):,} examples")

    records = []
    skipped = 0

    for example in dataset:
        conversations = example.get("conversations") or []
        if not conversations:
            skipped += 1
            continue

        messages = []

        # Add system prompt — use `or ""` to handle None values
        system = (example.get("system_prompt") or "").strip()
        messages.append({
            "role": "system",
            "content": system if system else DEFAULT_SYSTEM,
        })

        valid = True
        for turn in conversations:
            # Use `or ""` to handle None values in role and content fields
            role    = (turn.get("from") or turn.get("role") or "").lower()
            content = (turn.get("value") or turn.get("content") or "").strip()

            if not content:
                valid = False
                break

            if role in ("human", "user"):
                messages.append({"role": "user", "content": content})
            elif role in ("gpt", "assistant"):
                messages.append({"role": "assistant", "content": content})

        if not valid or len(messages) < 2:
            skipped += 1
            continue

        # Must end with assistant turn
        if messages[-1]["role"] != "assistant":
            skipped += 1
            continue

        records.append({
            "conversations": messages,
            "source": "openhermes",
            "sft_type": "general_assistant",
        })

    handcrafted_records = build_handcrafted_response_control_records()
    records.extend(handcrafted_records)
    log.info(f"Added generated response-control chat examples: {len(handcrafted_records):,}")
    log.info(f"Processed: {len(records):,} kept, {skipped:,} skipped")

    # Split
    n_val = max(1000, int(len(records) * val_fraction))
    random.seed(42)
    random.shuffle(records)
    val_records   = records[:n_val]
    train_records = records[n_val:]

    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    log.info(f"Chat SFT: {len(train_records):,} train, {len(val_records):,} val")


# ── Code SFT — Magicoder-OSS-Instruct ─────────────────────────────────────────

def prepare_code(val_fraction: float) -> None:
    """
    Download and format Magicoder-OSS-Instruct-75K for code SFT.

    Magicoder generates coding problems inspired by real open-source code,
    then generates solutions. Each example is a single-turn instruction/response
    pair.

    This stage keeps only examples whose assistant response contains real code.
    Obvious prose-only / explanation-only examples are dropped. For code-output
    task types, fenced code is normalized to code-only targets so code SFT
    teaches the model to produce code directly when code is requested.
    """
    from datasets import load_dataset

    out_dir    = SFT_DIR / "code"
    train_path = out_dir / "train.jsonl"
    val_path   = out_dir / "val.jsonl"

    if train_path.exists() and val_path.exists():
        log.info(f"Code SFT data already exists at {out_dir}")
        return

    log.info("Loading Magicoder-OSS-Instruct-75K...")
    dataset = load_dataset("ise-uiuc/Magicoder-OSS-Instruct-75K", split="train")
    log.info(f"Magicoder: {len(dataset):,} examples")

    records = []
    skipped_reasons = Counter()
    type_counts = Counter()
    normalized_count = 0

    for example in dataset:
        # Use `or ""` to handle None values
        instruction = (example.get("problem") or "").strip()
        solution    = (example.get("solution") or "").strip()

        if not instruction:
            skipped_reasons["missing_instruction"] += 1
            continue

        if not solution:
            skipped_reasons["missing_solution"] += 1
            continue

        sft_type = classify_code_sft_type(instruction, solution)

        if sft_type == "code_explanation":
            if looks_like_mostly_code(solution):
                skipped_reasons["code_explanation_is_code"] += 1
                continue
        else:
            if is_prose_heavy_without_code(solution):
                skipped_reasons["prose_only"] += 1
                continue

            if not looks_like_code(solution):
                skipped_reasons["no_code_detected"] += 1
                continue

        normalized_solution, normalized = normalize_code_solution(solution, sft_type)

        if not normalized_solution:
            skipped_reasons["empty_after_normalization"] += 1
            continue

        if normalized:
            normalized_count += 1

        type_counts[sft_type] += 1

        messages = [
            {"role": "system",    "content": CODE_SYSTEM},
            {"role": "user",      "content": instruction},
            {"role": "assistant", "content": normalized_solution},
        ]

        records.append({
            "conversations": messages,
            "source": "magicoder",
            "sft_type": sft_type,
            "normalized": normalized,
        })

    handcrafted_records = build_handcrafted_function_completion_records()
    records.extend(handcrafted_records)
    type_counts["function_completion"] += len(handcrafted_records)
    log.info(
        f"Added handcrafted function-completion examples: "
        f"{len(handcrafted_records):,}"
    )

    handcrafted_explanation_base = build_handcrafted_code_explanation_records()
    handcrafted_explanation_records = (
        handcrafted_explanation_base * HANDCRAFTED_CODE_EXPLANATION_REPEAT
    )
    records.extend(handcrafted_explanation_records)
    type_counts["code_explanation"] += len(handcrafted_explanation_records)
    log.info(
        f"Added handcrafted code-explanation examples: "
        f"{len(handcrafted_explanation_records):,} "
        f"({len(handcrafted_explanation_base):,} unique × "
        f"{HANDCRAFTED_CODE_EXPLANATION_REPEAT})"
    )

    skipped = sum(skipped_reasons.values())
    log.info(f"Processed: {len(records):,} kept, {skipped:,} skipped")
    log.info(f"Normalized code-only outputs: {normalized_count:,}")
    if skipped_reasons:
        log.info("Skipped by reason:")
        for reason, count in skipped_reasons.most_common():
            log.info(f"  {reason}: {count:,}")
    if type_counts:
        log.info("Kept by sft_type:")
        for sft_type, count in type_counts.most_common():
            log.info(f"  {sft_type}: {count:,}")

    if not records:
        raise RuntimeError(
            "Code SFT filtering removed all examples. Relax looks_like_code() "
            "or inspect Magicoder schema changes."
        )

    # Split
    n_val = max(500, int(len(records) * val_fraction))
    n_val = min(n_val, max(1, len(records) - 1))
    random.seed(42)
    random.shuffle(records)
    val_records   = records[:n_val]
    train_records = records[n_val:]

    write_jsonl(train_records, train_path)
    write_jsonl(val_records, val_path)

    log.info(f"Code SFT: {len(train_records):,} train, {len(val_records):,} val")

# ── CLI ────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Prepare SFT datasets")
    parser.add_argument(
        "--stage",
        choices=["chat", "code", "both"],
        default="both",
        help="Which SFT stage to prepare",
    )
    parser.add_argument(
        "--val-fraction",
        type=float,
        default=None,
        help=(
            "Override validation fraction. If unset, uses stage-specific "
            f"defaults: {DEFAULT_VAL_FRACTION}"
        ),
    )
    args = parser.parse_args()

    if args.stage in ("chat", "both"):
        log.info("=== Preparing Chat SFT data (OpenHermes-2.5) ===")
        frac = args.val_fraction if args.val_fraction is not None else DEFAULT_VAL_FRACTION["chat"]
        prepare_chat(val_fraction=frac)

    if args.stage in ("code", "both"):
        log.info("=== Preparing Code SFT data (Magicoder-OSS-Instruct) ===")
        frac = args.val_fraction if args.val_fraction is not None else DEFAULT_VAL_FRACTION["code"]
        prepare_code(val_fraction=frac)

    log.info("SFT data preparation complete.")
    log.info(f"Output: {SFT_DIR}")


if __name__ == "__main__":
    main()
