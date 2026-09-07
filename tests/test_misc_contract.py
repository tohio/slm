"""Cross-cutting path and inference contract tests."""

from pathlib import Path
from types import SimpleNamespace

import torch

from config.paths import BASE_EXPORTS_DIR, export_dir
from inference.chat import _prepare_chat_input_ids


ROOT = Path(__file__).resolve().parents[1]


class _ChatTokenizer:
    def __init__(self):
        self.truncation_side = "right"
        self.observed = None

    def apply_chat_template(self, messages, **kwargs):
        self.observed = {
            "messages": messages,
            "truncation_side": self.truncation_side,
            **kwargs,
        }
        return torch.tensor([[1, 2, 3]])


def test_export_dir_uses_the_export_root():
    assert export_dir("125m") == BASE_EXPORTS_DIR / "125m"


def test_chat_reserves_generation_budget_and_restores_tokenizer_state():
    model = SimpleNamespace(
        config=SimpleNamespace(max_position_embeddings=4096),
        device=torch.device("cpu"),
    )
    tokenizer = _ChatTokenizer()
    messages = [{"role": "user", "content": "hello"}]

    input_ids = _prepare_chat_input_ids(
        model,
        tokenizer,
        messages,
        max_new_tokens=512,
    )

    assert input_ids.tolist() == [[1, 2, 3]]
    assert tokenizer.observed["max_length"] == 3584
    assert tokenizer.observed["truncation_side"] == "left"
    assert tokenizer.truncation_side == "right"


def test_make_recipes_do_not_bypass_configured_results_root():
    recipe_lines = [
        line
        for line in (ROOT / "Makefile").read_text(encoding="utf-8").splitlines()
        if line.startswith("\t")
    ]
    assert not any("results/runs/" in line for line in recipe_lines)


def test_training_launcher_preserves_requested_multi_gpu_process_count():
    makefile = (ROOT / "Makefile").read_text(encoding="utf-8")

    assert (
        "ACCELERATE = $(_ACCELERATE) launch --config_file \"$(ACCELERATE_CONFIG)\" --num_processes $(GPUS) "
        "--num_machines 1 --mixed_precision bf16 --dynamo_backend no"
        in makefile
    )


def test_tool_turn_is_bounded_and_preserves_no_tool_chat():
    import pytest
    from config.chat import TOOL_OPEN, TOOL_CLOSE, safe_json
    from inference.tools import run_tool_turn

    history = [{"role": "user", "content": "Search for an example"}]
    call = TOOL_OPEN + safe_json({"name": "web_search", "arguments": {"query": "example"}}) + TOOL_CLOSE
    generated = iter([call, "The search had no results."])
    queries = []
    def search(query):
        queries.append(query)
        return []
    messages = run_tool_turn(lambda _: next(generated), history, search)
    assert [m["role"] for m in messages] == ["assistant", "tool", "assistant"]
    assert queries == ["example"] and len(history) == 1
    assert run_tool_turn(lambda _: "hello", history, search) == [{"role": "assistant", "content": "hello"}]
    assert queries == ["example"]
    with pytest.raises(ValueError, match="budget"):
        run_tool_turn(lambda _: call, history, search)
    bad = TOOL_OPEN + safe_json({"name": "execute", "arguments": {"query": "x"}}) + TOOL_CLOSE
    with pytest.raises(ValueError, match="Only web_search"):
        run_tool_turn(lambda _: bad, history, search)
