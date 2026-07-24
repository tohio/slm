import ast
from pathlib import Path

import yaml

from config import ALL_SOURCES, DEDUP_PRIORITY, consumed_tokens


def test_dedup_priority_contains_every_source_exactly_once():
    assert len(DEDUP_PRIORITY) == len(set(DEDUP_PRIORITY))
    assert set(DEDUP_PRIORITY) == set(ALL_SOURCES)


def test_static_pretrain_configs_match_consumed_token_contract():
    for size in ("125m", "350m", "1b"):
        path = Path("pretrain/configs") / f"gpt_{size}.yaml"
        config = yaml.safe_load(path.read_text())
        training = config["training"]
        model = config["model"]
        tokens_per_step = (
            training["micro_batch_size"]
            * training["gradient_accumulation_steps"]
            * model["max_position_embeddings"]
        )
        assert training["max_steps"] == consumed_tokens(size) // tokens_per_step


def test_curator_build_source_dispatch_matches_configured_sources():
    module = ast.parse(Path("curator/scripts/curate.py").read_text())
    function = next(
        node
        for node in module.body
        if isinstance(node, ast.FunctionDef) and node.name == "_build_source"
    )
    dispatched = [
        node.comparators[0].value
        for node in ast.walk(function)
        if (
            isinstance(node, ast.Compare)
            and isinstance(node.left, ast.Name)
            and node.left.id == "name"
            and len(node.ops) == 1
            and isinstance(node.ops[0], ast.Eq)
            and len(node.comparators) == 1
            and isinstance(node.comparators[0], ast.Constant)
            and isinstance(node.comparators[0].value, str)
        )
    ]
    assert len(dispatched) == len(set(dispatched))
    assert set(dispatched) == set(ALL_SOURCES)
