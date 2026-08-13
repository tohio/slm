import ast
from pathlib import Path

import yaml

from config import (
    ALL_SOURCES,
    DEDUP_PRIORITY,
    FILTER_SOURCE_FAMILIES,
    consumed_tokens,
    source_filter_family,
)


def test_dedup_priority_contains_every_source_exactly_once():
    assert len(DEDUP_PRIORITY) == len(set(DEDUP_PRIORITY))
    assert set(DEDUP_PRIORITY) == set(ALL_SOURCES)


def test_filter_source_families_contain_every_source_exactly_once():
    routed = [
        source
        for sources in FILTER_SOURCE_FAMILIES.values()
        for source in sources
    ]
    assert len(routed) == len(set(routed))
    assert set(routed) == set(ALL_SOURCES)
    assert {
        source: source_filter_family(source) for source in ALL_SOURCES
    } == {
        source: family
        for family, sources in FILTER_SOURCE_FAMILIES.items()
        for source in sources
    }


def test_static_pretrain_configs_match_consumed_token_contract():
    for size in ("125m", "350m", "1b"):
        path = Path("pretrain/configs") / f"gpt_{size}.yaml"
        config = yaml.safe_load(path.read_text())
        training = config["training"]
        model = config["model"]
        assert training["schedule_from_realized_tokens"] is True
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
