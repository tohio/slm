"""Focused tests for pretraining document-boundary tokenization."""

import json

import numpy as np
import pytest
from tokenizers import Tokenizer, decoders, models, pre_tokenizers, trainers

from pretrain.data import tokenize_data


SPECIAL_TOKENS = ["<PAD>", "<UNK>", "<BOS>", "<EOS>", "<|system|>"]


def _write_test_tokenizer(path):
    tokenizer = Tokenizer(models.BPE(unk_token="<UNK>"))
    tokenizer.pre_tokenizer = pre_tokenizers.ByteLevel(add_prefix_space=False)
    tokenizer.decoder = decoders.ByteLevel()
    tokenizer.train_from_iterator(
        [
            "ordinary text with literal angle brackets and marker names",
            "BOS EOS PAD UNK system",
        ],
        trainers.BpeTrainer(
            vocab_size=300,
            special_tokens=SPECIAL_TOKENS,
            initial_alphabet=pre_tokenizers.ByteLevel.alphabet(),
        ),
    )
    tokenizer.save(str(path))
    return tokenizer


def test_tokenize_chunk_keeps_reserved_literals_out_of_structural_ids(tmp_path):
    tokenizer_path = tmp_path / "tokenizer.json"
    tokenizer = _write_test_tokenizer(tokenizer_path)
    bos_id = tokenizer.token_to_id("<BOS>")
    eos_id = tokenizer.token_to_id("<EOS>")
    system_id = tokenizer.token_to_id("<|system|>")
    text = "This mentions <BOS>, <EOS>, and <|system|> as ordinary text."

    tokenize_data._worker_init(str(tokenizer_path), bos_id, eos_id)
    tokens, n_docs, source_counts = tokenize_data._tokenize_chunk(
        [(text, "wikipedia")]
    )

    assert n_docs == 1
    assert tokens[0] == bos_id
    assert tokens[-1] == eos_id
    assert tokens.count(bos_id) == 1
    assert tokens.count(eos_id) == 1
    assert system_id not in tokens[1:-1]
    assert source_counts == {
        "wikipedia": {"documents": 1, "tokens": len(tokens)}
    }
    assert tokenize_data._worker_tokenizer.decode(
        tokens[1:-1],
        skip_special_tokens=False,
    ) == text


def test_verify_dataset_rejects_embedded_structural_token_ids(tmp_path):
    bin_path = tmp_path / "train.bin"
    meta_path = tmp_path / "train.json"
    np.array([2, 10, 3, 3], dtype=np.uint16).tofile(bin_path)
    meta_path.write_text(
        json.dumps({
            "n_tokens": 4,
            "n_docs": 1,
            "bos_id": 2,
            "eos_id": 3,
            "format_version": tokenize_data.TOKENIZED_FORMAT_VERSION,
        })
    )

    with pytest.raises(RuntimeError, match="EOS count mismatch"):
        tokenize_data.verify_dataset(bin_path, meta_path)
