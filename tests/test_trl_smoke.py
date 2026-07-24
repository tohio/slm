"""One-step synthetic SFT/DPO integration tests for the pinned TRL stack."""

from copy import deepcopy

import torch
from datasets import Dataset
from tokenizers import Tokenizer
from tokenizers.models import WordLevel
from tokenizers.pre_tokenizers import Whitespace
from transformers import PreTrainedTokenizerFast
from trl import DPOConfig, DPOTrainer, SFTConfig, SFTTrainer

from model import SLMConfig, SLMForCausalLM


def _tokenizer() -> PreTrainedTokenizerFast:
    vocab = {
        "<PAD>": 0,
        "<BOS>": 1,
        "<EOS>": 2,
        "<UNK>": 3,
        "<|system|>": 4,
        "<|user|>": 5,
        "<|assistant|>": 6,
        "<|endofturn|>": 7,
        "You": 8,
        "are": 9,
        "helpful": 10,
        "Say": 11,
        "hello": 12,
        "Hello": 13,
        "world": 14,
        "Good": 15,
        "Bad": 16,
    }
    backend = Tokenizer(WordLevel(vocab, unk_token="<UNK>"))
    backend.pre_tokenizer = Whitespace()
    tokenizer = PreTrainedTokenizerFast(
        tokenizer_object=backend,
        pad_token="<PAD>",
        bos_token="<BOS>",
        eos_token="<EOS>",
        unk_token="<UNK>",
    )
    tokenizer.chat_template = (
        "{{ bos_token }}"
        "{% for message in messages %}"
        "{% if message['role'] == 'system' %}"
        "<|system|> {{ message['content'] }} <|endofturn|>"
        "{% elif message['role'] == 'user' %}"
        "<|user|> {{ message['content'] }} <|endofturn|>"
        "{% elif message['role'] == 'assistant' %}"
        "<|assistant|> {% generation %}{{ message['content'] }}"
        "{{ eos_token }}{% endgeneration %} <|endofturn|>"
        "{% endif %}"
        "{% endfor %}"
        "{% if add_generation_prompt %}<|assistant|>{% endif %}"
    )
    return tokenizer


def _model() -> SLMForCausalLM:
    config = SLMConfig(
        vocab_size=17,
        hidden_size=32,
        intermediate_size=64,
        num_hidden_layers=1,
        num_attention_heads=4,
        num_key_value_heads=2,
        max_position_embeddings=64,
        pad_token_id=0,
        bos_token_id=1,
        eos_token_id=2,
    )
    return SLMForCausalLM(config)


def test_one_step_sft_and_dpo(tmp_path):
    tokenizer = _tokenizer()
    sft_rows = [
        {
            "messages": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Say hello"},
                {"role": "assistant", "content": "Hello world"},
            ]
        }
    ] * 2
    sft_args = SFTConfig(
        output_dir=str(tmp_path / "sft"),
        max_steps=1,
        per_device_train_batch_size=1,
        optim="adamw_torch",
        report_to="none",
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="no",
        max_length=64,
        assistant_only_loss=True,
        gradient_checkpointing=False,
        use_cpu=True,
        disable_tqdm=True,
    )
    sft_trainer = SFTTrainer(
        model=_model(),
        args=sft_args,
        train_dataset=Dataset.from_list(sft_rows),
        processing_class=tokenizer,
    )
    labels = sft_trainer.train_dataset[0]["labels"]
    assert -100 in labels
    assert any(token != -100 for token in labels)
    sft_result = sft_trainer.train()
    assert torch.isfinite(torch.tensor(sft_result.training_loss))

    policy = _model()
    dpo_rows = [
        {
            "prompt": [
                {"role": "system", "content": "You are helpful"},
                {"role": "user", "content": "Say hello"},
            ],
            "chosen": [{"role": "assistant", "content": "Good hello"}],
            "rejected": [{"role": "assistant", "content": "Bad hello"}],
        }
    ] * 2
    dpo_args = DPOConfig(
        output_dir=str(tmp_path / "dpo"),
        max_steps=1,
        per_device_train_batch_size=1,
        optim="adamw_torch",
        report_to="none",
        logging_strategy="no",
        save_strategy="no",
        eval_strategy="no",
        max_length=64,
        gradient_checkpointing=False,
        use_cpu=True,
        disable_tqdm=True,
    )
    dpo_result = DPOTrainer(
        model=policy,
        ref_model=deepcopy(policy),
        args=dpo_args,
        train_dataset=Dataset.from_list(dpo_rows),
        processing_class=tokenizer,
    ).train()
    assert torch.isfinite(torch.tensor(dpo_result.training_loss))
