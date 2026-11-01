from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import torch
from torch.utils.data import Dataset


class MetaMathSFTDataset(Dataset):
    """Chat-formatted MetaMathQA examples with assistant-only labels."""

    def __init__(
        self,
        jsonl_path: str | Path,
        tokenizer,
        max_seq_len: int = 2048,
        enable_thinking: bool = False,
    ):
        self.tokenizer = tokenizer
        self.max_seq_len = max_seq_len
        self.enable_thinking = enable_thinking
        self.examples: list[dict[str, Any]] = []
        with open(jsonl_path, "r", encoding="utf-8") as f:
            for line in f:
                self.examples.append(json.loads(line))

    def __len__(self) -> int:
        return len(self.examples)

    def _apply_template(self, messages: list[dict[str, str]], add_generation_prompt: bool) -> list[int]:
        kwargs = {
            "tokenize": True,
            "add_generation_prompt": add_generation_prompt,
            "return_dict": False,
        }
        try:
            return self.tokenizer.apply_chat_template(
                messages,
                enable_thinking=self.enable_thinking,
                **kwargs,
            )
        except TypeError:
            return self.tokenizer.apply_chat_template(messages, **kwargs)

    def __getitem__(self, idx: int) -> dict[str, torch.Tensor]:
        ex = self.examples[idx]
        question = ex["query"]
        response = ex["response"]

        prompt_messages = [{"role": "user", "content": question}]
        full_messages = [
            {"role": "user", "content": question},
            {"role": "assistant", "content": response},
        ]

        prompt_ids = self._apply_template(prompt_messages, add_generation_prompt=True)
        full_ids = self._apply_template(full_messages, add_generation_prompt=False)

        if len(full_ids) > self.max_seq_len:
            # Should not happen after filtering; hard-fail rather than silently truncate answers.
            raise ValueError(
                f"Example {idx} length {len(full_ids)} exceeds max_seq_len {self.max_seq_len}"
            )

        labels = list(full_ids)
        prompt_len = min(len(prompt_ids), len(full_ids))
        for i in range(prompt_len):
            labels[i] = -100

        attention_mask = [1] * len(full_ids)
        return {
            "input_ids": torch.tensor(full_ids, dtype=torch.long),
            "attention_mask": torch.tensor(attention_mask, dtype=torch.long),
            "labels": torch.tensor(labels, dtype=torch.long),
            "source_index": torch.tensor(ex["source_index"], dtype=torch.long),
            "num_response_tokens": torch.tensor(
                sum(1 for x in labels if x != -100), dtype=torch.long
            ),
        }


def sft_collate_fn(features: list[dict[str, torch.Tensor]], pad_token_id: int) -> dict[str, torch.Tensor]:
    max_len = max(f["input_ids"].size(0) for f in features)
    batch_size = len(features)

    input_ids = torch.full((batch_size, max_len), pad_token_id, dtype=torch.long)
    attention_mask = torch.zeros((batch_size, max_len), dtype=torch.long)
    labels = torch.full((batch_size, max_len), -100, dtype=torch.long)
    source_index = torch.zeros(batch_size, dtype=torch.long)
    num_response_tokens = torch.zeros(batch_size, dtype=torch.long)

    for i, f in enumerate(features):
        n = f["input_ids"].size(0)
        input_ids[i, :n] = f["input_ids"]
        attention_mask[i, :n] = f["attention_mask"]
        labels[i, :n] = f["labels"]
        source_index[i] = f["source_index"]
        num_response_tokens[i] = f["num_response_tokens"]

    return {
        "input_ids": input_ids,
        "attention_mask": attention_mask,
        "labels": labels,
        "source_index": source_index,
        "num_response_tokens": num_response_tokens,
    }
