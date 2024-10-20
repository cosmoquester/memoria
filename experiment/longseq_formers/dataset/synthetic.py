import json
from typing import Dict, List, Tuple

import torch


def parse_syntetic_data(
    path: str,
) -> Tuple[int, int, list[dict[str, list[int]]], list[dict[str, list[int]]], list[dict[str, list[int]]]]:
    with open(path, "r") as f:
        data = json.load(f)

    prompt_length = data["prompt_length"]
    vocab_size = data["vocab_size"]
    train_examples = data["train"]
    dev_examples = data["dev"]
    test_examples = data["test"]

    return prompt_length, vocab_size, train_examples, dev_examples, test_examples


class SyntheticDataset(torch.utils.data.Dataset):
    def __init__(self, examples: list[dict[str, list[int]]]) -> None:
        super().__init__()

        self.data = examples

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        example = self.data[index]
        input_ids = example["prompt_ids"] + example["target_ids"][:-1]
        labels = [-100] * (len(example["prompt_ids"]) - 1) + example["target_ids"]
        return {"input_ids": torch.tensor(input_ids), "labels": torch.tensor(labels)}
