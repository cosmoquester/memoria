from typing import Dict, List, Optional

import datasets
import torch
from transformers import AutoTokenizer


def text_to_tokens(
    dataset: datasets.Dataset,
    tokenizer: AutoTokenizer,
    batch_size: int,
    max_length: int,
    batch_size_per_device: Optional[int] = None,
) -> datasets.Dataset:
    """Tokenize a series of text into tokens and chunk
    the processed datasets will be cached automatically.

    Args:
        dataset: huggingface dataset containing "text" field
        tokenizer: huggingface tokenizer
        batch_size: batch size, in same batch, there's sequential dataset.
        max_length: max length of each example. the remainder will be dropped.
        batch_size_per_device: batch size per device with using DDP.
    Return:
        huggingface input dictionary.
        the values shaped [NumExamples, BatchSize, MaxLength]
    """

    def _tokenize(example):
        return tokenizer(example["text"])

    token_dataset = dataset.map(_tokenize, remove_columns=dataset.column_names, load_from_cache_file=True)

    def _segment(example):
        num_segments = len(example["input_ids"]) // max_length
        return {
            "data": [
                {k: v[i * max_length : (i + 1) * max_length] for k, v in example.items()} for i in range(num_segments)
            ],
            "is_end": [False] * (num_segments - 1) + [True] if num_segments else [],
        }

    segment_dataset = token_dataset.map(_segment, remove_columns=token_dataset.column_names, load_from_cache_file=True)

    def _merge(examples):
        data = examples["data"]
        is_ends = examples["is_end"]
        merged = {k: [example[k] for datum in data for example in datum] for k in data[0][0].keys()}
        merged["is_end"] = [v for is_end in is_ends for v in is_end]
        return merged

    merge_dataset = segment_dataset.map(
        _merge,
        remove_columns=segment_dataset.column_names,
        load_from_cache_file=True,
        batched=True,
        batch_size=len(segment_dataset),
    )

    num_examples = len(merge_dataset) // batch_size

    def _batching(example):
        return {
            k: [v[i : num_examples * batch_size : num_examples] for i in range(num_examples)]
            for k, v in example.items()
        }

    batch_dataset = merge_dataset.map(_batching, load_from_cache_file=True, batched=True, batch_size=len(merge_dataset))

    def _rebatching_for_multi_device(example):
        return {
            k: [v[0][i : i + batch_size_per_device] for i in range(0, batch_size, batch_size_per_device)]
            for k, v in example.items()
        }

    if batch_size_per_device is not None and batch_size != batch_size_per_device:
        batch_dataset = batch_dataset.map(
            _rebatching_for_multi_device, load_from_cache_file=True, batched=True, batch_size=1
        )
    batch_dataset.set_format(type="torch", columns=batch_dataset.column_names)
    return batch_dataset


class LanguageModelingDataset(torch.utils.data.Dataset):
    def __init__(self, data: datasets.Dataset) -> None:
        super().__init__()

        self.data = {k: data[k] for k in data.column_names}

    def __len__(self) -> int:
        return len(self.data["input_ids"])

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        # [BatchSize, MaxLength]
        inputs = {k: v[index] for k, v in self.data.items()}
        inputs["labels"] = inputs["input_ids"]
        return inputs

    @staticmethod
    def collate_fn(batches: list[dict[str, torch.Tensor]]) -> dict[str, torch.Tensor]:
        """Select first item becuase batch size is 1"""
        assert len(batches) == 1
        return batches[0]
