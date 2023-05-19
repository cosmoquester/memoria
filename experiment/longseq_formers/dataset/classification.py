from typing import Dict, List, TypedDict

import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer


class ClassificationDatum(TypedDict):
    text: str
    label: int


class ClassificationDataset(torch.utils.data.Dataset):
    """ClassificationDataset

    Attributes:
        data: data for text classification
        tokenizer: huggingface tokenizer
        max_length: token max length
    """

    def __init__(
        self, data: List[ClassificationDatum], tokenizer: AutoTokenizer, max_length: int, truncation: bool = True
    ) -> None:
        super().__init__()

        self.data = data
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.truncation = truncation

    def __len__(self) -> int:
        return len(self.data)

    def __getitem__(self, index: int) -> Dict[str, torch.Tensor]:
        item = self.data[index]
        text = item["text"]
        label = item["label"]

        inputs = self.tokenizer(
            text,
            add_special_tokens=True,
            max_length=self.max_length,
            truncation=self.truncation,
            padding="max_length",
            return_token_type_ids=True,
            return_tensors="pt",
        )
        inputs = {k: v.squeeze(dim=0) for k, v in inputs.items()}
        inputs["labels"] = torch.tensor(label)

        return inputs

    @staticmethod
    def pad_collate_fn(batch: List[Dict[str, torch.Tensor]]) -> Dict[str, torch.Tensor]:
        # [NumTimeSteps, BatchSize, MaxSequenceLength]
        padded_batch = {k: [item[k] for item in batch] for k in batch[0].keys()}
        for k in padded_batch:
            if k == "labels":
                padded_batch[k] = torch.stack(padded_batch[k], dim=0)
            else:
                padded_batch[k] = pad_sequence(padded_batch[k], batch_first=True)

        return padded_batch
