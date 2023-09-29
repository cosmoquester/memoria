import argparse
from functools import partial
from typing import Dict

import pytorch_lightning as pl
import torch
from longseq_formers.data import (
    CLASSIFICATION_DATASETS,
    load_20news_data,
    load_ecthr_data,
    load_hyperpartisan_data,
    load_mimic3_data,
)
from longseq_formers.dataset import ClassificationDataset, MultiLabelClassificationDataset
from longseq_formers.task import Classification, MultiLabelClassification
from longseq_formers.utils import get_logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# fmt: off
parser = argparse.ArgumentParser(prog="train_classification", description="Train & Test Long Sequence Classification")

g = parser.add_argument_group("Train Parameter")
g.add_argument("--model", type=str, required=True, help="lightning checkpoint")
g.add_argument("--tokenizer", type=str, required=True, help="huggingface tokenizer")
g.add_argument("--dataset", type=str, default="hyperpartisan", choices=CLASSIFICATION_DATASETS, help="dataset name")
g.add_argument("--valid-batch-size", type=int, default=1, help="validation batch size")
g.add_argument("--max-length", type=int, default=512, help="max sequence length")
g.add_argument("--memory-length", type=int, default=512, help="max sequence length for bert one inference on infinity former")
g.add_argument("--seed", type=int, default=42, help="random seed")
g.add_argument("--not-truncate", action="store_false", dest="truncation", help="not truncate sequence")
g.add_argument("--segment-size", type=int, help="segment size for infinity former")
# fmt: on


def main(args: argparse.Namespace) -> Dict[str, float]:
    logger = get_logger("evaluate_classification")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    pl.seed_everything(args.seed, workers=True)

    logger.info(f'[+] Load Tokenizer: "{args.tokenizer}"')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info(f'[+] Use Dataset: "{args.dataset}"')
    if args.dataset in ("ecthr", "mimic3"):
        if args.dataset == "ecthr":
            datasets, label_to_index = load_ecthr_data()
        elif args.dataset == "mimic3":
            datasets, label_to_index = load_mimic3_data()
        num_labels = len(label_to_index)
        dataset_cls = partial(
            MultiLabelClassificationDataset,
            tokenizer=tokenizer,
            max_length=args.max_length,
            num_labels=num_labels,
            truncation=args.truncation,
        )
        task_cls = MultiLabelClassification
    else:
        dataset_cls = partial(
            ClassificationDataset, tokenizer=tokenizer, max_length=args.max_length, truncation=args.truncation
        )
        if args.dataset == "hyperpartisan":
            datasets = load_hyperpartisan_data()
        if args.dataset == "20news":
            datasets = load_20news_data()
        task_cls = Classification

    valid_dataset = dataset_cls(datasets["dev"])
    test_dataset = dataset_cls(datasets["test"])

    logger.info(f"[+] # of valid examples: {len(valid_dataset)}")
    logger.info(f"[+] # of test examples: {len(test_dataset)}")

    logger.info(f'[+] Load Model: "{args.model}"')
    classification = task_cls.load_from_checkpoint(args.model)

    collate_fn = ClassificationDataset.pad_collate_fn if not args.truncation else None
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size, collate_fn=collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=args.valid_batch_size, collate_fn=collate_fn)

    tester = pl.Trainer(accelerator="gpu" if torch.cuda.device_count() else None, devices=1)

    pl.seed_everything(args.seed, workers=True)
    result1 = tester.test(classification, valid_dataloader)[0]

    pl.seed_everything(args.seed, workers=True)
    result2 = tester.test(classification, test_dataloader)[0]

    print(result1)
    print(result2)


if __name__ == "__main__":
    main(parser.parse_args())
    exit(0)
