import argparse
from typing import Dict

import pytorch_lightning as pl
import torch
from longseq_formers.data import CLASSIFICATION_DATASETS, load_hyperpartisan_data
from longseq_formers.dataset import ClassificationDataset
from longseq_formers.task import Classification
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
    if args.dataset == "hyperpartisan":
        datasets = load_hyperpartisan_data()

    valid_dataset = ClassificationDataset(datasets["dev"])
    test_dataset = ClassificationDataset(datasets["test"])

    logger.info(f"[+] # of valid examples: {len(valid_dataset)}")
    logger.info(f"[+] # of test examples: {len(test_dataset)}")

    logger.info(f'[+] Load Model: "{args.model}"')
    classification = Classification.load_from_checkpoint(
        args.model, tokenizer=tokenizer, max_length=args.max_length, truncation=args.truncation
    )

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
