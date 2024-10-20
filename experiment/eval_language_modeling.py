import argparse
from typing import Dict

import pytorch_lightning as pl
import torch
from longseq_formers.data import (
    LANGUAGE_MODELING_DATASETS,
    enwik8_tokenize,
    load_enwik8_data,
    load_pg19_data,
    load_wikitext103_data,
)
from longseq_formers.dataset import LanguageModelingDataset, text_to_tokens
from longseq_formers.task import LanguageModeling
from longseq_formers.utils import get_logger
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

# fmt: off
parser = argparse.ArgumentParser(prog="evaluate", description="Evaluate Language Modeling")

g = parser.add_argument_group("Eval Parameter")
g.add_argument("--model", type=str, required=True, help="huggingface model")
g.add_argument("--tokenizer", type=str, default="gpt2", help="huggingface tokenizer")
g.add_argument("--dataset", type=str, default="wikitext103", choices=LANGUAGE_MODELING_DATASETS, help="dataset name")
g.add_argument("--valid-batch-size", type=int, default=1, help="validation batch size")
g.add_argument("--max-length", type=int, default=512, help="max sequence length")
g.add_argument("--seed", type=int, default=42, help="random seed")
# fmt: on


def main(args: argparse.Namespace) -> dict[str, float]:
    logger = get_logger("test_language_modeling")

    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    pl.seed_everything(args.seed, workers=True)

    gpus = torch.cuda.device_count()
    logger.info(f"[+] GPU: {gpus}")

    if args.tokenizer is None:
        logger.info(f"[+] Use tokenizer same as model: {args.model}")
        args.tokenizer = args.model
    if args.dataset == "enwik8":
        logger.info(f"[+] Use character tokenizer for enwik8 dataset")
        tokenizer = enwik8_tokenize
    else:
        logger.info(f'[+] Load Tokenizer: "{args.tokenizer}"')
        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info(f'[+] Use Dataset: "{args.dataset}"')
    if args.dataset == "wikitext103":
        data = load_wikitext103_data()
    elif args.dataset == "pg19":
        data = load_pg19_data()
    elif args.dataset == "enwik8":
        data = load_enwik8_data()
    else:
        raise ValueError(f"dataset `{args.dataset}` is not valid!")

    dev_tokens = text_to_tokens(data["dev"], tokenizer, args.valid_batch_size, args.max_length)
    test_tokens = text_to_tokens(data["test"], tokenizer, args.valid_batch_size, args.max_length)

    valid_dataset = LanguageModelingDataset(dev_tokens)
    test_dataset = LanguageModelingDataset(test_tokens)

    logger.info(f"[+] # of batched valid examples: {len(valid_dataset)}")
    logger.info(f"[+] # of batched test examples: {len(test_dataset)}")

    language_modeling = LanguageModeling.load_from_checkpoint(args.model)

    # Use batch size as 1 because already batched
    # train_dataloader = DataLoader(train_dataset, batch_size=1, collate_fn=LanguageModelingDataset.collate_fn)
    valid_dataloader = DataLoader(valid_dataset, batch_size=1, collate_fn=LanguageModelingDataset.collate_fn)
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=LanguageModelingDataset.collate_fn)
    tester = pl.Trainer(accelerator="gpu" if gpus else None, devices=1)

    pl.seed_everything(args.seed, workers=True)
    result1 = tester.test(language_modeling, valid_dataloader)[0]

    pl.seed_everything(args.seed, workers=True)
    result2 = tester.test(language_modeling, test_dataloader)[0]

    print(result1)
    print(result2)


if __name__ == "__main__":
    main(parser.parse_args())
    exit(0)
