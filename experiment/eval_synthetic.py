import argparse
from typing import Dict

import pytorch_lightning as pl
import torch
from longseq_formers.dataset.synthetic import SyntheticDataset, parse_syntetic_data
from longseq_formers.task import Synthetic
from longseq_formers.utils import get_logger
from torch.utils.data import DataLoader

# fmt: off
parser = argparse.ArgumentParser(prog="train_synthetic", description="Train & Test Synthetic Task")

g = parser.add_argument_group("Train Parameter")
g.add_argument("--model", type=str, required=True, help="model checkpoint")
g.add_argument("--dataset", type=str, required=True, help="dataset name")
g.add_argument("--valid-batch-size", type=int, default=1, help="validation batch size")
g.add_argument("--max-length", type=int, default=150, help="max sequence length")
g.add_argument("--seed", type=int, default=42, help="random seed")
g.add_argument("--shuffle", action="store_true", help="shuffle data order")
# fmt: on


def main(args: argparse.Namespace) -> dict[str, float]:
    logger = get_logger("eval_synthetic_task")

    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    pl.seed_everything(args.seed, workers=True)

    logger.info(f'[+] Use Dataset: "{args.dataset}"')
    _, vocab_size, _, dev_examples, test_examples = parse_syntetic_data(args.dataset)

    valid_dataset = SyntheticDataset(dev_examples)
    test_dataset = SyntheticDataset(test_examples)

    logger.info(f"[+] # of batched valid examples: {len(valid_dataset)}")
    logger.info(f"[+] # of batched test examples: {len(test_dataset)}")

    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size)
    test_dataloader = DataLoader(test_dataset, batch_size=args.valid_batch_size)

    synthetic_task = Synthetic.load_from_checkpoint(args.model, vocab_size=vocab_size)

    logger.info(f"[+] Start Evaluation")

    tester = pl.Trainer(accelerator="gpu" if torch.cuda.device_count() else None, devices=1)

    pl.seed_everything(args.seed, workers=True)
    result1 = tester.test(synthetic_task, valid_dataloader)[0]

    pl.seed_everything(args.seed, workers=True)
    result2 = tester.test(synthetic_task, test_dataloader)[0]

    print(result1)
    print(result2)


if __name__ == "__main__":
    main(parser.parse_args())
    exit(0)
