import argparse
import os
import tempfile
from typing import Dict

import pytorch_lightning as pl
import torch
import wandb
from longseq_formers.data import CLASSIFICATION_DATASETS, load_hyperpartisan_data
from longseq_formers.dataset import ClassificationDataset
from longseq_formers.task import Classification
from longseq_formers.utils import get_logger
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForSequenceClassification, AutoTokenizer

# fmt: off
parser = argparse.ArgumentParser(prog="train_classification", description="Train & Test Long Sequence Classification")

g = parser.add_argument_group("Train Parameter")
g.add_argument("--model", type=str, required=True, help="huggingface model")
g.add_argument("--model-type", type=str, help="specific model type")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--dataset", type=str, default="hyperpartisan", choices=CLASSIFICATION_DATASETS, help="dataset name")
g.add_argument("--batch-size", type=int, default=8, help="global training batch size")
g.add_argument("--valid-batch-size", type=int, default=32, help="validation batch size")
g.add_argument("--accumulate-grad-batches", type=int, default=1, help="the number of gradident accumulation steps")
g.add_argument("--max-length", type=int, default=512, help="max sequence length")
g.add_argument("--memory-length", type=int, default=512, help="max sequence length for bert one inference on infinity former")
g.add_argument("--epochs", type=int, default=20, help="the number of training epochs")
g.add_argument("--learning-rate", type=float, default=3e-5, help="learning rate")
g.add_argument("--warmup-rate", type=float, help="warmup step rate")
g.add_argument("--seed", type=int, default=42, help="random seed")
g.add_argument("--test-ckpt", type=str, default="last", choices=["best", "last"], help="checkpoint type for testing")
g.add_argument("--not-truncate", action="store_false", dest="truncation", help="not truncate sequence")
g.add_argument("--segment-size", type=int, help="segment size for infinity former")

g = parser.add_argument_group("Personal Options")
g.add_argument("--output-dir", type=str, help="output directory path to save artifacts")
g.add_argument("--gpus", type=int, help="the number of gpus, use all devices by default")
g.add_argument("--logging-interval", type=int, default=10, help="logging interval")

g = parser.add_argument_group("Wandb Options")
g.add_argument("--wandb-run-name", type=str, help="wanDB run name")
g.add_argument("--wandb-entity", type=str, help="wanDB entity name")
g.add_argument("--wandb-project", type=str, help="wanDB project name")
# fmt: on


def main(args: argparse.Namespace) -> dict[str, float]:
    logger = get_logger("train_classification")

    if args.output_dir:
        os.makedirs(args.output_dir)
        logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    pl.seed_everything(args.seed, workers=True)

    logger.info(f"[+] GPU: {args.gpus}")

    if args.tokenizer is None:
        if args.model:
            logger.info(f"[+] Use tokenizer same as model: {args.model}")
            args.tokenizer = args.model
        else:
            raise ValueError("you should set `--tokenizer` when use `--model-config`!")
    logger.info(f'[+] Load Tokenizer: "{args.tokenizer}"')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer)

    logger.info(f'[+] Use Dataset: "{args.dataset}"')
    if args.dataset == "hyperpartisan":
        datasets = load_hyperpartisan_data()
        num_classes = 2

    train_dataset = ClassificationDataset(
        datasets["train"], tokenizer=tokenizer, max_length=args.max_length, truncation=args.truncation
    )
    valid_dataset = ClassificationDataset(
        datasets["dev"], tokenizer=tokenizer, max_length=args.max_length, truncation=args.truncation
    )
    test_dataset = ClassificationDataset(
        datasets["test"], tokenizer=tokenizer, max_length=args.max_length, truncation=args.truncation
    )

    logger.info(f"[+] # of train examples: {len(train_dataset)}")
    logger.info(f"[+] # of valid examples: {len(valid_dataset)}")
    logger.info(f"[+] # of test examples: {len(test_dataset)}")

    logger.info(f'[+] Load Model: "{args.model}"')
    if args.model_type:
        model_cls = type(AutoModelForSequenceClassification.from_config(AutoConfig.for_model(args.model_type)))
    else:
        model_cls = AutoModelForSequenceClassification
    model = model_cls.from_pretrained(args.model, num_labels=num_classes)

    if args.gpus is None:
        args.gpus = torch.cuda.device_count()
    num_parallels = max(args.gpus, 1)
    distributed = num_parallels > 1
    batch_size_per_device = max(args.batch_size // num_parallels, 1)
    global_batch_size = batch_size_per_device * args.gpus
    valid_batch_size_per_device = max(args.valid_batch_size // num_parallels, 1)
    global_valid_batch_size = valid_batch_size_per_device * num_parallels
    if args.batch_size != global_batch_size:
        logger.warning(f"[-] Batch size {args.batch_size} isn't dividable by {args.gpus}!")
        logger.warning(f"[-] Use batch size as {batch_size_per_device} per device, {global_batch_size} global")
    if args.valid_batch_size != global_valid_batch_size:
        logger.warning(f"[-] Valid Batch size {args.valid_batch_size} isn't dividable by {args.gpus}!")
        logger.warning(
            f"[-] Use batch size as {valid_batch_size_per_device} per device, {global_valid_batch_size} global"
        )

    collate_fn = ClassificationDataset.pad_collate_fn if not args.truncation else None
    train_dataloader = DataLoader(
        train_dataset,
        shuffle=True,
        batch_size=batch_size_per_device,
        num_workers=os.cpu_count() // 2,
        pin_memory=True,
        collate_fn=collate_fn,
    )
    valid_dataloader = DataLoader(
        valid_dataset,
        batch_size=args.valid_batch_size // num_parallels,
        collate_fn=collate_fn,
    )
    test_dataloader = DataLoader(
        test_dataset,
        batch_size=args.valid_batch_size // num_parallels,
        collate_fn=collate_fn,
    )

    total_steps = len(train_dataloader) * args.epochs

    classification = Classification(
        model=model,
        total_steps=total_steps,
        learning_rate=args.learning_rate,
        warmup_rate=args.warmup_rate,
        segment_size=args.segment_size,
        num_classes=num_classes,
    )

    if args.output_dir:
        train_loggers = [TensorBoardLogger(args.output_dir, "", "logs")]
        model_dir = os.path.join(args.output_dir, "checkpoint")
    else:
        train_loggers = []
        tmp_dir = tempfile.TemporaryDirectory()
        model_dir = tmp_dir.name

    logger.info(f"[+] Start Training")
    if args.wandb_project and (args.wandb_run_name or args.output_dir):
        wandb_logger = WandbLogger(
            name=args.wandb_run_name or os.path.basename(args.output_dir),
            project=args.wandb_project,
            entity=args.wandb_entity,
            save_dir=args.output_dir if args.output_dir else None,
        )
        wandb_logger.log_hyperparams({"train_arguments": vars(args)})
        train_loggers.append(wandb_logger)

    model_checkpoint_callback = pl.callbacks.ModelCheckpoint(
        model_dir, mode="max", monitor="val/f1_final", save_last=True, auto_insert_metric_name=True
    )
    callbacks = [model_checkpoint_callback]

    if train_loggers:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    trainer = pl.Trainer(
        logger=train_loggers,
        max_epochs=args.epochs,
        log_every_n_steps=args.logging_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        strategy="ddp_fork" if distributed else None,
        accelerator="gpu" if args.gpus else None,
        devices=num_parallels,
    )
    trainer.fit(classification, train_dataloader, valid_dataloader)

    # Use seperated initialized trainer (https://github.com/Lightning-AI/lightning/issues/8375)
    tester = pl.Trainer(
        logger=train_loggers,
        callbacks=callbacks,
        accelerator="gpu" if args.gpus else None,
        devices=1,
    )
    result = tester.test(classification, test_dataloader, ckpt_path=args.test_ckpt)[0]

    wandb.finish()

    if not args.output_dir:
        tmp_dir.cleanup()

    return result


if __name__ == "__main__":
    main(parser.parse_args())
    exit(0)
