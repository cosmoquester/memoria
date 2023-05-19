import argparse
import os
import tempfile
from typing import Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM

import wandb
from longseq_formers.dataset.synthetic import SyntheticDataset, parse_syntetic_data
from longseq_formers.task import Synthetic
from longseq_formers.utils import get_logger

# fmt: off
parser = argparse.ArgumentParser(prog="train_synthetic", description="Train & Test Synthetic Task")

g = parser.add_argument_group("Train Parameter")
g.add_argument("--model-config", type=str, required=True, help="huggingface model config")
g.add_argument("--dataset", type=str, required=True, help="dataset name")
g.add_argument("--batch-size", type=int, default=32, help="global training batch size")
g.add_argument("--valid-batch-size", type=int, default=1, help="validation batch size")
g.add_argument("--accumulate-grad-batches", type=int, default=1, help="the number of gradident accumulation steps")
g.add_argument("--epochs", type=int, default=1, help="the number of training epochs")
g.add_argument("--learning-rate", type=float, default=2e-4, help="learning rate")
g.add_argument("--warmup-rate", type=float, default=0.06, help="warmup step rate")
g.add_argument("--max-grad-norm", type=float, default=1.0, help="maximum gradient norm")
g.add_argument("--seed", type=int, default=42, help="random seed")
g.add_argument("--test-ckpt", type=str, default="last", choices=["best", "last"], help="checkpoint type for testing")
g.add_argument("--segment-size", type=int, required=True, help="segment size for infinity former")

g = parser.add_argument_group("Personal Options")
g.add_argument("--output-dir", type=str, help="output directory path to save artifacts")
g.add_argument("--gpus", type=int, help="the number of gpus, use all devices by default")
g.add_argument("--logging-interval", type=int, default=100, help="logging interval")
g.add_argument("--valid-interval", type=float, default=1.0, help="validation interval rate")

g = parser.add_argument_group("Wandb Options")
g.add_argument("--wandb-run-name", type=str, help="wanDB run name")
g.add_argument("--wandb-entity", type=str, default="cosmoquester", help="wanDB entity name")
g.add_argument("--wandb-project", type=str, default="long-sequence-formers", help="wanDB project name")
# fmt: on


def main(args: argparse.Namespace) -> Dict[str, float]:
    logger = get_logger("train_synthetic_task")

    if args.output_dir:
        os.makedirs(args.output_dir)
        logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    pl.seed_everything(args.seed, workers=True)

    logger.info(f"[+] GPU: {args.gpus}")

    logger.info(f'[+] Use Dataset: "{args.dataset}"')
    _, vocab_size, train_examples, dev_examples, test_examples = parse_syntetic_data(args.dataset)

    if args.gpus is None:
        args.gpus = torch.cuda.device_count()
    num_parallels = max(args.gpus, 1)
    distributed = num_parallels > 1
    batch_size_per_device = max(args.batch_size // num_parallels, 1)
    global_batch_size = batch_size_per_device * num_parallels
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

    train_dataset = SyntheticDataset(train_examples)
    valid_dataset = SyntheticDataset(dev_examples)
    test_dataset = SyntheticDataset(test_examples)

    logger.info(f"[+] # of batched train examples: {len(train_dataset)}")
    logger.info(f"[+] # of batched valid examples: {len(valid_dataset)}")
    logger.info(f"[+] # of batched test examples: {len(test_dataset)}")

    logger.info(f'[+] Initialize Model with Config: "{args.model_config}"')
    config = AutoConfig.from_pretrained(args.model_config, trust_remote_code=True, vocab_size=vocab_size)
    model = AutoModelForCausalLM.from_config(config)

    train_dataloader = DataLoader(
        train_dataset, shuffle=True, batch_size=batch_size_per_device, num_workers=os.cpu_count() // 2, pin_memory=True
    )
    valid_dataloader = DataLoader(valid_dataset, batch_size=args.valid_batch_size // num_parallels)
    test_dataloader = DataLoader(test_dataset, batch_size=args.valid_batch_size // num_parallels)
    total_steps = len(train_dataloader) * args.epochs

    synthetic_task = Synthetic(
        model=model,
        total_steps=total_steps,
        learning_rate=args.learning_rate,
        warmup_rate=args.warmup_rate,
        segment_size=args.segment_size,
        vocab_size=vocab_size,
        max_grad_norm=args.max_grad_norm,
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
        model_dir, mode="max", monitor="val/acc-final", save_last=True, auto_insert_metric_name=True
    )
    callbacks = [model_checkpoint_callback]

    if train_loggers:
        callbacks.append(LearningRateMonitor(logging_interval="step"))
    trainer = pl.Trainer(
        logger=train_loggers,
        max_epochs=args.epochs,
        log_every_n_steps=args.logging_interval,
        val_check_interval=args.valid_interval,
        accumulate_grad_batches=args.accumulate_grad_batches,
        callbacks=callbacks,
        strategy="ddp_fork" if distributed else None,
        accelerator="gpu" if args.gpus else None,
        devices=num_parallels,
        replace_sampler_ddp=False,
    )
    trainer.fit(synthetic_task, train_dataloader, valid_dataloader)

    # Use seperated initialized trainer (https://github.com/Lightning-AI/lightning/issues/8375)
    # Use batch size as 1 because already batched
    tester = pl.Trainer(
        logger=train_loggers,
        callbacks=callbacks,
        accelerator="gpu" if args.gpus else None,
        devices=1,
    )
    result = tester.test(synthetic_task, test_dataloader, ckpt_path=args.test_ckpt)[0]

    wandb.finish()

    if not args.output_dir:
        tmp_dir.cleanup()

    return result


if __name__ == "__main__":
    main(parser.parse_args())
    exit(0)
