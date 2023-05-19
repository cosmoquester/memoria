import argparse
import os
import tempfile
from typing import Dict

import pytorch_lightning as pl
import torch
from pytorch_lightning.callbacks import LearningRateMonitor
from pytorch_lightning.loggers import TensorBoardLogger, WandbLogger
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer

import wandb
from longseq_formers.data import (
    LANGUAGE_MODELING_DATASETS,
    enwik8_tokenize,
    load_enwik8_data,
    load_pg19_data,
    load_wikitext103_data,
)
from longseq_formers.dataset import LanguageModelingDataset, text_to_tokens
from longseq_formers.task import LanguageModeling
from longseq_formers.utils import BatchedDataModule, get_logger

# fmt: off
parser = argparse.ArgumentParser(prog="train", description="Train & Test Language Modeling")

g = parser.add_argument_group("Train Parameter")
g.add_argument("--model-config", type=str, help="huggingface model config")
g.add_argument("--model", type=str, help="huggingface model")
g.add_argument("--model-type", type=str, help="specific model type")
g.add_argument("--tokenizer", type=str, help="huggingface tokenizer")
g.add_argument("--dataset", type=str, default="wikitext103", choices=LANGUAGE_MODELING_DATASETS, help="dataset name")
g.add_argument("--batch-size", type=int, default=8, help="global training batch size")
g.add_argument("--valid-batch-size", type=int, default=1, help="validation batch size")
g.add_argument("--accumulate-grad-batches", type=int, default=1, help="the number of gradident accumulation steps")
g.add_argument("--max-length", type=int, default=150, help="max sequence length")
g.add_argument("--epochs", type=int, default=6, help="the number of training epochs")
g.add_argument("--learning-rate", type=float, default=2e-4, help="learning rate")
g.add_argument("--warmup-rate", type=float, default=0.06, help="warmup step rate")
g.add_argument("--max-grad-norm", type=float, default=1.0, help="maximum gradient norm")
g.add_argument("--seed", type=int, default=42, help="random seed")
g.add_argument("--shuffle", action="store_true", help="shuffle data order")
g.add_argument("--test-ckpt", type=str, default="last", choices=["best", "last"], help="checkpoint type for testing")

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
    logger = get_logger("train_language_modeling")

    if args.output_dir:
        os.makedirs(args.output_dir)
        logger.info(f'[+] Save output to "{args.output_dir}"')

    logger.info(" ====== Arguements ======")
    for k, v in vars(args).items():
        logger.info(f"{k:25}: {v}")

    logger.info(f"[+] Set Random Seed to {args.seed}")
    pl.seed_everything(args.seed, workers=True)

    logger.info(f"[+] GPU: {args.gpus}")

    if args.tokenizer is None and args.dataset != "enwik8":
        if args.model:
            logger.info(f"[+] Use tokenizer same as model: {args.model}")
            args.tokenizer = args.model
        else:
            raise ValueError("you should set `--tokenizer` when use `--model-config`!")
    logger.info(f'[+] Load Tokenizer: "{args.tokenizer}"')
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer) if args.tokenizer else enwik8_tokenize

    logger.info(f'[+] Use Dataset: "{args.dataset}"')
    if args.dataset == "wikitext103":
        data = load_wikitext103_data()
    elif args.dataset == "pg19":
        data = load_pg19_data()
    elif args.dataset == "enwik8":
        data = load_enwik8_data()
    else:
        raise ValueError(f"dataset `{args.dataset}` is not valid!")

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

    train_tokens = text_to_tokens(data["train"], tokenizer, global_batch_size, args.max_length, batch_size_per_device)
    dev_tokens = text_to_tokens(
        data["dev"], tokenizer, global_valid_batch_size, args.max_length, valid_batch_size_per_device
    )
    test_tokens = text_to_tokens(data["test"], tokenizer, args.valid_batch_size, args.max_length)

    train_dataset = LanguageModelingDataset(train_tokens)
    valid_dataset = LanguageModelingDataset(dev_tokens)
    test_dataset = LanguageModelingDataset(test_tokens)

    logger.info(f"[+] # of batched train examples: {len(train_dataset)}")
    logger.info(f"[+] # of batched valid examples: {len(valid_dataset)}")
    logger.info(f"[+] # of batched test examples: {len(test_dataset)}")

    if args.model:
        logger.info(f'[+] Load Model: "{args.model}"')
        if args.model_type:
            model_cls = type(AutoModelForCausalLM.from_config(AutoConfig.for_model(args.model_type)))
            logger.info(f"[+] Use model type: {args.model_type}")
        else:
            model_cls = AutoModelForCausalLM
        model = model_cls.from_pretrained(args.model)
    elif args.model_config:
        logger.info(f'[+] Initialize Model with Config: "{args.model_config}"')
        config = AutoConfig.from_pretrained(args.model_config, trust_remote_code=True)
        model = AutoModelForCausalLM.from_config(config)
    else:
        raise ValueError("you should set `--model` or `--model-config` argument!")

    total_steps = len(train_tokens["input_ids"]) // num_parallels * args.epochs

    language_modeling = LanguageModeling(
        model=model,
        total_steps=total_steps,
        learning_rate=args.learning_rate,
        warmup_rate=args.warmup_rate,
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
        model_dir, mode="min", monitor="val/ppl", save_last=True, auto_insert_metric_name=True
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
        gradient_clip_val=args.max_grad_norm,
        callbacks=callbacks,
        strategy="ddp_fork" if distributed else None,
        accelerator="gpu" if args.gpus else None,
        devices=num_parallels,
        replace_sampler_ddp=False,
    )
    trainer.fit(
        language_modeling,
        datamodule=BatchedDataModule(train_dataset, valid_dataset, args.shuffle, distributed),
    )

    # Use seperated initialized trainer (https://github.com/Lightning-AI/lightning/issues/8375)
    # Use batch size as 1 because already batched
    test_dataloader = DataLoader(test_dataset, batch_size=1, collate_fn=test_dataset.collate_fn)
    tester = pl.Trainer(
        logger=train_loggers,
        callbacks=callbacks,
        accelerator="gpu" if args.gpus else None,
        devices=1,
    )
    result = tester.test(language_modeling, test_dataloader, ckpt_path=args.test_ckpt)[0]

    wandb.finish()

    if not args.output_dir:
        tmp_dir.cleanup()

    return result


if __name__ == "__main__":
    main(parser.parse_args())
    exit(0)
