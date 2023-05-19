from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy
from transformers import AutoConfig, AutoModelForCausalLM, get_linear_schedule_with_warmup


class Synthetic(pl.LightningModule):
    """Synthetic Task

    Attributes:
        model: model for classification
        num_classes: the number of classes
        total_steps: total training steps for lr scheduling
        learning_rate: Max LR
        warmup_rate: warmup step rate
        segment_size: segment size
        vocab_size: vocab size
    """

    def __init__(
        self,
        model: AutoModelForCausalLM,
        total_steps: int,
        learning_rate: float,
        warmup_rate: float,
        segment_size: int,
        vocab_size: int,
        max_grad_norm: Optional[float] = None,
    ):
        super().__init__()

        self.model = model
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.warmup_rate = warmup_rate
        self.segment_size = segment_size
        self.vocab_size = vocab_size
        self.max_grad_norm = max_grad_norm
        self.automatic_optimization = False

        self.train_acc = Accuracy(task="multiclass", top_k=1, num_classes=vocab_size, ignore_index=-100)
        self.valid_acc = Accuracy(task="multiclass", top_k=1, num_classes=vocab_size, ignore_index=-100)
        self.test_acc = Accuracy(task="multiclass", top_k=1, num_classes=vocab_size, ignore_index=-100)
        self.accs = {"train": self.train_acc, "val": self.valid_acc, "test": self.test_acc}

        self.save_hyperparameters(
            {
                "model": None,
                "model_config": model.config.to_dict() if model is not None else None,
                "total_steps": total_steps,
                "learning_rate": learning_rate,
                "warmup_rate": warmup_rate,
                "segment_size": segment_size,
                "vocab_size": vocab_size,
                "max_grad_norm": max_grad_norm,
            }
        )

    def _step(self, batch: Dict[str, torch.Tensor], batch_idx: int, prefix: str) -> Dict[str, float]:
        """Train step function"""
        batch_size, length = batch["input_ids"].size()
        num_valid_labels = (batch["labels"] != -100).sum(dim=1)
        indices = range(0, length, self.segment_size)
        loss_mean = 0.0
        acc = self.accs[prefix]
        for i in indices:
            segment_batch = {k: v[:, i : i + self.segment_size] for k, v in batch.items()}
            labels = segment_batch.pop("labels")
            if hasattr(self, "_mems"):
                segment_batch["mems"] = self._mems
            if hasattr(self, "_cmems"):
                segment_batch["cmems"] = self._cmems

            use_grad = prefix == "train" and (labels != -100).any().item()
            with torch.set_grad_enabled(use_grad):
                outputs = self.model(**segment_batch)
            if self.model.config.model_type in ["transfo-xl", "memoria-xl"]:
                self._mems = outputs.mems

            loss = (
                F.cross_entropy(
                    outputs.logits.view(-1, outputs.logits.size(-1)),
                    labels.reshape(-1),
                    ignore_index=-100,
                    reduction="none",
                )
                .view(batch_size, -1)
                .sum(dim=1)
                / num_valid_labels
            ).mean()

            if use_grad:
                self.manual_backward(loss)

            loss_mean += loss.item()
            preds = outputs.logits.argmax(dim=-1)
            acc.update(preds=preds, target=labels)

        metrics = {"loss": loss_mean, "acc": acc.compute()}
        metrics = {f"{prefix}/{k}": v for k, v in metrics.items()}
        return metrics

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Train step function"""
        opt = self.optimizers()
        sch = self.lr_schedulers()

        metrics = self._step(batch=batch, batch_idx=batch_idx, prefix="train")

        if self.max_grad_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        opt.step()
        if sch is not None:
            sch.step()
            opt.zero_grad()

        self.train_acc.reset()
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        return metrics

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Validation step function"""
        metrics = self._step(batch=batch, batch_idx=batch_idx, prefix="val")
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        return metrics

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Validation step function"""
        metrics = self._step(batch=batch, batch_idx=batch_idx, prefix="test")
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, on_epoch=False, sync_dist=True)
        return metrics

    def validation_epoch_end(self, outputs):
        val_acc = self.valid_acc.compute()
        self.valid_acc.reset()
        self.log("val/acc-final", val_acc, logger=True, on_step=False, sync_dist=True)

    def test_epoch_end(self, outputs):
        test_acc = self.test_acc.compute()
        self.test_acc.reset()
        self.log("test/acc-final", test_acc, logger=True, on_step=False, sync_dist=True)

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        optimizers = {"optimizer": optimizer}

        if self.warmup_rate is not None:
            scheduler = get_linear_schedule_with_warmup(
                optimizer,
                num_warmup_steps=int(self.total_steps * self.warmup_rate),
                num_training_steps=self.total_steps,
            )
            optimizers["lr_scheduler"] = {"scheduler": scheduler, "interval": "step", "name": "Learning Rate"}

        return optimizers

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint["model_config"] = self.model.config.to_dict()
        checkpoint["model_type"] = self.model.config.model_type

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        config_dict = checkpoint["model_config"]
        config_cls = AutoConfig.for_model(checkpoint["model_type"])
        config = config_cls.from_dict(config_dict)
        self.model = AutoModelForCausalLM.from_config(config)
        return super().on_load_checkpoint(checkpoint)

    def reset_memories(self) -> None:
        if self.model.config.model_type in ["gpt2_with_memoria", "memoria-xl"]:
            self.model.transformer.memoria.reset_memory()
            self.model.transformer.prev_hidden = None
        if self.model.config.model_type in ["transfo-xl", "memoria-xl"] and hasattr(self, "_mems"):
            del self._mems
        if self.model.config.model_type == "compressive_transformer":
            if hasattr(self, "_mems"):
                del self._mems
            if hasattr(self, "_cmems"):
                del self._cmems
        if self.model.config.model_type == "infinity_gpt2":
            self.model.reset_memories()

    def on_train_batch_start(self, batch, batch_idx) -> None:
        self.reset_memories()

    def on_validation_batch_start(self, batch, batch_idx, dataloader_idx) -> None:
        self.reset_memories()

    def on_test_batch_start(self, batch, batch_idx, dataloader_idx) -> None:
        self.reset_memories()
