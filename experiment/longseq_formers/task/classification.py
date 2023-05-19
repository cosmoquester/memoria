from dataclasses import dataclass
from typing import Any, Dict, Literal, Optional, Tuple

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, MulticlassF1Score
from torchmetrics.collections import MetricCollection
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup
from transformers.utils import ModelOutput


@dataclass
class SequenceClassifierOutputWithKLLoss(ModelOutput):
    loss: Optional[torch.FloatTensor] = None
    kl_regularization_loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class Classification(pl.LightningModule):
    """Classification

    Attributes:
        model: model for classification
        num_classes: the number of classes
        total_steps: total training steps for lr scheduling
        learning_rate: Max LR
        warmup_rate: warmup step rate
    """

    def __init__(
        self,
        model: AutoModelForSequenceClassification,
        num_classes: int,
        total_steps: int,
        learning_rate: float,
        warmup_rate: float,
        segment_size: Optional[int] = None,
        aggregate: Literal["mean", "last"] = "mean",
        eval_aggregate: Literal["mean", "last"] = "last",
    ):
        super().__init__()

        self.model = model
        self.num_classes = num_classes
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.warmup_rate = warmup_rate
        self.segment_size = segment_size
        self.aggregate = aggregate
        self.eval_aggregate = eval_aggregate
        self.automatic_optimization = False

        metric_collection = MetricCollection(
            {
                "acc": Accuracy(task="multiclass", top_k=1, num_classes=self.num_classes),
                "f1": MulticlassF1Score(task="multiclass", num_classes=self.num_classes, average="macro"),
            }
        )
        self.train_metrics = metric_collection.clone(prefix="train/")
        self.val_metrics = metric_collection.clone(prefix="val/")
        self.test_metrics = metric_collection.clone(prefix="test/")
        self.metrics = {"train/": self.train_metrics, "val/": self.val_metrics, "test/": self.test_metrics}

        self.save_hyperparameters(
            {
                "model": None,
                "model_config": model.config.to_dict() if model is not None else None,
                "num_classes": num_classes,
                "total_steps": total_steps,
                "learning_rate": learning_rate,
                "warmup_rate": warmup_rate,
                "segment_size": segment_size,
                "aggregate": aggregate,
                "eval_aggregate": eval_aggregate,
            }
        )

    def _single_step(self, batch: Dict[str, torch.Tensor], batch_idx: int, prefix="") -> Dict[str, float]:
        """Common step function

        Args:
            batch: training batch input/label
        Returns:
            metrics dictionary of this train step
        """
        labels = batch.pop("labels")

        outputs = self.model(**batch)
        logits = outputs.logits

        ce_loss = F.cross_entropy(logits, labels)
        loss = ce_loss
        other_metrics = {"ce_loss": ce_loss}
        if isinstance(outputs, SequenceClassifierOutputWithKLLoss) and outputs.kl_regularization_loss is not None:
            kl_loss = outputs.kl_regularization_loss.mean(dim=0)
            other_metrics["kl_loss"] = kl_loss
            loss += kl_loss
        if self.model.config.model_type == "memoria_bert":
            ltm_mask = self.model.bert.encoder.memoria.engrams.longterm_memory_mask
            other_metrics["num_ltms_per_batch"] = (
                ltm_mask.sum(dim=1).float().mean(dim=0) if ltm_mask.numel() > 0 else 0.0
            )
        other_metrics["loss"] = loss

        metrics = self.metrics[prefix](logits, labels)
        self.metrics[prefix].update(logits, labels)
        metrics.update(**{prefix + k: v for k, v in other_metrics.items()})
        return metrics

    def _segment_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        prefix="",
        aggregate: Literal["mean", "last"] = "mean",
    ) -> Dict[str, float]:
        length = batch["input_ids"].shape[1]
        metrics = []
        indices = list(range(0, length, self.segment_size))
        for i in indices:
            segment_batch = {k: v[:, i : i + self.segment_size] if k != "labels" else v for k, v in batch.items()}
            segment_metrics = self._single_step(segment_batch, batch_idx, prefix)
            if "train/loss" in segment_metrics:
                self.manual_backward(segment_metrics["train/loss"] / len(indices))
            metrics.append(segment_metrics)
        if aggregate == "mean":
            metrics = {k: torch.stack([m[k] for m in metrics], dim=0).mean(dim=0) for k in metrics[0].keys()}
        elif aggregate == "last":
            metrics = metrics[-1]
        else:
            raise ValueError(f"Unknown aggregate method: {aggregate}")
        return metrics

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Train step function"""
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()

        if self.segment_size:
            metrics = self._segment_step(batch=batch, batch_idx=batch_idx, aggregate=self.aggregate, prefix="train/")
        else:
            metrics = self._single_step(batch=batch, batch_idx=batch_idx, prefix="train/")
            self.manual_backward(metrics["train/loss"])

        opt.step()
        if sch is not None:
            sch.step()

        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        return metrics

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Validation step function"""
        if self.segment_size:
            metrics = self._segment_step(batch=batch, batch_idx=batch_idx, prefix="val/", aggregate=self.eval_aggregate)
        else:
            metrics = self._single_step(batch=batch, batch_idx=batch_idx, prefix="val/")
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        return metrics

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Test step function"""
        if self.segment_size:
            metrics = self._segment_step(
                batch=batch, batch_idx=batch_idx, prefix="test/", aggregate=self.eval_aggregate
            )
        else:
            metrics = self._single_step(batch=batch, batch_idx=batch_idx, prefix="test/")
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        return metrics

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
        self.model = AutoModelForSequenceClassification.from_config(config)
        return super().on_load_checkpoint(checkpoint)

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if self.model.config.model_type == "memoria_bert":
            self.model.bert.encoder.memoria.reset_memory()

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self.model.config.model_type == "memoria_bert":
            self.model.bert.encoder.memoria.reset_memory()

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self.model.config.model_type == "memoria_bert":
            self.model.bert.encoder.memoria.reset_memory()

    def _epoch_end(self, outputs, prefix: str = "") -> None:
        results = self.metrics[prefix].compute()
        results = {k + "_final": v for k, v in results.items()}
        self.metrics[prefix].reset()
        self.log_dict(results, logger=True, sync_dist=True)

    def training_epoch_end(self, outputs) -> None:
        return self._epoch_end(outputs, prefix="train/")

    def validation_epoch_end(self, outputs) -> None:
        return self._epoch_end(outputs, prefix="val/")

    def test_epoch_end(self, outputs) -> None:
        return self._epoch_end(outputs, prefix="test/")
