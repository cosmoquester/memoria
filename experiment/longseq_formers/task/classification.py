from typing import Any, Dict, Literal, Optional

import pytorch_lightning as pl
import torch
import torch.nn.functional as F
from torchmetrics.classification import Accuracy, MulticlassF1Score
from torchmetrics.collections import MetricCollection
from transformers import AutoConfig, AutoModelForSequenceClassification, get_linear_schedule_with_warmup


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

        ce_loss = F.cross_entropy(logits, labels, reduction="none")
        loss = ce_loss
        other_metrics = {"ce_loss": ce_loss.mean()}
        if self.model.config.model_type == "memoria_bert":
            ltm_mask = self.model.bert.encoder.memoria.engrams.longterm_memory_mask
            other_metrics["num_ltms_per_batch"] = (
                ltm_mask.sum(dim=1).float().mean(dim=0)
                if ltm_mask.numel() > 0
                else torch.tensor(0.0, device=loss.device)
            )
        if self.model.config.model_type == "memoria_roberta":
            ltm_mask = self.model.roberta.encoder.memoria.engrams.longterm_memory_mask
            other_metrics["num_ltms_per_batch"] = (
                ltm_mask.sum(dim=1).float().mean(dim=0)
                if ltm_mask.numel() > 0
                else torch.tensor(0.0, device=loss.device)
            )
        other_metrics["loss"] = loss

        other_metrics = {prefix + k: v for k, v in other_metrics.items()}
        return other_metrics, logits.detach(), labels.detach()

    def _segment_step(
        self,
        batch: Dict[str, torch.Tensor],
        batch_idx: int,
        aggregate: Literal["mean", "last"],
        prefix="",
    ) -> Dict[str, float]:
        batch_size, length = batch["input_ids"].shape
        num_valid_segments = batch["attention_mask"][:, :: self.segment_size].sum(dim=1)
        all_metrics = []
        all_probs = []
        indices = list(range(0, length, self.segment_size))
        prev_indices = [None] + indices[:-1]
        post_indices = indices[1:] + [None]
        final_loss = 0.0
        for pre_i, i, post_i in zip(prev_indices, indices, post_indices):
            segment_batch = {k: v[:, i : i + self.segment_size] if k != "labels" else v for k, v in batch.items()}
            pre_batch = (
                {k: v[:, pre_i : pre_i + self.segment_size] if k != "labels" else v for k, v in batch.items()}
                if pre_i is not None
                else None
            )
            post_batch = (
                {k: v[:, post_i : post_i + self.segment_size] if k != "labels" else v for k, v in batch.items()}
                if post_i is not None
                else None
            )

            current_valid = segment_batch["attention_mask"].bool().any(dim=1)
            is_last = current_valid
            if pre_batch is not None:
                pre_valid = pre_batch["attention_mask"].bool().any(dim=1)
                is_last &= pre_valid
            if post_batch is not None:
                post_valid = post_batch["attention_mask"].bool().any(dim=1)
                is_last &= ~post_valid

            segment_metrics, logits, labels = self._single_step(segment_batch, batch_idx, prefix)
            if aggregate == "last":
                loss = segment_metrics[f"{prefix}loss"] / batch_size
                loss = loss[is_last].sum()
                final_loss += loss.item()

                if logits[is_last].numel():
                    self.metrics[prefix].update(logits[is_last], labels[is_last])
                segment_metrics[f"{prefix}loss"] = loss
            elif aggregate == "mean":
                loss = segment_metrics[f"{prefix}loss"].mean() / len(indices)
                final_loss += loss.item()

                probs = logits.softmax(dim=-1)
                probs[~current_valid] = 0.0
                all_probs.append(probs)

                segment_metrics[f"{prefix}loss"] = loss
            else:
                raise ValueError(f"Unknown aggregate method: {aggregate}")

            if prefix == "train/":
                self.manual_backward(loss)

            all_metrics.append(segment_metrics)
        if aggregate == "mean":
            all_metrics = {
                k: torch.stack([m[k] for m in all_metrics], dim=0).mean(dim=0) for k in all_metrics[0].keys()
            }
            mean_logits = torch.stack(all_probs, dim=-1).mean(dim=-1)
            self.metrics[prefix].update(mean_logits, labels)
            segment_metrics = all_metrics

        segment_metrics.update(self.metrics[prefix].compute())
        return segment_metrics

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Train step function"""
        opt = self.optimizers()
        sch = self.lr_schedulers()
        opt.zero_grad()

        if self.segment_size:
            metrics = self._segment_step(batch=batch, batch_idx=batch_idx, aggregate=self.aggregate, prefix="train/")
        else:
            metrics, logits, labels = self._single_step(batch=batch, batch_idx=batch_idx, prefix="train/")
            self.manual_backward(metrics["train/loss"])
            metrics.update(self.metrics["train/"](logits, labels))

        opt.step()
        if sch is not None:
            sch.step()

        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        return metrics

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Validation step function"""
        if self.segment_size:
            metrics = self._segment_step(batch=batch, batch_idx=batch_idx, aggregate=self.eval_aggregate, prefix="val/")
        else:
            metrics, logits, labels = self._single_step(batch=batch, batch_idx=batch_idx, prefix="val/")
            metrics.update(self.metrics["val/"](logits, labels))
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        return metrics

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Test step function"""
        if self.segment_size:
            metrics = self._segment_step(
                batch=batch, batch_idx=batch_idx, aggregate=self.eval_aggregate, prefix="test/"
            )
        else:
            metrics, logits, labels = self._single_step(batch=batch, batch_idx=batch_idx, prefix="test/")
            metrics.update(self.metrics["test/"](logits, labels))
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
        self.metrics["train/"].reset()
        if self.model.config.model_type == "memoria_bert":
            self.model.bert.encoder.memoria.reset_memory()
        if self.model.config.model_type == "memoria_roberta":
            self.model.roberta.encoder.memoria.reset_memory()

    def on_validation_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self.model.config.model_type == "memoria_bert":
            self.model.bert.encoder.memoria.reset_memory()
        if self.model.config.model_type == "memoria_roberta":
            self.model.roberta.encoder.memoria.reset_memory()

    def on_test_batch_start(self, batch: Any, batch_idx: int, dataloader_idx: int) -> None:
        if self.model.config.model_type == "memoria_bert":
            self.model.bert.encoder.memoria.reset_memory()
        if self.model.config.model_type == "memoria_roberta":
            self.model.roberta.encoder.memoria.reset_memory()

    def _epoch_end(self, outputs, prefix: str = "") -> None:
        results = self.metrics[prefix].compute()
        results = {k + "_final": v for k, v in results.items()}
        self.metrics[prefix].reset()
        self.log_dict(results, logger=True, sync_dist=True)

    def validation_epoch_end(self, outputs) -> None:
        return self._epoch_end(outputs, prefix="val/")

    def test_epoch_end(self, outputs) -> None:
        return self._epoch_end(outputs, prefix="test/")
