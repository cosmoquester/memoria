from typing import Any, Dict, Optional

import pytorch_lightning as pl
import torch
from transformers import AutoConfig, AutoModelForCausalLM, get_linear_schedule_with_warmup


class LanguageModeling(pl.LightningModule):
    """LanguageModeling

    Attributes:
        model: model for language modeling
        total_steps: total training steps for lr scheduling
        learning_rate: Max LR
        warmup_rate: warmup step rate
    """

    def __init__(
        self, model: Optional[AutoModelForCausalLM], total_steps: int, learning_rate: float, warmup_rate: float
    ):
        super().__init__()

        self.model = model
        self.total_steps = total_steps
        self.learning_rate = learning_rate
        self.warmup_rate = warmup_rate

        self.save_hyperparameters(
            {
                "model": None,
                "total_steps": total_steps,
                "learning_rate": learning_rate,
                "warmup_rate": warmup_rate,
                "model_config": model.config.to_dict() if model is not None else None,
            }
        )

    def _step(self, batch: Dict[str, torch.Tensor], batch_idx: int, prefix="") -> Dict[str, float]:
        """Common step function

        Args:
            batch: training batch input/label
        Returns:
            metrics dictionary of this train step
        """
        is_end = batch.pop("is_end", None)

        if self.model.config.model_type in ["transfo-xl", "memoria-xl"]:
            del batch["attention_mask"]
        if hasattr(self, "_mems"):
            batch["mems"] = self._mems
        if hasattr(self, "_cmems"):
            batch["cmems"] = self._cmems
        outputs = self.model(**batch)
        lm_loss = outputs.loss

        if self.model.config.model_type in ["compressive_transformer"]:
            lm_loss = outputs.lm_loss
        if hasattr(outputs, "mems"):
            self._mems = outputs.mems
        if hasattr(outputs, "cmems"):
            self._cmems = outputs.cmems

        loss = outputs.loss
        ppl = lm_loss.detach().exp()
        metrics = {"loss": loss, "lm_loss": lm_loss, "ppl": ppl}
        if self.model.config.model_type in ["gpt2_with_memoria"]:
            ltm_mask = self.model.transformer.memoria.engrams.longterm_memory_mask
            metrics["num_ltms_per_batch"] = ltm_mask.sum(dim=1).float().mean(dim=0) if ltm_mask.numel() > 0 else 0.0
        metrics = {prefix + k: v for k, v in metrics.items()}
        return metrics

    def training_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Train step function"""
        metrics = self._step(batch=batch, batch_idx=batch_idx, prefix="")
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        return metrics

    def validation_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Validation step function"""
        metrics = self._step(batch=batch, batch_idx=batch_idx, prefix="val/")
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        return metrics

    def test_step(self, batch: Dict[str, torch.Tensor], batch_idx: int) -> Dict[str, float]:
        """Test step function"""
        metrics = self._step(batch=batch, batch_idx=batch_idx, prefix="test/")
        self.log_dict(metrics, prog_bar=True, logger=True, on_step=True, sync_dist=True)
        return metrics

    def configure_optimizers(self) -> Dict:
        optimizer = torch.optim.Adam(params=self.model.parameters(), lr=self.learning_rate)
        optimizers = {"optimizer": optimizer}

        scheduler = get_linear_schedule_with_warmup(
            optimizer,
            num_warmup_steps=int(self.total_steps * self.warmup_rate) if self.warmup_rate else 0,
            num_training_steps=self.total_steps,
        )
        optimizers["lr_scheduler"] = {"scheduler": scheduler, "interval": "step", "name": "Learning Rate"}

        return optimizers

    def reset_memories(self) -> None:
        if self.model.config.model_type in ["gpt2_with_memoria"]:
            self.model.transformer.memoria.reset_memory()
            self.model.transformer.prev_hidden = None
        if self.model.config.model_type in ["transfo-xl"] and hasattr(self, "_mems"):
            del self._mems
        if self.model.config.model_type == "compressive_transformer":
            if hasattr(self, "_mems"):
                del self._mems
            if hasattr(self, "_cmems"):
                del self._cmems

    def on_train_start(self) -> None:
        self.reset_memories()

    def on_train_end(self) -> None:
        self.reset_memories()

    def on_validation_start(self) -> None:
        self.reset_memories()

    def on_validation_end(self) -> None:
        self.reset_memories()

    def on_test_start(self) -> None:
        self.reset_memories()

    def on_test_end(self) -> None:
        self.reset_memories()

    def on_train_batch_start(self, batch: Any, batch_idx: int) -> None:
        if self.model.config.model_type in ["gpt2_with_memoria"]:
            if batch_idx % self.model.config.memoria_reset_period == 0:
                self.model.transformer.memoria.reset_memory()
                self.model.transformer.prev_hidden = None

    def on_save_checkpoint(self, checkpoint: Dict[str, Any]):
        checkpoint["model_config"] = self.model.config.to_dict()
        checkpoint["model_type"] = self.model.config.model_type

    def on_load_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        config_dict = checkpoint["model_config"]
        config_cls = AutoConfig.for_model(checkpoint["model_type"])
        config = config_cls.from_dict(config_dict)
        self.model = AutoModelForCausalLM.from_config(config)
        return super().on_load_checkpoint(checkpoint)
