from __future__ import annotations

import logging
from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler
from torch.utils.data import DataLoader

from knee_fairness.training.checkpoint import save_checkpoint

LOGGER = logging.getLogger(__name__)


@dataclass(frozen=True)
class TrainerConfig:
    epochs: int = 100
    accumulation_steps: int = 1
    gradient_clip_norm: float = 1.0
    precision: str = "fp32"
    log_interval: int = 50
    checkpoint_path: Path | None = None
    seed: int = 1701


@dataclass(frozen=True)
class EpochResult:
    loss: float
    accuracy: float
    examples: int
    steps: int


class Trainer:
    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        scheduler: LRScheduler | None,
        device: torch.device,
        config: TrainerConfig,
    ) -> None:
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.scheduler = scheduler
        self.device = device
        self.config = config
        self.global_step = 0
        self.best_metric = float("-inf")
        enabled = config.precision == "fp16" and device.type == "cuda"
        self.scaler = torch.cuda.amp.GradScaler(enabled=enabled)

    def _autocast(self) -> Any:
        enabled = self.config.precision in {"fp16", "bf16"} and self.device.type == "cuda"
        dtype = torch.float16 if self.config.precision == "fp16" else torch.bfloat16
        return torch.autocast(device_type=self.device.type, dtype=dtype, enabled=enabled)

    def train_epoch(self, loader: DataLoader[Mapping[str, Any]], epoch: int) -> EpochResult:
        self.model.train()
        self.optimizer.zero_grad(set_to_none=True)
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        steps = 0
        for batch_index, batch in enumerate(loader):
            images = batch["image"].to(self.device)
            labels = batch["grade"].to(self.device)
            with self._autocast():
                logits = self.model(images)
                loss = self.criterion(logits, labels)
                scaled_loss = loss / self.config.accumulation_steps
            self.scaler.scale(scaled_loss).backward()
            should_step = (batch_index + 1) % self.config.accumulation_steps == 0
            should_step |= batch_index + 1 == len(loader)
            if should_step:
                self.scaler.unscale_(self.optimizer)
                nn.utils.clip_grad_norm_(self.model.parameters(), self.config.gradient_clip_norm)
                self.scaler.step(self.optimizer)
                self.scaler.update()
                self.optimizer.zero_grad(set_to_none=True)
                if self.scheduler is not None:
                    self.scheduler.step()
                self.global_step += 1
                steps += 1
            count = labels.numel()
            total_loss += float(loss.detach()) * count
            total_correct += int((logits.argmax(1) == labels).sum())
            total_examples += count
            if self.global_step and self.global_step % self.config.log_interval == 0:
                LOGGER.info(
                    "epoch=%d step=%d loss=%.6f accuracy=%.4f",
                    epoch,
                    self.global_step,
                    total_loss / total_examples,
                    total_correct / total_examples,
                )
        return EpochResult(
            loss=total_loss / total_examples,
            accuracy=total_correct / total_examples,
            examples=total_examples,
            steps=steps,
        )

    @torch.no_grad()
    def evaluate(self, loader: DataLoader[Mapping[str, Any]]) -> EpochResult:
        self.model.eval()
        total_loss = 0.0
        total_correct = 0
        total_examples = 0
        steps = 0
        for batch in loader:
            images = batch["image"].to(self.device)
            labels = batch["grade"].to(self.device)
            with self._autocast():
                logits = self.model(images)
                loss = self.criterion(logits, labels)
            count = labels.numel()
            total_loss += float(loss) * count
            total_correct += int((logits.argmax(1) == labels).sum())
            total_examples += count
            steps += 1
        return EpochResult(
            loss=total_loss / total_examples,
            accuracy=total_correct / total_examples,
            examples=total_examples,
            steps=steps,
        )

    def fit(
        self,
        train_loader: DataLoader[Mapping[str, Any]],
        validation_loader: DataLoader[Mapping[str, Any]],
    ) -> list[tuple[EpochResult, EpochResult]]:
        history: list[tuple[EpochResult, EpochResult]] = []
        for epoch in range(self.config.epochs):
            train_result = self.train_epoch(train_loader, epoch)
            validation_result = self.evaluate(validation_loader)
            history.append((train_result, validation_result))
            LOGGER.info(
                "epoch=%d train_loss=%.6f validation_loss=%.6f validation_accuracy=%.4f",
                epoch,
                train_result.loss,
                validation_result.loss,
                validation_result.accuracy,
            )
            if validation_result.accuracy > self.best_metric:
                self.best_metric = validation_result.accuracy
                if self.config.checkpoint_path is not None:
                    save_checkpoint(
                        self.config.checkpoint_path,
                        self.model,
                        self.optimizer,
                        self.scheduler,
                        epoch,
                        self.global_step,
                        self.config.seed,
                        self.best_metric,
                    )
        return history
