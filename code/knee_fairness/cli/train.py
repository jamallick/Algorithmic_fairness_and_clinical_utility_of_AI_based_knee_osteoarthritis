from __future__ import annotations

import argparse
import logging
from pathlib import Path

import torch
from torch import nn
from torch.utils.data import DataLoader

from knee_fairness.config import load_config
from knee_fairness.data.dataset import KneeRadiographDataset
from knee_fairness.data.manifest import read_manifest
from knee_fairness.data.splits import participant_split
from knee_fairness.losses import OrdinalCrossEntropy
from knee_fairness.models import build_model
from knee_fairness.training import Trainer, TrainerConfig
from knee_fairness.training.optim import build_optimizer, build_scheduler
from knee_fairness.training.seed import set_seed, worker_seed


def parser() -> argparse.ArgumentParser:
    value = argparse.ArgumentParser(prog="knee-fairness-train")
    value.add_argument("--config", type=Path, required=True)
    value.add_argument("--manifest", type=Path, required=True)
    value.add_argument("--output", type=Path, required=True)
    value.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return value


def main() -> None:
    arguments = parser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = load_config(arguments.config)
    set_seed(config.seed)
    records = read_manifest(arguments.manifest)
    split = participant_split(
        records,
        config.test_fraction,
        config.validation_fraction,
        config.seed,
    )
    train_dataset = KneeRadiographDataset(split.train, config.image_size, augment=True)
    validation_dataset = KneeRadiographDataset(split.validation, config.image_size)
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.batch_size,
        shuffle=True,
        num_workers=config.workers,
        pin_memory=True,
        worker_init_fn=worker_seed,
    )
    validation_loader = DataLoader(
        validation_dataset,
        batch_size=config.batch_size,
        num_workers=config.workers,
        pin_memory=True,
        worker_init_fn=worker_seed,
    )
    model = build_model(config.model, config.num_classes)
    criterion: nn.Module = OrdinalCrossEntropy(config.num_classes, 0.5, 0.05)
    optimizer = build_optimizer(
        model.parameters(),
        config.optimizer,
        config.learning_rate,
        config.weight_decay,
    )
    total_steps = config.epochs * max(1, len(train_loader))
    warmup_steps = config.warmup_epochs * max(1, len(train_loader))
    scheduler = build_scheduler(optimizer, config.scheduler, total_steps, warmup_steps)
    trainer = Trainer(
        model,
        criterion,
        optimizer,
        scheduler,
        torch.device(arguments.device),
        TrainerConfig(
            epochs=config.epochs,
            gradient_clip_norm=config.grad_clip_norm,
            precision=config.precision,
            checkpoint_path=arguments.output,
            seed=config.seed,
        ),
    )
    trainer.fit(train_loader, validation_loader)


if __name__ == "__main__":
    main()
