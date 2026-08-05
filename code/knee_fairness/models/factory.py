from __future__ import annotations

from dataclasses import dataclass

from torch import nn

from knee_fairness.models.densenet import KneeDenseNet
from knee_fairness.models.ensemble import ProbabilityEnsemble
from knee_fairness.models.pim import PluginInteractionModel
from knee_fairness.models.siamese import SiameseKneeClassifier
from knee_fairness.models.vgg_ordinal import OrdinalVGG19


@dataclass(frozen=True)
class ModelSpec:
    name: str
    family: str
    reported_parameters_millions: float
    reported_accuracy: float
    reported_weighted_kappa: float | None


def model_specs() -> tuple[ModelSpec, ...]:
    return (
        ModelSpec("siamese", "Siamese CNN", 2.0, 0.664, 0.82),
        ModelSpec("vgg_ordinal", "VGG ordinal", 144.0, 0.698, 0.84),
        ModelSpec("densenet", "DenseNet", 8.0, 0.713, 0.86),
        ModelSpec("ensemble", "multi-architecture ensemble", 300.0, 0.768, 0.87),
        ModelSpec("ensemble_pim", "Swin and EfficientNet interaction", 200.0, 0.772, 0.87),
    )


def build_model(name: str, num_classes: int = 5, compact: bool = False) -> nn.Module:
    if name == "siamese":
        return SiameseKneeClassifier(num_classes, width=8 if compact else 32)
    if name == "vgg_ordinal":
        return OrdinalVGG19(num_classes)
    if name == "densenet":
        counts = (2, 2, 2, 2) if compact else (6, 12, 24, 16)
        growth = 8 if compact else 24
        initial = 16 if compact else 48
        return KneeDenseNet(num_classes, growth, counts, initial)
    if name == "ensemble_pim":
        return PluginInteractionModel(num_classes, 64 if compact else 256)
    if name == "ensemble":
        members = [
            SiameseKneeClassifier(num_classes, width=8 if compact else 32),
            KneeDenseNet(
                num_classes,
                8 if compact else 24,
                (2, 2, 2, 2) if compact else (6, 12, 24, 16),
                16 if compact else 48,
            ),
            PluginInteractionModel(num_classes, 64 if compact else 256),
        ]
        return ProbabilityEnsemble(members)
    raise ValueError(f"Unknown model: {name}")
