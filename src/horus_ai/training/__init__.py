"""Offline detector training and artifact creation."""

from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from .train import (
        TrainingConfig,
        TrainingResult,
        load_training_artifact,
        metadata_path_for,
        train,
    )

__all__ = [
    "TrainingConfig",
    "TrainingResult",
    "load_training_artifact",
    "metadata_path_for",
    "train",
]


def __getattr__(name):
    if name not in __all__:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from importlib import import_module

    training_module = import_module(".train", __name__)
    return getattr(training_module, name)
