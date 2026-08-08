from __future__ import annotations

from importlib import import_module

__all__ = [
    "TaskSpec",
    "train_model",
    "save_checkpoint",
    "load_checkpoint",
    "build_scheduler",
    "masked_ce_loss",
    "_ACTION_LABELS",
    "_MERGE_MAP",
    "_action_type_name",
]

_SPEC_EXPORTS = {"TaskSpec"}
_TRAINER_EXPORTS = {
    "train_model",
    "save_checkpoint",
    "load_checkpoint",
    "build_scheduler",
    "masked_ce_loss",
    "_ACTION_LABELS",
    "_MERGE_MAP",
    "_action_type_name",
}


def __getattr__(name: str):
    if name in _SPEC_EXPORTS:
        module = import_module("training.specs")
        value = getattr(module, name)
        globals()[name] = value
        return value
    if name in _TRAINER_EXPORTS:
        module = import_module("training.trainer")
        value = getattr(module, name)
        globals()[name] = value
        return value
    raise AttributeError(f"module 'training' has no attribute {name!r}")
