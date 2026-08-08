from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, Literal

import torch
import torch.nn as nn


BatchT = Any
UnpackBatchFn = Callable[[BatchT, torch.device], Dict[str, Any]]
ComputeExtraLossFn = Callable[
    [nn.Module, torch.device, Dict[str, Any], bool, int],
    tuple[torch.Tensor, Dict[str, float]],
]


@dataclass(frozen=True)
class TaskSpec:
    name: str
    unpack_batch: UnpackBatchFn
    compute_extra_loss: ComputeExtraLossFn
    log_metric_keys: tuple[str, ...] = ()
    best_metric_name: str = "ce"
    best_metric_mode: Literal["min", "max"] = "min"
