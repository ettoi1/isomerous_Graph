"""Experiment plan helpers."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product
from typing import Dict, Iterable, List


@dataclass
class AblationSetting:
    config: Dict[str, object]


def build_ablation_plan(options: Dict[str, Iterable[object]]) -> List[AblationSetting]:
    keys = list(options.keys())
    plan = []
    for values in product(*options.values()):
        config = dict(zip(keys, values))
        plan.append(AblationSetting(config=config))
    return plan


__all__ = ["AblationSetting", "build_ablation_plan"]
