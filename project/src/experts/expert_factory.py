"""Factory for constructing edge experts."""
from __future__ import annotations

from typing import Dict, List

from project.src.experts.expert_cross_system import CrossSystemExpert
from project.src.experts.expert_direct_coupling import DirectCouplingExpert
from project.src.experts.expert_linear_sync import LinearSyncExpert
from project.src.experts.expert_nonlinear import NonlinearExpert


def build_experts(cfg: Dict[str, int | float]) -> List:
    """Instantiate the four experts required by the routing pipeline."""

    top_k = int(cfg.get("top_k", 200))
    experts = [
        LinearSyncExpert(top_k=top_k, expert_id=0),
        DirectCouplingExpert(top_k=top_k, expert_id=1),
        NonlinearExpert(top_k=top_k, expert_id=2),
        CrossSystemExpert(top_k=top_k, expert_id=3),
    ]
    return experts
