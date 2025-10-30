"""Generate a toy textual report."""
from __future__ import annotations


def render_report(z_comm, z_graph):
    return {
        "z_comm_norm": float(z_comm.norm(dim=-1).mean().item()),
        "z_graph_norm": float(z_graph.norm(dim=-1).mean().item()),
    }
