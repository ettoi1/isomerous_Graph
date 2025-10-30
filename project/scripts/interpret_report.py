"""Generate a dummy interpretability report."""
from __future__ import annotations

import torch

from project.src.interpret.brain_system_report import render_report


def main():
    z_comm = torch.randn(1, 32)
    z_graph = torch.randn(1, 32)
    print(render_report(z_comm, z_graph))


if __name__ == "__main__":
    main()
