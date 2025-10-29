"""Entry point for preprocessing raw fMRI volumes."""

from __future__ import annotations

import argparse
from pathlib import Path

from src.dataio import preprocess


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    args = parser.parse_args()
    print(f"Preprocessing with config {args.config}")


if __name__ == "__main__":  # pragma: no cover
    main()
