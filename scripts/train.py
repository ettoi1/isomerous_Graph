"""Training entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    parser.add_argument("--train", type=Path, required=True)
    args = parser.parse_args()
    print(f"Training with dataset={args.dataset} model={args.model} train={args.train}")


if __name__ == "__main__":  # pragma: no cover
    main()
