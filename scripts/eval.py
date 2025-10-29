"""Evaluation entry point."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--ckpt", type=Path, required=True)
    parser.add_argument("--eval-config", type=Path, required=True)
    args = parser.parse_args()
    print(f"Evaluating checkpoint={args.ckpt} using config={args.eval_config}")


if __name__ == "__main__":  # pragma: no cover
    main()
