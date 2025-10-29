"""Build multi-view graph caches."""

from __future__ import annotations

import argparse
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--model", type=Path, required=True)
    args = parser.parse_args()
    print(f"Building graph cache with dataset={args.dataset} model={args.model}")


if __name__ == "__main__":  # pragma: no cover
    main()
