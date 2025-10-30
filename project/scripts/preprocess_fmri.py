"""Entry point for the preprocessing stage."""
from __future__ import annotations

import argparse

from project.src.dataio.preprocess import preprocess_dataset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=False, default="configs/dataset.yaml")
    args = parser.parse_args()
    preprocess_dataset({"config_path": args.config})


if __name__ == "__main__":
    main()
