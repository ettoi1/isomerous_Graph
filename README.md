# isomerous Graph

This repository implements a modular pipeline for building multi-view graphs
from resting-state fMRI data and training a Graph Transformer with
mixture-of-experts edge gating.  The code is organised into packages that mirror
the end-to-end workflow – from preprocessing to interpretability.

## Directory layout

```
project/
  configs/            # YAML configuration files for dataset/model/train
  data/               # Expected data caches (raw/interim/processed)
  src/                # Python source tree organised by responsibility
  scripts/            # Command-line entry points for the main stages
```

Each subpackage under `src/` exposes a documented API that other modules can
consume without tightly coupling to concrete implementations.

## Quick start

1. Preprocess the dataset to extract ROI time-series:

   ```bash
   python scripts/preprocess_fmri.py --config configs/dataset.yaml
   ```

2. Build multi-view graph caches:

   ```bash
   python scripts/build_graph_cache.py --dataset configs/dataset.yaml --model configs/model.yaml
   ```

3. Launch training:

   ```bash
   python scripts/train.py --dataset configs/dataset.yaml --model configs/model.yaml --train configs/train.yaml
   ```

4. Evaluate a trained checkpoint:

   ```bash
   python scripts/eval.py --ckpt path/to/checkpoint.ckpt --eval-config configs/train.yaml
   ```

## Development notes

- Optional dependencies such as `nibabel` and `nilearn` are imported lazily so
  unit tests can run with a light-weight environment.
- The mixture-of-experts gate supports soft and top-k routing, exposing both the
  fused edge weights and expert allocation probabilities for interpretability.
- Structural encodings follow the Graphormer design and are implemented in
  dedicated helper functions to ease experimentation.
