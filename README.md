# MAPPipe: Navigating to Optimal Data Preprocessing Pipelines via Structural Pruning and Knowledge-augmented MCTS

![MAPPipe Overview](./assets/framework-of-MAPPipe.png)

## Overview
`MAPPipe` is a knowledge-augmented, neural-free framework for automated preprocessing on tabular data.  
It targets the core AutoML bottleneck: finding high-quality preprocessing pipelines under strict evaluation budgets.

1. A knowledge-augmented, surrogate-free architecture that unifies pipeline recommendation, reconfiguration, and search, transforming exponential combinatorial optimization into guided exploration within a high-potential subspace.
2. A pruning strategy based on large-scale empirical analysis to aggressively compress the instantiation space by 99.58% while preserving near-optimal reachability.
3. A customized, neural-free MCTS with heuristic guidance for reliable near-optimal pipeline discovery under limited evaluation budgets.

## Key Ideas
- Three-stage optimization: **recommendation -> reconfiguration -> search**.
- Structural pruning to dramatically reduce search space while preserving high-value candidates.
- Heuristic-guided MCTS without neural surrogate models.
- Shared meta-knowledge for cross-dataset transfer.

## Repository Structure
```text
MAPPipe/
├── kamcts_main.py                                  # Main CLI entry
├── NewOperators.py                                 # Compatibility facade
├── mcts_refactor/
│   ├── recommendation.py                           # Stage 1: pipeline recommendation
│   ├── refactor_stage.py                           # Stage 2: pipeline reconfiguration
│   ├── search_stage.py                             # Stage 3: MCTS search
│   ├── orchestrator.py                             # Pipeline orchestration
│   └── common.py
├── new_operator_core/
│   ├── models.py
│   ├── pipeline_builder.py
│   └── preprocessing/
│       ├── imputation.py
│       ├── outlier.py
│       ├── encoding.py
│       ├── normalization.py
│       ├── feature_transform.py
│       └── feature_selection.py
├── dataset/
│   ├── README.md                                  # Download source and preparation guide
│   ├── diffprep/
│   └── deepline/
└── Knowledge/
    ├── metaldata/
    │   └── Newmetal_data.csv                       # Shared metadata for all downstream models
    └── model_metric/
        ├── LogisticRegression/accuracy/{csv,json}
        ├── KNN/accuracy/{csv,json}
        ├── DecisionTree/accuracy/{csv,json}
        └── SVM/accuracy/{csv,json}
```

## Quick Start
### 1) Create Conda environment
```bash
conda create -n mappipe python=3.10 -y
conda activate mappipe
python -m pip install --upgrade pip
pip install -r requirements.txt
```

Tested runtime:
- Python `3.10`
- See pinned packages in [`requirements.txt`](./requirements.txt)

### 2) Run on a single dataset
```bash
python kamcts_main.py \
  --dataset_path data/diffprep/egg.csv \
  --model LogisticRegression \
  --metric accuracy \
  --max_iter 30 \
  --subset_size 5000 \
  --output_dir output/single_run
```

Optional knowledge overrides:
```bash
python kamcts_main.py \
  --dataset_path path/to/dataset.csv \
  --knowledge_csv Knowledge/metaldata/Newmetal_data.csv \
  --knowledge_json_dir Knowledge/model_metric/LogisticRegression/accuracy/json
```

If not provided, MAPPipe auto-resolves:
- metadata: `Knowledge/metaldata/Newmetal_data.csv`
- profile json: `Knowledge/model_metric/<model>/<metric>/json`

## Input/Output
### Input assumptions
- CSV tabular data.
- Target column is the last column by default (or set `--target_column`).
- Main workflow splits train/test with ratio **4:1**.

### Outputs (`--output_dir`)
- `processed_train.csv`
- `processed_test.csv`
- `best_pipeline.json`
- `run_report.json`

## Reproducibility Notes
- Default seed is fixed in entry (`random.seed(0)`, plus deterministic split/CV seeds where used).
- For fair comparison, keep identical train/test split policy and search budget (`max_iter`, `subset_size`).
- Recommended dataset roots in this project: `dataset/diffprep`, `dataset/deepline`.
- Dataset download sources and OpenML-based preparation are documented in [`dataset/README.md`](./dataset/README.md).

## Citation
If you use this repository in research, please cite:

```bibtex
@misc{mappipe2026,
  title={MAPPipe: Navigating to Optimal Data Preprocessing Pipelines via Structural Pruning and Knowledge-augmented MCTS},
  year={2026}
}
```
