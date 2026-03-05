# Dataset Download Guide

This directory stores raw tabular datasets used by this project.

- `dataset/diffprep/`: datasets from the DiffPrep benchmark setting.
- `dataset/deepline/`: datasets from the DeepLine benchmark setting.

## Source
Both benchmark groups are downloaded from **OpenML**.

- OpenML portal: [https://www.openml.org/](https://www.openml.org/)
- OpenML Python API docs: [https://openml.github.io/openml-python/main/](https://openml.github.io/openml-python/main/)

## 1) DiffPrep datasets
The current project already includes the DiffPrep dataset ID mapping in
`mcts_refactor/common.py` (`DIFFPREP_DATASETS`).

OpenML dataset IDs used in this project:

- `obesity`: `40966`
- `eeg`: `1471`
- `ada_prior`: `715`
- `pbcseq`: `826`
- `jungle_chess_2pcs_raw_endgame_complete`: `182`
- `wall-robot-navigation`: `1504`
- `USCensus`: `4534`
- `house_prices`: `42165`
- `page-blocks`: `30`
- `microaggregation2`: `41156`
- `Run_or_walk_information`: `41161`
- `connect-4`: `40668`
- `shuttle`: `40685`
- `mozilla4`: `1045`
- `avila`: `40701`
- `google`: `41162`
- `pol`: `722`
- `abalone`: `183`

Recommended output location: `dataset/diffprep/`

## 2) DeepLine datasets
Use the OpenML dataset IDs released by the DeepLine experimental setup and
save them into `dataset/deepline/`.

If your DeepLine split/config already contains dataset IDs, download each ID
from OpenML and export to CSV into this folder.

## 3) Minimal download example (OpenML API)
```python
import openml

# example: one dataset
dataset = openml.datasets.get_dataset(40966)
X, y, _, _ = dataset.get_data(target=dataset.default_target_attribute)
df = X.copy()
df[dataset.default_target_attribute] = y

# save
import os
os.makedirs("dataset/diffprep", exist_ok=True)
df.to_csv("dataset/diffprep/40966.csv", index=False)
```

Install the helper package when needed:
```bash
pip install openml
```
