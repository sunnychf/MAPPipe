import pandas as pd
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from new_operator_core import build_pipeline, build_pipeline_test, support_models
def train_with_subset(processed_data, model_type, subset_size, random_state=0):
    """Train/evaluate a model on a sampled subset of processed data."""
    if processed_data.shape[1] <= 1:
        raise ValueError("processed_data must contain at least 1 feature column and 1 label column.")

    model = support_models[model_type]
    subset_size = min(processed_data.shape[0], subset_size)
    subset = processed_data.sample(n=subset_size, random_state=random_state)

    x = subset.iloc[:, :-1]
    y = subset.iloc[:, -1]
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.25, random_state=random_state)

    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)
    return accuracy_score(y_test, y_pred)


def process_row(row, target):
    logic_list = row["logic_pipe"].split("_")
    inter = [x for x in logic_list if x in target]

    if len(inter) == 0:
        return pd.Series({"logic": None, "phy": None, "acc": None})

    for i in range(len(target)):
        for j in range(i + 1, len(target)):
            a, b = target[i], target[j]
            if a in inter and b in inter and logic_list.index(a) > logic_list.index(b):
                return pd.Series({"logic": None, "phy": None, "acc": None})

    phy_raw = row["phy_pipe"]
    if isinstance(phy_raw, str):
        phy_list = phy_raw.split("_") if "_" in phy_raw else [phy_raw]
    else:
        phy_list = [phy_raw]

    indices = [logic_list.index(x) for x in inter]
    phy_new = [phy_list[i] if i < len(phy_list) else phy_list[-1] for i in indices]

    return pd.Series({"logic": inter, "phy": phy_new, "acc": row["accuracy"]})


def get_heuristic_table(dataset_names, target, csv_path="data/accuracy_of_every_pipe", top_k=50, verbose=False):
    """Build conditional transition heuristic table from historical pipeline results."""
    all_processed_rows = []

    for name in dataset_names:
        file_path = f"{csv_path}/{name}_jieguo.csv"
        try:
            df = pd.read_csv(file_path)
        except FileNotFoundError:
            if verbose:
                print(f"Warning: dataset {name} not found at {file_path}")
            continue

        df_processed = df.apply(lambda row: process_row(row, target), axis=1, result_type="expand")
        df_processed = df_processed.dropna()
        if not df_processed.empty:
            all_processed_rows.append(df_processed)

    if len(all_processed_rows) == 0:
        if verbose:
            print("Warning: no matching sub-pipeline found for the target in all datasets")
        return {}

    df_all = pd.concat(all_processed_rows, ignore_index=True)
    df_topk = df_all.sort_values(by="acc", ascending=False).head(top_k)

    transition_scores = {}
    start = "__START__"

    for _, row in df_topk.iterrows():
        phy = row["phy"]
        acc = row["acc"]

        first = phy[0]
        transition_scores.setdefault(start, {}).setdefault(first, []).append(acc)

        for prev, nxt in zip(phy[:-1], phy[1:]):
            transition_scores.setdefault(prev, {}).setdefault(nxt, []).append(acc)

    action_score_table = {
        prev_phy: {next_phy: max(values) for next_phy, values in next_dict.items()}
        for prev_phy, next_dict in transition_scores.items()
    }
    return action_score_table


__all__ = [
    "support_models",
    "build_pipeline",
    "build_pipeline_test",
    "train_with_subset",
    "process_row",
    "get_heuristic_table",
]
