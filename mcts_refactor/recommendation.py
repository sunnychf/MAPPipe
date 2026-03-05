import json

import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import StandardScaler

from extract_meta_features import extract_meta_features


def recommend_from_knowledge(
    data,
    dataset_id=None,
    knowledge_csv=None,
    knowledge_json_dir=None,
    top_n=5,
):
    """Stage 1: recommend historical logic/physical pipelines from nearest datasets."""
    if not knowledge_csv or not knowledge_json_dir:
        raise ValueError("knowledge_csv and knowledge_json_dir are required and cannot be None.")

    knowledge_database = pd.read_csv(knowledge_csv)
    x = knowledge_database.iloc[:, :-1]
    y = knowledge_database.iloc[:, -1]
    x = x.fillna(-1)

    meta_features, need_imv, need_o, need_e = extract_meta_features(data=data)

    if dataset_id in y.values:
        knowledge_database = knowledge_database[y != dataset_id]
        x = knowledge_database.iloc[:, :-1]
        y = knowledge_database.iloc[:, -1]

    query_df = pd.DataFrame([[meta_features[col] for col in x.columns]], columns=x.columns)
    query_df = query_df.fillna(-1)
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(x)
    query_scaled = scaler.transform(query_df)

    # distances = euclidean_distances(x_scaled, query_scaled).flatten()
    similarities = cosine_similarity(x_scaled, query_scaled).flatten()
    top_indices = np.argsort(similarities)[::-1][:top_n]
    top_labels = y.iloc[top_indices].tolist()

    config_pair = []
    pipe_configs = []
    for name in top_labels:
        fpath = f"{knowledge_json_dir}/{name}.json"
        with open(fpath, mode="r", encoding="utf-8") as file:
            text = json.load(file)
            pipe_list = next(iter(text.values()))
            for item in pipe_list:
                config_pair.append((item[1], item[2], [name]))
                complete_order = ["I"] + item[1]
                pipe_config = dict(zip(complete_order, item[2]))
                pipe_config["source_dataset"] = [name]
                pipe_configs.append(pipe_config)

    return {
        "similar_dataset_ids": [str(x) for x in top_labels],
        "meta_flags": {
            "need_IMV": bool(need_imv),
            "need_O": bool(need_o),
            "need_E": bool(need_e),
        },
        "need_flags": (need_imv, need_o, need_e),
        "config_pair": config_pair,
        "pipe_configs": pipe_configs,
    }
