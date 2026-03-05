import ast
import math
import os
import random
import warnings

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import accuracy_score
from sklearn.model_selection import StratifiedKFold, train_test_split

from NewOperators import build_pipeline, build_pipeline_test, get_heuristic_table, support_models
from heuristic import update_heuristic_table
from mcts_refactor.common import get_dataset_name_by_id, logger


class MCTSNode:
    def __init__(self, depth=0, config=None, parent=None):
        self.depth = depth
        self.config = config or []
        self.children = []
        self.parent = parent
        self.visits = 0
        self.score_sum = 0.0

    def ucb(self, c=2):
        if self.visits == 0:
            return float("inf")
        return self.score_sum + c * math.sqrt(math.log(self.parent.visits) / self.visits)

    def is_fully_expanded(self, choices):
        return len(self.children) == len(choices)


def weighted_choice(choices, scores, temperature=0.5, default_score=0.0):
    if not choices:
        raise ValueError("weighted_choice: choices cannot be empty")

    raw = [scores.get(a, default_score) for a in choices]
    if len(set(raw)) == 1:
        return random.choice(choices)

    if temperature is not None:
        max_s = max(raw)
        exps = [math.exp((s - max_s) / temperature) for s in raw]
        total = sum(exps)
        if total <= 0 or not math.isfinite(total):
            return random.choice(choices)
        probs = [e / total for e in exps]
    else:
        weights = [max(s, 0.0) for s in raw]
        total = sum(weights)
        if total <= 0:
            return random.choice(choices)
        probs = [w / total for w in weights]

    return random.choices(choices, probs)[0]


def build_pipeline_config(order, config):
    pipeline_config = {"flow": order}
    for name, value in zip(order, config):
        if name == "I":
            pipeline_config["I"] = {"strategy": value}
        elif name == "O":
            pipeline_config["O"] = {"method": value, "threshold": 3}
        elif name == "E":
            pipeline_config["E"] = {"method": value}
        elif name == "N":
            pipeline_config["N"] = {"method": value}
        elif name == "FT":
            pipeline_config["FT"] = {"method": value, "n_components": 2}
        elif name == "FS":
            pipeline_config["FS"] = {
                "method": value,
                "var_threshold": 0,
                "pearson_threshold": 0.1,
            }
    return pipeline_config


def format_best_pipeline(order, config):
    return build_pipeline_config(order, config)


def simulate_once(data, config_pair, model_type):
    warnings.filterwarnings("ignore")
    order, config = config_pair
    pipeline_config = build_pipeline_config(order, config)
    train_data, valid_data = train_test_split(data, test_size=0.25, random_state=0)

    try:
        model = support_models[model_type]
        processed_train, selected_features, _, encoding_feature, lda_transformer, context = build_pipeline(
            train_data, pipeline_config
        )
        processed_valid = build_pipeline_test(
            valid_data,
            pipeline_config,
            selected_features,
            encoding_feature=encoding_feature,
            lda_transformer=lda_transformer,
            context=context,
        )
        x_train = processed_train.iloc[:, :-1]
        y_train = processed_train.iloc[:, -1]
        x_valid = processed_valid.iloc[:, :-1]
        y_valid = processed_valid.iloc[:, -1]
        model.fit(x_train, y_train)
        y_pred = model.predict(x_valid)
        return accuracy_score(y_valid, y_pred)
    except Exception:
        return -1


def simulate_k_cv(data, config_pair, model_type, k=4):
    warnings.filterwarnings("ignore")
    order, config = config_pair
    pipeline_config = build_pipeline_config(order, config)

    x = data.iloc[:, :-1]
    y = data.iloc[:, -1]
    cv_scores = []
    skf = StratifiedKFold(n_splits=k, shuffle=True, random_state=0)

    for train_idx, valid_idx in skf.split(x, y):
        try:
            train_data = data.iloc[train_idx]
            valid_data = data.iloc[valid_idx]

            processed_train, selected_features, _, encoding_feature, lda_transformer, context = build_pipeline(
                train_data, pipeline_config
            )
            processed_valid = build_pipeline_test(
                valid_data,
                pipeline_config,
                selected_features,
                encoding_feature=encoding_feature,
                lda_transformer=lda_transformer,
                context=context,
            )
            x_train = processed_train.iloc[:, :-1]
            y_train = processed_train.iloc[:, -1]
            x_valid = processed_valid.iloc[:, :-1]
            y_valid = processed_valid.iloc[:, -1]

            model = support_models[model_type]
            model.fit(x_train, y_train)
            y_pred = model.predict(x_valid)
            cv_scores.append(accuracy_score(y_valid, y_pred))
        except Exception:
            cv_scores.append(-1)

    valid_scores = [s for s in cv_scores if s >= 0]
    return float(np.mean(valid_scores)) if valid_scores else -1.0


def evaluate_logic_candidates(data, candidate_config_pair, model_type):
    def evaluate_config_pair(config_pair):
        try:
            return config_pair, simulate_once(data, config_pair, model_type)
        except Exception as e:
            warnings.warn(f"Config {config_pair} failed with error: {e}")
            return config_pair, -1

    if not candidate_config_pair:
        return []

    jobs = max(1, len(candidate_config_pair))
    return Parallel(n_jobs=jobs)(delayed(evaluate_config_pair)(cfg) for cfg in candidate_config_pair)


def run_search_stage(
    data,
    model_type,
    max_iter,
    subset_size,
    dataset_id,
    candidate_config_pair,
    unique_logic,
    component_maps,
    dataset_name=None,
):
    """Stage 3: score logic candidates, run MCTS for top logic pipelines, finalize by CV."""
    if data.shape[0] > 10000:
        data = data.sample(n=subset_size, random_state=0).reset_index(drop=True)

    results = evaluate_logic_candidates(data, candidate_config_pair, model_type)
    top_k = min(len(results), 3)
    top_results = sorted(results, key=lambda x: x[1], reverse=True)[:top_k]

    if not top_results:
        raise RuntimeError("No valid candidate pipeline found in stage 3.")

    def _run_single(idx_cfg_score):
        idx, (cfg_pair, logic_score) = idx_cfg_score
        order = cfg_pair[0]
        seed_cfg = cfg_pair[1]

        def _simulate_once_local(config_pair):
            return simulate_once(data, config_pair, model_type)

        final_best_order = order
        root = MCTSNode()
        space = [component_maps[name] for name in final_best_order]
        epsilon = 0.2
        patience = 50
        no_improve_rounds = 0
        final_best_score = float("-inf")
        final_best_config = None

        dataset_names = unique_logic.get(tuple(order), tuple())
        action_score_table = get_heuristic_table(dataset_names=dataset_names, target=order, top_k=500)
        history_log = []

        for iteration in range(1, max_iter + 1):
            if no_improve_rounds >= patience:
                break

            if iteration == 1 and seed_cfg:
                current_config = seed_cfg
                node = root
                for depth, action in enumerate(current_config):
                    child = MCTSNode(depth=depth + 1, config=[action], parent=node)
                    node.children.append(child)
                    node = child
                score = logic_score
            else:
                node = root
                current_config = []

                for depth, choices in enumerate(space):
                    if not node.is_fully_expanded(choices):
                        tried = [child.config[0] for child in node.children]
                        untried = [c for c in choices if c not in tried]

                        if random.random() < epsilon:
                            action = random.choice(untried)
                        else:
                            if depth == 0:
                                heuristic_scores = action_score_table.get("__START__", {})
                            else:
                                heuristic_scores = action_score_table.get(current_config[-1], {})
                            action = weighted_choice(untried, heuristic_scores)

                        current_config.append(action)
                        child = MCTSNode(depth=depth + 1, config=[action], parent=node)
                        node.children.append(child)
                        node = child
                        break
                    else:
                        node = max(node.children, key=lambda n: n.ucb())
                        current_config.extend(node.config)

                while len(current_config) < len(space):
                    choices = space[len(current_config)]
                    action = weighted_choice(choices, action_score_table.get(current_config[-1], {}))
                    child = MCTSNode(depth=node.depth + 1, config=[action], parent=node)
                    node.children.append(child)
                    node = child
                    current_config.append(action)

                score = _simulate_once_local((order, current_config))
                update_heuristic_table(
                    action_score_table,
                    current_config,
                    score,
                    final_best_score,
                    allow_start_negative=False,
                    allow_negative=True,
                    alpha=1.0,
                )

            temp = node
            while temp:
                temp.visits += 1
                temp.score_sum = max(temp.score_sum, score)
                temp = temp.parent

            if score > final_best_score:
                final_best_score = score
                final_best_config = current_config
                no_improve_rounds = 0
            else:
                no_improve_rounds += 1

            history_log.append(
                {
                    "run_id": idx,
                    "iteration": iteration,
                    "order": "_".join(order),
                    "best_score": final_best_score,
                    "best_config": final_best_config,
                    "current_score": score,
                    "current_config": current_config,
                }
            )

        return format_best_pipeline(order, final_best_config), final_best_score, history_log

    results_parallel = Parallel(n_jobs=top_k)(
        delayed(_run_single)(item) for item in enumerate(top_results)
    )

    all_logs = []
    for _, _, history_log in results_parallel:
        all_logs.extend(history_log)

    fpath = f"logs/kamcts_subset{subset_size}"
    os.makedirs(fpath, exist_ok=True)
    os.makedirs(f"{fpath}/iter_detail", exist_ok=True)

    df_all = pd.DataFrame(all_logs)
    dataset_tag = dataset_name or get_dataset_name_by_id(dataset_id=dataset_id) or "dataset"
    path_detail = f"{fpath}/iter_detail/{dataset_tag}_{model_type}_{max_iter}_detail.csv"
    df_all.to_csv(path_detail, index=False)

    global_curve = []
    for it in sorted(df_all["iteration"].unique()):
        subset = df_all[df_all["iteration"] == it]
        best_row = subset.loc[subset["best_score"].idxmax()]
        global_curve.append(
            {
                "iteration": it,
                "global_best_score": best_row["best_score"],
                "best_order": best_row["order"],
                "best_config": best_row["best_config"],
            }
        )

    df_global = pd.DataFrame(global_curve)
    os.makedirs(f"{fpath}/global_iter_curve", exist_ok=True)
    path_global = f"{fpath}/global_iter_curve/{dataset_tag}_{model_type}_{max_iter}_global_iter_curve.csv"
    df_global.to_csv(path_global, index=False)

    df = df_all[["order", "best_score", "best_config"]].copy()
    df["best_config_tuple"] = df["best_config"].apply(
        lambda x: tuple(ast.literal_eval(x)) if isinstance(x, str) else tuple(x)
    )
    dedup = (
        df.sort_values("best_score", ascending=False)
        .groupby("best_config_tuple", as_index=False)
        .first()
    )
    dedup = dedup[["order", "best_score", "best_config"]]

    top5 = dedup.sort_values("best_score", ascending=False).head(5).reset_index(drop=True)
    score_max = top5.loc[0, "best_score"]
    final_candidates = top5[(top5["best_score"] >= score_max - 0.01)].reset_index(drop=True)

    cv_candidates = []
    for _, row in final_candidates.iterrows():
        order = row["order"].split("_")
        config = list(ast.literal_eval(row["best_config"])) if isinstance(row["best_config"], str) else list(row["best_config"])
        assert len(order) == len(config), f"Order/config length mismatch: {order} vs {config}"
        cv_candidates.append((order, config))

    cv_score = -1
    best = None
    for cand in cv_candidates:
        k_cv = simulate_k_cv(data, cand, model_type)
        if k_cv > cv_score:
            cv_score = k_cv
            best = cand

    best_pipeline = format_best_pipeline(best[0], best[1])

    stage3_details = {
        "top_logic_results": [
            {"order": cfg[0], "seed_ops": cfg[1], "score": score} for cfg, score in top_results
        ],
        "final_cv_candidates": [{"order": cand[0], "config": cand[1]} for cand in cv_candidates],
        "best_pipeline": best_pipeline,
        "best_score": float(cv_score),
    }

    logger.info(f"Stage 3 complete. Best pipeline: {best_pipeline}, score: {cv_score}")
    return best_pipeline, cv_score, stage3_details
