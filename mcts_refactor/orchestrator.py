import os

from mcts_refactor.common import logger
from mcts_refactor.recommendation import recommend_from_knowledge
from mcts_refactor.refactor_stage import refactor_candidates
from mcts_refactor.search_stage import run_search_stage


def _build_component_maps(model_type):
    i_map = ["mean", "median", "most_frequent", "knn"]
    o_map = [None, "zscore", "iqr"]
    e_map = ["onehot", "label", "binary", "frequency"]
    n_map = [None, "standard", "minmax", "maxabs", "robust", "normalizer", "power", "quantile", "kbins"]
    ft_map = [None, "pca", "kernalPCA", "polynomial", "InteractionFeatures", "IncrementalPCA", "TruncatedSVD", "RandomTrees"]
    fs_map = [None, "variance", "univariate", "pearson"]

    if model_type == "DecisionTree":
        n_map = [None]
        o_map = [None]
        ft_map = [None, "pca", "polynomial"]
    elif model_type in ("KNN", "SVM"):
        ft_map = [None, "pca", "kernel_pca"]
        e_map = ["onehot", "binary"]
    elif model_type == "LogisticRegression":
        i_map = ["mean"]
        o_map = ["zscore", "iqr"]
        e_map = ["onehot", "label"]
        fs_map = ["variance"]

    return {
        "I": i_map,
        "O": o_map,
        "E": e_map,
        "N": n_map,
        "FT": ft_map,
        "FS": fs_map,
    }


def _resolve_knowledge_paths(model_type, metric, knowledge_csv=None, knowledge_json_dir=None):
    if knowledge_csv and knowledge_json_dir:
        return knowledge_csv, knowledge_json_dir
    if knowledge_csv or knowledge_json_dir:
        raise ValueError("knowledge_csv and knowledge_json_dir must be provided together.")

    csv_path = os.path.join("Knowledge", "metaldata", "Newmetal_data.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            "Shared metadata file not found: Knowledge/metaldata/Newmetal_data.csv."
        )

    metric_candidates = [metric] if metric == "accuracy" else [metric, "accuracy"]

    for m in metric_candidates:
        json_dir = os.path.join("Knowledge", "model_metric", model_type, m, "json")
        if os.path.exists(csv_path) and os.path.isdir(json_dir):
            return csv_path, json_dir

    raise FileNotFoundError(
        "Knowledge base not found. Please provide data under "
        "Knowledge/model_metric/<model>/<metric>/json and ensure "
        "Knowledge/metaldata/Newmetal_data.csv exists, or explicitly pass "
        "knowledge_csv and knowledge_json_dir."
    )


def search_optimal_pipeline_by_MCTS(
    data,
    model_type="DecisionTree",
    metric="accuracy",
    max_iter=30,
    dataset_id=None,
    dataset_name=None,
    subset_size=5000,
    knowledge_csv=None,
    knowledge_json_dir=None,
    show_stages=True,
    return_stage_details=False,
):
    component_maps = _build_component_maps(model_type)
    resolved_knowledge_csv, resolved_knowledge_json_dir = _resolve_knowledge_paths(
        model_type=model_type,
        metric=metric,
        knowledge_csv=knowledge_csv,
        knowledge_json_dir=knowledge_json_dir,
    )

    stage_details = {
        "stage1_recommend": {},
        "stage2_refactor": {},
        "stage3_search": {},
    }

    if show_stages:
        logger.info("========== Stage 1: Pipeline Recommendation (similar datasets + historical candidates) ==========")
    stage1 = recommend_from_knowledge(
        data=data,
        dataset_id=dataset_id,
        knowledge_csv=resolved_knowledge_csv,
        knowledge_json_dir=resolved_knowledge_json_dir,
    )
    stage_details["stage1_recommend"]["similar_dataset_ids"] = stage1["similar_dataset_ids"]
    stage_details["stage1_recommend"]["meta_flags"] = stage1["meta_flags"]
    stage_details["stage1_recommend"]["knowledge_csv"] = resolved_knowledge_csv
    stage_details["stage1_recommend"]["knowledge_json_dir"] = resolved_knowledge_json_dir

    if show_stages:
        logger.info("========== Stage 2: Pipeline Refactoring (cleaning + deduplication) ==========")
    need_imv, need_o, need_e = stage1["need_flags"]
    stage2 = refactor_candidates(
        pipe_configs=stage1["pipe_configs"],
        config_pair=stage1["config_pair"],
        need_imv=need_imv,
        need_o=need_o,
        need_e=need_e,
    )
    stage_details["stage2_refactor"].update(stage2["stage2_stats"])
    stage_details["stage1_recommend"].update(stage2["stage1_candidate_stats"])

    if show_stages:
        logger.info("========== Stage 3: Pipeline Search (MCTS optimization for physical operators) ==========")
    best_pipeline, best_score, stage3_details = run_search_stage(
        data=data,
        model_type=model_type,
        max_iter=max_iter,
        subset_size=subset_size,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        candidate_config_pair=stage2["candidate_config_pair"],
        unique_logic=stage2["unique_logic"],
        component_maps=component_maps,
    )
    stage_details["stage3_search"].update(stage3_details)

    if return_stage_details:
        return best_pipeline, best_score, stage_details
    return best_pipeline, best_score
