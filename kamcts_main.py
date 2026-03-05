import argparse
import glob
import json
import os
import random
import time
import warnings

import pandas as pd
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from mcts_refactor.common import logger
from mcts_refactor.orchestrator import search_optimal_pipeline_by_MCTS
from NewOperators import build_pipeline, build_pipeline_test, support_models


def _infer_dataset_name(dataset_path):
    return os.path.splitext(os.path.basename(dataset_path))[0]


def preprocess_data(
    dataset_path,
    model,
    metric="accuracy",
    max_iter=100,
    dataset_id=None,
    dataset_name=None,
    subset_size=2000,
    knowledge_csv=None,
    knowledge_json_dir=None,
    target_column=None,
    random_state=0,
):
    """Load one CSV, split train/test by 4:1, search best pipeline, and return processed splits."""
    data = pd.read_csv(dataset_path)
    if data.shape[1] < 2:
        raise ValueError("Dataset must contain at least 2 columns (features + target).")

    if target_column is None:
        target_column = data.columns[-1]
    if target_column not in data.columns:
        raise ValueError(f"Target column `{target_column}` is not present in the dataset.")
    if data.columns[-1] != target_column:
        reordered_cols = [c for c in data.columns if c != target_column] + [target_column]
        data = data[reordered_cols]

    y_all = data.iloc[:, -1]
    stratify = y_all if y_all.nunique() > 1 else None
    train_data, test_data = train_test_split(
        data, test_size=0.2, random_state=random_state, stratify=stratify
    )
    dataset_name = dataset_name or _infer_dataset_name(dataset_path)

    best_pipeline, best_score = search_optimal_pipeline_by_MCTS(
        train_data,
        model_type=model,
        metric=metric,
        max_iter=max_iter,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        subset_size=subset_size,
        knowledge_csv=knowledge_csv,
        knowledge_json_dir=knowledge_json_dir,
        show_stages=True,
        return_stage_details=False,
    )

    processed_train, selected_features, _, encoding_feature, lda_transformer, context = build_pipeline(
        train_data, best_pipeline
    )
    processed_test = build_pipeline_test(
        test_data,
        best_pipeline,
        selected_features,
        encoding_feature=encoding_feature,
        lda_transformer=lda_transformer,
        context=context,
    )

    x_train = processed_train.iloc[:, :-1]
    y_train = processed_train.iloc[:, -1]
    x_test = processed_test.iloc[:, :-1]
    y_test = processed_test.iloc[:, -1]
    return x_train, y_train, x_test, y_test, best_pipeline, best_score


def preprocess_single_dataset(
    dataset_path,
    model="LogisticRegression",
    metric="accuracy",
    max_iter=30,
    subset_size=5000,
    dataset_id=None,
    dataset_name=None,
    knowledge_csv=None,
    knowledge_json_dir=None,
    output_dir="output/single_run",
    target_column=None,
    test_size=0.2,
    random_state=1,
):
    """Single-dataset entry for AutoML-style preprocessing search and export."""
    if model not in support_models:
        raise ValueError(f"Unsupported model: {model}. Available options: {list(support_models.keys())}")

    data = pd.read_csv(dataset_path)
    if data.shape[1] < 2:
        raise ValueError("Dataset must contain at least 2 columns (features + target).")

    if target_column is None:
        target_column = data.columns[-1]
    if target_column not in data.columns:
        raise ValueError(f"Target column `{target_column}` is not present in the dataset.")
    if data.columns[-1] != target_column:
        reordered_cols = [c for c in data.columns if c != target_column] + [target_column]
        data = data[reordered_cols]

    y_all = data.iloc[:, -1]
    stratify = y_all if y_all.nunique() > 1 else None
    train_data, test_data = train_test_split(
        data, test_size=test_size, random_state=random_state, stratify=stratify
    )
    dataset_name = dataset_name or _infer_dataset_name(dataset_path)

    best_pipeline, best_score, stage_details = search_optimal_pipeline_by_MCTS(
        train_data,
        model_type=model,
        metric=metric,
        max_iter=max_iter,
        dataset_id=dataset_id,
        dataset_name=dataset_name,
        subset_size=subset_size,
        knowledge_csv=knowledge_csv,
        knowledge_json_dir=knowledge_json_dir,
        show_stages=True,
        return_stage_details=True,
    )

    processed_train, selected_features, _, encoding_feature, lda_transformer, context = build_pipeline(
        train_data, best_pipeline
    )
    processed_test = build_pipeline_test(
        test_data,
        best_pipeline,
        selected_features,
        encoding_feature=encoding_feature,
        lda_transformer=lda_transformer,
        context=context,
    )

    x_train = processed_train.iloc[:, :-1]
    y_train = processed_train.iloc[:, -1]
    x_test = processed_test.iloc[:, :-1]
    y_test = processed_test.iloc[:, -1]

    model_inst = support_models[model]
    model_inst.fit(x_train, y_train)
    y_pred = model_inst.predict(x_test)

    metrics = {
        "accuracy": float(accuracy_score(y_test, y_pred)),
        "precision_macro": float(precision_score(y_test, y_pred, average="macro")),
        "precision_weighted": float(precision_score(y_test, y_pred, average="weighted")),
        "recall_macro": float(recall_score(y_test, y_pred, average="macro")),
        "recall_weighted": float(recall_score(y_test, y_pred, average="weighted")),
        "f1_macro": float(f1_score(y_test, y_pred, average="macro")),
        "f1_weighted": float(f1_score(y_test, y_pred, average="weighted")),
    }

    os.makedirs(output_dir, exist_ok=True)
    train_out = os.path.join(output_dir, "processed_train.csv")
    test_out = os.path.join(output_dir, "processed_test.csv")
    pipeline_out = os.path.join(output_dir, "best_pipeline.json")
    report_out = os.path.join(output_dir, "run_report.json")

    processed_train.to_csv(train_out, index=False)
    processed_test.to_csv(test_out, index=False)
    with open(pipeline_out, "w", encoding="utf-8") as f:
        json.dump(best_pipeline, f, indent=2, ensure_ascii=False)

    with open(report_out, "w", encoding="utf-8") as f:
        json.dump(
            {
                "input_dataset": dataset_path,
                "target_column": target_column,
                "model": model,
                "search_metric": metric,
                "max_iter": max_iter,
                "subset_size": subset_size,
                "best_pipeline": best_pipeline,
                "best_pipeline_score": float(best_score),
                "test_metrics": metrics,
                "stage_details": stage_details,
                "outputs": {
                    "processed_train": train_out,
                    "processed_test": test_out,
                    "best_pipeline": pipeline_out,
                    "run_report": report_out,
                },
            },
            f,
            indent=2,
            ensure_ascii=False,
        )

    logger.info("========== Run Completed ==========")
    logger.info(f"Best pipeline: {best_pipeline}")
    logger.info(f"Best search-stage score: {best_score}")
    logger.info(f"Test metrics: {metrics}")
    logger.info(f"Output files: {train_out}, {test_out}, {pipeline_out}, {report_out}")

    return {
        "x_train": x_train,
        "y_train": y_train,
        "x_test": x_test,
        "y_test": y_test,
        "best_pipeline": best_pipeline,
        "best_score": best_score,
        "metrics": metrics,
        "stage_details": stage_details,
    }


def process_one_task(arg):
    (datasetname, dataset_id), beishu, model_type, metric, subset_size, max_iter = arg
    dataset_roots = ["dataset/diffprep", "dataset/deepline"]
    candidate_files = []
    for root in dataset_roots:
        candidate_files.extend(
            [
                os.path.join(root, f"{datasetname}.csv"),
                os.path.join(root, f"{datasetname}_beishu{beishu}.csv"),
            ]
        )
        candidate_files.extend(glob.glob(os.path.join(root, f"{datasetname}*.csv")))

    dataset_path = next((p for p in candidate_files if os.path.exists(p)), None)
    if dataset_path is None:
        raise FileNotFoundError(
            f"Dataset `{datasetname}` not found. Searched roots: {dataset_roots}"
        )

    model_type = "LogisticRegression"
    start_time = time.time()
    x_train, y_train, x_test, y_test, best_pipe, best_score = preprocess_data(
        dataset_path,
        model_type,
        metric,
        max_iter=max_iter,
        dataset_id=dataset_id,
        dataset_name=datasetname,
        subset_size=subset_size,
    )
    end_time = time.time()

    father_folder = f"data/ablation/kamcts_{metric}_subset{subset_size}_iter{max_iter}"
    if not os.path.exists(father_folder):
        os.mkdir(father_folder)

    output_folder = father_folder + "/predict_" + datasetname + "_" + model_type + "/"
    if not os.path.exists(output_folder):
        os.mkdir(output_folder)

    output_path = output_folder + datasetname + "_beishu" + beishu + "_missing0.0_outlier0.0.json"
    logger.info(f"Preprocessing time: {end_time - start_time:.2f} seconds")

    model = support_models[model_type]
    model.fit(x_train, y_train)
    y_pred = model.predict(x_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision_macro = precision_score(y_test, y_pred, average="macro")
    precision_weighted = precision_score(y_test, y_pred, average="weighted")
    recall_macro = recall_score(y_test, y_pred, average="macro")
    recall_weighted = recall_score(y_test, y_pred, average="weighted")
    f1_macro = f1_score(y_test, y_pred, average="macro")
    f1_weighted = f1_score(y_test, y_pred, average="weighted")

    result = {
        "model": model_type,
        "preprocess_time_sec": round(end_time - start_time, 2),
        "best_pipe": best_pipe,
        "best_score": best_score,
        "test_accuracy": round(accuracy, 4),
        "test_precision_macro": round(precision_macro, 4),
        "test_precision_weighted": round(precision_weighted, 4),
        "test_recall_macro": round(recall_macro, 4),
        "test_recall_weighted": round(recall_weighted, 4),
        "test_f1_macro": round(f1_macro, 4),
        "test_f1_weighted": round(f1_weighted, 4),
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    logger.info(f"{datasetname}_{beishu}_{model_type} success ({dataset_path})")


if __name__ == "__main__":
    warnings.filterwarnings("ignore")

    parser = argparse.ArgumentParser(
        description="KAMCTS automated preprocessing: input one dataset + choose model/metric, output processed data and best pipeline"
    )
    parser.add_argument("--dataset_path", type=str, required=True, help="Input dataset CSV path")
    parser.add_argument(
        "--model",
        type=str,
        default="LogisticRegression",
        choices=list(support_models.keys()),
        help="Model type",
    )
    parser.add_argument("--metric", type=str, default="accuracy", help="Search metric, e.g. accuracy / f1-score")
    parser.add_argument("--max_iter", type=int, default=30, help="Maximum MCTS iterations")
    parser.add_argument("--subset_size", type=int, default=5000, help="Subsample size for large datasets")
    parser.add_argument("--dataset_id", type=int, default=None, help="Optional: knowledge-base dataset id")
    parser.add_argument("--knowledge_csv", type=str, default=None, help="Optional: knowledge metadata CSV path")
    parser.add_argument("--knowledge_json_dir", type=str, default=None, help="Optional: knowledge JSON directory path")
    parser.add_argument("--target_column", type=str, default=None, help="Optional: target column name, default is the last column")
    parser.add_argument("--test_size", type=float, default=0.2, help="Test split ratio")
    parser.add_argument("--random_state", type=int, default=0, help="Random seed")
    parser.add_argument("--output_dir", type=str, default="output/single_run", help="Output directory")
    args = parser.parse_args()

    preprocess_single_dataset(
        dataset_path=args.dataset_path,
        model=args.model,
        metric=args.metric,
        max_iter=args.max_iter,
        subset_size=args.subset_size,
        dataset_id=args.dataset_id,
        dataset_name=_infer_dataset_name(args.dataset_path),
        knowledge_csv=args.knowledge_csv,
        knowledge_json_dir=args.knowledge_json_dir,
        output_dir=args.output_dir,
        target_column=args.target_column,
        test_size=args.test_size,
        random_state=args.random_state,
    )
