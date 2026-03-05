import numpy as np
import pandas as pd
from sklearn.feature_selection import SelectKBest, VarianceThreshold, f_classif


def select_features(data, method: str = "variance", var_threshold: float = 0, pearson_threshold: float = 0.1, k: float = 0.8):
    """Feature selection on numeric columns while preserving categorical columns."""
    x_all = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    x_num = x_all.select_dtypes(include=[np.number])
    x_cat = x_all.select_dtypes(exclude=[np.number])
    selected_features = x_num.columns

    if method == "variance":
        try:
            selector = VarianceThreshold(threshold=var_threshold)
            x_selected = selector.fit_transform(x_num)
            selected_features = x_num.columns[selector.get_support()]
            x_num = pd.DataFrame(x_selected, columns=selected_features, index=x_num.index)
        except ValueError:
            x_num = x_num.copy()
    elif method == "univariate":
        selector = VarianceThreshold(threshold=0)
        x_filtered = selector.fit_transform(x_num)
        selected_var_features = x_num.columns[selector.get_support()]
        x_num = pd.DataFrame(x_filtered, columns=selected_var_features, index=x_num.index)

        k_count = int(x_num.shape[1] * k)
        if k_count == 0:
            k_count = 1
        selector = SelectKBest(score_func=f_classif, k=k_count)
        x_selected = selector.fit_transform(x_num, y)
        selected_features = x_num.columns[selector.get_support()]
        x_num = pd.DataFrame(x_selected, columns=selected_features, index=x_num.index)
    elif method == "pearson":
        corr = pd.concat([x_num, y], axis=1).corr()
        corr_target = corr.iloc[:-1, -1]
        selected_features = corr_target[abs(corr_target) > pearson_threshold].index.tolist()
        x_num = x_num[selected_features]

    x_final = pd.concat([x_num, x_cat], axis=1)
    selected_features = list(x_final.columns)
    x_final["label"] = y.values
    x_final.columns = x_final.columns.astype(str)

    return x_final, selected_features, var_threshold


def apply_fs_before_train(data, context, next_step: str, fs_config=None):
    """Apply feature selection before a given preprocessing step and record kept columns."""
    if "feature_history" not in context:
        context["feature_history"] = {}

    data, selected_features, _ = select_features(data, **(fs_config or {}))
    context["feature_history"][f"before_{next_step}"] = selected_features
    return data


def apply_fs_before_test(data, context, next_step: str):
    """Align test features using feature selection history from train stage."""
    key = f"before_{next_step}"
    if "feature_history" not in context or key not in context["feature_history"]:
        return data

    selected_features = context["feature_history"][key]
    y = data.iloc[:, -1]
    x = data.iloc[:, :-1]
    common = x.columns.intersection(selected_features)
    x = x[common].copy()
    x["label"] = y.values
    return x
