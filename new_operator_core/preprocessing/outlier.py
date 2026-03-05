import numpy as np

from .imputation import transform_with_imputers


def fit_outlier_detector(data, method: str = "zscore"):
    numeric_cols = data.iloc[:, :-1].select_dtypes(include="number").columns
    detector_params = {}

    if method == "zscore":
        mean = data[numeric_cols].mean()
        std = data[numeric_cols].std()
        detector_params = {"method": "zscore", "mean": mean, "std": std}
    elif method == "iqr":
        q1 = data[numeric_cols].quantile(0.25)
        q3 = data[numeric_cols].quantile(0.75)
        detector_params = {"method": "iqr", "Q1": q1, "Q3": q3}

    return detector_params


def clean_outliers(data, detector_params, threshold: float = 3):
    numeric_cols = data.iloc[:, :-1].select_dtypes(include="number").columns

    if detector_params["method"] == "zscore":
        mean = detector_params["mean"]
        std = detector_params["std"]
        z_scores = (data[numeric_cols] - mean) / std
        outliers = np.abs(z_scores) > threshold
    elif detector_params["method"] == "iqr":
        q1 = detector_params["Q1"]
        q3 = detector_params["Q3"]
        iqr = q3 - q1
        outliers = (data[numeric_cols] < (q1 - 1.5 * iqr)) | (data[numeric_cols] > (q3 + 1.5 * iqr))
    else:
        return data

    data[numeric_cols] = data[numeric_cols].mask(outliers)
    return data


def handle_outliers(data, imputer=None, method: str = "zscore", threshold: float = 3):
    detector_params = fit_outlier_detector(data, method)
    data = clean_outliers(data, detector_params=detector_params, threshold=threshold)
    if imputer is not None:
        data = transform_with_imputers(data, imputer)
    return data
