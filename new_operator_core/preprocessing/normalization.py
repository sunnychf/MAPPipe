import pandas as pd
from sklearn.preprocessing import (
    KBinsDiscretizer,
    MaxAbsScaler,
    MinMaxScaler,
    Normalizer,
    PowerTransformer,
    QuantileTransformer,
    RobustScaler,
    StandardScaler,
)


_SCALER_FACTORY = {
    "standard": StandardScaler,
    "minmax": MinMaxScaler,
    "maxabs": MaxAbsScaler,
    "power": PowerTransformer,
    "quantile": QuantileTransformer,
    "normalizer": Normalizer,
    "robust": RobustScaler,
    "kbins": lambda: KBinsDiscretizer(encode="ordinal"),
}


def normalize_features(data, method: str = "standard"):
    """Fit and apply normalization on numeric features for training data."""
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    x.columns = x.columns.astype(str)

    num_cols = x.select_dtypes(include=["number"]).columns.tolist()
    other_cols = [col for col in x.columns if col not in num_cols]
    if len(num_cols) == 0:
        raise ValueError("No numeric columns for normalization")

    if method not in _SCALER_FACTORY:
        raise ValueError(f"Unknown normalize method: {method}")

    scaler = _SCALER_FACTORY[method]()
    normalizer = scaler.fit(x[num_cols])
    x_trans = normalizer.transform(x[num_cols])

    try:
        feature_names = normalizer.get_feature_names_out(num_cols)
    except Exception:
        feature_names = num_cols

    x_trans_df = pd.DataFrame(x_trans, columns=feature_names, index=x.index)
    x_full = pd.concat([x_trans_df, x[other_cols]], axis=1)
    x_full["label"] = y.values
    x_full.columns = x_full.columns.astype(str)

    return x_full, normalizer, list(feature_names) + list(other_cols)


def normalize_features_for_test(data, normalizer, expected_columns):
    """Transform test data using training normalizer."""
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    x.columns = x.columns.astype(str)

    num_cols = normalizer.feature_names_in_.tolist()
    for col in num_cols:
        if col not in x.columns:
            x[col] = 0

    x_num = x[num_cols]
    x_trans = normalizer.transform(x_num)

    try:
        feature_names = normalizer.get_feature_names_out(num_cols)
    except Exception:
        feature_names = num_cols

    x_trans_df = pd.DataFrame(x_trans, columns=feature_names, index=x.index)
    other_cols = [col for col in x.columns if col not in num_cols]
    x_full = pd.concat([x_trans_df, x[other_cols]], axis=1)
    x_full = x_full[[col for col in expected_columns if col in x_full.columns]]
    x_full.columns = x_full.columns.astype(str)
    x_full["label"] = y.values
    return x_full
