from typing import Dict, Iterable, Tuple

from sklearn.experimental import enable_iterative_imputer  # noqa: F401
from sklearn.impute import IterativeImputer, KNNImputer, SimpleImputer


def fit_imputers(data, strategy: str = "mean") -> Tuple[Dict[str, object], Iterable[str], Iterable[str]]:
    """Fit imputers on training features only."""
    imputers: Dict[str, object] = {}
    x = data.iloc[:, :-1].copy()

    num_cols = x.select_dtypes(include="number").columns
    cat_cols = x.select_dtypes(exclude="number").columns

    if strategy == "knn":
        imputers["num"] = KNNImputer(n_neighbors=5)
    elif strategy == "iterative":
        imputers["num"] = IterativeImputer(random_state=42)
    else:
        imputers["num"] = SimpleImputer(strategy=strategy)

    if len(num_cols) > 0:
        imputers["num"].fit(x[num_cols])

    if len(cat_cols) > 0:
        imputers["cat"] = SimpleImputer(strategy="most_frequent")
        imputers["cat"].fit(x[cat_cols])

    return imputers, num_cols, cat_cols


def transform_with_imputers(data, imputers, num_cols=None, cat_cols=None):
    """Transform data with fitted imputers."""
    num_cols = list(num_cols) if num_cols is not None else []
    cat_cols = list(cat_cols) if cat_cols is not None else []

    if "num" in imputers and len(num_cols) > 0:
        data[num_cols] = imputers["num"].transform(data[num_cols])
    if "cat" in imputers and len(cat_cols) > 0:
        data[cat_cols] = imputers["cat"].transform(data[cat_cols])
    return data
