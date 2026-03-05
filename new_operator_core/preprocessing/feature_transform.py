import numpy as np
import pandas as pd
from sklearn.decomposition import IncrementalPCA, KernelPCA, PCA, TruncatedSVD
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis
from sklearn.ensemble import RandomTreesEmbedding
from sklearn.preprocessing import PolynomialFeatures


def _normalize_method_name(method: str) -> str:
    aliases = {
        "kernalPCA": "kernelPCA",
        "kernel_pca": "kernelPCA",
    }
    return aliases.get(method, method)


def transform_features(data, method: str = "pca", n_components: int = 2):
    """Fit and apply feature transformation on numeric columns."""
    method = _normalize_method_name(method)
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    x_num = x.select_dtypes(include=[np.number])
    x_cat = x.select_dtypes(exclude=[np.number])

    if x_num.shape[1] == 0:
        raise ValueError("No numeric columns for feature transform")

    if method == "pca":
        transformer = PCA(n_components=min(n_components, x_num.shape[1])).fit(x_num)
        x_trans = transformer.transform(x_num)
    elif method == "kernelPCA":
        transformer = KernelPCA(n_components=min(n_components, x_num.shape[1]), kernel="rbf").fit(x_num)
        x_trans = transformer.transform(x_num)
    elif method == "lda":
        transformer = LinearDiscriminantAnalysis(n_components=min(n_components, len(set(y)) - 1)).fit(x_num, y)
        x_trans = transformer.transform(x_num)
    elif method == "polynomial":
        transformer = PolynomialFeatures(include_bias=False).fit(x_num)
        x_trans = transformer.transform(x_num)
    elif method == "IncrementalPCA":
        transformer = IncrementalPCA(n_components=min(n_components, x_num.shape[1])).fit(x_num)
        x_trans = transformer.transform(x_num)
    elif method == "TruncatedSVD":
        transformer = TruncatedSVD(n_components=min(n_components, x_num.shape[1])).fit(x_num)
        x_trans = transformer.transform(x_num)
    elif method == "InteractionFeatures":
        transformer = PolynomialFeatures(interaction_only=True, include_bias=False).fit(x_num)
        x_trans = transformer.transform(x_num)
    elif method == "RandomTrees":
        transformer = RandomTreesEmbedding(random_state=0).fit(x_num)
        x_trans = transformer.transform(x_num).toarray()
    else:
        raise ValueError(f"Unknown feature transform method: {method}")

    x_trans_df = pd.DataFrame(x_trans, index=x.index)
    x_all = pd.concat([x_trans_df, x_cat], axis=1)
    x_all["label"] = y.values
    x_all.columns = x_all.columns.astype(str)
    return x_all, transformer


def transform_features_for_test_data(data, method: str = "pca", n_components: int = 2, lda_transformer=None):
    """Transform test data using fitted feature transformer."""
    _ = n_components
    method = _normalize_method_name(method)
    x, y = data.iloc[:, :-1], data.iloc[:, -1]
    x_num = x.select_dtypes(include=[np.number])
    x_cat = x.select_dtypes(exclude=[np.number])

    if x_num.shape[1] == 0:
        raise ValueError("No numeric columns for feature transform")

    transformer = lda_transformer
    if method == "RandomTrees":
        x_trans = transformer.transform(x_num).toarray()
    else:
        x_trans = transformer.transform(x_num)

    x_trans_df = pd.DataFrame(x_trans, index=x.index)
    x_all = pd.concat([x_trans_df, x_cat], axis=1)
    x_all["label"] = y.values
    x_all.columns = x_all.columns.astype(str)
    return x_all
