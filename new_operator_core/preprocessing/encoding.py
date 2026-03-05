import category_encoders as ce
import pandas as pd
from sklearn.preprocessing import LabelEncoder, OneHotEncoder


def encode_categorical_features(data, method: str = "onehot"):
    """Fit/transform categorical encoding on train data."""
    data = data.copy()
    target_column = data.columns[-1]
    y = data[target_column]

    x = data.iloc[:, :-1]
    categorical_features = x.select_dtypes(include=["object"]).columns
    encoder = None

    if len(categorical_features) > 0:
        if method == "onehot":
            encoder = OneHotEncoder(handle_unknown="ignore", sparse_output=False)
            x_cat_enc = encoder.fit_transform(x[categorical_features])
            enc_cols = encoder.get_feature_names_out(categorical_features)
            x_cat_enc = pd.DataFrame(x_cat_enc, columns=enc_cols, index=x.index)
            x = pd.concat([x.drop(categorical_features, axis=1), x_cat_enc], axis=1)
        elif method == "binary":
            encoder = ce.BinaryEncoder(cols=categorical_features)
            x = encoder.fit_transform(x)
        elif method == "frequency":
            for col in categorical_features:
                freq = x[col].value_counts(normalize=True)
                x[col] = x[col].map(freq)
        elif method == "label":
            encoder = {}
            for col in categorical_features:
                le = LabelEncoder()
                x[col] = le.fit_transform(x[col])
                encoder[col] = le
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    x[target_column] = y
    return x, x.columns[:-1], encoder


def encode_categorical_features_for_test(data, method: str = "onehot", encoder=None):
    """Transform categorical encoding on test data only."""
    if encoder is None:
        raise RuntimeError("Encoder must be provided for test data!")

    data = data.copy()
    target_column = data.columns[-1]
    y = data[target_column]

    x = data.iloc[:, :-1]
    categorical_features = x.select_dtypes(include=["object"]).columns

    if len(categorical_features) > 0:
        if method == "onehot":
            x_cat_enc = encoder.transform(x[categorical_features])
            enc_cols = encoder.get_feature_names_out(categorical_features)
            x_cat_enc = pd.DataFrame(x_cat_enc, columns=enc_cols, index=x.index)
            x = pd.concat([x.drop(categorical_features, axis=1), x_cat_enc], axis=1)
        elif method == "binary":
            x = encoder.transform(x)
        elif method == "frequency":
            pass
        elif method == "label":
            for col, le in encoder.items():
                # Handle unseen categories in test data silently:
                # map known labels to trained indices, and assign -1 to unknown labels.
                label_to_idx = {label: idx for idx, label in enumerate(le.classes_)}
                x[col] = x[col].map(label_to_idx).fillna(-1).astype(int)
        else:
            raise ValueError(f"Unknown encoding method: {method}")

    x[target_column] = y
    return x
