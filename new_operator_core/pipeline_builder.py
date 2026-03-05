from .preprocessing import (
    apply_fs_before_test,
    apply_fs_before_train,
    encode_categorical_features,
    encode_categorical_features_for_test,
    fit_imputers,
    handle_outliers,
    normalize_features,
    normalize_features_for_test,
    select_features,
    transform_features,
    transform_features_for_test_data,
    transform_with_imputers,
)


def build_pipeline(data, pipeline_config):
    """Build train preprocessing pipeline with dynamic step flow."""
    data = data.copy(deep=True)
    context = {}

    if pipeline_config.get("I", {}):
        imputers, num_cols, cat_cols = fit_imputers(data, **pipeline_config.get("I", {}))
        data = transform_with_imputers(data, imputers, num_cols, cat_cols)
    else:
        imputers, num_cols, cat_cols = fit_imputers(data, strategy="mean")

    context["imputer"] = imputers
    context["num_cols"] = num_cols
    context["cat_cols"] = cat_cols
    context["feature_history"] = {}

    selected_features = data.columns[:-1]
    var_threshold = pipeline_config.get("FS", {}).get("var_threshold", 0)
    encoding_feature = None
    lda_transformer = None

    flow = pipeline_config.get("flow", ["O", "E", "N", "FT", "FS"])

    for step in flow:
        if step == "O" and "O" in pipeline_config and pipeline_config["O"].get("method"):
            data = handle_outliers(data, imputer=imputers, **pipeline_config["O"])
        elif step == "E" and "E" in pipeline_config and pipeline_config["E"].get("method"):
            data = apply_fs_before_train(data, context, next_step=step)
            data, encoding_feature, encoder = encode_categorical_features(data, **pipeline_config["E"])
            context["encoder"] = encoder
        elif (
            step == "N"
            and "N" in pipeline_config
            and pipeline_config["N"].get("method")
            and data.iloc[:, :-1].select_dtypes(include=["number"]).shape[1] > 0
        ):
            data = apply_fs_before_train(data, context, next_step=step)
            data, normalizer, expected_columns = normalize_features(data, **pipeline_config["N"])
            context["normalize"] = normalizer
            context["expected_columns"] = expected_columns
        elif (
            step == "FT"
            and "FT" in pipeline_config
            and pipeline_config["FT"].get("method")
            and data.iloc[:, :-1].select_dtypes(include=["number"]).shape[1] > 0
        ):
            data = apply_fs_before_train(data, context, next_step=step)
            try:
                data, lda_transformer = transform_features(data, **pipeline_config["FT"])
            except Exception:
                continue

    data, selected_features, var_threshold = select_features(data)
    return data, selected_features, var_threshold, encoding_feature, lda_transformer, context


def build_pipeline_test(data, pipeline_config, selected_features, encoding_feature, lda_transformer=None, context=None):
    """Build test preprocessing pipeline with train-fitted states."""
    _ = encoding_feature
    data = data.copy(deep=True)

    if pipeline_config.get("I", {}):
        data = transform_with_imputers(data, context["imputer"], context["num_cols"], context["cat_cols"])

    flow = pipeline_config.get("flow", ["O", "E", "N", "FT", "FS"])

    for step in flow:
        if step == "O" and "O" in pipeline_config and pipeline_config["O"].get("method"):
            data = handle_outliers(data, context["imputer"], **pipeline_config["O"])
        elif step == "E" and "E" in pipeline_config and pipeline_config["E"].get("method"):
            data = apply_fs_before_test(data, context, next_step=step)
            data = encode_categorical_features_for_test(data, **pipeline_config["E"], encoder=context["encoder"])
        elif (
            step == "N"
            and "N" in pipeline_config
            and pipeline_config["N"].get("method")
            and data.iloc[:, :-1].select_dtypes(include=["number"]).shape[1] > 0
        ):
            data = apply_fs_before_test(data, context, next_step=step)
            data = normalize_features_for_test(data, context["normalize"], context["expected_columns"])
        elif (
            step == "FT"
            and "FT" in pipeline_config
            and pipeline_config["FT"].get("method")
            and data.iloc[:, :-1].select_dtypes(include=["number"]).shape[1] > 0
        ):
            data = apply_fs_before_test(data, context, next_step=step)
            data = transform_features_for_test_data(data, **pipeline_config["FT"], lda_transformer=lda_transformer)

    y = data.iloc[:, -1]
    x = data.iloc[:, :-1]
    common_features = x.columns.intersection(selected_features)
    data = data[common_features].copy()
    data["label"] = y.values
    return data
