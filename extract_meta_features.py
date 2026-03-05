import os
import pandas as pd
import numpy as np
from sklearn.feature_selection import mutual_info_classif
from sklearn.preprocessing import LabelEncoder

def extract_meta_features(data):
    """
    ，MCTS、
    :
        data: pd.DataFrame，（）
    :
        meta_features: dict，
        need_IMV：
        need_O：，（ z-score, IQR）
        need_E：，（ onehot, label）
    """
    meta_features = {}
    need_IMV = False
    need_O = False
    need_E = False
    n_instances = data.shape[0]
    meta_features['n_instances'] = n_instances

    n_features = data.shape[1] - 1
    meta_features['n_features'] = n_features

    X = data.iloc[:, :-1]
    y = data.iloc[:, -1]

    n_num_features = X.select_dtypes(include=[np.number]).shape[1]
    n_cat_features = X.select_dtypes(exclude=[np.number]).shape[1]
    meta_features['n_num_features'] = n_num_features
    meta_features['n_cat_features'] = n_cat_features
    need_E = n_cat_features > 0
    missing_ratio = X.isnull().sum().sum() / X.size
    meta_features['missing_ratio'] = missing_ratio
    need_IMV = missing_ratio > 0
    if y.dtype == 'object' or str(y.dtype) == 'category':
        y_encoded = LabelEncoder().fit_transform(y)
    else:
        y_encoded = y.values
    meta_features['target_num_classes'] = len(np.unique(y_encoded))

    feature_mean_skew = X.skew(numeric_only=True).mean()
    meta_features['feature_mean_skew'] = feature_mean_skew

    feature_mean_kurtosis = X.kurtosis(numeric_only=True).mean()
    meta_features['feature_mean_kurtosis'] = feature_mean_kurtosis

    # need_O = int(abs(feature_mean_skew) > 2 or feature_mean_kurtosis > 2)


    mean_feature_variance = X.var(numeric_only=True).mean()
    meta_features['mean_feature_variance'] = mean_feature_variance

    corr_matrix = X.corr(numeric_only=True).abs()
    upper_triangle = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
    mean_feature_correlation = upper_triangle.stack().mean()
    meta_features['mean_feature_correlation'] = mean_feature_correlation

    try:
        if not np.issubdtype(y_encoded.dtype, np.integer):
            y_encoded = LabelEncoder().fit_transform(y_encoded)
        mi_scores = mutual_info_classif(X.select_dtypes(include=[np.number]).fillna(0), y_encoded, discrete_features=False)
        meta_features['mean_mutual_info'] = np.mean(mi_scores)
        meta_features['max_mutual_info'] = np.max(mi_scores)
    except:
        meta_features['mean_mutual_info'] = 0.0
        meta_features['max_mutual_info'] = 0.0

    value_counts = pd.Series(y_encoded).value_counts(normalize=True)
    target_entropy = -np.sum(value_counts * np.log2(value_counts + 1e-9))
    meta_features['target_entropy'] = target_entropy

    return meta_features, need_IMV, need_O, need_E

def extract_column_meta(series):
    meta = {}
    meta['dtype'] = str(series.dtype)
    meta['nums'] = len(series)
    meta['n_missing'] = series.isnull().sum()
    meta['missing_ratio'] = series.isnull().mean()
    meta['n_unique'] = series.nunique(dropna=True)
    meta['is_constant'] = meta['n_unique'] == 1
    meta['is_binary'] = int(series.dropna().nunique() == 2)

    if pd.api.types.is_numeric_dtype(series):
        series_no_na = series.dropna()
        mean = series_no_na.mean()
        std = series_no_na.std()
        meta['mean'] = float(mean)
        meta['std'] = float(std)
        meta['skew'] = series_no_na.skew(skipna=True)
        meta['kurtosis'] = series_no_na.kurtosis(skipna=True)
        meta['min'] = series_no_na.min(skipna=True)
        meta['max'] = series_no_na.max(skipna=True)
        meta['range'] = meta['max'] - meta['min']
        q1 = series_no_na.quantile(0.25)
        q3 = series_no_na.quantile(0.75)
        meta['q1'] = q1
        meta['q2'] = series_no_na.median()
        meta['q3'] = q3
        meta['iqr'] = q3 - q1

        if std and np.isfinite(std) and std > 0:
            z = (series_no_na - mean) / std
            meta['zscore_outlier_ratio'] = float((z.abs() > 3).mean())
        else:
            meta['zscore_outlier_ratio'] = 0.0
        meta['has_negative'] = bool((series_no_na < 0).any())
        meta['sparsity'] = float((series_no_na == 0).mean())
    else:
        value_counts = series.value_counts(normalize=True, dropna=True)
        meta['mode_freq_ratio'] = value_counts.iloc[0] if not value_counts.empty else 0
        meta['value_counts_entropy'] = -np.sum(value_counts * np.log2(value_counts + 1e-8))
        meta['is_binary'] = meta['n_unique'] == 2
    return meta

def extract_missing_numeric_column_meta_table(data, target_col=None, save_path=None):
    """
    ，（CSV）

    ：
    - data: pd.DataFrame，
    - target_col: str  None，（，）
    - save_path: str  None，CSV

    ：
    - meta_df: DataFrame，
    """

    if target_col is None:
        target_col = data.columns[-1]

    meta_dict_list = []
    for col in data.columns:
        if col == target_col:
            continue
        series = data[col]
        if pd.api.types.is_numeric_dtype(series) and series.isnull().sum() > 0:
            meta = extract_column_meta(series)
            meta['feature_name'] = f'col_{col}'
            meta_dict_list.append(meta)

    meta_df = pd.DataFrame(meta_dict_list).set_index('feature_name')
    if save_path:
        meta_df.to_csv(save_path, index=True)
    return 

import pandas as pd



def extract_column_meta_table(data, target_col=None, save_path=None):
    """
    ，（CSV）

    ：
    - data: pd.DataFrame，
    - target_col: str  None，（，）
    - save_path: str  None，CSV

    ：
    - meta_df: DataFrame，
    """

    if target_col is None:
        target_col = data.columns[-1]

    meta_dict_list = []
    for col in data.columns:
        if col == target_col:
            continue
        series = data[col]
        if pd.api.types.is_numeric_dtype(series) > 0:
            meta = extract_column_meta(series)
            meta['feature_name'] = f'col_{col}'
            meta_dict_list.append(meta)

    meta_df = pd.DataFrame(meta_dict_list).set_index('feature_name')
    if save_path:
        meta_df.to_csv(save_path, index=True)
    return 


def compute_weighted_strategy_frequencies(topk_df, strategy_columns=None, accuracy_col='accuracy'):
    """
    （），（ accuracy ）

    ：
        topk_df:  accuracy  DataFrame
        strategy_columns: list or None， None  accuracy 
        accuracy_col: accuracy 

    ：
        DataFrame: ，，
    """
    if strategy_columns is None:
        strategy_columns = [col for col in topk_df.columns if col != accuracy_col]
    topk_df = topk_df.copy()
    topk_df[strategy_columns] = topk_df[strategy_columns].fillna('none')
    freq_dict = {}

    for col in strategy_columns:
        method_weights = topk_df.groupby(col)[accuracy_col].sum()
        freq_dict[col] = method_weights

    freq_df = pd.DataFrame(freq_dict).T.fillna(0)

    all_methods = ['none', 'standard', 'minmax', 'maxabs', 'robust', 'normalizer', 'power', 'quantile', 'kbins']
    for method in all_methods:
        if method not in freq_df.columns:
            freq_df[method] = 0

    freq_df = freq_df[all_methods]

    # freq_df = freq_df.div(row_sums, axis=0)
    return freq_df

import numpy as np
import pandas as pd

def compute_softmax_weighted_strategy_frequencies(topk_df, strategy_columns=None, accuracy_col='accuracy', beta=1.0):
    """
    ，， Accuracy × Softmax() 

    ：
        topk_df:  accuracy  DataFrame
        strategy_columns: （None ）
        accuracy_col: accuracy 
        beta:  softmax ， → 

    ：
        freq_dict: ，（ accuracy  softmax ）
    """
    if strategy_columns is None:
        strategy_columns = [col for col in topk_df.columns if col != accuracy_col]

    n = len(topk_df)

    ranks = np.arange(n)  # rank: 0, 1, 2, ...
    softmax_pos_weights = np.exp(-beta * ranks)
    softmax_pos_weights /= softmax_pos_weights.sum()

    final_weights = topk_df[accuracy_col].values * softmax_pos_weights

    freq_dict = {}
    for col in strategy_columns:
        # temp_df = topk_df[[col]].copy()
        temp_df = topk_df.copy()
        temp_df['weight'] = final_weights
        method_weights = temp_df.groupby(col)['weight'].sum()
        freq_dict[col] = method_weights

    freq_df = pd.DataFrame(freq_dict).T.fillna(0)
    all_methods = ['none', 'standard', 'minmax', 'maxabs', 'robust', 'normalizer', 'power', 'quantile', 'kbins']
    for method in all_methods:
        if method not in freq_df.columns:
            freq_df[method] = 0

    freq_df = freq_df[all_methods]

    row_sums = freq_df.sum(axis=1).replace(0, 1e-9)
    freq_df = freq_df.div(row_sums, axis=0)

    return freq_df


from concurrent.futures import ProcessPoolExecutor, ThreadPoolExecutor

def process_file(file, k=100):
    try:
        # meta_data_path = os.path.join('data/meta_col', file)
        meta_data_path = os.path.join('data/meta_data_of_columns', file)
        meta_data = pd.read_csv(meta_data_path)

        IFK = file.replace('missing', 'miss').replace('.csv', '_testscore.csv')
        # IFK_path = os.path.join('results/Impute_for_Knowledge', IFK)
        IFK_path = os.path.join('results/Normalize_for_Knowledge', IFK)
        IFK_data = pd.read_csv(IFK_path)
        IFK_data = IFK_data.fillna('none')
        topk = IFK_data.nlargest(k, 'accuracy')
        method_freq_per_column = compute_softmax_weighted_strategy_frequencies(topk)
        all_methods = ['none', 'standard', 'minmax', 'maxabs', 'robust', 'normalizer', 'power', 'quantile', 'kbins']
        for method in all_methods:
            if method not in method_freq_per_column.columns:
                method_freq_per_column[method] = 0
        method_freq_per_column = method_freq_per_column.fillna(0)
        method_freq_per_column = method_freq_per_column[all_methods]
        method_freq_per_column['col'] = method_freq_per_column.index.map(lambda x: f'col_{x}')

        method_freq_per_column['col'] = method_freq_per_column['col'].astype(str)

        meta_data['feature_name'] = meta_data['feature_name'].astype(str)

        last_4_col = method_freq_per_column.columns[-10:-1]
        rename_dict = {col: 'strategy_' + col for col in last_4_col}
        method_freq_per_column = method_freq_per_column.rename(columns=rename_dict)

        merged_df = pd.merge(meta_data, method_freq_per_column, left_on='feature_name', right_on='col', how='inner')
        merged_df = merged_df.drop(columns=['col', 'dtype', 'feature_name'])
        save_dir = 'results/Normalize_for_Knowledge/meta_with_strategy'
        os.makedirs(save_dir, exist_ok=True)
        merged_df.to_csv(os.path.join(save_dir, file), index=False)
        print(f'{file} ✅ success')
    except Exception as e:
        print(f'{file} ❌ failed: {e}')

import glob

def merge_all_csv(save_dir, output_path):
    all_files = glob.glob(os.path.join(save_dir, '*.csv'))
    df_list = [pd.read_csv(f) for f in all_files]
    merged_df = pd.concat(df_list, ignore_index=True)
    merged_df.to_csv(output_path, index=False)
    print(f"✅ ，：{output_path}")

if __name__ == "__main__":
#     # files = os.listdir('data/meta_col')
#     #     executor.map(process_file, files) 
    merge_all_csv(
        save_dir='results/Normalize_for_Knowledge/meta_with_strategy',
        output_path='results/Normalize_for_Knowledge/final_merged/meta_with_strategy_merged.csv'
    )
    output_path='results/Normalize_for_Knowledge/final_merged/meta_with_strategy_merged.csv'
    df = pd.read_csv(output_path)
    df = df.drop(columns=['is_constant'])
    df.to_csv(output_path, index=False)

# if __name__ == "__main__":
    # dir = '../effective_preprocessing_pipelines/resources/datasets'
    # files = sorted(os.listdir(dir))
    # datasetnames = [file.split('.')[0] for file in files]
    # dir = 'results/Normalize_for_Knowledge'
    # files = sorted(os.listdir(dir))
    # datasetnames = [file.split('_')[0] for file in files]
    # for dataset in datasetnames[:]:
    #     if dataset == 'final' or dataset == 'meta':
    #         continue
    #     train_path = f"data/train/predict_{dataset}_100/{dataset}_beishu100_missing0.0_outlier0.0.csv"
    #     data = pd.read_csv(train_path)
    #     extract_column_meta_table(data,save_path=f"data/meta_data_of_columns/{dataset}_beishu100_missing0.0_outlier0.0.csv")
    #     file = f'{dataset}_beishu100_missing0.0_outlier0.0.csv'
    #     process_file(file)
    # datasetnames = []
    # for dataset in datasetnames[:]:
    #     if dataset == 'final' or dataset == 'meta':
    #         continue
    #     train_path = f"data/train/predict_{dataset}_100/{dataset}_beishu100_missing0.0_outlier0.0.csv"
    #     data = pd.read_csv(train_path)
    #     extract_column_meta_table(data,save_path=f"data/meta_data_of_columns/{dataset}_beishu100_missing0.0_outlier0.0.csv")
    #     file = f'{dataset}_beishu100_missing0.0_outlier0.0.csv'
    #     process_file(file)