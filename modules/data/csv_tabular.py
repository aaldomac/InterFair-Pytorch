"""Local real-world tabular data using the same contract as Adult and synthetic data."""
import pandas as pd
from modules.utils.dataset_utils import DatasetSpec, LoadedDataset


def load_dataset(path, label_col, protected_cols, drop_feature_cols=(), log1p_cols=()):
    df = pd.read_csv(path)
    if not protected_cols or df[list(protected_cols)+[label_col]].isna().any().any():
        raise ValueError('Require nonempty protected_cols and complete protected attributes/labels')
    keys = df[list(protected_cols)].astype(str).apply(tuple, axis=1)
    categories = sorted(keys.unique())
    mapping = {key:i for i,key in enumerate(categories)}
    df['group_id'] = keys.map(mapping)
    names = {str(key):i for key,i in mapping.items()}
    df['group'] = keys.map(str)
    spec = DatasetSpec(name='csv_tabular', protected_cols=tuple(protected_cols),
        label_col=label_col, drop_feature_cols=tuple(drop_feature_cols), log1p_cols=tuple(log1p_cols))
    return LoadedDataset(df=df, original_columns=list(df.columns), group_id=names,
                         metadata={'spec':spec})
