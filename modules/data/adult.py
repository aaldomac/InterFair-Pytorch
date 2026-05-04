import os
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import kagglehub
import numpy as np
import pandas as pd
import torch

from torch.utils.data import Dataset, DataLoader, TensorDataset
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler

from modules.utils.tensor_utils import (
    _to_long_tensor,
    _to_float_tensor,
    ArrayLike,
    DeviceLike,
    _as_tensor,
)

from modules.utils.dataset_utils import (
    _validate_dataframe,
    _validate_columns_exist,
    _infer_cat_and_cont_cols,
    to_tensor_dataset_ar,
    to_tensor_dataset_flow,
    to_tensor_dataset_predictive,
    make_loader,
    split_df,
    split_df_with_indices,
    fit_schema_px,
    transform_px,
    fit_predictor_schema,
    transform_predictor,
    TabularDataset,
    compute_pg_dirichlet,
    compute_pg_dirichlet_from_groups
)


# -------------------------------------------------------------
# LOAD DATASETS AND PREPROCESSING
# -------------------------------------------------------------
def load_adult_income_dataset(drop_na: bool = True) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the Adult Income dataset from Kaggle.

    Args:
        drop_na: Whether to drop rows with missing values.

    Returns:
        df: Loaded DataFrame.
        original_columns: Original column names before preprocessing.
    """
    dataset_path = kagglehub.dataset_download("wenruliu/adult-income-dataset")
    print("Path to dataset files:", dataset_path)
    csv_file = os.path.join(dataset_path, "adult.csv")

    df = pd.read_csv(csv_file, na_values="?")
    if drop_na:
        df = df.dropna().reset_index(drop=True)

    original_columns = df.columns.tolist()
    return df, original_columns


def preprocess_adult_dataset(df: pd.DataFrame) -> Tuple[pd.DataFrame, Dict[str, int]]:
    """
    Preprocess the Adult Income dataset.

    This function:
    - strips column names and string values
    - encodes protected attributes
    - binarizes native-country into US vs non-US
    - encodes the label
    - creates an intersectional group and its integer ID

    Returns:
        df: Preprocessed DataFrame.
        group_id: Mapping from group string to integer group ID.
    """
    _validate_dataframe(df, name="Adult Income DataFrame")

    df = df.copy()
    df.columns = df.columns.str.strip()

    for col in df.columns:
        if df[col].dtype == "object":
            df[col] = df[col].astype(str).str.strip()

    required_cols = ["gender", "race", "native-country", "income"]
    _validate_columns_exist(df, required_cols)

    # Protected attributes
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    if df["gender"].isna().any():
        raise ValueError("Column 'gender' contains unexpected values.")

    df["race"] = df["race"].astype("category")

    df["native-country"] = df["native-country"].fillna("United-States")
    df["native-country"] = (df["native-country"] != "United-States").astype(int)

    # Label
    df["income"] = df["income"].map({"<=50K": 1, ">50K": 0}).astype(int)
    if df["income"].isna().any():
        raise ValueError("Column 'income' contains unexpected values.")

    # Intersectional group
    df["group"] = (
        df["gender"].astype(str)
        + "_"
        + df["race"].astype(str)
        + "_"
        + df["native-country"].astype(str)
    ).astype("category")

    group_id = {name: idx for idx, name in enumerate(df["group"].cat.categories)}
    df["group_id"] = df["group"].cat.codes.astype(int)

    return df, group_id


def load_dataset(dataset_name: str, drop_na: bool = True):
    """
    Load and preprocess a supported dataset, then estimate p(g).

    Supported values:
        - "adult_income"
        - "compas"
        - "law_school"

    Returns:
        df
        original_columns
        pg_table
        pg
    """
    if dataset_name == "adult_income":
        df, original_columns = load_adult_income_dataset(drop_na=drop_na)
        df, group_id = preprocess_adult_dataset(df)
    # elif dataset_name == "compas":
    #     df, original_columns = load_compas_dataset(drop_na=drop_na)
    #     df = preprocess_compas_dataset(df)
    # elif dataset_name == "law_school":
    #     df, original_columns = load_law_school_dataset(drop_na=drop_na)
    #     df = preprocess_law_school_dataset(df)
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")

    pg_table, pg = compute_pg_dirichlet(
        df=df,
        group_col="group",
        alpha=1.0,
        return_intervals=True,
        num_draws=20000,
        ci=0.95,
        seed=42,
    )

    return df, original_columns, pg_table, pg



# -------------------------------------------------------------
# DATA EXTRACTION
# -------------------------------------------------------------

def get_adult_sets(
    df: pd.DataFrame,
    protected_cols: Sequence[str]=("gender", "race", "native-country"),
    label_col: str="income",
    device: Optional[DeviceLike] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """
    Extract X, S, y from the Adult dataset as tensors.

    Returns:
        X: float32 tensor of features
        S: float32 tensor of protected attributes
        y: long tensor of labels
    """
    _validate_dataframe(df)
    _validate_columns_exist(df, list(protected_cols) + [label_col])

    df = df.copy()
    feature_cols = [col for col in df.columns if col not in list(protected_cols) + [label_col, "group", "group_id"]]

    # cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
    # num_cols = [c for c in feature_cols if df[c].dtype != "object"]

    cat_cols, cont_cols = _infer_cat_and_cont_cols(df, feature_cols)

    print(f"Categorical columns are: {cat_cols}")
    print(f"Continuous (numerical) columns are: {cont_cols}")

    df_cat = df[cat_cols]
    df_cont = df[cont_cols]

    one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    X_cat = one_hot_encoder.fit_transform(df_cat).astype(np.float32) if len(cat_cols) > 0 else np.zeros((len(df), 0), dtype=np.float32)

    scaler = StandardScaler()
    X_cont = scaler.fit_transform(df_cont).astype(np.float32) if len(cont_cols) > 0 else np.zeros((len(df), 0), dtype=np.float32)

    X_np = np.hstack([X_cont, X_cat]).astype(np.float32)
    y_np = df[label_col].astype(np.int64).values
    print(f"Label instances\n{df[label_col].value_counts(dropna=False)}")

    df["gender"] = df["gender"].astype("float32")
    df["native-country"] = df["native-country"].astype("float32")
    if str(df["race"].dtype) == "category":
        df["race"] = df["race"].cat.codes.astype("float32")
    else:
        df["race"] = df["race"].astype("category").cat.codes.astype("float32")

    S_np = df[list(protected_cols)].values.astype(np.float32)

    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    S = torch.tensor(S_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)

    return X, S, y


def get_sets(
    df: pd.DataFrame, 
    device: Optional[DeviceLike] = None):
    """
    Dispatch to the appropriate dataset-specific extractor based on label columns.
    """
    _validate_dataframe(df)

    df = df.copy()

    if "two_year_recid" in df.columns:
        return get_compas_sets(df, device=device)
    elif "income" in df.columns:
        return get_adult_sets(df, device=device)
    elif "pass_bar" in df.columns:
        return get_law_school_sets(df, device=device)
    else:
        raise ValueError("DataFrame does not contain recognized label columns.")