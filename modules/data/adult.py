from __future__ import annotations

import os
from typing import Any, Dict, List, Optional, Sequence, Tuple

import pandas as pd


from modules.utils.dataset_utils import (
    DatasetSpec,
    LoadedDataset,
    _validate_dataframe,
    _validate_columns_exist,
    compute_pg_dirichlet,
)

ADULT_SPEC = DatasetSpec(
    name="adult",
    protected_cols=("gender", "race", "native-country"),
    label_col="income",
    group_col="group",
    group_id_col="group_id",
    log1p_cols=("capital-gain", "capital-loss"),
)

# -------------------------------------------------------------
# LOAD DATASETS AND PREPROCESSING
# -------------------------------------------------------------
def load_raw_adult(drop_na: bool = True, csv_path=None) -> Tuple[pd.DataFrame, List[str]]:
    """
    Load the Adult Income dataset from Kaggle.

    Args:
        drop_na: Whether to drop rows with missing values.

    Returns:
        df: Loaded DataFrame.
        original_columns: Original column names before preprocessing.
    """
    if csv_path is None:
        import kagglehub
        dataset_path = kagglehub.dataset_download("wenruliu/adult-income-dataset")
        csv_file = os.path.join(dataset_path, "adult.csv")
    else:
        csv_file = csv_path

    df = pd.read_csv(csv_file, na_values=["?", " ?"], skipinitialspace=True)
    df.columns = df.columns.str.strip()
    if "sex" in df.columns and "gender" not in df.columns:
        df = df.rename(columns={"sex": "gender"})
    for col in df.select_dtypes(include=['object','string']):
        df[col] = df[col].str.strip().replace('?',pd.NA)
    if drop_na:
        df = df.dropna().reset_index(drop=True)

    original_columns = df.columns.tolist()
    return df, original_columns



def preprocess_adult(df: pd.DataFrame, positive_label: str = "<=50K", binary_attrs: bool = True) -> Tuple[pd.DataFrame, Dict[str, int]]:
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
            df[col] = df[col].astype("string").str.strip()

    required_cols = ["gender", "race", "native-country", "income"]
    _validate_columns_exist(df, required_cols)

    if positive_label not in ('<=50K', '>50K'):
        raise ValueError('positive label must be <=50K or >50K')
    if df[required_cols].isna().any().any():
        raise ValueError('Missing protected attributes/label; use drop_na=True')

    # Protected attributes
    df["gender"] = df["gender"].map({"Male": 0, "Female": 1})
    if df["gender"].isna().any():
        raise ValueError("Column 'gender' contains unexpected values.")

    if binary_attrs:
        df["race"] = df["race"].fillna("White")  # Fill missing race with "White" (most common)
        df["race"] = (df["race"] != "White").astype(int)
    else:
        df["race"] = df["race"].astype("category")
    
    df["native-country"] = df["native-country"].fillna("United-States")
    df["native-country"] = (df["native-country"] != "United-States").astype(int)

    # Label
    labels = df["income"].astype('string').str.rstrip('.')
    if not labels.isin(['<=50K','>50K']).all():
        raise ValueError('Unexpected income labels; expected <=50K or >50K')
    df['income'] = (labels == positive_label).astype(int)

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


def load_dataset(
        *, 
        drop_na: bool = True,
        csv_path=None,
        positive_label: str = "<=50K",
        compute_pg: bool = True,
        pg_alpha: float = 1.0,
        pg_num_draws: int = 20000,
        pg_ci: float = 0.95,
        seed: int = 42,
        binary_attrs: bool = True
    ) -> LoadedDataset:
    """
    Standard Adult loader used by `load_dataset_by_name("adult")`.
    """
    df, original_columns = load_raw_adult(drop_na=drop_na, csv_path=csv_path)
    df, group_id = preprocess_adult(df, positive_label=positive_label, binary_attrs=binary_attrs)

    pg_table = None
    pg = None
    if compute_pg:
        pg_table, pg = compute_pg_dirichlet(
            df=df,
            group_col=ADULT_SPEC.group_col,
            alpha=pg_alpha,
            return_intervals=True,
            num_draws=pg_num_draws,
            ci=pg_ci,
            seed=seed,
        )

    return LoadedDataset(
        df=df,
        original_columns=original_columns,
        group_id=group_id,
        pg_table=pg_table,
        pg=pg,
        metadata={"spec": ADULT_SPEC},
    )


# # -------------------------------------------------------------
# # DATA EXTRACTION
# # -------------------------------------------------------------

# def get_adult_sets(
#     df: pd.DataFrame,
#     protected_cols: Sequence[str]=("gender", "race", "native-country"),
#     label_col: str="income",
#     device: Optional[DeviceLike] = None,
# ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
#     """
#     Extract X, S, y from the Adult dataset as tensors.

#     Returns:
#         X: float32 tensor of features
#         S: float32 tensor of protected attributes
#         y: long tensor of labels
#     """
#     _validate_dataframe(df)
#     _validate_columns_exist(df, list(protected_cols) + [label_col])

#     df = df.copy()
#     feature_cols = [col for col in df.columns if col not in list(protected_cols) + [label_col, "group", "group_id"]]

#     # cat_cols = [c for c in feature_cols if df[c].dtype == "object"]
#     # num_cols = [c for c in feature_cols if df[c].dtype != "object"]

#     cat_cols, cont_cols = _infer_cat_and_cont_cols(df, feature_cols)

#     print(f"Categorical columns are: {cat_cols}")
#     print(f"Continuous (numerical) columns are: {cont_cols}")

#     df_cat = df[cat_cols]
#     df_cont = df[cont_cols]

#     one_hot_encoder = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
#     X_cat = one_hot_encoder.fit_transform(df_cat).astype(np.float32) if len(cat_cols) > 0 else np.zeros((len(df), 0), dtype=np.float32)

#     scaler = StandardScaler()
#     X_cont = scaler.fit_transform(df_cont).astype(np.float32) if len(cont_cols) > 0 else np.zeros((len(df), 0), dtype=np.float32)

#     X_np = np.hstack([X_cont, X_cat]).astype(np.float32)
#     y_np = df[label_col].astype(np.int64).values
#     print(f"Label instances\n{df[label_col].value_counts(dropna=False)}")

#     df["gender"] = df["gender"].astype("float32")
#     df["native-country"] = df["native-country"].astype("float32")
#     if str(df["race"].dtype) == "category":
#         df["race"] = df["race"].cat.codes.astype("float32")
#     else:
#         df["race"] = df["race"].astype("category").cat.codes.astype("float32")

#     S_np = df[list(protected_cols)].values.astype(np.float32)

#     X = torch.tensor(X_np, dtype=torch.float32, device=device)
#     S = torch.tensor(S_np, dtype=torch.float32, device=device)
#     y = torch.tensor(y_np, dtype=torch.long, device=device)

#     return X, S, y

