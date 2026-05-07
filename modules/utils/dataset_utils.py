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

# -------------------------------------------------------------
# INTERNAL HELPERS
# -------------------------------------------------------------
def _validate_dataframe(df: pd.DataFrame, name: str = "df") -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")


def _validate_columns_exist(df: pd.DataFrame, columns: Sequence[str], df_name: str = "df") -> None:
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {df_name}: {missing}")


def _infer_cat_and_cont_cols(df: pd.DataFrame, feature_cols: Sequence[str]) -> Tuple[List[str], List[str]]:
    cat_cols = [col for col in feature_cols if df[col].dtype == "object"]
    cont_cols = [col for col in feature_cols if df[col].dtype != "object"]
    return cat_cols, cont_cols


# -------------------------------------------------------------
# TENSOR DATASET HELPERS
# -------------------------------------------------------------
def to_tensor_dataset_ar(X_cat: Any, g: Any) -> TensorDataset:
    """
    Build a TensorDataset for autoregressive categorical modeling.

    Returns:
        TensorDataset(X_cat, g), with dtypes long and long.
    """
    X_cat_t = _to_long_tensor(X_cat)
    g_t = _to_long_tensor(g)
    if X_cat_t.shape[0] != g_t.shape[0]:
        raise ValueError("X_cat and g must have the same first dimension.")
    return TensorDataset(X_cat_t, g_t)


def to_tensor_dataset_flow(X_cat: Any, X_cont: Any, g: Any) -> TensorDataset:
    """
    Build a TensorDataset for conditional flow modeling.

    Returns:
        TensorDataset(X_cat, X_cont, g), with dtypes long, float32, long.
    """
    X_cat_t = _to_long_tensor(X_cat)
    X_cont_t = _to_float_tensor(X_cont)
    g_t = _to_long_tensor(g)

    n = X_cat_t.shape[0]
    if X_cont_t.shape[0] != n or g_t.shape[0] != n:
        raise ValueError("X_cat, X_cont, and g must have the same first dimension.")

    return TensorDataset(X_cat_t, X_cont_t, g_t)


def to_tensor_dataset_predictive(X: Any, y: Any) -> TensorDataset:
    """
    Build a TensorDataset for predictive modeling.

    Returns:
        TensorDataset(X, y), with dtypes float32 and float32.
    """
    X_t = _to_float_tensor(X)
    y_t = _to_float_tensor(y)
    if X_t.shape[0] != y_t.shape[0]:
        raise ValueError("X and y must have the same first dimension.")
    return TensorDataset(X_t, y_t)


def make_loader(dataset: Dataset, batch_size: int, shuffle: bool) -> DataLoader:
    """
    Create a DataLoader from a dataset.
    """
    if batch_size <= 0:
        raise ValueError("batch_size must be positive.")
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle, drop_last=False)

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


def split_df(
    df: pd.DataFrame, 
    test_size: float=0.25, 
    val_size: float=0.15, 
    seed: int=42, 
    stratify: str="group"
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame into train, validation, and test sets.

    Note:
        val_size is applied to the remaining training portion after removing test.
    """
    _validate_dataframe(df)
    _validate_columns_exist(df, [stratify])

    df_train, df_test = train_test_split(
        df,
        test_size=test_size,
        random_state=seed,
        stratify=df[stratify],
    )
    df_train, df_val = train_test_split(
        df_train,
        test_size=val_size,
        random_state=seed,
        stratify=df_train[stratify],
    )

    df_train = df_train.reset_index(drop=True)
    df_val = df_val.reset_index(drop=True)
    df_test = df_test.reset_index(drop=True)

    return df_train, df_val, df_test

def split_df_with_indices(
    df: pd.DataFrame, 
    test_size: float=0.25, 
    val_size: float=0.15, 
    seed: int=42, 
    stratify: str="group",
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray, np.ndarray, np.ndarray]:
    """
    Split a DataFrame into train, validation, and test sets, returning row indices too.

    Note:
        val_size is applied to the remaining training indices after removing test.
    """
    _validate_dataframe(df)
    _validate_columns_exist(df, [stratify])

    idx = np.arange(len(df))

    train_idx, test_idx = train_test_split(
        idx, test_size=test_size, random_state=seed, stratify=df[stratify].values
    )
    train_idx, val_idx = train_test_split(
        train_idx, test_size=val_size, random_state=seed, stratify=df.iloc[train_idx][stratify].values
    )

    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_val = df.iloc[val_idx].copy().reset_index(drop=True)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)

    return df_train, df_val, df_test, train_idx, val_idx, test_idx   

# -------------------------------------------------------------
# SCHEMA FIT / TRANSFORM FOR p(x)
# -------------------------------------------------------------

def fit_schema_px(
    df_train: pd.DataFrame,
    protected_cols: Sequence[str]=("gender", "race", "native-country"),
    label_col: str="income",
    group_id_col: str="group_id",
    log1p_cols: Sequence[str]=("capital-gain", "capital-loss"),
    drop_feature_cols: Sequence[str]=(),
) -> Dict[str, Any]:
    """
    Fit schema for p(x) modeling from the training split only.

    Returns a dictionary with:
        - cat_cols
        - cont_cols
        - cat_vocabs
        - scaler
        - log1p_cols
        - drop_feature_cols
    """
    _validate_dataframe(df_train)

    drop_cols = set(protected_cols) | {label_col, "group", group_id_col} | set(drop_feature_cols)
    feature_cols = [col for col in df_train.columns if col not in drop_cols]

    # cat_cols = [c for c in feature_cols if df_train[c].dtype == "object"]
    # cont_cols = [c for c in feature_cols if df_train[c].dtype != "object"]
    cat_cols, cont_cols = _infer_cat_and_cont_cols(df_train, feature_cols)

    cat_vocabs: Dict[str, List[str]] = {}
    for col in cat_cols:
        s = df_train[col].fillna("NA").astype(str)
        vocab = pd.Series(s).astype("category").cat.categories.tolist()
        cat_vocabs[col] = vocab

    scaler = StandardScaler()
    df_cont = df_train[cont_cols].astype(np.float32).copy()

    for col in log1p_cols:
        if col in df_cont.columns:
            df_cont[col] = np.log1p(df_cont[col])

    if len(cont_cols) > 0:
        scaler.fit(df_cont.values)

    schema = {
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "cat_vocabs": cat_vocabs,
        "scaler": scaler,
        "log1p_cols": set(log1p_cols),
        "drop_feature_cols": set(drop_feature_cols),
    }
    return schema


def transform_px(
    df: pd.DataFrame,
    schema: Mapping[str, Any],
    group_id_col: str = "group_id",
    unknown_token: str = "__UNK__",
    device: Optional[DeviceLike] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, List[int]]:
    """
    Apply a fitted p(x) schema.

    Returns:
        X_cat: [N, num_cat] long tensor
        X_cont: [N, num_cont] float32 tensor
        g_ids: [N] long tensor
        vocab_sizes: list of categorical vocabulary sizes
    """
    _validate_dataframe(df)
    _validate_columns_exist(df, [group_id_col])

    df = df.copy()
    cat_cols = list(schema["cat_cols"])
    cont_cols = list(schema["cont_cols"])
    cat_vocabs = dict(schema["cat_vocabs"])
    scaler = schema["scaler"]
    log1p_cols = schema["log1p_cols"]

    X_cat_list: List[np.ndarray] = []
    vocab_sizes: List[int] = []

    for col in cat_cols:
        vocab = list(cat_vocabs[col])
        if unknown_token not in vocab:
            vocab = vocab + [unknown_token]

        vocab_index = {tok: i for i, tok in enumerate(vocab)}
        values = df[col].fillna("NA").astype(str).values
        ids = np.array([vocab_index.get(tok, vocab_index[unknown_token]) for tok in values], dtype=np.int64)

        X_cat_list.append(ids)
        vocab_sizes.append(len(vocab))

    if len(cat_cols) > 0:
        X_cat_np = np.stack(X_cat_list, axis=1)
    else:
        X_cat_np = np.zeros((len(df), 0), dtype=np.int64)

    if len(cont_cols) > 0:
        X_cont_df = df[cont_cols].astype(np.float32).copy()
        for col in cont_cols:
            if col in log1p_cols:
                X_cont_df[col] = np.log1p(X_cont_df[col])
        X_cont_np = scaler.transform(X_cont_df.values).astype(np.float32)
    else:
        X_cont_np = np.zeros((len(df), 0), dtype=np.float32)

    g_ids_np = df[group_id_col].astype(np.int64).values

    X_cat = torch.tensor(X_cat_np, dtype=torch.long, device=device)
    X_cont = torch.tensor(X_cont_np, dtype=torch.float32, device=device)
    g_ids = torch.tensor(g_ids_np, dtype=torch.long, device=device)

    return X_cat, X_cont, g_ids, vocab_sizes


# -------------------------------------------------------------
# SCHEMA FIT / TRANSFORM FOR PREDICTOR
# -------------------------------------------------------------
def fit_predictor_schema(
    df_train: pd.DataFrame,
    protected_cols: Sequence[str]=("gender", "race", "native-country"),
    label_col: str="income",
    log1p_cols: Sequence[str]=("capital-gain", "capital-loss"),
) -> Dict[str, Any]:
    """
    Fit preprocessing schema for the predictive model.

    The predictor uses:
        - one-hot encoded categorical non-protected features
        - standardized continuous non-protected features
        - raw protected attributes appended at the end
    """
    _validate_dataframe(df_train)

    drop_cols = set(protected_cols) | {label_col, "group", "group_id"}
    feature_cols = [col for col in df_train.columns if col not in drop_cols]
    # cat_cols = [col for col in feature_cols if df_train[col].dtype == "object"]
    # cont_cols = [col for col in feature_cols if df_train[col].dtype != "object"]
    cat_cols, cont_cols = _infer_cat_and_cont_cols(df_train, feature_cols)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    scaler = StandardScaler()

    if len(cat_cols) > 0:
        ohe.fit(df_train[cat_cols].fillna("NA").astype(str))

    if len(cont_cols) > 0:
        Xc = df_train[cont_cols].astype(np.float32).copy()
        for col in log1p_cols:
            if col in Xc.columns:
                Xc[col] = np.log1p(Xc[col])
        scaler.fit(Xc.values)

    schema = {
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "ohe": ohe,
        "scaler": scaler,
        "log1p_cols": set(log1p_cols),
        "protected_cols": tuple(protected_cols),
        "label_col": label_col,
    }
    return schema


def transform_predictor(
    df: pd.DataFrame,
    schema: Mapping[str, Any],
    group_id_col: str = "group_id",
    protected_cols: Sequence[str]=("gender", "race", "native-country"),
    label_col: str="income",
    device: Optional[DeviceLike] = None,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Transform a DataFrame for predictive modeling.

    Returns:
        X: float32 tensor of shape [N, D]
        y: long tensor of shape [N]
    """
    _validate_dataframe(df)
    _validate_columns_exist(df, list(protected_cols) + [label_col])
    _validate_columns_exist(df, group_id_col)

    cat_cols = list(schema["cat_cols"])
    cont_cols = list(schema["cont_cols"])
    ohe = schema["ohe"]
    scaler = schema["scaler"]
    log1p_cols = set(schema["log1p_cols"])

    if len(cat_cols) > 0:
        X_cat = ohe.transform(df[cat_cols].fillna("NA").astype(str)).astype(np.float32)
    else:
        X_cat = np.zeros((len(df), 0), dtype=np.float32)

    if len(cont_cols) > 0:
        Xc = df[cont_cols].astype(np.float32).copy()
        for col in cont_cols:
            if col in log1p_cols:
                Xc[col] = np.log1p(Xc[col])
        X_cont = scaler.transform(Xc.values).astype(np.float32)
    else:
        X_cont = np.zeros((len(df), 0), dtype=np.float32)

    S = df[list(protected_cols)].copy()
    if str(S["race"].dtype) == "category":
        S["race"] = S["race"].cat.codes
    else:
        S["race"] = S["race"].astype("category").cat.codes
    S_np = S.astype(np.float32).values

    X_np = np.hstack([X_cont, X_cat, S_np]).astype(np.float32)
    y_np = df[label_col].astype(np.int64).values

    X = torch.tensor(X_np, dtype=torch.float32, device=device)
    y = torch.tensor(y_np, dtype=torch.long, device=device)

    # Compute group IDs for reference (not used in the predictor but may be useful for analysis)
    g_ids_np = df[group_id_col].astype(np.int64).values
    g_ids = torch.tensor(g_ids_np, dtype=torch.long, device=device)

    return X, y, g_ids


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


# def get_sets(
#     df: pd.DataFrame, 
#     device: Optional[DeviceLike] = None):
#     """
#     Dispatch to the appropriate dataset-specific extractor based on label columns.
#     """
#     _validate_dataframe(df)

#     df = df.copy()

#     if "two_year_recid" in df.columns:
#         return get_compas_sets(df, device=device)
#     elif "income" in df.columns:
#         return get_adult_sets(df, device=device)
#     elif "pass_bar" in df.columns:
#         return get_law_school_sets(df, device=device)
#     else:
#         raise ValueError("DataFrame does not contain recognized label columns.")


# -------------------------------------------------------------
# TORCH DATASET HELPERS
# -------------------------------------------------------------
class TabularDataset(Dataset):
    """
    Basic Dataset for named tensors.

    Example:
        ds = TabularDataset(X=X, y=y)
        sample = ds[0]  # {"X": X[0], "y": y[0]}
    """

    def __init__(self, **tensors: torch.Tensor) -> None:
        if len(tensors) == 0:
            raise ValueError("At least one tensor must be provided.")

        lengths = {name: tensor.shape[0] for name, tensor in tensors.items()}
        if len(set(lengths.values())) != 1:
            raise ValueError(f"All tensors must have the same first dimension. Got: {lengths}")

        self.tensors = tensors
        self.length = next(iter(lengths.values()))

    def __len__(self) -> int:
        return self.length

    def __getitem__(self, idx: int) -> Dict[str, torch.Tensor]:
        return {name: tensor[idx] for name, tensor in self.tensors.items()}


# -------------------------------------------------------------
# GROUP PROBABILITY ESTIMATION
# -------------------------------------------------------------
def compute_pg_dirichlet(
    df: pd.DataFrame,
    group_col: str = "group",
    alpha: float = 1.0,
    return_intervals: bool = False,
    num_draws: int = 20000,
    ci: float = 0.95,
    seed: int = 42,
    device: Optional[DeviceLike] = None,
) -> Tuple[pd.DataFrame, Dict[str, float]]:
    """
    Compute p(g) using Dirichlet-smoothed counts.

    Posterior mean:
        p_mean(g) = (count_g + alpha) / (N + K * alpha)

    Args:
        df: Input DataFrame.
        group_col: Name of the group column.
        alpha: Symmetric Dirichlet concentration parameter.
        return_intervals: Whether to estimate posterior credible intervals by sampling.
        num_draws: Number of Dirichlet posterior samples.
        ci: Credible interval mass.
        seed: Random seed for posterior sampling.
        device: Torch device.

    Returns:
        result: DataFrame with group, count, p_mean, and optional interval columns.
        pg_dict: Dict mapping group string to posterior mean probability.
    """
    _validate_dataframe(df)
    _validate_columns_exist(df, [group_col])

    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if return_intervals and num_draws <= 0:
        raise ValueError("num_draws must be positive when return_intervals=True.")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1).")

    g = df[group_col]

    if isinstance(g.dtype, pd.CategoricalDtype):
        categories = g.cat.categories
        counts_np = g.value_counts(sort=False).reindex(categories, fill_value=0).to_numpy()
        groups = categories.astype(str).to_numpy()
    else:
        vc = g.astype(str).value_counts()
        groups = vc.index.to_numpy()
        counts_np = vc.to_numpy()

    counts = torch.tensor(counts_np, dtype=torch.float32, device=device)
    N = counts.sum()
    K = counts.numel()

    if N.item() == 0 or K == 0:
        raise ValueError("No data or no groups found for computing p(g).")

    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)
    p_mean = (counts + alpha_t) / (N + K * alpha_t)

    result = pd.DataFrame({
        "group": groups,
        "count": counts.cpu().numpy().astype(int),
        "p_mean": p_mean.cpu().numpy(),
    })

    if return_intervals:
        torch.manual_seed(seed)
        posterior = torch.distributions.Dirichlet(counts + alpha_t)
        samples = posterior.sample((num_draws,))  # [num_draws, K]

        lo_q = (1.0 - ci) / 2.0
        hi_q = 1.0 - lo_q

        p_lo = torch.quantile(samples, lo_q, dim=0)
        p_hi = torch.quantile(samples, hi_q, dim=0)

        result["p_lo"] = p_lo.cpu().numpy()
        result["p_hi"] = p_hi.cpu().numpy()

    result = result.sort_values("p_mean", ascending=True).reset_index(drop=True)
    pg_dict = {row["group"]: float(row["p_mean"]) for _, row in result.iterrows()}

    return result, pg_dict


def compute_pg_dirichlet_from_groups(
    groups: Any,
    K: Optional[int] = None,
    alpha: float = 1.0,
    return_intervals: bool = False,
    num_draws: int = 20000,
    ci: float = 0.95,
    seed: int = 0,
    device: Optional[DeviceLike] = None,
):
    """
    Compute Dirichlet-smoothed p(g) from integer group IDs.

    Args:
        groups: 1D array-like of integer group IDs.
        K: Total number of groups. If None, inferred as max(groups) + 1.
        alpha: Symmetric Dirichlet concentration parameter.
        return_intervals: Whether to return posterior credible intervals.
        num_draws: Number of Dirichlet posterior samples.
        ci: Credible interval mass.
        seed: Random seed.
        device: Torch device.

    Returns:
        If return_intervals is False:
            counts_long, p_mean
        else:
            counts_long, p_mean, p_lo, p_hi
    """
    if alpha <= 0:
        raise ValueError("alpha must be positive.")
    if return_intervals and num_draws <= 0:
        raise ValueError("num_draws must be positive when return_intervals=True.")
    if not (0.0 < ci < 1.0):
        raise ValueError("ci must be in (0, 1).")

    groups_t = torch.as_tensor(groups, dtype=torch.long, device=device).view(-1)

    if groups_t.numel() == 0:
        raise ValueError("groups is empty.")

    if K is None:
        K = int(groups_t.max().item()) + 1
    if K <= 0:
        raise ValueError("K must be positive.")

    gmin = int(groups_t.min().item())
    gmax = int(groups_t.max().item())

    if gmin < 0 or gmax >= K:
        raise ValueError(f"Group IDs must be in [0, {K - 1}]. Found min={gmin}, max={gmax}.")

    counts = torch.bincount(groups_t, minlength=K).to(dtype=torch.float32, device=device)
    N = counts.sum()
    if N.item() == 0:
        raise ValueError("No observations found in groups.")

    alpha_t = torch.tensor(alpha, dtype=torch.float32, device=device)
    p_mean = (counts + alpha_t) / (N + K * alpha_t)

    if not return_intervals:
        return counts.to(torch.long), p_mean

    torch.manual_seed(seed)
    posterior = torch.distributions.Dirichlet(counts + alpha_t)
    samples = posterior.sample((num_draws,))

    lo_q = (1.0 - ci) / 2.0
    hi_q = 1.0 - lo_q
    p_lo = torch.quantile(samples, lo_q, dim=0)
    p_hi = torch.quantile(samples, hi_q, dim=0)

    return counts.to(torch.long), p_mean, p_lo, p_hi