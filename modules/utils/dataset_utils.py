from __future__ import annotations

import importlib
from dataclasses import dataclass, field
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import torch

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from torch.utils.data import Dataset, DataLoader, TensorDataset

from modules.utils.tensor_utils import (
    _to_long_tensor,
    _to_float_tensor,
    DeviceLike,
)

# ----------------------------
# DATASET SPECIFICATION
# ----------------------------
@dataclass(frozen=True)
class DatasetSpec:
    """
    Generic metadata needed by reusable dataset utilities.
    Dataset-specific files should define one of these, for example:
    ADULT_SPEC = DatasetSpec(
        name="adult_income",
        protected_cols=["gender", "race", "native-country"],      
        label_col="income",
        group_col="group",
        group_id_col="group_id",
        log1p_cols=["capital-gain", "capital-loss"],
        )
    """
    name: str
    protected_cols: Sequence[str, ...]
    label_col: str
    group_col: str = "group"
    group_id_col: str = "group_id"
    log1p_cols: Tuple[str, ...] = ()
    drop_feature_cols: Tuple[str, ...] = ()

@dataclass
class LoadedDataset:
    """
    Standard return object for dataset loaders.
    """
    df: pd.DataFrame
    original_columns: List[str]
    group_id: Dict[str, int] = field(default_factory=dict)
    pg_table: Optional[pd.DataFrame] = None
    pg: Optional[Dict[str, float]] = None
    metadata: Dict[str, Any] = field(default_factory=dict)

# -------------------------------------------------------------
# DYNAMIC DATASET LOADING
# -------------------------------------------------------------
def load_dataset_by_name(
        dataset_name: str,
        *,
        package: str = "modules.data",
        **kwargs: Any,
) -> LoadedDataset:
    """
    Load a dataset by dynamically importing `modules.data.<dataset_name>`.

    This avoids editing `dataset_utils.py` whenever a new dataset is added.

    Requirements for each dataset module:
        - define `load_dataset(**kwargs) -> LoadedDataset`

    Example:
        adult = load_dataset_by_name("adult", drop_na=True)
        compas = load_dataset_by_name("compas", drop_na=True)
    """
    if not dataset_name.isidentifier():
        raise ValueError(
            "dataset_name must be a valid Python module name, e.g. 'adult' or 'law_school'."
        )

    module = importlib.import_module(f"{package}.{dataset_name}")

    if not hasattr(module, "load_dataset"):
        raise AttributeError(
            f"Dataset module '{package}.{dataset_name}' must define a load_dataset(...) function."
        )

    loaded = module.load_dataset(**kwargs)
    if not isinstance(loaded, LoadedDataset):
        raise TypeError(
            f"{package}.{dataset_name}.load_dataset(...) must return LoadedDataset, "
            f"got {type(loaded)!r}."
        )

    return loaded

# -------------------------------------------------------------
# VALIDATION HELPERS
# -------------------------------------------------------------
def _validate_dataframe(df: pd.DataFrame, name: str = "df") -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError(f"{name} must be a pandas DataFrame.")


def _validate_columns_exist(df: pd.DataFrame, columns: Sequence[str], df_name: str = "df") -> None:
    # if not isinstance(columns, str):
    #     columns = [columns]
    missing = [col for col in columns if col not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {df_name}: {missing}")


def _infer_cat_and_cont_cols(
    df: pd.DataFrame,
    feature_cols: Sequence[str],
) -> Tuple[List[str], List[str]]:
    cat_cols = []
    cont_cols = []

    for col in feature_cols:
        if pd.api.types.is_object_dtype(df[col]) or isinstance(df[col].dtype, pd.CategoricalDtype):
            cat_cols.append(col)
        else:
            cont_cols.append(col)

    return cat_cols, cont_cols

def _apply_log1p(
    df: pd.DataFrame,
    cols: Sequence[str],
) -> pd.DataFrame:
    df = df.copy()
    for col in cols:
        if col in df.columns:
            df[col] = np.log1p(df[col].astype(np.float32))
    return df


def _feature_columns_from_spec(df: pd.DataFrame, spec: DatasetSpec) -> List[str]:
    drop_cols = (
        set(spec.protected_cols)
        | {spec.label_col, spec.group_col, spec.group_id_col}
        | set(spec.drop_feature_cols)
    )
    return [col for col in df.columns if col not in drop_cols]

def _series_to_str_with_na(s: pd.Series, missing_token: str = "NA") -> pd.Series:
    """
    Convert any Series, including categorical Series, to string values with a stable
    missing-value token.

    This avoids pandas Categorical fillna errors when missing_token is not already
    an existing category.
    """
    return s.astype("object").where(s.notna(), missing_token).astype(str)

# -------------------------------------------------------------
# TENSOR DATASET HELPERS
# -------------------------------------------------------------
def to_tensor_dataset_predictive(X: Any, y: Any) -> TensorDataset:
    """
    Build a TensorDataset for predictive modeling.

    Returns:
        TensorDataset(X, y), with dtypes float32 and long.
    """
    X_t = _to_float_tensor(X)
    y_t = _to_long_tensor(y)
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
# TORCH DATASET HELPERS
# -------------------------------------------------------------
class TabularDataset(Dataset):
    """Basic Dataset for named tensors."""

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
# LOAD DATASETS AND PREPROCESSING
# -------------------------------------------------------------
def split_df(
    df: pd.DataFrame,
    *, 
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
    *,
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
# SCHEMA FIT / TRANSFORM FOR PREDICTOR
# -------------------------------------------------------------
def _is_numeric_series(s: pd.Series) -> bool:
    return pd.api.types.is_numeric_dtype(s)

def _fit_protected_schema(
    df_train: pd.DataFrame,
    protected_cols: Sequence[str],
    unknown_token: str,
) -> Dict[str, Any]:
    protected_schema: Dict[str, Any] = {}

    for col in protected_cols:
        s = df_train[col]
        if _is_numeric_series(s):
            protected_schema[col] = {"kind": "numeric"}
        else:
            # values = s.fillna("NA").astype(str)
            values = _series_to_str_with_na(s)
            categories = pd.Series(values).astype("category").cat.categories.tolist()
            if unknown_token not in categories:
                categories.append(unknown_token)
            protected_schema[col] = {"kind": "categorical", "categories": categories}

    return protected_schema

def _transform_protected(
    df: pd.DataFrame,
    protected_schema: Mapping[str, Any],
    protected_cols: Sequence[str],
    unknown_token: str,
) -> np.ndarray:
    columns: List[np.ndarray] = []

    for col in protected_cols:
        schema = protected_schema[col]
        if schema["kind"] == "numeric":
            values = df[col].astype(np.float32).to_numpy()
        else:
            categories = list(schema["categories"])
            category_to_id = {category: idx for idx, category in enumerate(categories)}
            unk_id = category_to_id[unknown_token]
            # values_str = df[col].fillna("NA").astype(str).to_numpy()
            values_str = _series_to_str_with_na(df[col]).to_numpy()
            values = np.array([category_to_id.get(v, unk_id) for v in values_str], dtype=np.float32)

        columns.append(values.reshape(-1, 1))

    if len(columns) == 0:
        return np.zeros((len(df), 0), dtype=np.float32)

    return np.hstack(columns).astype(np.float32)


def fit_predictor_schema(
    df_train: pd.DataFrame,
    spec: DatasetSpec,
    *,
    unknown_token: str = "__UNK__",
    append_protected: bool = True,
) -> Dict[str, Any]:
    """Fit preprocessing schema for a predictive model.

    The predictor uses:
        - one-hot encoded categorical non-protected features
        - standardized continuous non-protected features
        - optionally encoded protected attributes appended at the end
    """
    _validate_dataframe(df_train)
    _validate_columns_exist(df_train, list(spec.protected_cols) + [spec.label_col])

    feature_cols = _feature_columns_from_spec(df_train, spec)
    cat_cols, cont_cols = _infer_cat_and_cont_cols(df_train, feature_cols)

    ohe = OneHotEncoder(sparse_output=False, handle_unknown="ignore")
    scaler = StandardScaler()

    if len(cat_cols) > 0:
        # ohe.fit(df_train[cat_cols].fillna("NA").astype(str))
        ohe.fit(
            pd.DataFrame(
                {col: _series_to_str_with_na(df_train[col]) for col in cat_cols},
                index=df_train.index,
            )
        )

    if len(cont_cols) > 0:
        Xc = df_train[cont_cols].astype(np.float32).copy()
        Xc = _apply_log1p(Xc, spec.log1p_cols)
        scaler.fit(Xc.values)

    protected_schema = _fit_protected_schema(
        df_train=df_train,
        protected_cols=spec.protected_cols,
        unknown_token=unknown_token,
    )

    return {
        "spec": spec,
        "feature_cols": feature_cols,
        "cat_cols": cat_cols,
        "cont_cols": cont_cols,
        "ohe": ohe,
        "scaler": scaler,
        "protected_schema": protected_schema,
        "append_protected": append_protected,
        "unknown_token": unknown_token,
    }


def transform_predictor_schema(
    df: pd.DataFrame,
    schema: Mapping[str, Any],
    *,
    device: Optional[DeviceLike] = None,
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Transform a DataFrame for predictive modeling.

    Returns:
        X: float32 tensor of shape [N, D]
        y: long tensor of shape [N]
        g_ids: long tensor of shape [N]
    """
    _validate_dataframe(df)

    spec: DatasetSpec = schema["spec"]
    _validate_columns_exist(df, list(spec.protected_cols) + [spec.label_col, spec.group_id_col])

    cat_cols = list(schema["cat_cols"])
    cont_cols = list(schema["cont_cols"])
    ohe = schema["ohe"]
    scaler = schema["scaler"]

    if len(cat_cols) > 0:
        # X_cat = ohe.transform(df[cat_cols].fillna("NA").astype(str)).astype(np.float32)
        X_cat_df = pd.DataFrame(
            {col: _series_to_str_with_na(df[col]) for col in cat_cols},
            index=df.index,
        )
        X_cat = ohe.transform(X_cat_df).astype(np.float32)
    else:
        X_cat = np.zeros((len(df), 0), dtype=np.float32)

    if len(cont_cols) > 0:
        Xc = df[cont_cols].astype(np.float32).copy()
        Xc = _apply_log1p(Xc, spec.log1p_cols)
        X_cont = scaler.transform(Xc.values).astype(np.float32)
    else:
        X_cont = np.zeros((len(df), 0), dtype=np.float32)

    blocks = [X_cont, X_cat]

    if schema.get("append_protected", True):
        S_np = _transform_protected(
            df=df,
            protected_schema=schema["protected_schema"],
            protected_cols=spec.protected_cols,
            unknown_token=schema["unknown_token"],
        )
        blocks.append(S_np)

    X_np = np.hstack(blocks).astype(np.float32)
    y_np = df[spec.label_col].astype(np.int64).values
    g_ids_np = df[spec.group_id_col].astype(np.int64).values

    return (
        torch.tensor(X_np, dtype=torch.float32, device=device),
        torch.tensor(y_np, dtype=torch.long, device=device),
        torch.tensor(g_ids_np, dtype=torch.long, device=device),
    )



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
    if not 0.0 < ci < 1.0:
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

    result = pd.DataFrame(
        {
            "group": groups,
            "count": counts.cpu().numpy().astype(int),
            "p_mean": p_mean.cpu().numpy(),
        }
    )

    if return_intervals:
        torch.manual_seed(seed)
        posterior = torch.distributions.Dirichlet(counts + alpha_t)
        samples = posterior.sample((num_draws,))

        lo_q = (1.0 - ci) / 2.0
        hi_q = 1.0 - lo_q

        result["p_lo"] = torch.quantile(samples, lo_q, dim=0).cpu().numpy()
        result["p_hi"] = torch.quantile(samples, hi_q, dim=0).cpu().numpy()

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
    if not 0.0 < ci < 1.0:
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