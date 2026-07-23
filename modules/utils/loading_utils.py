import json
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Tuple, Union

import numpy as np
import pandas as pd
import sklearn.preprocessing
import torch
from torch import nn
import yaml

from modules.models.predictive_models import Classifier, evaluate_ensemble
from modules.utils.dataset_utils import (
    fit_predictor_schema,
    load_dataset,
    make_loader,
    to_tensor_dataset_predictive,
    transform_predictor,
)


PathLike = Union[str, Path]
DeviceLike = Union[str, torch.device]

# -------------------------------------------------------------
# BASIC FILE HELPERS
# -------------------------------------------------------------

def _as_path(path: PathLike) -> Path:
    return Path(path)


def _load_json(path: PathLike) -> Dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        return json.load(f)


def _resolve_existing_path(*candidates: PathLike) -> Path:
    """
    Return the first existing path among candidates.
    """
    for candidate in candidates:
        path = Path(candidate)
        if path.exists():
            return path
    raise FileNotFoundError(
        "None of the candidate paths exists:\n" + "\n".join(str(Path(c)) for c in candidates)
    )

    
# -------------------------------------------------------------
# CONFIG / METADATA / ARTIFACTS
# -------------------------------------------------------------

def load_config_from_experiment(exp_folder: PathLike) -> Dict[str, Any]:
    """
    Load experiment config from resolved_config.yaml or config.json.
    """
    exp_path = _as_path(exp_folder)
    yaml_path = exp_path / "resolved_config.yaml"
    json_path = exp_path / "config.json"

    if yaml_path.exists():
        with yaml_path.open("r", encoding="utf-8") as f:
            return yaml.safe_load(f)

    if json_path.exists():
        return _load_json(json_path)

    raise FileNotFoundError(f"No resolved_config.yaml or config.json found in {exp_path}")


def load_split_indices(folder: PathLike) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Load train/val/test indices from split_indices.npz.
    """
    folder_path = _as_path(folder)
    data = np.load(folder_path / "split_indices.npz")
    return data["train_idx"], data["val_idx"], data["test_idx"]


def load_rng_states(folder: PathLike) -> Dict[str, Any]:
    """
    Load RNG states and restore Python / NumPy / Torch RNGs.

    Returns the loaded state dictionary.
    """
    folder_path = _as_path(folder)
    rng_states = torch.load(folder_path / "rng_states.pt", weights_only=False)

    if "python_random_state" in rng_states:
        random.setstate(rng_states["python_random_state"])

    if "numpy_random_state" in rng_states:
        np.random.set_state(rng_states["numpy_random_state"])

    if "torch_random_state" in rng_states:
        torch_state = rng_states["torch_random_state"]
        if isinstance(torch_state, np.ndarray):
            torch_state = torch.from_numpy(torch_state)
        torch.set_rng_state(torch_state.cpu())

    if rng_states.get("torch_cuda_random_state") is not None and torch.cuda.is_available():
        for i, state in enumerate(rng_states["torch_cuda_random_state"]):
            torch.cuda.set_rng_state(state, device=i)

    return rng_states


def load_optimizer_state(
    optimizer: torch.optim.Optimizer,
    checkpoint_path: PathLike,
    device: DeviceLike = "cpu",
) -> torch.optim.Optimizer:
    """
    Load optimizer state from checkpoint_path.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    optimizer.load_state_dict(state)
    return optimizer


def load_model_state(
    model: nn.Module,
    checkpoint_path: PathLike,
    device: DeviceLike = "cpu",
) -> nn.Module:
    """
    Load model state from checkpoint_path and put model in eval mode.
    """
    state = torch.load(checkpoint_path, map_location=device, weights_only=False)
    model.load_state_dict(state)
    model.to(device)
    model.eval()
    return model


def load_preprocessing_artifact(folder: PathLike) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load preprocessing artifacts and reconstruct a schema dict compatible with transform_px.
    """
    folder_path = _as_path(folder)
    artifacts = _load_json(folder_path / "preprocessing.json")

    scaler = sklearn.preprocessing.StandardScaler()
    if artifacts.get("scaler_mean") is not None and artifacts.get("scaler_scale") is not None:
        scaler.mean_ = np.array(artifacts["scaler_mean"], dtype=np.float64)
        scaler.scale_ = np.array(artifacts["scaler_scale"], dtype=np.float64)

        if artifacts.get("scaler_var") is not None:
            scaler.var_ = np.array(artifacts["scaler_var"], dtype=np.float64)

        if artifacts.get("scaler_n_features_in") is not None:
            scaler.n_features_in_ = int(artifacts["scaler_n_features_in"])

    schema = {
        "cat_cols": list(artifacts.get("cat_cols", [])),
        "cont_cols": list(artifacts.get("cont_cols", [])),
        "cat_vocabs": {k: list(v) for k, v in artifacts.get("cat_vocabs", {}).items()},
        "log1p_cols": set(artifacts.get("log1p_cols", [])),
        "drop_feature_cols": set(artifacts.get("drop_feature_cols", [])),
        "protected_cols": tuple(artifacts.get("protected_cols", [])),
        "label_col": artifacts.get("label_col"),
        "group_col": artifacts.get("group_col"),
        "group_id_col": artifacts.get("group_id_col"),
        "scaler": scaler,
    }
    return schema, artifacts


def load_predictor_schema(folder: PathLike) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """
    Load predictor preprocessing schema if it was saved separately.

    Expected file:
        predictor_schema.json

    This reconstructs:
        - cat_cols
        - cont_cols
        - log1p_cols
        - protected_cols
        - label_col
        - ohe
        - scaler
    """
    folder_path = _as_path(folder)
    artifacts = _load_json(folder_path / "predictor_schema.json")

    scaler = sklearn.preprocessing.StandardScaler()
    if artifacts.get("scaler_mean") is not None and artifacts.get("scaler_scale") is not None:
        scaler.mean_ = np.array(artifacts["scaler_mean"], dtype=np.float64)
        scaler.scale_ = np.array(artifacts["scaler_scale"], dtype=np.float64)
        if artifacts.get("scaler_var") is not None:
            scaler.var_ = np.array(artifacts["scaler_var"], dtype=np.float64)
        if artifacts.get("scaler_n_features_in") is not None:
            scaler.n_features_in_ = int(artifacts["scaler_n_features_in"])

    ohe = sklearn.preprocessing.OneHotEncoder(
        sparse_output=False,
        handle_unknown="ignore",
    )
    if "ohe_categories" in artifacts:
        ohe.categories_ = [np.array(cat, dtype=object) for cat in artifacts["ohe_categories"]]
        ohe._infrequent_enabled = False
        ohe.feature_names_in_ = np.array(artifacts.get("cat_cols", []), dtype=object)
        ohe.n_features_in_ = len(artifacts.get("cat_cols", []))

    schema = {
        "feature_cols": list(artifacts.get("feature_cols", [])),
        "cat_cols": list(artifacts.get("cat_cols", [])),
        "cont_cols": list(artifacts.get("cont_cols", [])),
        "ohe": ohe,
        "scaler": scaler,
        "log1p_cols": set(artifacts.get("log1p_cols", [])),
        "protected_cols": tuple(artifacts.get("protected_cols", [])),
        "label_col": artifacts.get("label_col"),
    }
    return schema, artifacts


def load_model_metadata(exp_folder: PathLike) -> Dict[str, Any]:
    """
    Load model metadata from models/model_metadata.json.
    """
    exp_path = _as_path(exp_folder)
    return _load_json(exp_path / "models" / "model_metadata.json")              


# -------------------------------------------------------------
# DATA RECONSTRUCTION
# -------------------------------------------------------------

def reconstruct_test_data(
    exp_folder: PathLike,
    cfg: Mapping[str, Any],
    device: Optional[DeviceLike] = None,
) -> Dict[str, Any]:
    """
    Reconstruct train/val/test splits and all tensors needed for testing.

    This is aligned with the cleaned dataset/preprocessing utilities.
    """
    exp_path = _as_path(exp_folder)

    df, _, _, _ = load_dataset(cfg["data"]["dataset_name"])
    train_idx, val_idx, test_idx = load_split_indices(exp_path / "splits")

    df_train = df.iloc[train_idx].copy().reset_index(drop=True)
    df_val = df.iloc[val_idx].copy().reset_index(drop=True)
    df_test = df.iloc[test_idx].copy().reset_index(drop=True)

    schema_px, px_artifacts = load_preprocessing_artifact(exp_path / "preprocessing")

    group_id_col = cfg["data"].get("group_id_col", px_artifacts.get("group_id_col", "group_id"))

    pred_schema_file = exp_path / "predictor_schema.json"
    if pred_schema_file.exists():
        pred_schema, _ = load_predictor_schema(exp_path)
    else:
        pred_schema = fit_predictor_schema(
            df_train=df_train,
            protected_cols=tuple(cfg["data"]["protected_attributes"]),
            label_col=cfg["data"]["label_col"],
            log1p_cols=tuple(cfg["preprocessing"]["log1p_cols"]),
        )

    X_pred_train, y_train = transform_predictor(
        df_train,
        pred_schema,
        protected_cols=tuple(cfg["data"]["protected_attributes"]),
        label_col=cfg["data"]["label_col"],
        device=device,
    )
    X_pred_val, y_val = transform_predictor(
        df_val,
        pred_schema,
        protected_cols=tuple(cfg["data"]["protected_attributes"]),
        label_col=cfg["data"]["label_col"],
        device=device,
    )
    X_pred_test, y_test = transform_predictor(
        df_test,
        pred_schema,
        protected_cols=tuple(cfg["data"]["protected_attributes"]),
        label_col=cfg["data"]["label_col"],
        device=device,
    )

    num_groups = int(df[group_id_col].nunique())

    return {
        "df_train": df_train,
        "df_val": df_val,
        "df_test": df_test,
        "X_pred_train": X_pred_train,
        "y_train": y_train,
        "X_pred_val": X_pred_val,
        "y_val": y_val,
        "X_pred_test": X_pred_test,
        "y_test": y_test,
        "num_groups": num_groups,
    }


# -------------------------------------------------------------
# MODEL RECONSTRUCTION
# -------------------------------------------------------------
def build_predictive_models_from_metadata(
    model_metadata: Mapping[str, Any],
    device: DeviceLike,
) -> list[Classifier]:
    """
    Build the predictive ensemble from saved metadata.
    """
    pred_meta = model_metadata["predictive_model"]
    ensemble_size = int(pred_meta.get("ensemble_size", 1))

    hidden = tuple(pred_meta.get("hidden", pred_meta.get("hidden_dims", [256, 128])))
    if len(hidden) != 2:
        raise ValueError("Predictive model metadata must define exactly two hidden dimensions.")

    models = []
    for _ in range(ensemble_size):
        model = Classifier(
            input_dim=int(pred_meta["input_dim"]),
            hidden=hidden,
            dropout=float(pred_meta["dropout"]),
        ).to(device)
        models.append(model)

    return models

def _resolve_generative_paths(exp_folder: PathLike) -> Dict[str, Path]:
    """
    Resolve generative model checkpoint paths for both the new and old folder layouts.
    """
    exp_path = _as_path(exp_folder)

    ar_path = _resolve_existing_path(
        exp_path / "generative" / "ar_model.pt",
        exp_path / "generative" / "run_0" / "ar_model.pt",
    )
    flow_path = _resolve_existing_path(
        exp_path / "generative" / "flow_model.pt",
        exp_path / "generative" / "run_0" / "flow_model.pt",
    )

    ctx_candidates = [
        exp_path / "generative" / "ctx_model.pt",
        exp_path / "generative" / "run_0" / "ctx_model.pt",
    ]
    ctx_path = None
    for candidate in ctx_candidates:
        if candidate.exists():
            ctx_path = candidate
            break

    return {
        "ar": ar_path,
        "flow": flow_path,
        "ctx": ctx_path,
    }


def _resolve_predictive_folder(exp_folder: PathLike) -> Path:
    """
    Resolve predictive ensemble folder for both the new and old folder layouts.
    """
    exp_path = _as_path(exp_folder)
    return _resolve_existing_path(
        exp_path / "predictive_ensemble",
        exp_path / "predictive_ensemble" / "ensemble_0",
    )


def load_all_models(
    exp_folder: PathLike,
    cfg: Mapping[str, Any],
    device: DeviceLike,
    debug_flow: bool = False,
    keep_last_batch_flow: bool = False,
) -> Tuple[list[Classifier], Dict[str, Any]]:
    """
    Rebuild and load all saved models.
    """
    model_metadata = load_model_metadata(exp_folder)

    predictive_models = build_predictive_models_from_metadata(model_metadata, device)
    pred_folder = _resolve_predictive_folder(exp_folder)

    for i, model in enumerate(predictive_models):
        model_path = pred_folder / f"predictor_{i}.pt"
        predictive_models[i] = load_model_state(model, model_path, device)

    return predictive_models, model_metadata


# -------------------------------------------------------------
# OPTIONAL OPTIMIZER LOADING
# -------------------------------------------------------------
def load_optimizers_if_available(
    exp_folder: PathLike,
    predictive_models: Sequence[Classifier],
    cfg: Mapping[str, Any],
    device: DeviceLike = "cpu",
) -> Dict[str, Any]:
    """
    Rebuild optimizers and load states if checkpoint files are available.
    """
    exp_path = _as_path(exp_folder)
    optimizers: Dict[str, Any] = {}

    pred_folder = _resolve_predictive_folder(exp_folder)

    predictive_optimizers = []
    for i, model in enumerate(predictive_models):
        opt = torch.optim.Adam(model.parameters(), lr=cfg["predictive_model"]["lr"])
        opt_path = pred_folder / f"predictor_optimizer_{i}.pt"
        if opt_path.exists():
            opt = load_optimizer_state(opt, opt_path, device=device)
        predictive_optimizers.append(opt)

    optimizers["predictive_optimizers"] = predictive_optimizers
    return optimizers

# -------------------------------------------------------------
# TESTING
# -------------------------------------------------------------
def test_predictive_ensemble(
    predictive_models: Sequence[Classifier],
    data_dict: Mapping[str, Any],
    device: DeviceLike,
    batch_size: int,
) -> Dict[str, Any]:
    """
    Evaluate the predictive ensemble on the test split.
    """
    pred_test_loader = make_loader(
        to_tensor_dataset_predictive(data_dict["X_pred_test"], data_dict["y_test"]),
        batch_size=batch_size,
        shuffle=False,
    )

    predictions, entropy, aleatoric, epistemic, mean_probs_2c = evaluate_ensemble(
        models=predictive_models,
        loader=pred_test_loader,
        device=device,
    )

    y_test_np = data_dict["y_test"].detach().cpu().numpy()
    pred_labels = (predictions >= 0.5).astype(np.int64)
    accuracy = float((pred_labels == y_test_np).mean())

    return {
        "predictions": predictions,
        "pred_labels": pred_labels,
        "entropy": entropy,
        "aleatoric": aleatoric,
        "epistemic": epistemic,
        "y_test": y_test_np,
        "accuracy": accuracy,
        "mean_probs_2c": mean_probs_2c,
    }