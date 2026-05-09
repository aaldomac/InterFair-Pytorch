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
    to_tensor_dataset_ar,
    to_tensor_dataset_flow,
    to_tensor_dataset_predictive,
    transform_predictor,
    transform_px,
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

    X_cat_train, X_cont_train, g_train, vocab_sizes = transform_px(
        df_train,
        schema_px,
        group_id_col=group_id_col,
        device=device,
    )
    X_cat_val, X_cont_val, g_val, _ = transform_px(
        df_val,
        schema_px,
        group_id_col=group_id_col,
        device=device,
    )
    X_cat_test, X_cont_test, g_test, _ = transform_px(
        df_test,
        schema_px,
        group_id_col=group_id_col,
        device=device,
    )

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
        "X_cat_train": X_cat_train,
        "X_cont_train": X_cont_train,
        "g_train": g_train,
        "X_cat_val": X_cat_val,
        "X_cont_val": X_cont_val,
        "g_val": g_val,
        "X_cat_test": X_cat_test,
        "X_cont_test": X_cont_test,
        "g_test": g_test,
        "X_pred_train": X_pred_train,
        "y_train": y_train,
        "X_pred_val": X_pred_val,
        "y_val": y_val,
        "X_pred_test": X_pred_test,
        "y_test": y_test,
        "vocab_sizes": vocab_sizes,
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
) -> Tuple[ARModel, Optional[ContextEncoder], ConditionalRealNVPFlow, list[Classifier], Dict[str, Any]]:
    """
    Rebuild and load all saved models.
    """
    model_metadata = load_model_metadata(exp_folder)

    ar_meta = model_metadata["ar_model"]
    flow_meta = model_metadata["flow_model"]
    ctx_meta = model_metadata.get("ctx_model")

    ar_model = ARModel(
        vocab_sizes=ar_meta["vocab_sizes"],
        num_groups=int(ar_meta["num_groups"]),
        g_emb_dim=int(ar_meta["g_emb_dim"]),
        x_emb_dim=int(ar_meta["x_emb_dim"]),
        hidden=int(ar_meta["hidden"]),
        dropout=float(ar_meta.get("dropout", 0.0)),
        num_layers=int(ar_meta.get("num_layers", 1)),
    ).to(device)

    ctx_model = None
    if ctx_meta is not None:
        ctx_model = ContextEncoder(
            vocab_sizes=ctx_meta["vocab_sizes"],
            num_groups=int(ctx_meta["num_groups"]),
            g_emb_dim=int(ctx_meta["g_emb_dim"]),
            x_emb_dim=int(ctx_meta["x_emb_dim"]),
            hidden=int(ctx_meta["hidden"]),
            out_dim=int(ctx_meta["out_dim"]),
        ).to(device)

    flow_model = ConditionalRealNVPFlow(
        dim=int(flow_meta["dim"]),
        context_dim=int(flow_meta["context_dim"]),
        num_couplings=int(flow_meta["num_couplings"]),
        hidden=int(flow_meta["hidden"]),
        seed=flow_meta.get("seed"),
        debug=debug_flow,
        keep_last_batch=keep_last_batch_flow,
    ).to(device)

    gen_paths = _resolve_generative_paths(exp_folder)
    ar_model = load_model_state(ar_model, gen_paths["ar"], device)
    flow_model = load_model_state(flow_model, gen_paths["flow"], device)

    if ctx_model is not None and gen_paths["ctx"] is not None:
        ctx_model = load_model_state(ctx_model, gen_paths["ctx"], device)

    predictive_models = build_predictive_models_from_metadata(model_metadata, device)
    pred_folder = _resolve_predictive_folder(exp_folder)

    for i, model in enumerate(predictive_models):
        model_path = pred_folder / f"predictor_{i}.pt"
        predictive_models[i] = load_model_state(model, model_path, device)

    return ar_model, ctx_model, flow_model, predictive_models, model_metadata


# -------------------------------------------------------------
# OPTIONAL OPTIMIZER LOADING
# -------------------------------------------------------------
def load_optimizers_if_available(
    exp_folder: PathLike,
    ar_model: ARModel,
    ctx_model: Optional[ContextEncoder],
    flow_model: ConditionalRealNVPFlow,
    predictive_models: Sequence[Classifier],
    cfg: Mapping[str, Any],
    device: DeviceLike = "cpu",
) -> Dict[str, Any]:
    """
    Rebuild optimizers and load states if checkpoint files are available.
    """
    exp_path = _as_path(exp_folder)
    optimizers: Dict[str, Any] = {}

    gen_folder = _resolve_existing_path(
        exp_path / "generative",
        exp_path / "generative" / "run_0",
    )
    pred_folder = _resolve_predictive_folder(exp_folder)

    ar_optimizer = torch.optim.Adam(ar_model.parameters(), lr=cfg["ar_model"]["lr"])

    if ctx_model is None:
        flow_params = list(flow_model.parameters())
    else:
        flow_params = list(flow_model.parameters()) + list(ctx_model.parameters())

    flow_optimizer = torch.optim.Adam(flow_params, lr=cfg["flow_model"]["lr"])

    ar_opt_path = gen_folder / "ar_optimizer.pt"
    flow_opt_path = gen_folder / "flow_optimizer.pt"

    if ar_opt_path.exists():
        ar_optimizer = load_optimizer_state(ar_optimizer, ar_opt_path, device=device)
    if flow_opt_path.exists():
        flow_optimizer = load_optimizer_state(flow_optimizer, flow_opt_path, device=device)

    predictive_optimizers = []
    for i, model in enumerate(predictive_models):
        opt = torch.optim.Adam(model.parameters(), lr=cfg["predictive_model"]["lr"])
        opt_path = pred_folder / f"predictor_optimizer_{i}.pt"
        if opt_path.exists():
            opt = load_optimizer_state(opt, opt_path, device=device)
        predictive_optimizers.append(opt)

    optimizers["ar_optimizer"] = ar_optimizer
    optimizers["flow_optimizer"] = flow_optimizer
    optimizers["predictive_optimizers"] = predictive_optimizers
    return optimizers

# -------------------------------------------------------------
# TESTING
# -------------------------------------------------------------

def test_generative_models(
    ar_model: ARModel,
    ctx_model: ContextEncoder,
    flow_model: ConditionalRealNVPFlow,
    data_dict: Mapping[str, Any],
    device: DeviceLike,
    batch_size: int,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Evaluate AR and flow models on the test split.

    Returns:
        gen_df: per-sample DataFrame
        summary: grouped summary by test group
    """
    ar_loader = make_loader(
        to_tensor_dataset_ar(data_dict["X_cat_test"], data_dict["g_test"]),
        batch_size=batch_size,
        shuffle=False,
    )
    flow_loader = make_loader(
        to_tensor_dataset_flow(
            data_dict["X_cat_test"],
            data_dict["X_cont_test"],
            data_dict["g_test"],
        ),
        batch_size=batch_size,
        shuffle=False,
    )

    ar_losses = []
    for x_cat, g in ar_loader:
        loss = eval_step_ar(ar_model, x_cat, g, device)
        ar_losses.append(loss)

    flow_losses = []
    for x_cat, x_cont, g in flow_loader:
        loss = eval_step_flow(flow_model, ctx_model, x_cont, x_cat, g, device)
        flow_losses.append(loss)

    reset_flow_debug_stats(flow_model)

    with torch.no_grad():
        ar_model.eval()
        ctx_model.eval()
        flow_model.eval()

        x_cat_test_t = data_dict["X_cat_test"].to(device=device, dtype=torch.long)
        g_test_t = data_dict["g_test"].to(device=device, dtype=torch.long)
        x_cont_test_t = data_dict["X_cont_test"].to(device=device, dtype=torch.float32)

        logp_cat_test = ar_model.log_prob(x_cat_test_t, g_test_t).detach().cpu().numpy()
        context_test = ctx_model(x_cat_test_t, g_test_t)
        logp_cont_test = flow_model.log_prob(x_cont_test_t, context_test).detach().cpu().numpy()
        logp_x_given_g_test = logp_cat_test + logp_cont_test

    print_flow_debug_summary(flow_model)

    groups_np = data_dict["g_test"].detach().cpu().numpy()

    gen_df = pd.DataFrame(
        {
            "group": groups_np,
            "logp_cat_test": logp_cat_test,
            "logp_cont_test": logp_cont_test,
            "logp_x_given_g_test": logp_x_given_g_test,
        }
    )

    gen_df["ar_test_nll"] = float(np.mean(ar_losses))
    gen_df["flow_test_nll"] = float(np.mean(flow_losses))

    summary = (
        gen_df.groupby("group", dropna=False)
        .agg(
            count=("group", "size"),
            mean_logp_cat=("logp_cat_test", "mean"),
            mean_logp_cont=("logp_cont_test", "mean"),
            mean_logp_total=("logp_x_given_g_test", "mean"),
            std_logp_total=("logp_x_given_g_test", "std"),
        )
        .reset_index()
    )

    return gen_df, summary


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