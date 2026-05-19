from __future__ import annotations

import json
import os
import platform
import random
import sys
from dataclasses import asdict, is_dataclass
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

import joblib
import numpy as np
import sklearn
import torch
import torch.nn as nn

try:
    import pandas as pd
except ImportError:  # pragma: no cover
    pd = None

try:
    from modules.pipelines.train_predictive_pipeline import PipelineResult
except ImportError:  # pragma: no cover
    PipelineResult = Any


PathLike = Union[str, os.PathLike]

# ---------------------------------------------------------------------
# GENERIC FILE HELPERS
# ---------------------------------------------------------------------
def _ensure_dir(path: PathLike) -> Path:
    """
    Create a directory if it does not exist and return it as a Path.
    """
    path_obj = Path(path)
    path_obj.mkdir(parents=True, exist_ok=True)
    return path_obj


def _count_existing_experiments(base_folder: PathLike, prefix: str = "run_") -> int:
    """
    Count entries in base_folder whose names start with the given prefix.
    """
    base_path = Path(base_folder)
    if not base_path.exists():
        return 0
    return sum(1 for entry in base_path.iterdir() if entry.name.startswith(prefix))
    

def _jsonable(value: Any) -> Any:
    """Convert common experiment objects into JSON-serializable values."""
    if is_dataclass(value):
        return _jsonable(asdict(value))

    if isinstance(value, Path):
        return str(value)

    if isinstance(value, Mapping):
        return {str(k): _jsonable(v) for k, v in value.items()}

    if isinstance(value, tuple):
        return [_jsonable(v) for v in value]

    if isinstance(value, list):
        return [_jsonable(v) for v in value]

    if isinstance(value, set):
        return sorted(_jsonable(v) for v in value)

    if isinstance(value, np.ndarray):
        return value.tolist()

    if isinstance(value, np.generic):
        return value.item()

    if torch.is_tensor(value):
        return value.detach().cpu().tolist()

    if isinstance(value, (str, int, float, bool)) or value is None:
        return value

    return str(value)


def _json_dump(data: Any, path: PathLike) -> None:
    """
    Save data as formatted JSON.
    """
    path_obj = Path(path)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(_jsonable(data), f, indent=2)

def make_experiment_folder(
    base_folder: PathLike = "saved_models",
    prefix: str = "run_",
    exact_path: Optional[PathLike] = None
) -> Path:
    """
    Create and return an experiment folder.
    If exact_path is given, use it directly.
    Otherwise create base_folder/prefix{k}
    """
    if exact_path is not None:
        folder = Path(exact_path)
    else:
        base_path = _ensure_dir(base_folder)
        k = _count_existing_experiments(base_path, prefix)
        folder = base_path / f"{prefix}{k}"

    return _ensure_dir(folder)

# ---------------------------------------------------------------------
# MODEL SAVING
# ---------------------------------------------------------------------
def _model_metadata(model: nn.Module, index: int) -> Dict[str, Any]:
    """Best-effort metadata for a PyTorch model."""
    metadata: Dict[str, Any] = {
        "index": index,
        "class": model.__class__.__name__,
        "module": model.__class__.__module__,
        "num_parameters": int(sum(p.numel() for p in model.parameters())),
        "num_trainable_parameters": int(sum(p.numel() for p in model.parameters() if p.requires_grad)),
    }

    # Compatible with the MLPClassifier proposed earlier.
    for attr in ("input_dim", "num_outputs"):
        if hasattr(model, attr):
            metadata[attr] = _jsonable(getattr(model, attr))

    return metadata

def save_predictive_ensemble(
    ensemble_models: Sequence[nn.Module], 
    ensemble_folder: Optional[PathLike]=None,
    exact_path: Optional[PathLike]=None,
    ensemble_optimizers: Optional[Sequence[torch.optim.Optimizer]]=None,
    create_run_subfolder: bool=True,
    metadata: Optional[Mapping[str, Any]] = None,
) -> Path:
    """
    Save a list of predictive models and optional optimizers.
    Files:
        predictor_0.pt, predictor_1.pt, ...
        optimizer_0.pt, optimizer_1.pt, ... (optional)
        predictive_ensemble_metadata.json
    """
    if len(ensemble_models) == 0:
        raise ValueError("ensemble_models must be non-empty.")

    if ensemble_optimizers is not None and len(ensemble_optimizers) != len(ensemble_models):
        raise ValueError("ensemble_optimizers must have the same length as ensemble_models.")

    if create_run_subfolder:
        save_folder = make_experiment_folder(
            base_folder="saved_models" if ensemble_folder is None else ensemble_folder,
            prefix="ensemble_",
            exact_path=exact_path,
        )
    else:
        target = exact_path if exact_path is not None else ensemble_folder
        if target is None:
            raise ValueError("Either ensemble_folder or exact_path must be provided.")
        save_folder = _ensure_dir(target)

    model_metadata = []
    for i, model in enumerate(ensemble_models):
        model_path = save_folder / f"predictor_{i}.pt"
        torch.save(model.state_dict(), model_path)

        item = _model_metadata(model, i)
        item["state_dict_file"] = model_path.name
        model_metadata.append(item)

    optimizer_metadata = []
    if ensemble_optimizers is not None:
        for i, optimizer in enumerate(ensemble_optimizers):
            optimizer_path = save_folder / f"predictor_optimizer_{i}.pt"
            torch.save(optimizer.state_dict(), optimizer_path)
            optimizer_metadata.append(
                {
                    "index": i,
                    "class": optimizer.__class__.__name__,
                    "state_dict_file": optimizer_path.name,
                }
            )

    _json_dump(
        {
            "models": model_metadata,
            "optimizers": optimizer_metadata,
            "extra_metadata": dict(metadata or {}),
        },
        save_folder / "predictive_ensemble_metadata.json",
    )

    return save_folder

# ---------------------------------------------------------------------
# CONFIGS, RESULTS, HISTORIES
# ---------------------------------------------------------------------
def save_experiment_config(
    config: Mapping[str, Any] | Any,
    config_folder: PathLike, 
    save_as_npy: bool = True
) -> None:
    """
    Save a config as:
        - config.json
        - config.txt
        - config.npy (optional)

        Accepts plain mappings or dataclass configs such as PipelineConfig
    """
    config_path = _ensure_dir(config_folder)
    config_obj = _jsonable(config)

    _json_dump(config_obj, config_path / "config.json")

    with (config_path / "config.txt").open("w", encoding="utf-8") as f:
        if isinstance(config_obj, Mapping):
            for key, value in config_obj.items():
                f.write(f"{key}: {value}\n")
        else:
            f.write(str(config_obj) + "\n")

    if save_as_npy:
        np.save(config_path / "config.npy", config_obj, allow_pickle=True)

def save_results_dict(
    results_dict: Mapping[str, Any], 
    results_folder: PathLike
) -> None:
    """
    Save each item in results_dict into results_folder.

    Rules:
        - np.ndarray -> .npy
        - torch.Tensor -> .pt and, if possible, .npy
        - pandas DataFrame -> .csv and .pkl
        - int/float/str/bool/list/dict -> results.json (aggregated when JSON-safe)
        - fallback -> results.txt
    """
    results_path = _ensure_dir(results_folder)

    json_safe: Dict[str, Any] = {}
    fallback_lines = []

    for key, value in results_dict.items():
        path_base = results_path / key

        if isinstance(value, np.ndarray):
            np.save(path_base.with_suffix(".npy"), value)

        elif torch.is_tensor(value):
            torch.save(value, path_base.with_suffix(".pt"))
            try:
                np.save(path_base.with_suffix(".npy"), value.detach().cpu().numpy())
            except Exception:
                fallback_lines.append(f"{key}: tensor saved as .pt only")

        elif pd is not None and isinstance(value, pd.DataFrame):
            value.to_csv(path_base.with_suffix(".csv"), index=False)
            value.to_pickle(path_base.with_suffix(".pkl"))

        elif isinstance(value, (int, float, str, bool)) or value is None:
            json_safe[key] = value

        elif isinstance(value, (list, tuple, dict)):
            try:
                json_safe[key] = _jsonable(value)
                json.dumps(json_safe[key])
            except TypeError:
                fallback_lines.append(f"{key}: {str(value)}")

        else:
            fallback_lines.append(f"{key}: {str(value)}")

    if json_safe:
        _json_dump(json_safe, results_path / "results.json")

    if fallback_lines:
        with (results_path / "results.txt").open("w", encoding="utf-8") as f:
            for line in fallback_lines:
                f.write(line + "\n")

def save_training_histories(
        histories: Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Sequence[Any]]], 
        folder: PathLike
    ) -> None:
    """
    Save training histories.

    Accepts either:
        - one flattened mapping: {"train_loss": [...], "val_loss": [...]}
        - one history per model: [{"train_loss": [...]}, {"train_loss": [...]}]
    """
    folder_path = _ensure_dir(folder)

    if isinstance(histories, Mapping):
        normalized: Dict[str, Sequence[Any]] = dict(histories)
    else:
        normalized = {}
        for model_idx, history in enumerate(histories):
            for key, values in history.items():
                normalized[f"predictive_{model_idx}_{key}"] = values
                
    json_safe: Dict[str, Any] = {}
    for key, values in normalized.items():
        arr = np.asarray(values)
        # np.save(folder_path / f"{key}.npy", arr)
        json_safe[key] = arr.tolist()

    _json_dump(json_safe, folder_path / "histories.json")

# ---------------------------------------------------------------------
# PREPROCESSING ARTIFACTS
# ---------------------------------------------------------------------
def _extract_spec_info(schema: Mapping[str, Any]) -> Dict[str, Any]:
    spec = schema.get("spec")
    if spec is None:
        return {}

    return {
        "dataset_name": getattr(spec, "name", None),
        "protected_cols": list(getattr(spec, "protected_cols", [])),
        "label_col": getattr(spec, "label_col", None),
        "group_col": getattr(spec, "group_col", None),
        "group_id_col": getattr(spec, "group_id_col", None),
        "log1p_cols": list(getattr(spec, "log1p_cols", [])),
        "drop_feature_cols": list(getattr(spec, "drop_feature_cols", [])),
    }


def _extract_scaler_info(scaler: Any) -> Dict[str, Any]:
    return {
        "scaler_mean": scaler.mean_.tolist() if scaler is not None and hasattr(scaler, "mean_") else None,
        "scaler_scale": scaler.scale_.tolist() if scaler is not None and hasattr(scaler, "scale_") else None,
        "scaler_var": scaler.var_.tolist() if scaler is not None and hasattr(scaler, "var_") else None,
        "scaler_n_features_in": int(scaler.n_features_in_) if scaler is not None and hasattr(scaler, "n_features_in_") else None,
    }


def _extract_ohe_info(ohe: Any) -> Dict[str, Any]:
    categories = None
    if ohe is not None and hasattr(ohe, "categories_"):
        categories = [list(cat) for cat in ohe.categories_]

    return {
        "ohe_categories": categories,
        "ohe_n_features_in": int(ohe.n_features_in_) if ohe is not None and hasattr(ohe, "n_features_in_") else None,
    }

def save_preprocessing_artifacts(
    schema: Mapping[str, Any],
    folder: PathLike,
    name: str = "preprocessing",
    save_joblib: bool = True,
) -> None:
    """
    Save preprocessing artifacts needed to inspect or reconstruct preprocessing.

    Compatible with both schemas from the latest dataset utilities:
        - predictor_schema from fit_predictor_schema
        - px_schema from fit_schema_px

    JSON stores human-readable metadata.
    Joblib stores the fitted sklearn objects and full schema for exact reuse.
    """
    folder_path = _ensure_dir(folder)

    scaler = schema.get("scaler")
    ohe = schema.get("ohe")

    artifacts: Dict[str, Any] = {
        "schema_type": "predictor" if "ohe" in schema else "px",
        "feature_cols": list(schema.get("feature_cols", [])),
        "cat_cols": list(schema.get("cat_cols", [])),
        "cont_cols": list(schema.get("cont_cols", [])),
        "cat_vocabs": {k: list(v) for k, v in schema.get("cat_vocabs", {}).items()},
        "protected_schema": schema.get("protected_schema", None),
        "append_protected": schema.get("append_protected", None),
        "unknown_token": schema.get("unknown_token", None),
        **_extract_spec_info(schema),
        **_extract_scaler_info(scaler),
        **_extract_ohe_info(ohe),
    }

    _json_dump(artifacts, folder_path / f"{name}.json")
    np.save(folder_path / f"{name}.npy", _jsonable(artifacts), allow_pickle=True)

    if save_joblib:
        joblib.dump(dict(schema), folder_path / f"{name}_schema.joblib")

def save_all_preprocessing_artifacts(
    predictor_schema: Mapping[str, Any],
    folder: PathLike,
    px_schema: Optional[Mapping[str, Any]] = None,
) -> None:
    folder_path = _ensure_dir(folder)
    save_preprocessing_artifacts(
        predictor_schema,
        folder_path / "predictor",
        name="predictor_preprocessing",
    )

    if px_schema is not None:
        save_preprocessing_artifacts(
            px_schema,
            folder_path / "px",
            name="px_preprocessing",
        )

# ---------------------------------------------------------------------
# SPLITS / RNG / ENVIRONMENT
# ---------------------------------------------------------------------
def save_split_indices(
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    folder: PathLike,
) -> None:
    """
    Save dataset split indices into split_indices.npz.
    """
    folder_path = _ensure_dir(folder)
    np.savez(
        folder_path / "split_indices.npz",
        train_idx=np.asarray(train_idx, dtype=np.int64),
        val_idx=np.asarray(val_idx, dtype=np.int64),
        test_idx=np.asarray(test_idx, dtype=np.int64),
    )


def save_rng_states(folder: PathLike) -> None:
    """
    Save Python, NumPy, and Torch RNG states.
    """
    folder_path = _ensure_dir(folder)

    rng_states = {
        "python_random_state": random.getstate(),
        "numpy_random_state": np.random.get_state(),
        "torch_random_state": torch.get_rng_state().cpu(),
        "torch_cuda_random_state": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
        "platform": platform.platform(),
        "python_version": sys.version,
    }

    torch.save(rng_states, folder_path / "rng_states.pt")


def save_environment_info(folder: PathLike) -> None:
    """
    Save basic environment and dependency information.
    """
    folder_path = _ensure_dir(folder)

    env = {
        "python_version": sys.version,
        "platform": platform.platform(),
        "torch_version": torch.__version__,
        "sklearn_version": sklearn.__version__,
        "numpy_version": np.__version__,
        "cuda_available": torch.cuda.is_available(),
        "cuda_version": torch.version.cuda,
        "cudnn_version": torch.backends.cudnn.version() if torch.backends.cudnn.is_available() else None,
        "device_count": torch.cuda.device_count(),
        "cudnn_benchmark": torch.backends.cudnn.benchmark,
        "cudnn_deterministic": torch.backends.cudnn.deterministic,
    }

    if torch.cuda.is_available():
        env["gpu_names"] = [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())]

    _json_dump(env, folder_path / "environment.json")


def save_model_metadata(folder: PathLike, metadata: Mapping[str, Any]) -> None:
    """
    Save model metadata as JSON.
    """
    folder_path = _ensure_dir(folder)
    _json_dump(dict(metadata), folder_path / "model_metadata.json")


# ---------------------------------------------------------------------
# FULL EXPERIMENT SAVING
# ---------------------------------------------------------------------
def save_full_experiment(
    predictive_models: Sequence[nn.Module],
    config: Optional[Mapping[str, Any] | Any]=None,
    results_dict: Optional[Mapping[str, Any]]=None,
    base_folder: PathLike="saved_models",
    exact_path: Optional[PathLike]=None,
    predictive_optimizers: Optional[Sequence[torch.optim.Optimizer]]=None,
    predictor_schema: Optional[Mapping[str, Any]] = None,
    px_schema: Optional[Mapping[str, Any]] = None,
    histories: Optional[Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Sequence[Any]]]] = None,
    model_metadata: Optional[Mapping[str, Any]] = None,
    data_info: Optional[Mapping[str, Any]] = None,
) -> Path:
    """
    Create one experiment folder and save all artifacts inside it.
    """
    exp_folder = make_experiment_folder(
        base_folder=base_folder,
        prefix="experiment_",
        exact_path=exact_path,
    )

    pred_folder = _ensure_dir(exp_folder / "predictive_ensemble")
    save_predictive_ensemble(
        ensemble_models=predictive_models,
        ensemble_folder=pred_folder,
        ensemble_optimizers=predictive_optimizers,
        create_run_subfolder=False,
        metadata=model_metadata,
    )

    if config is not None:
        save_experiment_config(config, config_folder=exp_folder)

    if results_dict is not None:
        save_results_dict(results_dict, exp_folder / "results")

    if predictor_schema is not None:
        save_all_preprocessing_artifacts(
            predictor_schema=predictor_schema,
            px_schema=px_schema,
            folder=exp_folder / "preprocessing",
        )

    if histories is not None:
        save_training_histories(histories, exp_folder / "histories")

    if model_metadata is not None:
        save_model_metadata(exp_folder / "models", model_metadata)

    if data_info is not None:
        _ensure_dir(exp_folder / "data")
        _json_dump(dict(data_info), exp_folder / "data" / "data_info.json")

    save_rng_states(exp_folder / "rng")
    save_environment_info(exp_folder / "environment")

    return exp_folder







# Example of model metadata
# model_metadata = {
#     "ar_model": {
#         "class": "ARModel",
#         "vocab_sizes": vocab_sizes,
#         "num_groups": num_groups,
#         "g_emb_dim": 16,
#         "x_emb_dim": 32,
#         "hidden": 128,
#         "dropout": 0.1,
#     },
#     "ctx_model": {
#         "class": "ContextEncoder",
#         "vocab_sizes": vocab_sizes,
#         "num_groups": num_groups,
#         "g_emb_dim": 16,
#         "x_emb_dim": 16,
#         "hidden": 128,
#         "out_dim": 128,
#     },
#     "flow_model": {
#         "class": "ConditionalRealNVPFlow",
#         "dim": X_cont_train.shape[1],
#         "context_dim": 128,
#         "num_couplings": 6,
#         "hidden": 128,
#         "seed": 0,
#     },
#     "predictive_model": {
#         "class": "PredictiveMLP",
#         "input_dim": int(X_pred_train.shape[1]),
#         "hidden_dims": [256, 128],
#         "dropout": 0.2,
#     }
# }



def save_full_reproducibility_bundle(
    experiment_folder: PathLike,
    config: Mapping[str, Any] | Any,
    schema: Mapping[str, Any],
    train_idx: Optional[Sequence[int]] = None,
    val_idx: Optional[Sequence[int]] = None,
    test_idx: Optional[Sequence[int]] = None,
    model_metadata: Optional[Mapping[str, Any]] = None,
    histories: Optional[Mapping[str, Sequence[Any]] | Sequence[Mapping[str, Sequence[Any]]]] = None,
    data_info: Optional[Mapping[str, Any]] = None,
    px_schema: Optional[Mapping[str, Any]] = None,
) -> None:
    """Save the main reproducibility artifacts for an experiment."""
    experiment_path = _ensure_dir(experiment_folder)

    save_experiment_config(config, experiment_path)
    save_all_preprocessing_artifacts(schema, experiment_path / "preprocessing", px_schema=px_schema)

    if train_idx is not None and val_idx is not None and test_idx is not None:
        save_split_indices(train_idx, val_idx, test_idx, experiment_path / "splits")

    if model_metadata is not None:
        save_model_metadata(experiment_path / "models", model_metadata)

    save_rng_states(experiment_path / "rng")
    save_environment_info(experiment_path / "environment")

    if histories is not None:
        save_training_histories(histories, experiment_path / "histories")

    if data_info is not None:
        _ensure_dir(experiment_path / "data")
        _json_dump(dict(data_info), experiment_path / "data" / "data_info.json")

# ---------------------------------------------------------------------
# PIPELINE-RESULT SAVING
# ---------------------------------------------------------------------

def save_pipeline_result(
    result: PipelineResult,
    base_folder: PathLike = "saved_models",
    exact_path: Optional[PathLike] = None,
) -> Path:
    """Save the output of `run_predictive_pipeline(...)`."""
    data = result.data

    results_dict: Dict[str, Any] = {
        "test_metrics": result.test_metrics,
        "group_metrics": result.group_metrics,
    }

    if result.ensemble_metrics is not None:
        for key, value in result.ensemble_metrics.items():
            results_dict[f"ensemble_{key}"] = value

    model_metadata = {
        "dataset_name": data.spec.name,
        "num_features": data.num_features,
        "num_classes": data.num_classes,
        "binary": data.binary,
        "n_models": len(result.models),
    }

    data_info = {
        "dataset_name": data.spec.name,
        "n_total": len(data.loaded.df),
        "n_train": len(data.train_df),
        "n_val": len(data.val_df),
        "n_test": len(data.test_df),
        "original_columns": data.loaded.original_columns,
        "group_id": data.loaded.group_id,
    }

    if data.loaded.pg_table is not None:
        results_dict["pg_table"] = data.loaded.pg_table

    if data.loaded.pg is not None:
        results_dict["pg"] = data.loaded.pg

    return save_full_experiment(
        predictive_models=result.models,
        config=result.config,
        results_dict=results_dict,
        base_folder=base_folder,
        exact_path=exact_path,
        predictor_schema=data.predictor_schema,
        histories=result.histories,
        model_metadata=model_metadata,
        data_info=data_info,
    )
