import os
import json
import numpy as np
import torch
import sklearn
import sys
import platform
import random
from pathlib import Path
from typing import Any, Dict, Mapping, Optional, Sequence, Union

PathLike = Union[str, os.PathLike]

from modules.models.generative_models import (
    ARModel,
    ConditionalRealNVPFlow,
    ContextEncoder
)
from modules.models.predictive_models import (
    Classifier,
)

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
    

def _json_dump(data: Any, path: PathLike) -> None:
    """
    Save data as formatted JSON using default=str for non-native objects.
    """
    path_obj = Path(path)
    with path_obj.open("w", encoding="utf-8") as f:
        json.dump(data, f, indent=2, default=str)

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

def save_generative_models(
    ar_model: ARModel, 
    flow_model:  ConditionalRealNVPFlow, 
    ctx_model: Optional[ContextEncoder] = None, 
    folder: Optional[PathLike]=None, 
    exact_path: Optional[PathLike]=None, 
    ar_optimizer: Optional[torch.optim.Optimizer]=None, 
    flow_optimizer: Optional[torch.optim.Optimizer]=None,
    create_run_subfolder: bool=True
) -> Path:
    """
    Save AR model, Flow model, optional context encoder, and optional optimizers.

    If create_run_subfolder is True, creates a run_*/ folder inside base folder unless
    exact_path is provided. If False, saves directly into folder/exact_path.
    """
    if create_run_subfolder:
        save_folder = make_experiment_folder(
            base_folder="saved_models" if folder is None else folder,
            prefix="run_",
            exact_path=exact_path,
        )
    else:
        target = exact_path if exact_path is not None else folder
        if target is None:
            raise ValueError("Either folder or exact_path must be provided.")
        save_folder = _ensure_dir(target)

    torch.save(ar_model.state_dict(), save_folder / "ar_model.pt")
    torch.save(flow_model.state_dict(), save_folder / "flow_model.pt")

    if ctx_model is not None:
        torch.save(ctx_model.state_dict(), save_folder / "ctx_model.pt")

    if ar_optimizer is not None:
        torch.save(ar_optimizer.state_dict(), save_folder / "ar_optimizer.pt")

    if flow_optimizer is not None:
        torch.save(flow_optimizer.state_dict(), save_folder / "flow_optimizer.pt")

    return save_folder

def save_predictive_ensemble(
    ensemble_models: Sequence[Classifier], 
    ensemble_folder: Optional[PathLike]=None,
    exact_path: Optional[PathLike]=None,
    ensemble_optimizers: Optional[Sequence[torch.optim.Optimizer]]=None,
    create_run_subfolder: bool=True,
) -> Path:
    """
    Save a list of predictive models and optional optimizers.
    Files:
        predictor_0.pt, predictor_1.pt, ...
        optimizer_0.pt, optimizer_1.pt, ... (optional)
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

    for i, model in enumerate(ensemble_models):
        torch.save(model.state_dict(), save_folder / f"predictor_{i}.pt")

    if ensemble_optimizers is not None:
        for i, optimizer in enumerate(ensemble_optimizers):
            torch.save(optimizer.state_dict(), save_folder / f"predictor_optimizer_{i}.pt")

    return save_folder

def save_experiment_config(
    config: Mapping[str, Any],
    config_folder: PathLike, 
    save_as_npy: bool = True
) -> None:
    """
    Save a config as:
        - config.json
        - config.txt
        - config.npy (optional)
    """
    config_path = _ensure_dir(config_folder)

    _json_dump(dict(config), config_path / "config.json")

    with (config_path / "config.txt").open("w", encoding="utf-8") as f:
        for key, value in config.items():
            f.write(f"{key}: {value}\n")

    if save_as_npy:
        np.save(config_path / "config.npy", dict(config), allow_pickle=True)

def save_results_dict(
    results_dict: Mapping[str, Any], 
    results_folder: PathLike
) -> None:
    """
    Save each item in results_dict into results_folder.

    Rules:
        - np.ndarray -> .npy
        - torch.Tensor -> .pt and, if possible, .npy
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

        elif isinstance(value, (int, float, str, bool)) or value is None:
            json_safe[key] = value

        elif isinstance(value, (list, dict)):
            try:
                json.dumps(value)
                json_safe[key] = value
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

def save_full_experiment(
    ar_model: ARModel,
    flow_model: ConditionalRealNVPFlow,
    predictive_models: Sequence[Classifier],
    ctx_model: Optional[ContextEncoder]=None,
    config: Optional[Mapping[str, Any]]=None,
    results_dict: Optional[Mapping[str, Any]]=None,
    base_folder: PathLike="saved_models",
    exact_path: Optional[PathLike]=None,
    ar_optimizer: Optional[torch.optim.Optimizer]=None,
    flow_optimizer: Optional[torch.optim.Optimizer]=None,
    predictive_optimizers: Optional[Sequence[torch.optim.Optimizer]]=None,
) -> Path:
    """
    Create one experiment folder and save all artifacts inside it.
    """
    exp_folder = make_experiment_folder(
        base_folder=base_folder,
        prefix="experiment_",
        exact_path=exact_path,
    )

    gen_folder = _ensure_dir(exp_folder / "generative")
    save_generative_models(
        ar_model=ar_model,
        flow_model=flow_model,
        ctx_model=ctx_model,
        folder=gen_folder,
        ar_optimizer=ar_optimizer,
        flow_optimizer=flow_optimizer,
        create_run_subfolder=False,
    )

    pred_folder = _ensure_dir(exp_folder / "predictive_ensemble")
    save_predictive_ensemble(
        ensemble_models=predictive_models,
        ensemble_folder=pred_folder,
        ensemble_optimizers=predictive_optimizers,
        create_run_subfolder=False,
    )

    if config is not None:
        save_experiment_config(config, config_folder=exp_folder)

    if results_dict is not None:
        save_results_dict(results_dict, exp_folder / "results")

    return exp_folder


def save_preprocessing_artifacts(
    schema: Mapping[str, Any],
    folder: PathLike,
    protected_cols: Sequence[str] = ("gender", "race", "native-country"),
    label_col: str = "income",
    group_col: str = "group",
    group_id_col: str = "group_id",
) -> None:
    """
    Save preprocessing artifacts needed to reconstruct preprocessing.
    """
    folder_path = _ensure_dir(folder)

    scaler = schema.get("scaler", None)

    artifacts = {
        "cat_cols": list(schema.get("cat_cols", [])),
        "cont_cols": list(schema.get("cont_cols", [])),
        "cat_vocabs": {
            k: list(v) for k, v in schema.get("cat_vocabs", {}).items()
        },
        "log1p_cols": list(schema.get("log1p_cols", [])),
        "drop_feature_cols": list(schema.get("drop_feature_cols", [])),
        "protected_cols": list(protected_cols),
        "label_col": label_col,
        "group_col": group_col,
        "group_id_col": group_id_col,
        "scaler_mean": scaler.mean_.tolist() if scaler is not None and hasattr(scaler, "mean_") else None,
        "scaler_scale": scaler.scale_.tolist() if scaler is not None and hasattr(scaler, "scale_") else None,
        "scaler_var": scaler.var_.tolist() if scaler is not None and hasattr(scaler, "var_") else None,
        "scaler_n_features_in": int(scaler.n_features_in_) if scaler is not None and hasattr(scaler, "n_features_in_") else None,
    }

    _json_dump(artifacts, folder_path / "preprocessing.json")
    np.save(folder_path / "preprocessing.npy", artifacts, allow_pickle=True)


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

def save_training_histories(histories: Mapping[str, Sequence[Any]], folder: PathLike) -> None:
    """
    histories example:
    {
        "ar_train_loss": [...],
        "ar_val_loss": [...],
        "flow_train_loss": [...],
        "flow_val_loss": [...],
        "predictive_0_train_loss": [...],
        ...
    }
    """
    folder_path = _ensure_dir(folder)

    json_safe: Dict[str, Any] = {}
    for key, values in histories.items():
        arr = np.asarray(values)
        np.save(folder_path / f"{key}.npy", arr)
        json_safe[key] = arr.tolist()

    _json_dump(json_safe, folder_path / "histories.json")

def save_full_reproducibility_bundle(
    experiment_folder: PathLike,
    config: Mapping[str, Any],
    schema: Mapping[str, Any],
    train_idx: Sequence[int],
    val_idx: Sequence[int],
    test_idx: Sequence[int],
    model_metadata: Mapping[str, Any],
    histories: Optional[Mapping[str, Sequence[Any]]] = None,
    data_info: Optional[Mapping[str, Any]] = None,
) -> None:
    """
    Save the main reproducibility artifacts for an experiment.
    """
    experiment_path = _ensure_dir(experiment_folder)

    save_experiment_config(config, experiment_path)
    save_preprocessing_artifacts(schema, experiment_path / "preprocessing")
    save_split_indices(train_idx, val_idx, test_idx, experiment_path / "splits")
    save_model_metadata(experiment_path / "models", model_metadata)
    save_rng_states(experiment_path / "rng")
    save_environment_info(experiment_path / "environment")

    if histories is not None:
        save_training_histories(histories, experiment_path / "histories")

    if data_info is not None:
        _ensure_dir(experiment_path / "data")
        _json_dump(dict(data_info), experiment_path / "data" / "data_info.json")