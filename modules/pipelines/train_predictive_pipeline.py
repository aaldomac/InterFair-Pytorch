from __future__ import annotations

import random
from dataclasses import dataclass, field, asdict, replace
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from modules.utils.dataset_utils import (
    DatasetSpec,
    LoadedDataset,
    compute_pg_dirichlet,
    fit_predictor_schema,
    load_dataset_by_name,
    make_loader,
    split_df,
    transform_predictor_schema,
)
from modules.predictive.models import MLPClassifier
from modules.predictive.losses import (
    BinaryClassificationLoss,
    CompositeLoss,
    MulticlassClassificationLoss,
    Regularizer,
)
from modules.predictive.metrics import entropy, logits_to_probs, probs_to_labels
from modules.predictive.trainer import PredictiveTrainer, TrainConfig
from modules.predictive.ensemble import evaluate_ensemble


# ---------------------------------------------------------------------
# CONFIGURATION
# ---------------------------------------------------------------------

@dataclass(frozen=True)
class SplitConfig:
    test_size: float = 0.25
    val_size: float = 0.15
    seed: int = 42
    stratify: Optional[str] = None
    rare_group_policy: Optional[str] = None


@dataclass(frozen=True)
class LoaderConfig:
    batch_size: int = 256
    eval_batch_size: int = 512
    shuffle_train: bool = True


@dataclass(frozen=True)
class ModelConfig:
    hidden_dims: Tuple[int, ...] = (256, 128)
    dropout: float = 0.2


@dataclass(frozen=True)
class OptimizerConfig:
    lr: float = 1e-3
    weight_decay: float = 1e-4


@dataclass(frozen=True)
class PipelineConfig:
    dataset_name: str = "adult"
    dataset_kwargs: Mapping[str, Any] = field(default_factory=lambda: {"drop_na": True})

    split: SplitConfig = field(default_factory=SplitConfig)
    loader: LoaderConfig = field(default_factory=LoaderConfig)
    model: ModelConfig = field(default_factory=ModelConfig)
    optimizer: OptimizerConfig = field(default_factory=OptimizerConfig)
    train: TrainConfig = field(default_factory=lambda: TrainConfig(epochs=300, patience=30, binary=True))

    model_seed: Optional[int] = None
    n_models: int = 5
    append_protected_to_predictor: bool = False
    store_member_probs: bool = False
    device: Optional[str] = None


@dataclass
class PreparedData:
    loaded: LoadedDataset
    spec: DatasetSpec

    train_df: pd.DataFrame
    val_df: pd.DataFrame
    test_df: pd.DataFrame

    predictor_schema: Dict[str, Any]
    train_loader: DataLoader
    val_loader: DataLoader
    test_loader: DataLoader

    num_features: int
    num_classes: int
    binary: bool

@dataclass
class PipelineResult:
    config: PipelineConfig
    data: PreparedData
    models: List[nn.Module]
    histories: List[Dict[str, List[float]]]
    test_metrics: Dict[str, float]
    group_metrics: pd.DataFrame
    ensemble_metrics: Optional[Dict[str, np.ndarray]] = None


# ---------------------------------------------------------------------
# REPRODUCIBILITY
# ---------------------------------------------------------------------

def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


# ---------------------------------------------------------------------
# DATA PREPARATION
# ---------------------------------------------------------------------

def _get_spec(loaded: LoadedDataset) -> DatasetSpec:
    try:
        spec = loaded.metadata["spec"]
    except KeyError as exc:
        raise KeyError(
            "LoadedDataset.metadata must contain a DatasetSpec under metadata['spec']. "
            "The dataset-specific module should return LoadedDataset(..., metadata={'spec': SPEC})."
        ) from exc

    if not isinstance(spec, DatasetSpec):
        raise TypeError(f"metadata['spec'] must be a DatasetSpec, got {type(spec)!r}.")

    return spec


def _infer_num_classes(train_df: pd.DataFrame, label_col: str) -> int:
    labels = sorted(pd.Series(train_df[label_col]).dropna().unique().tolist())
    if len(labels) < 2:
        raise ValueError(f"Need at least two classes in '{label_col}', found {labels}.")
    return len(labels)


def _validate_labels_for_training(train_df: pd.DataFrame, spec: DatasetSpec, *, binary: bool, num_classes: int) -> None:
    labels = sorted(pd.Series(train_df[spec.label_col]).dropna().unique().tolist())

    if binary:
        if set(labels) != {0, 1}:
            raise ValueError(
                f"Binary BCE training expects labels exactly {{0, 1}} in '{spec.label_col}', found {labels}. "
                "Either remap labels in the dataset-specific preprocess function or use multiclass training."
            )
        return

    expected = set(range(num_classes))
    if set(labels) != expected:
        raise ValueError(
            f"Multiclass CrossEntropyLoss expects integer labels {sorted(expected)}, found {labels}. "
            "Remap labels in the dataset-specific preprocess function."
        )


def _make_supervised_loader(
    X: torch.Tensor,
    y: torch.Tensor,
    g: torch.Tensor,
    *,
    batch_size: int,
    shuffle: bool,
) -> DataLoader:
    # The trainer consumes x, y and ignores extra fields via `*rest`.
    # Keeping g in the dataset makes group-aware evaluation easy later.
    dataset = TensorDataset(X, y, g)
    return make_loader(dataset, batch_size=batch_size, shuffle=shuffle)

def prepare_data(config: PipelineConfig) -> PreparedData:
    """Download/load data, split it, fit preprocessing schemas, and build loaders."""
    seed_everything(config.split.seed)

    if config.dataset_name.lower() == 'celeba':
        from modules.data.CelebA import prepare_image_data
        return prepare_image_data(config)
    loaded = load_dataset_by_name(config.dataset_name, **dict(config.dataset_kwargs))
    spec = _get_spec(loaded)

    stratify_col = config.split.stratify or spec.group_col

    fixed = loaded.metadata.get("split_indices")
    if fixed is not None:
        names = ("train", "validation", "audit" if "audit" in fixed else "test")
        indices = [np.asarray(fixed[name]) for name in names]
        all_indices = np.concatenate([np.asarray(v) for v in fixed.values()])
        if (not np.issubdtype(all_indices.dtype, np.integer)
                or len(np.unique(all_indices)) != len(all_indices)
                or (all_indices < 0).any() or (all_indices >= len(loaded.df)).any()
                or any(len(v) == 0 for v in indices)):
            raise ValueError("Fixed splits must be nonempty, disjoint positional indices in range.")
        train_df, val_df, test_df = [loaded.df.iloc[idx].copy() for idx in indices]
    else:
        train_df, val_df, test_df = split_df(
            loaded.df, test_size=config.split.test_size,
            val_size=config.split.val_size, seed=config.split.seed,
            stratify=stratify_col, rare_group_policy=config.split.rare_group_policy,
        )
    loaded.pg_table, loaded.pg = compute_pg_dirichlet(train_df)

    num_classes = _infer_num_classes(train_df, spec.label_col)
    binary = bool(config.train.binary)
    _validate_labels_for_training(train_df, spec, binary=binary, num_classes=num_classes)

    predictor_schema = fit_predictor_schema(
        train_df,
        spec,
        append_protected=config.append_protected_to_predictor,
    )
    if config.dataset_name == "synthetic_data" and not config.append_protected_to_predictor:
        forbidden = {"Gender", "Race", "S1", "S2"}
        exposed = forbidden.intersection(predictor_schema["feature_cols"])
        if exposed or predictor_schema["append_protected"]:
            raise RuntimeError(f"Protected attributes exposed to synthetic preditor: {exposed}.")

    X_train, y_train, g_train = transform_predictor_schema(train_df, predictor_schema)
    X_val, y_val, g_val = transform_predictor_schema(val_df, predictor_schema)
    X_test, y_test, g_test = transform_predictor_schema(test_df, predictor_schema)

    train_loader = _make_supervised_loader(
        X_train,
        y_train,
        g_train,
        batch_size=config.loader.batch_size,
        shuffle=config.loader.shuffle_train,
    )
    val_loader = _make_supervised_loader(
        X_val,
        y_val,
        g_val,
        batch_size=config.loader.eval_batch_size,
        shuffle=False,
    )
    test_loader = _make_supervised_loader(
        X_test,
        y_test,
        g_test,
        batch_size=config.loader.eval_batch_size,
        shuffle=False,
    )

    return PreparedData(
        loaded=loaded,
        spec=spec,
        train_df=train_df,
        val_df=val_df,
        test_df=test_df,
        predictor_schema=predictor_schema,
        train_loader=train_loader,
        val_loader=val_loader,
        test_loader=test_loader,
        num_features=int(X_train.shape[1]),
        num_classes=num_classes,
        binary=binary,
    )


# ---------------------------------------------------------------------
# MODEL / LOSS / OPTIMIZER FACTORIES
# ---------------------------------------------------------------------

def build_model(data: PreparedData, config: PipelineConfig) -> nn.Module:
    num_outputs = 1 if data.binary else data.num_classes

    if data.predictor_schema.get('modality') == 'image':
        from modules.predictive.models import ImageClassifier
        return ImageClassifier(num_outputs=num_outputs, dropout=config.model.dropout)
    return MLPClassifier(
        input_dim=data.num_features,
        num_outputs=num_outputs,
        hidden_dims=config.model.hidden_dims,
        dropout=config.model.dropout,
    )


def build_optimizer(model: nn.Module, config: PipelineConfig) -> torch.optim.Optimizer:
    return torch.optim.AdamW(
        model.parameters(),
        lr=config.optimizer.lr,
        weight_decay=config.optimizer.weight_decay,
    )


def build_loss(
    data: PreparedData,
    *,
    regularizer: Optional[Regularizer] = None,
    regularizer_weight: float = 1.0,
) -> CompositeLoss:
    task_loss: nn.Module
    if data.binary:
        task_loss = BinaryClassificationLoss()
    else:
        task_loss = MulticlassClassificationLoss()

    return CompositeLoss(
        task_loss=task_loss,
        regularizer=regularizer,
        regularizer_weight=regularizer_weight,
    )


def _training_config_for_data(config: PipelineConfig, data: PreparedData) -> TrainConfig:
    return replace(config.train, binary=data.binary)


# ---------------------------------------------------------------------
# TRAINING
# ---------------------------------------------------------------------

def train_one_model(
    data: PreparedData,
    config: PipelineConfig,
    *,
    seed: int,
    regularizer: Optional[Regularizer] = None,
    regularizer_weight: float = 1.0,
    fair_loader: Optional[DataLoader] = None,
) -> Tuple[nn.Module, Dict[str, List[float]]]:
    seed_everything(seed)

    model = build_model(data, config)
    optimizer = build_optimizer(model, config)
    loss_fn = build_loss(
        data,
        regularizer=regularizer,
        regularizer_weight=regularizer_weight,
    )

    trainer = PredictiveTrainer(
        model=model,
        optimizer=optimizer,
        loss_fn=loss_fn,
        config=_training_config_for_data(config, data),
        device=config.device,
        fair_loader=fair_loader,
    )

    return trainer.fit(data.train_loader, data.val_loader)


def train_models(
    data: PreparedData,
    config: PipelineConfig,
    *,
    regularizer: Optional[Regularizer] = None,
    regularizer_weight: float = 1.0,
    fair_loader: Optional[DataLoader] = None,
) -> Tuple[List[nn.Module], List[Dict[str, List[float]]]]:
    if config.n_models <= 0:
        raise ValueError("config.n_models must be positive.")

    models: List[nn.Module] = []
    histories: List[Dict[str, List[float]]] = []

    for model_idx in range(config.n_models):
        if config.train.verbose:
            print(f"[Pipeline] Training model {model_idx + 1}/{config.n_models}")

        model, history = train_one_model(
            data,
            config,
            seed=(config.model_seed if config.model_seed is not None else config.split.seed) + model_idx,
            regularizer=regularizer,
            regularizer_weight=regularizer_weight,
            fair_loader=fair_loader,
        )
        models.append(model)
        histories.append(history)

    return models, histories


# ---------------------------------------------------------------------
# EVALUATION
# ---------------------------------------------------------------------
def _group_metrics_from_outputs(outputs, labels, groups, data, config):
    from modules.metrics.ensemble_quality import ensemble_quality
    from modules.metrics.distribution_decomposition_metrics import group_entropy_decompositions
    names = {int(g): str(name) for name, g in data.loaded.group_id.items()}
    _, frame = ensemble_quality(outputs, labels, groups, binary=data.binary,
        threshold=config.train.threshold, group_names=names)
    decompositions = group_entropy_decompositions(outputs['mean_probs'], groups)
    frame['group_mix_entropy'] = frame['group_id'].map(
        {g: dec.entropy_of_mean for g, dec in decompositions.items()})
    frame['group_het_entropy'] = frame['group_id'].map(
        {g: dec.entropy_of_mean-dec.mean_entropy for g, dec in decompositions.items()})
    return frame.sort_values('group_id').reset_index(drop=True)


def evaluate_models(
    models: Sequence[nn.Module],
    data: PreparedData,
    config: PipelineConfig,
) -> Tuple[Dict[str, float], pd.DataFrame, Optional[Dict[str, np.ndarray]]]:
    if len(models) == 0:
        raise ValueError("At least one trained model is required for evaluation.")

    from modules.metrics.ensemble_quality import ensemble_quality
    outputs = evaluate_ensemble(models, data.test_loader, binary=data.binary,
        device=config.device, threshold=config.train.threshold,
        store_member_probs=config.store_member_probs)
    labels = data.test_df[data.spec.label_col].to_numpy(dtype=int)
    groups = data.test_df[data.spec.group_id_col].to_numpy(dtype=int)
    metrics, _ = ensemble_quality(outputs, labels, groups, threshold=config.train.threshold, binary=data.binary)
    # Keep the loss key for older callers; evaluation loss is ensemble NLL.
    metrics['loss'] = metrics['nll']
    group_metrics = _group_metrics_from_outputs(outputs, labels, groups, data, config)
    return metrics, group_metrics, outputs


# ---------------------------------------------------------------------
# END-TO-END PIPELINE
# ---------------------------------------------------------------------

def run_predictive_pipeline(
    config: PipelineConfig,
    *,
    regularizer: Optional[Regularizer] = None,
    regularizer_weight: float = 1.0,
    fair_loader: Optional[DataLoader] = None,
) -> PipelineResult:
    """Run the complete pipeline.

    Steps:
        1. Download/load dataset through `modules.data.<dataset_name>`.
        2. Split into train/validation/test.
        3. Fit preprocessing on train only.
        4. Build PyTorch DataLoaders.
        5. Train one model or an ensemble.
        6. Evaluate on the test set.

    Regularizers require an explicit fair_loader. Synthetic fixed splits are
    respected; image adapters use their official partitions.
    """
    from modules.pipelines.experiment_config import validate_pipeline_options
    validate_pipeline_options({key: value for key, value in asdict(config).items()
                               if key in ('split', 'loader', 'model', 'optimizer', 'train',
                                          'n_models', 'append_protected_to_predictor', 'store_member_probs')})
    
    data = prepare_data(config)

    if regularizer is not None and fair_loader is None:
        raise ValueError("Pass an explicit fair_loader when using a regularizer.")

    models, histories = train_models(
        data,
        config,
        regularizer=regularizer,
        regularizer_weight=regularizer_weight,
        fair_loader=fair_loader,
    )

    test_metrics, group_metrics, ensemble_metrics = evaluate_models(models, data, config)

    return PipelineResult(
        config=config,
        data=data,
        models=models,
        histories=histories,
        test_metrics=test_metrics,
        group_metrics=group_metrics,
        ensemble_metrics=ensemble_metrics,
    )

