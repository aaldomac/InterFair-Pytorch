import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import copy
import numpy as np

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from modules.utils.tensor_utils import (
    ArrayLike,
    DeviceLike,
    _as_tensor,
    _default_device,
)

class Classifier(nn.Module):
    """
    Binary classifier MLP that returns probabilities in [0, 1].

    Args:
        input_dim: Number of input features.
        hidden: Two hidden layer sizes.
        dropout: Dropout probability.
    """
    def __init__(
        self, 
        input_dim: int, 
        hidden: Tuple[int, int]=(256, 128), 
        dropout: float=0.2
    ) -> None:
        super().__init__()

        if input_dim <= 0:
            raise ValueError("input_dim must be a positive integer.")
        if len(hidden) != 2:
            raise ValueError("hidden must contain exactly two layer sizes.")
        if hidden[0] <= 0 or hidden[1] <= 0:
            raise ValueError("All hidden layer sizes must be positive.")
        if not (0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in [0, 1).")

        self.input_dim = input_dim
        self.hidden = hidden
        self.dropout_p = dropout

        self.layer1 = nn.Linear(input_dim, hidden[0])
        self.layer2 = nn.Linear(hidden[0], hidden[1])
        self.output_layer = nn.Linear(hidden[1], 1)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the classifier.
        Args:
            x: Input tensor of shape [B, input_dim].
        Returns:
            Tensor of shape [B, 1] with predicted probabilities in [0, 1].
        """
        if x.ndim != 2:
            raise ValueError(f"x must have shape [B, input_dim], got {tuple(x.shape)}.")
        if x.shape[1] != self.input_dim:
            raise ValueError(f"x must have shape [B, {self.input_dim}], got {tuple(x.shape)}.")

        x = F.relu(self.layer1(x))
        x = self.dropout(x)
        x = F.relu(self.layer2(x))
        x = self.dropout(x)
        x = torch.sigmoid(self.output_layer(x))
        return x

    @torch.no_grad()
    def predict_probs(self, x: ArrayLike) -> torch.Tensor:
        """
        Predict probabilities.

        Args:
            x: Input features of shape [B, input_dim].

        Returns:
            Tensor of shape [B, 1] with probabilities.
        """
        self.eval()
        device = next(self.parameters()).device
        x_t = _as_tensor(x, dtype=torch.float32, device=device, name="x")
        return self.forward(x_t)

    @torch.no_grad()
    def predict(self, x: ArrayLike, threshold: float=0.5) -> torch.Tensor:
        """
        Predict binary labels using a threshold on predicted probabilities.

        Args:
            x: Input features.
            threshold: Decision threshold in [0, 1].

        Returns:
            Long tensor of shape [B, 1].
        """
        if not (0.0 <= threshold <= 1.0):
            raise ValueError("threshold must be in [0, 1].")
        probs = self.predict_probs(x)
        return (probs >= threshold).long()

    @torch.no_grad()
    def accuracy(self, x: ArrayLike, y_true: ArrayLike, threshold: float=0.5) -> float:
        """
        Computes accuracy of the model on the given data tensor.
        """
        self.eval()
        device = next(self.parameters()).device
        y_pred = self.predict(x, threshold=threshold).view(-1)
        y_true_t = _as_tensor(y_true, dtype=torch.long, device=device, name="y_true").view(-1)
        return float((y_pred == y_true).float().mean().item())

def train_predictive_model_step(
    model: Classifier, 
    optimizer: torch.optim.Optimizer, 
    criterion: nn.Module, 
    x_batch: ArrayLike, 
    y_batch: ArrayLike, 
    device: DeviceLike
) -> Tuple[torch.Tensor, float]:
    """
    Perform one training step.

    Returns:
        y_pred: Predicted probabilities of shape [B].
        loss_value: Scalar loss.
    """
    x_batch_t = _as_tensor(x_batch, dtype=torch.float32, device=device, name="x_batch")
    y_batch_t = _as_tensor(y_batch, dtype=torch.float32, device=device, name="y_batch")

    model.train()
    optimizer.zero_grad(set_to_none=True)

    y_pred = model(x_batch_t).squeeze(-1)
    loss = criterion(y_pred, y_batch_t)

    loss.backward()
    optimizer.step()

    return y_pred.detach(), float(loss.item())

@torch.no_grad()
def eval_predictive_model_step(
    model: Classifier, 
    criterion: nn.Module, 
    x_batch: ArrayLike, 
    y_batch: ArrayLike, 
    device: DeviceLike
) -> float:
    """
    Perform one evaluation step and return the loss.
    """
    x_batch_t = _as_tensor(x_batch, dtype=torch.float32, device=device, name="x_batch")
    y_batch_t = _as_tensor(y_batch, dtype=torch.float32, device=device, name="y_batch")

    model.eval()
    y_pred = model(x_batch_t).squeeze(-1)
    loss = criterion(y_pred, y_batch_t)
    return float(loss.item())

@torch.no_grad()
def evaluate_predictive_model(
    model: Classifier, 
    loader: torch.utils.data.DataLoader, 
    device: DeviceLike, 
    threshold: float=0.5, 
    criterion: Optional[nn.Module]=None
) -> Dict[str, float]:
    """
    Evaluate a binary predictive model on a dataloader.

    Args:
        model: Binary classifier.
        loader: DataLoader yielding (x, y).
        device: Target device.
        threshold: Classification threshold.
        criterion: Optional loss function.

    Returns:
        Dictionary containing accuracy, and loss if criterion is provided.
    """
    if not (0.0 <= threshold <= 1.0):
        raise ValueError("threshold must be in [0, 1].")

    model.eval()

    total_correct = 0
    total_samples = 0
    total_loss = 0.0

    for x, y in loader:
        x_t = _as_tensor(x, dtype=torch.float32, device=device, name="x")
        y_t = _as_tensor(y, dtype=torch.float32, device=device, name="y")

        probs = model(x_t).squeeze(-1)
        preds = (probs >= threshold).long()

        total_correct += (preds == y_t.long()).sum().item()
        total_samples += y_t.size(0)

        if criterion is not None:
            loss = criterion(probs, y_t)
            total_loss += float(loss.item()) * y_t.size(0)
    if total_samples == 0:
        raise ValueError("No samples found in the dataloader; cannot compute accuracy.")

    accuracy = total_correct / total_samples

    if criterion is None:
        return {"accuracy": accuracy}

    mean_loss = total_loss / total_samples
    return {"accuracy": accuracy, "loss": mean_loss}

def _df_soft_penalty_from_probs(
    probs_positive: torch.Tensor,
    group_ids: torch.Tensor,
    *,
    alpha: float = 1.0,
    epsilon_target: float = 0.0,
    eps: float = 1e-12,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Differentiable DF penalty from soft positive probabilities.

    probs_positive: shape [N]
    group_ids: shape [N], integer intersectional group IDs
    """
    probs_positive = probs_positive.view(-1).clamp(min=eps, max=1.0 - eps)
    group_ids = group_ids.view(-1).long()

    num_groups = int(group_ids.max().item()) + 1

    counts = torch.bincount(group_ids, minlength=num_groups).to(
        device=probs_positive.device, dtype=probs_positive.dtype
    )

    positive_mass = torch.zeros(
        num_groups, device=probs_positive.device, dtype=probs_positive.dtype
    )
    positive_mass.scatter_add_(0, group_ids, probs_positive)

    # Eq. (7)-style Dirichlet smoothing for binary Y: denominator N_s + 2*alpha
    p_pos = (positive_mass + alpha) / (counts + 2.0 * alpha)
    p_neg = 1.0 - p_pos

    log_pos = torch.log(p_pos.clamp_min(eps))
    log_neg = torch.log(p_neg.clamp_min(eps))

    eps_pos = torch.abs(log_pos[:, None] - log_pos[None, :])
    eps_neg = torch.abs(log_neg[:, None] - log_neg[None, :])

    epsilon_df = torch.maximum(eps_pos, eps_neg).max()
    penalty = torch.relu(epsilon_df - epsilon_target)

    return penalty, epsilon_df


def _compute_df_penalty_on_loader(
    model: Classifier,
    fair_loader: torch.utils.data.DataLoader,
    *,
    device: DeviceLike,
    alpha: float,
    epsilon_target: float,
) -> Tuple[torch.Tensor, torch.Tensor]:
    """
    Compute the DF penalty on a full, non-shuffled fair_loader yielding (x, y, g).
    """
    was_training = model.training
    model.eval()

    probs_all: List[torch.Tensor] = []
    groups_all: List[torch.Tensor] = []

    for x_fair, _, g_fair in fair_loader:
        x_fair_t = _as_tensor(x_fair, dtype=torch.float32, device=device, name="x_fair")
        g_fair_t = _as_tensor(g_fair, dtype=torch.long, device=device, name="g_fair").view(-1)

        probs = model(x_fair_t).squeeze(-1)
        probs_all.append(probs)
        groups_all.append(g_fair_t)

    probs_all_t = torch.cat(probs_all, dim=0)
    groups_all_t = torch.cat(groups_all, dim=0)

    penalty, epsilon_df = _df_soft_penalty_from_probs(
        probs_all_t,
        groups_all_t,
        alpha=alpha,
        epsilon_target=epsilon_target,
    )

    if was_training:
        model.train()

    return penalty, epsilon_df

def train_single_predictive_model(
    model: Classifier,
    optimizer: torch.optim.Optimizer,
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    epochs: int=50,
    patience: int=5,
    device: Optional[DeviceLike]=None,
    df_fair_loader: Optional[torch.utils.data.DataLoader] = None,
    df_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[Classifier, Dict[str, List[float]]]:
    """
    Train a single predictive model with early stopping on validation loss.

    Args:
        model: Binary classifier.
        optimizer: Optimizer.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        epochs: Maximum number of epochs.
        patience: Early stopping patience.
        device: Target device.
        df_fair_loader: Optional dataloader yielding (x, y, g) for computing DF penalty.
        df_cfg: Optional dictionary containing DF regularization config

    Returns:
        Trained model with best validation weights restored, and training history.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")
    if patience <= 0:
        raise ValueError("patience must be positive.")

    if device is None:
        device = _default_device()
    device = torch.device(device)

    model = model.to(device)
    criterion = nn.BCELoss()

    # If DF regularizer
    use_df = df_cfg is not None and bool(df_cfg.get("enabled", False)) and df_fair_loader is not None
    df_lambda = float(df_cfg.get("lambda", 0.0)) if df_cfg is not None else 0.0
    df_alpha = float(df_cfg.get("alpha", 1.0)) if df_cfg is not None else 1.0
    df_epsilon_target = float(df_cfg.get("epsilon_target", 0.0)) if df_cfg is not None else 0.0
    df_burn_in_epochs = int(df_cfg.get("burn_in_epochs", 0)) if df_cfg is not None else 0

    best_val_loss = float("inf")
    bad_epochs = 0
    best_state = copy.deepcopy(model.state_dict())

    history: Dict[str, List[float]] = {
        "train_loss": [],
        "val_loss": [],
        "train_acc": [],
        "val_acc": [],
        "df_penalty": [],
        "df_epsilon": [],
    }

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_losses: List[float] = []
        train_correct = 0
        train_total = 0

        epoch_df_penalties: List[float] = []
        epoch_df_epsilons: List[float] = []

        for x_batch, y_batch in train_loader:
            x_batch_t = _as_tensor(x_batch, dtype=torch.float32, device=device, name="x_batch")
            y_batch_t = _as_tensor(y_batch, dtype=torch.float32, device=device, name="y_batch")

            model.train()
            optimizer.zero_grad(set_to_none=True)

            y_pred = model(x_batch_t).squeeze(-1)
            task_loss = criterion(y_pred, y_batch_t)

            df_penalty = torch.tensor(0.0, device=device)
            df_epsilon = torch.tensor(0.0, device=device)
            
            if use_df and epoch > df_burn_in_epochs:
                df_penalty, df_epsilon = _compute_df_penalty_on_loader(
                    model=model,
                    fair_loader=df_fair_loader,
                    device=device,
                    alpha=df_alpha,
                    epsilon_target=df_epsilon_target,
                )
                loss = task_loss + df_lambda * df_penalty
            
            else:
                loss = task_loss
            
            loss.backward()
            optimizer.step()

            epoch_train_losses.append(float(loss.item()))
            epoch_df_penalties.append(float(df_penalty.item()))
            epoch_df_epsilons.append(float(df_epsilon.item()))

            preds = (y_pred.detach() >= 0.5).long()
            train_correct += (preds == y_batch_t.long()).sum().item()
            train_total += y_batch_t.size(0)

            # y_pred, loss = train_predictive_model_step(
            #     model=model,
            #     optimizer=optimizer,
            #     criterion=criterion,
            #     x_batch=x_batch,
            #     y_batch=y_batch,
            #     device=device,
            # )

        if train_total == 0:
            raise ValueError("No samples found in the training dataloader; cannot compute training accuracy.")

        train_loss = float(np.mean(epoch_train_losses))
        train_acc = train_correct / train_total

        # ---- Validation ----
        val_metrics = evaluate_predictive_model(
            model=model,
            loader=val_loader,
            device=device,
            threshold=0.5,
            criterion=criterion,
        )
        val_loss = val_metrics["loss"]
        val_acc = val_metrics["accuracy"]

        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_loss)
        history["train_acc"].append(train_acc)
        history["val_acc"].append(val_acc)
        history["df_penalty"].append(float(np.mean(epoch_df_penalties)) if len(epoch_df_penalties) > 0 else 0.0)
        history["df_epsilon"].append(float(np.mean(epoch_df_epsilons)) if len(epoch_df_epsilons) > 0 else 0.0)
        print(
            f"Epoch {epoch}/{epochs} | "
            f"Loss: {train_loss:.4f}, Acc: {train_acc:.4f}, "
            f"DF_penalty: {history['df_penalty'][-1]:.4f}, DF_eps: {history['df_epsilon'][-1]:.4f} | "
            f"Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}"
        )
        # ---- Early stopping ----
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            bad_epochs = 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad_epochs += 1
            if bad_epochs >= patience:
                print("Early stopping triggered.")
                break

    model.load_state_dict(best_state)
    return model, history


@torch.no_grad()
def evaluate_ensemble(
    models: Sequence[Classifier], 
    loader: torch.utils.data.DataLoader, 
    device: Optional[DeviceLike]=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """
    Evaluate a uniform ensemble of binary classifiers.

    Each model is assumed to output either:
    - a sigmoid probability of shape [B, 1] or [B], or
    - two logits of shape [B, 2], in which case class-1 probability is taken.

    Args:
        models: Sequence of trained models.
        loader: DataLoader yielding (x, y) or compatible tuples where x is first.
        device: Target device.

    Returns:
        predictions: Mean class-1 probabilities, shape [N].
        entropy: Predictive entropy, shape [N].
        aleatoric: Expected per-model entropy, shape [N].
        epistemic: entropy - aleatoric, shape [N].
    """
    if len(models) == 0:
        raise ValueError("At least one model is required for ensemble evaluation.")
    if device is None:
        device = _default_device()
    device = torch.device(device)

    all_probs: List[torch.Tensor] = []

    for model in models:
        model = model.to(device)
        model.eval()

        batch_probs: List[torch.Tensor] = []

        for batch in loader:
            if len(batch) < 1:
                raise ValueError("Each batch must contain at least the input tensor.")
            
            x_batch = batch[0]
            x_batch_t = _as_tensor(x_batch, dtype=torch.float32, device=device, name="x_batch")

            outputs = model(x_batch_t)

            # binary classifier returning sigmoid probabilities
            if outputs.ndim == 2 and outputs.shape[1] == 1:
                p1 = outputs.squeeze(1)
            elif outputs.ndim == 1:
                p1 = outputs
            elif outputs.ndim == 2 and outputs.shape[1] == 2:
                p1 = torch.softmax(outputs, dim=1)[:, 1]
            else:
                raise ValueError(f"Model outputs must be either [B, 1], [B], or [B, 2], got {tuple(outputs.shape)}.")

            batch_probs.append(p1.detach().cpu())  # keep as tensor

        if len(batch_probs) == 0:
            raise ValueError("No batches found in the dataloader; cannot compute ensemble predictions.")

        probs = torch.cat(batch_probs, dim=0)  # [N] still tensor
        all_probs.append(probs)

    # Shape: [num_modes, num_samples, num_classes]
    ensemble_probs = torch.stack(all_probs, dim=0)  # [M, N]
    mean_probs = ensemble_probs.mean(dim=0)  # [N]

    mean_probs_2c = torch.stack([1.0 - mean_probs, mean_probs], dim=1)  # [N, 2]
    per_model_probs_2c = torch.stack([1.0 - ensemble_probs, ensemble_probs], dim=2)  # [M, N, 2]

    eps = 1e-10
    entropy = -torch.sum(mean_probs_2c * torch.log(mean_probs_2c + eps), dim=1)  # [N]
    # Aleatoric
    per_model_entropy = -torch.sum(per_model_probs_2c * torch.log(per_model_probs_2c + eps), dim=2)  # [M, N]
    aleatoric = per_model_entropy.mean(dim=0)  # [N]
    # Epistemic
    epistemic = entropy - aleatoric  # [N]

    predictions = mean_probs.numpy()
    entropy = entropy.numpy()
    aleatoric = aleatoric.numpy()
    epistemic = epistemic.numpy()

    return predictions, entropy, aleatoric, epistemic, mean_probs_2c.numpy()

def train_predictive_ensemble(
    models: Sequence[Classifier],
    optimizers: Sequence[torch.optim.Optimizer],
    train_loader: torch.utils.data.DataLoader,
    val_loader: torch.utils.data.DataLoader,
    cfg: Dict[str, Any],
    device: DeviceLike,
    df_fair_loader: Optional[torch.utils.data.DataLoader] = None,
    df_cfg: Optional[Dict[str, Any]] = None,
) -> Tuple[List[Classifier], List[Dict[str, List[float]]]]:
    """
    Train an ensemble of predictive models independently.

    Args:
        models: Sequence of untrained models.
        optimizers: Sequence of optimizers, one per model.
        train_loader: Training dataloader.
        val_loader: Validation dataloader.
        cfg: Configuration dictionary containing cfg["predictive_model"]["epochs"]
             and optionally cfg["predictive_model"]["patience"].
        device: Target device.

    Returns:
        trained_models: List of trained models.
        all_histories: Training history for each model.
    """
    if len(models) != len(optimizers):
        raise ValueError("models and optimizers must have the same length.")

    if "predictive_model" not in cfg:
        raise KeyError("cfg must contain the key 'predictive_model'.")
    if "epochs" not in cfg["predictive_model"]:
        raise KeyError("cfg['predictive_model'] must contain the key 'epochs'.")

    epochs = int(cfg["predictive_model"]["epochs"])
    patience = int(cfg["predictive_model"].get("patience", 5))

    trained_models: List[Classifier] = []
    all_histories: List[Dict[str, List[float]]] = []

    for i, (model, optimizer) in enumerate(zip(models, optimizers), start=1):
        print(f"[Predictive] Training model {i}/{len(models)}")

        model, history = train_single_predictive_model(
            model=model,
            optimizer=optimizer,
            train_loader=train_loader,
            val_loader=val_loader,
            epochs=epochs,
            patience=patience,
            device=device,
            df_fair_loader=df_fair_loader,
            df_cfg=df_cfg,
        )

        trained_models.append(model)
        all_histories.append(history)

    return trained_models, all_histories