from __future__ import annotations

import copy
from dataclasses import dataclass, field
from typing import Any, Optional

import torch
import torch.nn as nn

from modules.utils.tensor_utils import ArrayLike, DeviceLike, _as_tensor, _default_device
from .losses import CompositeLoss, RegularizerContext
from .metrics import accuracy_from_logits, logits_to_probs, probs_to_labels

@dataclass
class TrainConfig:
    epochs: int = 50
    patience: int = 5
    binary: bool = True
    threshold: float = 0.5
    grad_clip_norm: Optional[float] = None
    restore_best: bool = True
    verbose: bool = True
    def validate(self) -> None:
        if self.epochs <= 0:
            raise ValueError("epochs must be positive.")
        if self.patience < 0:
            raise ValueError("patience must be non-negative.")
        if not 0.0 <= self.threshold <= 1.0:
            raise ValueError("threshold must be in [0, 1].")
        if self.grad_clip_norm is not None and self.grad_clip_norm <= 0.0:
            raise ValueError("grad_clip_norm must be positive if specified.")
        
@dataclass 
class History:
    values: dict[str, list[float]] = field(default_factory=dict)

    def append(self, metric: dict[str, float]) -> None:
        for key, value in metric.items():
            self.values.setdefault(key, []).append(float(value))

def _mean_weighted(total: float, n: int, name: str) -> float:
    if n == 0:
        raise ValueError(f"No samples found for {name}; cannot compute mean.")
    return total / n

class PredictiveTrainer:
    def __init__(
        self,
        model: nn.Module,
        optimizer: torch.optim.Optimizer,
        loss_fn: CompositeLoss,
        config: TrainConfig,
        device: Optional[DeviceLike] = None,
        fair_loader: Optional[torch.utils.data.DataLoader] = None,
    ) -> None:
        config.validate()
        self.device = torch.device(device if device is not None else _default_device())
        self.model = model.to(self.device)
        self.optimizer = optimizer
        self.loss_fn = loss_fn
        self.config = config
        self.fair_loader = fair_loader
        self.global_step = 0

    def fit(
        self,
        train_loader: torch.utils.data.DataLoader,
        val_loader: torch.utils.data.DataLoader,
    ) -> tuple[nn.Module, dict[str, list[float]]]:
        best_val_loss = float("inf")
        bad_epochs = 0
        best_state = copy.deepcopy(self.model.state_dict())
        history = History()

        for epoch in range(1, self.config.epochs + 1):
            train_metrics = self.train_epoch(train_loader, epoch)
            val_metrics = self.evaluate(val_loader)

            epoch_metrics = {
                **{f"train_{k}": v for k, v in train_metrics.items()},
                **{f"val_{k}": v for k, v in val_metrics.items()},
            }
            history.append(epoch_metrics)

            if self.config.verbose:
                self._print_epoch(epoch, epoch_metrics)

            val_loss = val_metrics["loss"]
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                bad_epochs = 0
                best_state = copy.deepcopy(self.model.state_dict())
            else:
                bad_epochs += 1
                if bad_epochs >= self.config.patience:
                    if self.config.verbose:
                        print(f"Early stopping triggered after {epoch} epochs.")
                    break

        if self.config.restore_best:
            self.model.load_state_dict(best_state)

        return self.model, history.values
    
    def train_epoch(
        self,
        loader: torch.utils.data.DataLoader,
        epoch: int,
    ) -> dict[str, float]:
        self.model.train()

        totals: dict[str, float] = {}
        correct = 0
        total_samples = 0

        for x_batch, y_batch, *rest in loader:
            batch_metrics, batch_correct, batch_size = self.train_step(x_batch, y_batch, epoch)

            for key, value in batch_metrics.items():
                totals[key] = totals.get(key, 0.0) + value * batch_size

            correct += batch_correct
            total_samples += batch_size

        metrics = {key: _mean_weighted(value, total_samples, key) for key, value in totals.items()}
        metrics["accuracy"] = _mean_weighted(float(correct), total_samples, "accuracy")
        return metrics
    
    def train_step(self, x_batch: ArrayLike, y_batch: ArrayLike, epoch: int) -> tuple[dict[str, float], int, int]:
        x = _as_tensor(x_batch, dtype=torch.float32, device=self.device, name="x_batch")
        target_dtype = torch.float32 if self.config.binary else torch.long
        y = _as_tensor(y_batch, dtype=target_dtype, device=self.device, name="y_batch")

        self.optimizer.zero_grad()
        logits = self.model(x)

        context = RegularizerContext(
            model=self.model,
            device=self.device,
            epoch=epoch,
            global_step=self.global_step,
            fair_loader=self.fair_loader,
        )

        
        loss, metrics = self.loss_fn(logits, y, context)
        loss.backward()

        if self.config.grad_clip_norm is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.grad_clip_norm)

        self.optimizer.step()
        self.global_step += 1

        batch_size = y.view(-1).size(0)
        correct = int(
            accuracy_from_logits(
                logits.detach(),
                y.detach(),
                binary=self.config.binary,
                threshold=self.config.threshold,
            )
            * batch_size
        )
        return metrics, correct, batch_size

    @torch.no_grad()
    def evaluate(self, loader: torch.utils.data.DataLoader) -> dict[str, float]:
        self.model.eval()

        total_loss = 0.0
        correct = 0
        total_samples = 0

        for x_batch, y_batch, *rest in loader:
            x = _as_tensor(x_batch, dtype=torch.float32, device=self.device, name="x_batch")
            target_dtype = torch.float32 if self.config.binary else torch.long
            y = _as_tensor(y_batch, dtype=target_dtype, device=self.device, name="y_batch")

            logits = self.model(x)
            context = RegularizerContext(
                model=self.model,
                device=self.device,
                epoch=-1,
                global_step=self.global_step,
                fair_loader=self.fair_loader,
            )
            loss, _ = self.loss_fn(logits, y, context)

            batch_size = y.view(-1).size(0)
            total_loss += float(loss.item()) * batch_size
            correct += int(
                accuracy_from_logits(
                    logits,
                    y,
                    binary=self.config.binary,
                    threshold=self.config.threshold,
                )
                * batch_size
            )
            total_samples += batch_size

        return {
            "loss": _mean_weighted(total_loss, total_samples, "loss"),
            "accuracy": _mean_weighted(float(correct), total_samples, "accuracy"),
        }
    
    @torch.no_grad()
    def predict_proba(self, x: ArrayLike) -> torch.Tensor:
        self.model.eval()
        x_t = _as_tensor(x, dtype=torch.float32, device=self.device, name="x")
        logits = self.model(x_t)
        return logits_to_probs(logits, binary=self.config.binary)
    
    @torch.no_grad()
    def predict(self, x: ArrayLike) -> torch.Tensor:
        probs = self.predict_proba(x)
        return probs_to_labels(probs, binary=self.config.binary, threshold=self.config.threshold)
    
    @staticmethod
    def _print_epoch(epoch: int, metrics: dict[str, float]) -> None:
        pieces = [f"Epoch {epoch}"]
        for key, value in metrics.items():
            pieces.append(f"{key}: {value:.4f}")
        print(" | ".join(pieces))
