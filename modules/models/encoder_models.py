from dataclasses import dataclass
from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn
import torch.nn.functional as F 
from torchvision.models import resnet18, ResNet18_Weights

from modules.utils.tensor_utils import (
    _to_long_tensor,
    _to_float_tensor,
)

def binary_entropy_from_probs(probs: torch.Tensor, eps: float = 1e-8) -> torch.Tensor:
    """
    Compute binary entropy from probabilities.
    probs: [B]
    returns: [B]
    """
    probs = torch.clamp(probs, min=eps, max=1-eps)
    return -probs * torch.log(probs) - (1.0-probs) * torch.log(1.0-probs)


class SharedImageEncoder(nn.Module):
    """
    Shared CNN encoder for all ensemble heads and the density estimator.

    Output:
        z: [B, D] latent representation
    """
    def __init__(
        self,
        backbone_name: str = "resnet18",
        pretrained: bool = True,
        projection_dim: int = 256,
        train_backbone: bool = True,
    ) -> None:
        super().__init__()

        if backbone_name != "resnet18":
            raise ValueError(f"Unsupported backbone_name={backbone_name}. Currently only supporting 'resnet18'.")

        weights = ResNet18_Weights.DEFAULT if pretrained else None
        backbone = resnet18(weights=weights)

        in_feature = backbone.fc.in_features
        backbone.fc = nn.Identity()

        self.backbone = nn.Sequential(
            nn.Linear(in_features, projection_dim),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(projection_dim, projection_dim),
        )
        self.feature_dim = projection_dim

        if not train_backbone:
            for p in self.backbone.parameters():
                p.requires_grad = False

    def forward(self, images: torch.Tensor) -> torch.Tensor:
        """
        images: [B, 3, H, W]
        returns: [B, feature_dim]
        """
        if images.ndim != 4:
            raise ValueError(f"images must have shape [B, 3, H, W], got {tuple(images.shape)}")
        x = self.backbone(images)
        z = self.projection(x)
        return z



####### ALL FRAMEWORK HERE TO BE IMPLEMENTED IN FUTURE WORK #######
class BinaryPredictionHead(nn.Module):
    """
    One binary classifier head operating on shared latent features z and optional g embedding.
    """
    def __init__(
        self,
        in_dim: int,
        hidden_dim: int = 128,
        dropout: float = 0.2,
    ) -> None:
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, 1),   
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        x: [B, in_dim]
        returns: [B] probabilities
        """
        logits = self.net(x).squeeze(-1)
        return logits

class SharedEncoderEnsembleClassifier(nn.Module):
    """
    Shared encoder + ensemble of binary heads
    If use_group_embedding=True, each head sees [z, emb(g)]
    """
    def __init_(
        self,
        encoder: SharedImageEncoder,
        num_heads: int,
        num_groups: int,
        use_group_embedding: bool = True,
        group_emb_dim: int = 16,
        head_hidden_dim: int = 128,
        head_dropout: float = 0.2,
    ) -> None:
        super().__init__()

        if num_heads < 1:
            raise ValueError(f"num_heads must be >= 1, got {num_heads}")
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")

        self.encoder = encoder
        self.num_heads = num_heads
        self.use_group_embedding = use_group_embedding

        if use_group_embedding:
            self.group_emb = nnEmbedding(num_groups, group_emb_dim)
            head_in_dim = encoder.feature_dim + group_emb_dim
        else:
            self.group_emb = None
            head_in_dim = encoder.feature_dim

        self.heads = nn.ModuleList([
            BinaryPredictionHead(
                in_dim=head_in_dim,
                hidden_dim=head_hidden_dim,
                dropout=head_dropout,
            )
            for _ in range(num_heads)
        ])

    def extract_features(self, images: torch.Tensor) -> torch.Tensor:
        return self.encoder(images)

    def _build_head_input(self, z: torch.Tensor, group_ids: Optional[torch.Tensor]) -> torch.Tensor:
        if self.use_group_embedding:
            if group_ids is None:
                raise ValueError("group_ids must be provided when use_group_embedding=True")
            g_emb = self.group_emb(group_ids)
            head_input = torch.cat([z, g_emb], dim=-1)
        else:
            head_input = z
        return head_input

    def forward_logits(self, images: torch.Tensor, group_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        returns logits of shape [M, B]
        """
        z = self.extract_features(images)
        head_input = self._build_head_input(z, group_ids)
        logits = torch.stack([head(head_input) for head in self.heads], dim=0)
        return logits

    def forward_probs(self, images: torch.Tensor, group_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        returns probs of shape [M, B]
        """
        return torch.sigmoid(self.forward_logits(images, group_ids))

    @torch.no_grad()
    def predictive_statistics(
        self,
        images: torch.Tensor,
        group_ids: Optional[torch.Tensor] = None,
    ) -> Dict[str, torch.Tensor]:
        """
        Compute predictive statistics from ensemble outputs.
        Returns a dict with keys:
            - mean_prob: [B] mean predicted probability across heads
            - pred_entropy: [B] entropy of mean prediction
            - aleatoric: [B] mean binary entropy of individual head predictions
            - epistemic: [B] difference between entropy of mean prediction and mean aleatoric uncertainty
            - per_head_probs: [M, B] predicted probabilities from each head
        """
        self.eval()
        probs = self.forward_probs(images, group_ids)  # [M, B]
        mean_prob = probs.mean(dim=0)  # [B]

        entropy = binary_entropy_from_probs(mean_prob)  # [B]
        aleatoric = binary_entropy_from_probs(probs).mean(dim=0)  # [B]
        epistemic = entropy - aleatoric  # [B]

        return {
            "mean_prob": mean_prob,
            "entropy": entropy,
            "aleatoric": aleatoric,
            "epistemic": epistemic,
            "per_head_probs": probs,
        }
    
# ============================================================
# Density model on the shared latent space z
# ============================================================
class ConditionalDiagonalGaussian(nn.Module):
    """
    Group-conditional Gaussian density model on latent features z.

    p(z|g) = N(\mu_g, diag(var_g)) if use_mlp=False
    or a learned context network from g embedding if use_mlp=True

    This is the right first density model before upgrading to flows.
    """
    def __init__(
        self,
        latent_dim: int,
        num_groups: int,
        group_emb_dim: int = 16,
        hidden_dim: int = 128,
        use_mlp: bool = True,
        min_logvar: float = -6.0,
        max_logvar: float = 4.0,
    ) -> None:
        super().__init__()

        if latent_dim < 1:
            raise ValueError(f"latent_dim must be >= 1, got {latent_dim}")
        if num_groups < 1:
            raise ValueError(f"num_groups must be >= 1, got {num_groups}")

        self.latent_dim = latent_dim
        self.num_groups = num_groups
        self.use_mlp = use_mlp
        self.min_logvar = min_logvar
        self.max_logvar = max_logvar

        if use_mlp:
            self.group_emb = nn.Embedding(num_groups, group_emb_dim)
            self.net = nn.Sequential(
                nn.Linear(group_emb_dim, hidden_dim),
                nn.ReLU(),
                nn.Linear(hidden_dim, 2 * latent_dim),  # output both mean and logvar
            )
        else:
            self.group_emb = None
            self.mean = nn.Parameter(torch.zeros(num_groups, latent_dim))
            self.logvar = nn.Parameter(torch.zeros(num_groups, latent_dim))

        logvar = torch.clamp(logvar, min=self.min_logvar, max=self.max_logvar)
        return mu, logvar

    def log_prob(self, z: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(z|g) for each example in the batch.
        z: [B, latent_dim]
        group_ids: [B]
        returns: [B] log probabilities
        """
        um, logvar = self._params(group_ids)
        inv_var = torch.exp(-logvar)

        # diagonal Gaussian log-density
        logp = -0.5 * (
            ((z - mu) ** 2 * inv_var).sum(dim=-1)
            + logvar.sum(dim=-1)
            + z.shape[1] * torch.log(torch.tensor(2.0 * torch.pi, device=z.device))
        )
        return logp

    def nll(self, z: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
        """
        Negative log-likelihood of z under p(z|g).
        """
        return -self.log_prob(z, group_ids).mean()


# ============================================================
# Full framework wrapper
# ============================================================

class ImageFairnessFramework(nn.Module):
    """
    One shared encoder, one ensemble of heads, one density estimator on z.
    """
    def __init__(
        self,
        classifier: SharedEncoderEnsembleClassifier,
        density_model: ConditionalDiagonalGaussian,
    ) -> None:
        super().__init__()
        self.classifier = classifier
        self.density_model = density_model

        if self.classifier.encoder.feature_dim != self.density_model.latent_dim:
            raise ValueError(
                f"Encoder feature_dim={self.classifier.encoder.feature_dim} must match "
                f"density_model latent_dim={self.density_model.latent_dim}"
            )

        def extract_features(self, images: torch.Tensor) -> torch.Tensor:
            return self.classifier.extract_features(images)

        def predicive_logits(self, images: torch.Tensor, group_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
            return self.classifier.forward_logits(images, group_ids)

        def predictive_probs(self, images: torch.Tensor, group_ids: Optional[torch.Tensor] = None) -> torch.Tensor:
            return self.classifier.forward_probs(images, group_ids)

        def predictive_statistics(self, images: torch.Tensor, group_ids: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
            return self.classifier.predictive_statistics(images, group_ids)

        def latent_log_prob(self, images: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
            z = self.extract_features(images)
            return self.density_model.log_prob(z, group_ids)

        def latent_log_prob_from_z(self, z: torch.Tensor, group_ids: torch.Tensor) -> torch.Tensor:
            return self.density_model.log_prob(z, group_ids)

# ============================================================
# Training: classifier first
# ============================================================

@dataclass
class PredictiveBatchOutput:
    loss: float,
    mean_prob: torch.Tensor,
    per_head_probs: torch.Tensor

def train_predictive_step(
    model: ImageFairnessFramework,
    optimizer: torch.optim.Optimizer,
    images: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    device: DeviceLike,
    pos_weight: Optional[torch.Tensor] = None,
) -> PredictiveBatchOutput:
    model.train()

    images = _as_float_tensor(images, device=device)
    labels = _as_float_tensor(labels, device).view(-1)
    group_ids = _as_long_tensor(group_ids, device=device).view(-1)

    optimizer.zero_grad(set_to_none=True)

    logits = model.predictive_logits(images, group_ids) # [M, B]

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = 0.0
    for m in range(logits.shape[0]):
        loss= loss + criterion(logits[m], labels)
    loss = loss / logits.shape[0]

    loss.backward()
    optimizer.step()

    probs = torch.sigmoid(logits).detach()
    mean_prob = probs.mean(dim=0)

    return PredictiveBatchOutput(
        loss=float(loss.item()),
        mean_prob=mean_prob,
        per_head_probs=probs,
    )

@torch.no_grad()
def eval_predictive_step(
    model: ImageFairnessFramework,
    images: torch.Tensor,
    labels: torch.Tensor,
    group_ids: torch.Tensor,
    device: DeviceLike,
    pos_weight: Optional[torch.Tensor] = None,
) -> Dict[str, float]:
    model.eval()

    images = _as_float_tensor(images, device=device)
    labels = _as_float_tensor(labels, device=device).view(-1)
    group_ids = _as_long_tensor(group_ids, device=device).view(-1)

    logits = model.predictive_logits(images, group_ids) # [M, B]

    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    loss = 0.0
    for m in range(logits.shape[0]):
        loss= loss + criterion(logits[m], labels)
    loss = loss / logits.shape[0]

    probs = torch.sigmoid(logits).mean(dim=0)
    preds = (probs >= 0.5).long()
    acc = (preds == labels.long()).float().mean().item()

    return {
        "loss": float(loss.item()),
        "accuracy": float(acc),
    }

# ============================================================
# Training: density model second (encoder frozen)
# ============================================================   

def freeze_encoder(model: ImageFairnessFramework) -> None:
    for param in model.classifier.encoder.parameters():
        param.requires_grad = False

def unfreeze_encoder(model: ImageFairnessFramework) -> None:
    for param in model.classifier.encoder.parameters():
        param.requires_grad = True

def train_density_step(
    model: ImageFairnessFramework,
    optimizer: Optimizer, 
    images: torch.Tensor,
    group_ids: torch.Tensor,
    device: DeviceLike,
) -> float:
    """
    Train only the density model on frozen encoder features.
    """
    model.train()

    images = _as_float_tensor(images, device=device)
    group_ids = _as_long_tensor(group_ids, device=device).view(-1)

    optimizer.zero_grad(set_to_none=True)

    with torch.no_grad():
        z = model.extract_features(images)

    loss = model.density_model.nll(z, group_ids)
    loss.backward()
    optimizer.step()

    return float(loss.item())

@torch.no_grad()
def eval_density_step(
    model: ImageFairnessFramework,
    images: torch.Tensor,
    group_ids: torch.Tensor,
    device: DeviceLike,
) -> float:
    model.eval()

    images = _as_float_tensor(images, device=device)
    group_ids = _as_long_tensor(group_ids, device=device).view(-1)

    z = model.extract_features(images)
    loss = model.density_model.nll(z, group_ids)

    return float(loss.item())

# ============================================================
# Test-time inspection
# ============================================================
@torch.no_grad()
def score_batch(
    model: ImageFairnessFramework,
    images: torch.Tensor,
    group_ids: torch.Tensor,
    device: DeviceLike,
) -> Dict[str, torch.Tensor]:
    """
    Compute per-example predictive and density quantities.
    """
    model.eval()

    images = _as_float_tensor(images, device)
    group_ids = _as_long_tensor(group_ids, device).view(-1)

    z = model.extract_features(images)
    stats = model.predictive_statistics(images, group_ids)
    logp_z_given_g = model.latent_log_prob_from_z(z, group_ids)

    return {
        "z": z,
        "mean_prob": stats["mean_prob"],
        "pred_entropy": stats["pred_entropy"],
        "expected_entropy": stats["expected_entropy"],
        "epistemic_mi": stats["epistemic_mi"],
        "logp_z_given_g": logp_z_given_g,
    }

@torch.no_grad()
def compare_two_examples(
    model: ImageFairnessFramework,
    images: Tensor,
    group_ids: Tensor,
    idx_a: int,
    idx_b: int,
    device: torch.device,
) -> Dict[str, Dict[str, float]]:
    """
    Compare two examples by support and predictive uncertainty.
    """
    batch_images = torch.stack([images[idx_a], images[idx_b]], dim=0)
    batch_groups = torch.tensor([group_ids[idx_a], group_ids[idx_b]], dtype=torch.long)

    out = score_batch(model, batch_images, batch_groups, device)

    result = {
        "a": {
            "index": idx_a,
            "group_id": int(batch_groups[0].item()),
            "mean_prob": float(out["mean_prob"][0].item()),
            "pred_entropy": float(out["pred_entropy"][0].item()),
            "epistemic_mi": float(out["epistemic_mi"][0].item()),
            "logp_z_given_g": float(out["logp_z_given_g"][0].item()),
        },
        "b": {
            "index": idx_b,
            "group_id": int(batch_groups[1].item()),
            "mean_prob": float(out["mean_prob"][1].item()),
            "pred_entropy": float(out["pred_entropy"][1].item()),
            "epistemic_mi": float(out["epistemic_mi"][1].item()),
            "logp_z_given_g": float(out["logp_z_given_g"][1].item()),
        },
    }
    return result



# # ============================================================
# # EXAMPLE OF USAGE
# # ============================================================

# # Construction
# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# encoder = SharedImageEncoder(
#     backbone_name="resnet18",
#     pretrained=True,
#     projection_dim=256,
#     train_backbone=True,
# )

# classifier = SharedEncoderEnsembleClassifier(
#     encoder=encoder,
#     num_heads=5,
#     num_groups=8,              # example
#     use_group_embedding=True,
#     group_emb_dim=16,
#     head_hidden_dim=128,
#     head_dropout=0.2,
# )

# density_model = ConditionalDiagonalGaussian(
#     latent_dim=256,
#     num_groups=8,
#     group_emb_dim=16,
#     hidden_dim=128,
#     use_mlp=True,
# )

# model = ImageFairnessFramework(
#     classifier=classifier,
#     density_model=density_model,
# ).to(device)

# # Predictive stage training
# predictive_params = (
#     list(model.classifier.encoder.parameters())
#     + list(model.classifier.heads.parameters())
#     + (list(model.classifier.group_emb.parameters()) if model.classifier.group_emb is not None else [])
# )

# pred_optimizer = torch.optim.Adam(predictive_params, lr=1e-4, weight_decay=1e-4)

# for epoch in range(10):
#     train_losses = []
#     for images, labels, group_ids in train_loader:
#         out = train_predictive_step(
#             model=model,
#             optimizer=pred_optimizer,
#             images=images,
#             labels=labels,
#             group_ids=group_ids,
#             device=device,
#         )
#         train_losses.append(out.loss)

#     val_metrics = []
#     for images, labels, group_ids in val_loader:
#         metrics = eval_predictive_step(
#             model=model,
#             images=images,
#             labels=labels,
#             group_ids=group_ids,
#             device=device,
#         )
#         val_metrics.append(metrics)

#     mean_val_loss = np.mean([m["loss"] for m in val_metrics])
#     mean_val_acc = np.mean([m["accuracy"] for m in val_metrics])

#     print(
#         f"[Predictive] epoch={epoch+1:03d} "
#         f"train_loss={np.mean(train_losses):.4f} "
#         f"val_loss={mean_val_loss:.4f} "
#         f"val_acc={mean_val_acc:.4f}"
#     )

# # Density stage training
# freeze_encoder(model)

# density_optimizer = torch.optim.Adam(model.density_model.parameters(), lr=1e-3, weight_decay=1e-5)

# for epoch in range(10):
#     train_losses = []
#     for images, _, group_ids in train_loader:
#         loss = train_density_step(
#             model=model,
#             optimizer=density_optimizer,
#             images=images,
#             group_ids=group_ids,
#             device=device,
#         )
#         train_losses.append(loss)

#     val_losses = []
#     for images, _, group_ids in val_loader:
#         loss = eval_density_step(
#             model=model,
#             images=images,
#             group_ids=group_ids,
#             device=device,
#         )
#         val_losses.append(loss)

#     print(
#         f"[Density] epoch={epoch+1:03d} "
#         f"train_nll={np.mean(train_losses):.4f} "
#         f"val_nll={np.mean(val_losses):.4f}"
#     )

# # Test-time scoring
# images, labels, group_ids = next(iter(test_loader))

# out = score_batch(model, images, group_ids, device=device)

# print("mean_prob:", out["mean_prob"][:5])
# print("epistemic_mi:", out["epistemic_mi"][:5])
# print("logp_z_given_g:", out["logp_z_given_g"][:5])

# comparison = compare_two_examples(
#     model=model,
#     images=images,
#     group_ids=group_ids,
#     idx_a=3,
#     idx_b=17,
#     device=device,
# )

# print(comparison)