import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.distributions as dist
import copy

from typing import Any, Dict, List, Optional, Sequence, Tuple, Union

from modules.utils.tensor_utils import (
    ArrayLike,
    DeviceLike,
    _as_tensor, 
    _validate_vocab_sizes, 
    _validate_2d_long_tensor, 
    _validate_1d_long_tensor, 
    _validate_2d_float_tensor
)

class ARModel(nn.Module):
    """
    Autoregressive model over categorical features conditioned on group identity.

    Each categorical feature x_j is predicted as:
        p(x_j | x_<j, g)
    using feature embeddings, a prepended start token embedding (zeros),
    a repeated group embedding, and a GRU backbone.
    """
    def __init__(
        self, 
        vocab_sizes: Sequence[int], 
        num_groups: int, 
        g_emb_dim: int=16, 
        x_emb_dim: int=32, 
        hidden: int=128, 
        dropout: float=0.1, 
        num_layers: int=1
    ) -> None:
        super().__init__()

        vocab_sizes = _validate_vocab_sizes(vocab_sizes)
        if num_groups <= 0:
            raise ValueError("num_groups must be a positive integer.")
        if g_emb_dim <= 0 or x_emb_dim <= 0 or hidden <= 0:
            raise ValueError("Embedding and hidden dimensions must be positive integers.")
        if num_layers <= 0:
            raise ValueError("num_layers must be a positive integer.")
        if not(0.0 <= dropout < 1.0):
            raise ValueError("dropout must be in the range [0.0, 1.0).")

        self.vocab_sizes: List[int] = list(vocab_sizes)
        self.num_features: int = len(vocab_sizes)

        self.g_emb = nn.Embedding(num_groups, g_emb_dim)
        self.x_embs = nn.ModuleList([nn.Embedding(vocab_size, x_emb_dim) for vocab_size in self.vocab_sizes])  # embeddings for each categorical feature
        gru_dropout = dropout if num_layers > 1 else 0.0
        self.gru = nn.GRU(
            input_size=x_emb_dim + g_emb_dim, 
            hidden_size=hidden, 
            num_layers=num_layers, 
            batch_first=True, 
            dropout=gru_dropout
        )
        self.heads = nn.ModuleList([nn.Linear(hidden, vocab_size) for vocab_size in self.vocab_sizes])  # output heads for each categorical feature

    def forward(self, x_cat, g_ids):
        """
        Compute logits for each categorical feature.

        Args:
            x_cat: Long tensor of shape [B, D].
            g_ids: Long tensor of shape [B].

        Returns:
            List of D tensors, where the j-th tensor has shape [B, vocab_sizes[j]].
        """
        device = self.g_emb.weight.device
        x_cat = _as_tensor(x_cat, dtype=torch.long, device=device, name="x_cat")
        g_ids = _as_tensor(g_ids, dtype=torch.long, device=device, name="g_ids")

        _validate_2d_long_tensor(x_cat, expected_second_dim=self.num_features, name="x_cat")
        _validate_1d_long_tensor(g_ids, name="g_ids")

        if x_cat.shape[0] != g_ids.shape[0]:
            raise ValueError(f"Batch size mismatch: x_cat has batch size {x_cat.shape[0]}, but g_ids has batch size {g_ids.shape[0]}.")

        batch_size = x_cat.shape[0]  # B
        g = self.g_emb(g_ids)  # [B, g_emb_dim]

        # Get the embeddings for each categorical feature
        x_steps = [emb(x_cat[:, j]) for j, emb in enumerate(self.x_embs)]  # D * [B, x_emb_dim]
        x_seq = torch.stack(x_steps, dim=1)  # [B, D, x_emb_dim]
        start = torch.zeros(batch_size, 1, x_seq.size(-1), device=x_seq.device, dtype=x_seq.dtype)
        x_in = torch.cat([start, x_seq[:, :-1, :]], dim=1)  # [B, D, x_emb_dim]

        # Concatenate group embedding to each timestep
        # g_rep = g.unsqueeze(1).repeat(1, x_seq.size(1), 1)  # [B, D, g_emb_dim]
        g_rep = g.unsqueeze(1).expand(-1, self.num_features, -1)  # [B, D, g_emb_dim]
        seq_in = torch.cat([x_in, g_rep], dim=-1)  # [B, D, x_emb_dim + g_emb_dim]

        # Process through GRU
        h, _ = self.gru(seq_in)  # [B, D, hidden]
        # Get logits for each categorical feature
        logits_list = [head(h[:, j, :]) for j, head in enumerate(self.heads)]
        return logits_list

    def log_prob(self, x_cat: torch.Tensor, g_ids: torch.Tensor) -> torch.Tensor:
        """
        Compute autoregressive log-probability for each sample.

        Args:
            x_cat: Long tensor of shape [B, D].
            g_ids: Long tensor of shape [B].

        Returns:
            Tensor of shape [B] with log p(x | g).
        """
        device = self.g_emb.weight.device
        x_cat = _as_tensor(x_cat, dtype=torch.long, device=device, name="x_cat")
        g_ids = _as_tensor(g_ids, dtype=torch.long, device=device, name="g_ids")

        logits_list = self.forward(x_cat, g_ids)

        logp = torch.zeros(x_cat.size(0), device=device, dtype=torch.float32)
        for j, logits in enumerate(logits_list):
            ce = F.cross_entropy(logits, x_cat[:, j], reduction="none") # ce = -log p(x_j | x_<j, g)
            logp -= ce  # accumulate log-probability
        return logp

class ContextEncoder(nn.Module):
    """
    Encodes categorical features and group identity into a dense context vector.
    """
    def __init__(
        self, 
        vocab_sizes: Sequence[int], 
        num_groups: int, 
        g_emb_dim: int=16, 
        x_emb_dim: int=16, 
        hidden: int=128, 
        out_dim: int=128
    ) -> None:
        super().__init__()

        vocab_sizes = _validate_vocab_sizes(vocab_sizes)
        if num_groups <= 0:
            raise ValueError("num_groups must be a positive integer.")
        if g_emb_dim <= 0 or x_emb_dim <= 0 or hidden <= 0 or out_dim <= 0:
            raise ValueError("Embedding, hidden, and output dimensions must be positive integers.")

        self.vocab_sizes: List[int] = list(vocab_sizes)  # D
        self.num_features: int = len(vocab_sizes)

        self.g_emb = nn.Embedding(num_groups, g_emb_dim)
        self.x_embs = nn.ModuleList([nn.Embedding(vocab_size, x_emb_dim) for vocab_size in vocab_sizes])
        
        in_dim = g_emb_dim + self.num_features * x_emb_dim
        self.proj = nn.Sequential(
            nn.Linear(in_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, out_dim)
        )

    def forward(self, x_cat: torch.Tensor, g_ids: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x_cat: Long tensor of shape [B, D].
            g_ids: Long tensor of shape [B].

        Returns:
            Tensor of shape [B, out_dim].
        """
        device = self.g_emb.weight.device
        x_cat = _as_tensor(x_cat, dtype=torch.long, device=device, name="x_cat")
        g_ids = _as_tensor(g_ids, dtype=torch.long, device=device, name="g_ids")

        _validate_2d_long_tensor(x_cat, expected_second_dim=self.num_features, name="x_cat")
        _validate_1d_long_tensor(g_ids, name="g_ids")

        if x_cat.shape[0] != g_ids.shape[0]:
            raise ValueError("x_cat and g_ids must have the same batch size.")

        g = self.g_emb(g_ids)  # [B, g_emb_dim]
        x_parts = [emb(x_cat[:, j]) for j, emb in enumerate(self.x_embs)]  # D * [B, x_emb_dim]
        x_cat_emb = torch.cat(x_parts, dim=-1)  # [B, D * x_emb_dim]
        context = torch.cat([g, x_cat_emb], dim=-1)  # [B, g_emb_dim + D * x_emb_dim]
        context = self.proj(context)  # [B, out_dim]
        return context


class ConditionalAffineCoupling(nn.Module):
    """
    Conditional affine coupling layer for RealNVP-style flows.

    Splits x = [x_a, x_b]. Leaves x_a unchanged and transforms x_b using
    shift/log_scale predicted from [x_a, context].
    """
    def __init__(
        self, 
        dim: int, 
        split: int, 
        context_dim: int, 
        hidden: int=128, 
        debug: bool=False, 
        keep_last_batch: bool=False
    ) -> None:
        super().__init__()
        
        if dim <= 1:
            raise ValueError("dim must be > 1.")
        if not (0 < split < dim):
            raise ValueError("split must satisfy 0 < split < dim.")
        if context_dim <= 0 or hidden <= 0:
            raise ValueError("context_dim and hidden must be positive.")

        self.dim = dim
        self.split = split
        self.debug = debug
        self.keep_last_batch = keep_last_batch

        self.net = nn.Sequential(
            nn.Linear(split + context_dim, hidden),
            nn.ReLU(),
            # nn.Linear(hidden, hidden),
            # nn.ReLU(),
            nn.Linear(hidden, 2 * (dim - split))  # outputs both scale and shift
        )

        self.reset_debug_stats()

    def reset_debug_stats(self) -> None:
        self.debug_stats: Dict[str, float] = {
            "num_calls": 0,

            "shift_mean_sum": 0.0,
            "shift_std_sum": 0.0,
            "shift_absmax_sum": 0.0,
            "shift_min_sum": 0.0,
            "shift_max_sum": 0.0,

            "log_scale_mean_sum": 0.0,
            "log_scale_std_sum": 0.0,
            "log_scale_absmax_sum": 0.0,
            "log_scale_min_sum": 0.0,
            "log_scale_max_sum": 0.0,

            "scale_mean_sum": 0.0,
            "scale_std_sum": 0.0,
            "scale_absmax_sum": 0.0,
            "scale_min_sum": 0.0,
            "scale_max_sum": 0.0,

            # fraction of entries close to tanh saturation
            "log_scale_sat_frac_sum": 0.0,
        }

        self.last_batch_debug: Optional[Dict[str, torch.Tensor]] = None

    @staticmethod
    def _safe_std(x: torch.Tensor) -> float:
        if x.numel() <= 1:
            return 0.0
        return x.std(unbiased=False).item()
    
    @torch.no_grad()
    def _update_debug_stats(self, shift: torch.Tensor, log_scale: torch.Tensor) -> None:
        scale = torch.exp(log_scale)

        self.debug_stats["num_calls"] += 1

        self.debug_stats["shift_mean_sum"] += shift.mean().item()
        self.debug_stats["shift_std_sum"] += self._safe_std(shift)
        self.debug_stats["shift_absmax_sum"] += shift.abs().max().item()
        self.debug_stats["shift_min_sum"] += shift.min().item()
        self.debug_stats["shift_max_sum"] += shift.max().item()

        self.debug_stats["log_scale_mean_sum"] += log_scale.mean().item()
        self.debug_stats["log_scale_std_sum"] += self._safe_std(log_scale)
        self.debug_stats["log_scale_absmax_sum"] += log_scale.abs().max().item()
        self.debug_stats["log_scale_min_sum"] += log_scale.min().item()
        self.debug_stats["log_scale_max_sum"] += log_scale.max().item()

        self.debug_stats["scale_mean_sum"] += scale.mean().item()
        self.debug_stats["scale_std_sum"] += self._safe_std(scale)
        self.debug_stats["scale_absmax_sum"] += scale.abs().max().item()
        self.debug_stats["scale_min_sum"] += scale.min().item()
        self.debug_stats["scale_max_sum"] += scale.max().item()

        # saturation indicator: near tanh limits
        sat_frac = (log_scale.abs() > 0.95).float().mean().item()
        self.debug_stats["log_scale_sat_frac_sum"] += sat_frac

        if self.keep_last_batch:
            self.last_batch_debug = {
                "shift": shift.detach().cpu(),
                "log_scale": log_scale.detach().cpu(),
                "scale": scale.detach().cpu(),
            }

    def get_debug_summary(self) -> Dict[str, float]:
        n = int(self.debug_stats["num_calls"])
        if n == 0:
            return {"num_calls": 0}

        return {
            "num_calls": n,

            "shift_mean": self.debug_stats["shift_mean_sum"] / n,
            "shift_std": self.debug_stats["shift_std_sum"] / n,
            "shift_absmax": self.debug_stats["shift_absmax_sum"] / n,
            "shift_min": self.debug_stats["shift_min_sum"] / n,
            "shift_max": self.debug_stats["shift_max_sum"] / n,

            "log_scale_mean": self.debug_stats["log_scale_mean_sum"] / n,
            "log_scale_std": self.debug_stats["log_scale_std_sum"] / n,
            "log_scale_absmax": self.debug_stats["log_scale_absmax_sum"] / n,
            "log_scale_min": self.debug_stats["log_scale_min_sum"] / n,
            "log_scale_max": self.debug_stats["log_scale_max_sum"] / n,

            "scale_mean": self.debug_stats["scale_mean_sum"] / n,
            "scale_std": self.debug_stats["scale_std_sum"] / n,
            "scale_absmax": self.debug_stats["scale_absmax_sum"] / n,
            "scale_min": self.debug_stats["scale_min_sum"] / n,
            "scale_max": self.debug_stats["scale_max_sum"] / n,

            "log_scale_sat_frac": self.debug_stats["log_scale_sat_frac_sum"] / n,
        }

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Forward transformation y = f(x; context).

        Args:
            x: Tensor of shape [B, dim].
            context: Tensor of shape [B, context_dim].

        Returns:
            y: Tensor of shape [B, dim].
            log_det: Tensor of shape [B].
        """
        if x.ndim != 2:
            raise ValueError(f"x must have shape [B, dim], got {tuple(x.shape)}.")
        if context.ndim != 2:
            raise ValueError(f"context must have shape [B, context_dim], got {tuple(context.shape)}.")
        if x.shape[0] != context.shape[0]:
            raise ValueError("x and context must have the same batch size.")
        if x.shape[1] != self.dim:
            raise ValueError(f"x must have second dimension {self.dim}, got {x.shape[1]}.")

        x_a = x[:, :self.split]
        x_b = x[:, self.split:]

        h = torch.cat([x_a, context], dim=-1)
        params = self.net(h)
        shift, log_scale = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale)  # constrain log_scale to a reasonable range
        
        if self.debug:
            self._update_debug_stats(shift, log_scale)

        y_b = x_b * torch.exp(log_scale) + shift
        y = torch.cat([x_a, y_b], dim=-1)
        log_det = log_scale.sum(dim=-1)  # sum over transformed dimensions
        return y, log_det

    def inverse(self, y: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Inverse transformation x = f^{-1}(y; context).

        Args:
            y: Tensor of shape [B, dim].
            context: Tensor of shape [B, context_dim].

        Returns:
            x: Tensor of shape [B, dim].
            log_det: Tensor of shape [B].
        """
        if y.ndim != 2:
            raise ValueError(f"y must have shape [B, dim], got {tuple(y.shape)}.")
        if context.ndim != 2:
            raise ValueError(f"context must have shape [B, context_dim], got {tuple(context.shape)}.")
        if y.shape[0] != context.shape[0]:
            raise ValueError("y and context must have the same batch size.")
        if y.shape[1] != self.dim:
            raise ValueError(f"y must have second dimension {self.dim}, got {y.shape[1]}.")

        y_a = y[:, :self.split]
        y_b = y[:, self.split:]

        h = torch.cat([y_a, context], dim=-1)
        params = self.net(h)
        shift, log_scale = params.chunk(2, dim=-1)
        log_scale = torch.tanh(log_scale)

        if self.debug:
            self._update_debug_stats(shift, log_scale)

        x_b = (y_b - shift) * torch.exp(-log_scale)
        x = torch.cat([y_a, x_b], dim=-1)
        log_det = -log_scale.sum(dim=-1)  # inverse transformation
        return x, log_det


class ConditionalRealNVPFlow(nn.Module):
    """
    Conditional RealNVP flow with random fixed permutations between coupling layers.
    """
    def __init__(
        self, 
        dim: int, 
        context_dim: int, 
        num_couplings: int=6, 
        hidden: int=128, 
        seed: Optional[int]=0, 
        debug: bool=False, 
        keep_last_batch: bool=False
    ) -> None:
        super().__init__()

        if dim <= 1:
            raise ValueError("dim must be > 1.")
        if context_dim <= 0:
            raise ValueError("context_dim must be positive.")
        if num_couplings <= 0:
            raise ValueError("num_couplings must be positive.")
        if hidden <= 0:
            raise ValueError("hidden must be positive.")

        self.dim = dim
        self.context_dim = context_dim
        self.num_couplings = num_couplings

        self.couplings = nn.ModuleList()

        generator = None
        if seed is not None:
            generator = torch.Generator()
            generator.manual_seed(seed)

        split = dim // 2
        for i in range(num_couplings):
            perm = torch.randperm(dim, generator=generator)
            inv = torch.argsort(perm)

            self.register_buffer(f"perm_{i}", perm)
            self.register_buffer(f"inv_perm_{i}", inv)

            self.couplings.append(
                ConditionalAffineCoupling(dim=dim, split=split, context_dim=context_dim, hidden=hidden, debug=debug, keep_last_batch=keep_last_batch)
            )

        self.register_buffer("base_loc", torch.zeros(self.dim))
        self.register_buffer("base_cov", torch.eye(self.dim))
        # self.base = dist.MultivariateNormal(torch.zeros(dim), torch.eye(dim))
    
    def _base_dist(self) -> dist.MultivariateNormal:
        return dist.MultivariateNormal(self.base_loc, self.base_cov)

    def forward(self, x: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map x to latent z.

        Args:
            x: Tensor of shape [B, dim].
            context: Tensor of shape [B, context_dim].

        Returns:
            z: Tensor of shape [B, dim].
            log_det_total: Tensor of shape [B].
        """
        if x.ndim != 2:
            raise ValueError(f"x must have shape [B, dim], got {tuple(x.shape)}.")
        if context.ndim != 2:
            raise ValueError(f"context must have shape [B, context_dim], got {tuple(context.shape)}.")
        if x.shape[0] != context.shape[0]:
            raise ValueError("x and context must have the same batch size.")
        if x.shape[1] != self.dim:
            raise ValueError(f"x must have second dimension {self.dim}, got {x.shape[1]}.")

        log_det_total = torch.zeros(x.size(0), device=x.device, dtype=x.dtype)
        z = x 
        for i, coupling in enumerate(self.couplings):
            perm = getattr(self, f"perm_{i}")
            z = z.index_select(1, perm)
            z, log_det = coupling(z, context)
            log_det_total += log_det
        return z, log_det_total

    def inverse(self, z: torch.Tensor, context: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Map latent z back to x.
        """
        if z.ndim != 2:
            raise ValueError(f"z must have shape [B, dim], got {tuple(z.shape)}.")
        if context.ndim != 2:
            raise ValueError(f"context must have shape [B, context_dim], got {tuple(context.shape)}.")
        if z.shape[0] != context.shape[0]:
            raise ValueError("z and context must have the same batch size.")
        if z.shape[1] != self.dim:
            raise ValueError(f"z must have second dimension {self.dim}, got {z.shape[1]}.")

        log_det_total = torch.zeros(z.size(0), device=z.device, dtype=z.dtype)
        x = z

        for i in reversed(range(self.num_couplings)):
            coupling = self.couplings[i]
            x, log_det = coupling.inverse(x, context)
            inv_perm = getattr(self, f"inv_perm_{i}")
            x = x.index_select(1, inv_perm)
            log_det_total += log_det

        return x, log_det_total

    def log_prob(self, x: torch.Tensor, context: torch.Tensor) -> torch.Tensor:
        """
        Compute log p(x | context).
        """
        z, log_det = self.forward(x, context)
        log_base = self._base_dist().log_prob(z)
        return log_base + log_det

def train_step_ar(
    ar_model: ARModel, 
    optimizer: torch.optim.Optimizer, 
    x_cat: ArrayLike, 
    g_ids: ArrayLike, 
    device: DeviceLike
) -> float:
    """
    One optimization step for the autoregressive model. Compute the NLL loss and update the model parameters.
    """
    x_cat_t = _as_tensor(x_cat, dtype=torch.long, device=device, name="x_cat")
    g_ids_t = _as_tensor(g_ids, dtype=torch.long, device=device, name="g_ids")

    optimizer.zero_grad(set_to_none=True)
    logits_list = ar_model(x_cat_t, g_ids_t)

    nll = torch.zeros(x_cat_t.size(0), dtype=torch.float32, device=device)
    for j, logits in enumerate(logits_list):
        nll += F.cross_entropy(logits, x_cat_t[:, j], reduction="none")

    loss = nll.mean()
    loss.backward()
    optimizer.step()
    return float(loss.item())

@torch.no_grad()
def eval_step_ar(
    ar_model: ARModel, 
    x_cat: ArrayLike, 
    g_ids: ArrayLike, 
    device: DeviceLike
) -> float:
    """
    One evaluation step for the autoregressive model.
    """
    x_cat_t = _as_tensor(x_cat, dtype=torch.long, device=device, name="x_cat")
    g_ids_t = _as_tensor(g_ids, dtype=torch.long, device=device, name="g_ids")

    logits_list = ar_model(x_cat_t, g_ids_t)

    nll = torch.zeros(x_cat_t.size(0), dtype=torch.float32, device=device)
    for j, logits in enumerate(logits_list):
        nll += F.cross_entropy(logits, x_cat_t[:, j], reduction="none")

    return float(nll.mean().item())

def train_ar_model(
    model: ARModel, 
    optimizer: torch.optim.Optimizer, 
    train_loader: Any, 
    val_loader: Any, 
    epochs: int, 
    device: DeviceLike
) -> Tuple[nn.Module, List[float], List[float]]:
    """
    Train an autoregressive model. Use train_step_ar and eval_step_ar for the training and evaluation steps.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    train_losses: List[float] = []
    val_losses: List[float] = []

    best_val = float("inf")
    best_state: Optional[Dict[str, Tensor]] = None

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_train_losses: List[float] = []

        for x_cat, g_ids in train_loader:
            loss = train_step_ar(model, optimizer, x_cat, g_ids, device)
            epoch_train_losses.append(loss)

        model.eval()
        epoch_val_losses: List[float] = []

        for x_cat, g_ids in val_loader:
            loss = eval_step_ar(model, x_cat, g_ids, device)
            epoch_val_losses.append(loss)

        train_loss = float(np.mean(epoch_train_losses))
        val_loss = float(np.mean(epoch_val_losses))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[AR] Epoch {epoch:03d}: train_NLL={train_loss:.4f}, val_NLL={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            best_state = copy.deepcopy(model.state_dict())

    if best_state is not None:
        model.load_state_dict(best_state)

    return model, train_losses, val_losses


def train_step_flow(
    flow_model: ConditionalRealNVPFlow, 
    ctx_model: ContextEncoder, 
    optimizer: torch.optim.Optimizer, 
    x_cont: ArrayLike, 
    x_cat: ArrayLike, 
    g_ids: ArrayLike, 
    device: DeviceLike
) -> float:
    """
    One optimization step for the conditional flow model.
    """
    x_cont_t = _as_tensor(x_cont, dtype=torch.float32, device=device, name="x_cont")
    x_cat_t = _as_tensor(x_cat, dtype=torch.long, device=device, name="x_cat")
    g_ids_t = _as_tensor(g_ids, dtype=torch.long, device=device, name="g_ids")

    optimizer.zero_grad(set_to_none=True)
    context = ctx_model(x_cat_t, g_ids_t)
    logp = flow_model.log_prob(x_cont_t, context)

    loss = -logp.mean()
    loss.backward()
    optimizer.step()
    return float(loss.item())

@torch.no_grad()
def eval_step_flow(
    flow_model: ConditionalRealNVPFlow,
    ctx_model: ContextEncoder,
    x_cont: ArrayLike,
    x_cat: ArrayLike,
    g_ids: ArrayLike,
    device: DeviceLike,
) -> float:
    """
    One evaluation step for the conditional flow model.
    """
    x_cont_t = _as_tensor(x_cont, dtype=torch.float32, device=device, name="x_cont")
    x_cat_t = _as_tensor(x_cat, dtype=torch.long, device=device, name="x_cat")
    g_ids_t = _as_tensor(g_ids, dtype=torch.long, device=device, name="g_ids")

    context = ctx_model(x_cat_t, g_ids_t)
    logp = flow_model.log_prob(x_cont_t, context)

    loss = -logp.mean()
    return float(loss.item())

def train_flow_model(
    flow_model: ConditionalRealNVPFlow, 
    ctx_model: ContextEncoder, 
    optimizer: torch.optim.Optimizer, 
    train_loader: Any, 
    val_loader: Any, 
    epochs: int, 
    device: DeviceLike
) -> Tuple[ConditionalRealNVPFlow, ContextEncoder, List[float], List[float]]:
    """
    Train a flow model. Use train_step_flow and eval_step_flow for the training and evaluation steps.
    Keep track of the best validation loss and restore the best model at the end.
    """
    if epochs <= 0:
        raise ValueError("epochs must be positive.")

    train_losses: List[float] = []
    val_losses: List[float] = []

    best_val = float("inf")
    best_flow_state: Optional[Dict[str, Tensor]] = None
    best_ctx_state: Optional[Dict[str, Tensor]] = None

    for epoch in range(1, epochs + 1):
        flow_model.train()
        ctx_model.train()
        epoch_train_losses: List[float] = []

        for x_cat, x_cont, g_ids in train_loader:
            loss = train_step_flow(flow_model, ctx_model, optimizer, x_cont, x_cat, g_ids, device)
            epoch_train_losses.append(loss)

        flow_model.eval()
        ctx_model.eval()
        epoch_val_losses: List[float] = []

        for x_cat, x_cont, g_ids in val_loader:
            loss = eval_step_flow(flow_model, ctx_model, x_cont, x_cat, g_ids, device)
            epoch_val_losses.append(loss)

        train_loss = float(np.mean(epoch_train_losses))
        val_loss = float(np.mean(epoch_val_losses))

        train_losses.append(train_loss)
        val_losses.append(val_loss)

        print(f"[Flow] Epoch {epoch:03d}: train_NLL={train_loss:.4f}, val_NLL={val_loss:.4f}")

        if val_loss < best_val:
            best_val = val_loss
            # best_flow_state = {k: v.detach().cpu().clone() for k, v in flow_model.state_dict().items()}
            # best_ctx_state = {k: v.detach().cpu().clone() for k, v in ctx_model.state_dict().items()}
            best_flow_state = copy.deepcopy(flow_model.state_dict())
            best_ctx_state = copy.deepcopy(ctx_model.state_dict())

    if best_flow_state is not None:
        flow_model.load_state_dict(best_flow_state)
    if best_ctx_state is not None:
        ctx_model.load_state_dict(best_ctx_state)

    return flow_model, ctx_model, train_losses, val_losses

def print_flow_debug_summary(flow_model: ConditionalRealNVPFlow) -> None:
    # print("=== Flow debug summary ===")
    for i, coupling in enumerate(flow_model.couplings):
        if hasattr(coupling, "get_debug_summary"):
            summary = coupling.get_debug_summary()
            print(f"Coupling layer {i}: {summary}")

def reset_flow_debug_stats(flow_model: ConditionalRealNVPFlow) -> None:
    for coupling in flow_model.couplings:
        if hasattr(coupling, "reset_debug_stats"):
            coupling.reset_debug_stats()