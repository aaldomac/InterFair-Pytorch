import numpy as np
import torch

from modules.models.generative_models import (
    ARModel,
    ContextEncoder,
    ConditionalRealNVPFlow,
)

from typing import Any, Dict, Mapping, Optional, Sequence, Union
DeviceLike = Union[torch.device, str]

@torch.no_grad()
def inspect_test_examples(
    ar_model: ARModel,
    ctx_model: ContextEncoder,
    flow_model: ConditionalRealNVPFlow,
    data_dict: dict,
    indices: Sequence[int],
    device: DeviceLike,
) -> Sequence[Dict[str, Any]]:
    ar_model.eval()
    ctx_model.eval()
    flow_model.eval()

    idx = torch.as_tensor(indices, dtype=torch.long)

    x_cat = data_dict["X_cat_test"][idx].to(device=device, dtype=torch.long)
    x_cont = data_dict["X_cont_test"][idx].to(device=device, dtype=torch.float32)
    g_ids = data_dict["g_test"][idx].to(device=device, dtype=torch.long)

    logp_cat = ar_model.log_prob(x_cat, g_ids)
    context = ctx_model(x_cat, g_ids)
    logp_cont = flow_model.log_prob(x_cont, context)
    logp_total = logp_cat + logp_cont
    
    results = []

    for k, original_idx in enumerate(indices):
        results.append({
            "test_index": int(original_idx),
            "group_id": int(g_ids[k].item()),
            "logp_cat": float(logp_cat[k].item()),
            "logp_cont": float(logp_cont[k].item()),
            "logp_total": float(logp_total[k].item()),
        })
    return results

@torch.no_grad()
def compare_two_test_examples(
    i: int,
    j: int,
    ar_model: ARModel,
    ctx_model: ContextEncoder,
    flow_model: ConditionalRealNVPFlow,
    data_dict: dict,
    device: DeviceLike,
):
    ar_model.eval()
    ctx_model.eval()
    flow_model.eval()

    idx = torch.tensor([i, j], dtype=torch.long)

    x_cat = data_dict["X_cat_test"][idx].to(device=device, dtype=torch.long)
    x_cont = data_dict["X_cont_test"][idx].to(device=device, dtype=torch.float32)
    g_ids = data_dict["g_test"][idx].to(device=device, dtype=torch.long)

    logp_cat = ar_model.log_prob(x_cat, g_ids)
    context = ctx_model(x_cat, g_ids)
    logp_cont = flow_model.log_prob(x_cont, context)
    logp_total = logp_cat + logp_cont

    rows = data_dict["df_test"].iloc[[i, j]]

    out = []

    for local_k, global_idx in enumerate([i, j]):
        out.append({
            "test_index": global_idx,
            "group_id": int(g_ids[local_k].item()),
            "logp_cat": float(logp_cat[local_k].item()),
            "logp_cont": float(logp_cont[local_k].item()),
            "logp_total": float(logp_total[local_k].item()),
            "row": rows.iloc[local_k].to_dict(),
        })

    print(f"Example A:")
    print(out[0]["row"])
    print(
        f"logp_cat: {out[0]['logp_cat']:.4f}, "
        f"logp_cont: {out[0]['logp_cont']:.4f}, "
        f"logp_total: {out[0]['logp_total']:.4f}"
    )

    print(f"\nExample B:")
    print(out[1]["row"])
    print(
        f"logp_cat: {out[1]['logp_cat']:.4f}, "
        f"logp_cont: {out[1]['logp_cont']:.4f}, "
        f"logp_total: {out[1]['logp_total']:.4f}"
    )   
    if out[0]["logp_total"] < out[1]["logp_total"]:
        print(f"\nExample A (test index {i}) is less likely under the model than Example B.")
    elif out[0]["logp_total"] > out[1]["logp_total"]:
        print(f"\nExample A (test index {i}) is more likely under the model than Example B.")
    else:
        print(f"\nExample A (test index {i}) and Example B have the same likelihood under the model.")
    
    return out


@torch.no_grad()
def rank_test_examples_by_density(
    ar_model: ARModel,
    ctx_model: ContextEncoder,
    flow_model: ConditionalRealNVPFlow,
    data_dict: dict,
    device: DeviceLike,
    top_k: Optional[int] = None,
) -> Sequence[Dict[str, Any]]:
    ar_model.eval()
    ctx_model.eval()
    flow_model.eval()

    x_cat = data_dict["X_cat_test"].to(device=device, dtype=torch.long)
    x_cont = data_dict["X_cont_test"].to(device=device, dtype=torch.float32)
    g_ids = data_dict["g_test"].to(device=device, dtype=torch.long)

    logp_cat = ar_model.log_prob(x_cat, g_ids)
    context = ctx_model(x_cat, g_ids)
    logp_cont = flow_model.log_prob(x_cont, context)
    logp_total = logp_cat + logp_cont

    order = torch.argsort(logp_total)  # lowest first

    if top_k is None:
        top_k = -1
        
    ranked = {
        "order": order.cpu().numpy()[:top_k],
        "logp_cat": logp_cat.cpu().numpy()[:top_k],
        "logp_cont": logp_cont.cpu().numpy()[:top_k],
        "logp_total": logp_total.cpu().numpy()[:top_k],
    }

    return ranked