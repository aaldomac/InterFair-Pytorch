import numpy as np
import torch
import os
import sys
import argparse
import random

sys.path.append('/home/aaldoma/Projects/InterFairPytorch')

from modules.utils.loading_utils import (
    load_config_from_experiment,
    load_rng_states,
    reconstruct_test_data,
    load_all_models,
    load_optimizers_if_available,
    test_generative_models,
    test_predictive_ensemble,
)
from modules.models.generative_models import (
    reset_flow_debug_stats, 
    print_flow_debug_summary
)
from modules.utils.testing_utils import (
    inspect_test_examples, 
    compare_two_test_examples, 
    rank_test_examples_by_density
)

from modules.metrics.fairness_metrics import (
    # equalized_odds_difference,
    # equal_opportunity_difference,
    differential_fairness,
    uncertainty_difference,
)

from modules.utils.tensor_utils import (
    _as_tensor,
    _to_long_tensor,
    )

from modules.utils.plotting_utils import (
    plot_fairness_matrix,
    mix_fairness_matrices,
    scatter_fairness_matrices,
)

def set_global_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def main(args) -> None:
    exp_folder = args.experiment_folder
    cfg = load_config_from_experiment(exp_folder)

    device = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")
    print(f"Using device: {device}")

    rng_path = os.path.join(exp_folder, "rng", "rng_states.pt")
    if os.path.exists(rng_path):
        load_rng_states(os.path.join(exp_folder, "rng"))
        print(f"Loaded RNG states from {rng_path}")
    else:
        seed = int(cfg["experiment"]["seed"])
        set_global_seed(seed)
        print(f"Loaded seed from config: {seed}")

    data_dict = reconstruct_test_data(exp_folder, cfg, device=device)

    ar_model, ctx_model, flow_model, predictive_models, model_metadata = load_all_models(
        exp_folder=exp_folder, 
        cfg=cfg, 
        device=device, 
        debug_flow=True, 
        keep_last_batch_flow=True
    )

    batch_size = args.batch_size_eval

    optimizers = load_optimizers_if_available(exp_folder, ar_model, ctx_model, flow_model, predictive_models, cfg)

    # gen_results, summary = test_generative_models(
    #     ar_model=ar_model,
    #     ctx_model=ctx_model,
    #     flow_model=flow_model,
    #     data_dict=data_dict,
    #     device=device,
    #     batch_size=batch_size,
    # )

    pred_results = test_predictive_ensemble(
        predictive_models=predictive_models,
        data_dict=data_dict,
        device=device,
        batch_size=batch_size,
    )

    # print("\n=== Generative Model Results ===")
    # print(gen_results.head())
    # print(f"AR test NLL: {gen_results['ar_test_nll'].iloc[0]:.4f}")
    # print(f"Flow test NLL: {gen_results['flow_test_nll'].iloc[0]:.4f}")
    # print("Grouped summary:")
    # print(summary)

    print("\n=== Predictive Ensemble Results ===")
    print(f"Predictions shape: {np.asarray(pred_results['predictions']).shape}")
    print(f"Entropy shape: {np.asarray(pred_results['entropy']).shape}")
    print(
        f"Predicted labels [:10] - True labels [:10]: "
        f"{pred_results['pred_labels'][:10]} - {pred_results['y_test'][:10]}"
    )
    print(
        f"Aleatoric and epistemic uncertainty [:10]:\n"
        f"Aleatoric: {pred_results['aleatoric'][:10]}\n"
        f"Epistemic: {pred_results['epistemic'][:10]}"
    )
    print(f"Accuracy of ensemble: {pred_results['accuracy']:.4f}")

    print(f"Mean probs for first test example: {pred_results['mean_probs_2c'][0]}")
    
    g_test_np = data_dict["g_test"].detach().cpu().numpy()
    available_groups = np.unique(g_test_np)
    print(f"Groups in test set: {available_groups}")

    # Let's try comparing groups 4 and 14
    group_A = 4
    group_B = 14
    group_A_mask = data_dict["g_test"] == group_A
    group_B_mask = data_dict["g_test"] == group_B
    print(f"\nComparing group {group_A} and group {group_B}: with number of samples {group_A_mask.sum().item()} and {group_B_mask.sum().item()} respectively.")

    # Take probabilities for group 4 and group 14
    group_A_probs = pred_results["mean_probs_2c"][group_A_mask.cpu().numpy()]
    group_B_probs = pred_results["mean_probs_2c"][group_B_mask.cpu().numpy()]

    # Average predictive vector for each group
    group_A_mean_prob = group_A_probs.mean(axis=0)
    group_B_mean_prob = group_B_probs.mean(axis=0)

    print(f"Average predictive vector for group {group_A}: {group_A_mean_prob}")
    print(f"Average predictive vector for group {group_B}: {group_B_mean_prob}")

    # Compute the epsilons
    epsilon_vector = np.log(group_A_mean_prob / group_B_mean_prob)
    print(f"Epsilon vector (log ratio of mean probabilities): {epsilon_vector}")
    # Compute the average epsilon across the two classes averaged over the predictivevectors
    average_epsilon_A = np.abs(np.sum(epsilon_vector * group_A_mean_prob))
    average_epsilon_B = np.abs(np.sum(epsilon_vector * group_B_mean_prob))
    print(f"Average epsilon for group {group_A}: {average_epsilon_A:.4f}")
    print(f"Average epsilon for group {group_B}: {average_epsilon_B:.4f}")

    # Compute the KL divergence between the two average predictive vectors
    kl_AB = np.sum(group_A_mean_prob * np.log(group_A_mean_prob / group_B_mean_prob))
    kl_BA = np.sum(group_B_mean_prob * np.log(group_B_mean_prob / group_A_mean_prob))

    print(f"KL divergence from A to B: {kl_AB:.4f}")
    print(f"KL divergence from B to A: {kl_BA:.4f}")

    # Compute the cross entropy between the two average predictive vectors
    cross_entropy_AB = -np.sum(group_A_mean_prob * np.log(group_B_mean_prob))
    cross_entropy_BA = -np.sum(group_B_mean_prob * np.log(group_A_mean_prob))

    # Compute the entropy of each average predictive vector
    entropy_A = -np.sum(group_A_mean_prob * np.log(group_A_mean_prob))
    entropy_B = -np.sum(group_B_mean_prob * np.log(group_B_mean_prob))

    # Compute the average entropy of the predictive vectors in each group
    average_entropy_A = -np.mean(np.sum(group_A_probs * np.log(group_A_probs + 1e-12), axis=1))
    average_entropy_B = -np.mean(np.sum(group_B_probs * np.log(group_B_probs + 1e-12), axis=1))
    # Compute the average aleatoric and epistemic uncertainty of the predictive vectors in each group
    average_aleatoric_A = pred_results["aleatoric"][group_A_mask.cpu().numpy()].mean()
    average_aleatoric_B = pred_results["aleatoric"][group_B_mask.cpu().numpy()].mean()
    average_epistemic_A = pred_results["epistemic"][group_A_mask.cpu().numpy()].mean()
    average_epistemic_B = pred_results["epistemic"][group_B_mask.cpu().numpy()].mean()

    # Compute the KL divergence between the predictive vector of each example and the average predictive vector of the its group, averaged over examples
    kl_A = np.mean(np.sum(group_A_probs * np.log(group_A_probs / group_A_mean_prob), axis=1))
    kl_B = np.mean(np.sum(group_B_probs * np.log(group_B_probs / group_B_mean_prob), axis=1))

    print(f"Difference of average entropies (A - B): {average_entropy_A:.4f} - {average_entropy_B:.4f} = {(average_entropy_A - average_entropy_B):.4f}")
    print(f"Difference of average epistemic uncertainty (A - B): {average_epistemic_A:.4f} - {average_epistemic_B:.4f} = {(average_epistemic_A - average_epistemic_B):.4f}")
    print(f"Difference of average aleatoric uncertainty (A - B): {average_aleatoric_A:.4f} - {average_aleatoric_B:.4f} = {(average_aleatoric_A - average_aleatoric_B):.4f}")
    print(f"Difference of KL divergences to group mean (A - B): {kl_A:.4f} - {kl_B:.4f} = {(kl_A - kl_B):.4f}")
    print(f"Sum of the previous three quantities: {(average_entropy_A - average_entropy_B) + (kl_A - kl_B):.4f}")
    print(f"Difference of entropies (A - B): {entropy_A:.4f} - {entropy_B:.4f} = {(entropy_A - entropy_B):.4f}")
    print(f"Difference of cross-entropies (AB - BA): {cross_entropy_AB:.4f} - {cross_entropy_BA:.4f} = {(cross_entropy_AB - cross_entropy_BA):.4f}")
    print(f"Difference of average epsilons (B - A): {average_epsilon_B:.4f} - {average_epsilon_A:.4f} = {(average_epsilon_B - average_epsilon_A):.4f}")
    print(f"Sum of Delta cross-entropy and Delta entropy: {((cross_entropy_AB - cross_entropy_BA) + (average_epsilon_B - average_epsilon_A)):.4f}")

    # Compute DF for all groups
    aggregate_DF, DF_epsilon_matrix, DF_groups, DF_group_rates = differential_fairness(torch.as_tensor(pred_results["predictions"], dtype=torch.float32), _to_long_tensor(g_test_np))
    print(f"Aggregate Differential Fairness (DF): {aggregate_DF:.4f}")
    print(f"DF groups: {DF_groups}, DF group rates: {DF_group_rates}")

    # Compute uncertainty difference for all groups
    # aggregate_UF, UF_epsilon_matrix, UF_groups, UF_group_uncertainties = uncertainty_difference(pred_results["epistemic"], _to_long_tensor(g_test_np))
    aggregate_UF_tot, UF_epsilon_matrix_tot, UF_groups, UF_group_uncertainties = uncertainty_difference(pred_results["entropy"], _to_long_tensor(g_test_np))
    aggregate_UF_ale, UF_epsilon_matrix_ale, UF_groups, UF_group_uncertainties = uncertainty_difference(pred_results["aleatoric"], _to_long_tensor(g_test_np))
    aggregate_UF_epi, UF_epsilon_matrix_epi, UF_groups, UF_group_uncertainties = uncertainty_difference(pred_results["epistemic"], _to_long_tensor(g_test_np))
    print(f"Aggregate Uncertainty Fairness (UF): {aggregate_UF_tot:.4f} - {aggregate_UF_ale:.4f} (aleatoric) - {aggregate_UF_epi:.4f} (epistemic)")
    print(f"UF groups: {UF_groups}, UF group uncertainties: {UF_group_uncertainties}")

    # Mix the DF and UF epsilon matrices for visualization
    mixed_matrix = mix_fairness_matrices(DF_epsilon_matrix, UF_epsilon_matrix_tot)
    fig = plot_fairness_matrix(mixed_matrix, DF_groups, "Differential Fairness (upper) and Uncertainty Fairness (lower)")
    figures_folder = os.path.join(exp_folder, "figures")
    os.makedirs(figures_folder, exist_ok=True)
    fig.savefig(os.path.join(figures_folder, "fairness_matrices_DF_TU.png"))

    # Check if there is order correlation
    scatter_fig, stats = scatter_fairness_matrices(DF_epsilon_matrix, UF_epsilon_matrix_tot, correlation="spearman", plot_mode="ranks", return_data=True)
    scatter_fig.savefig(os.path.join(figures_folder, "fairness_scatter_DF_TU.png"))
    print(f"DF and UF ranking correlation: {stats['correlation_value']}")

    # if args.target_group is not None:
    #     target_group = args.target_group
    #     if target_group not in available_groups:
    #         raise ValueError(
    #             f"target_group={target_group} not in test set. Available groups: {available_groups}"
    #         )

    #     group_mask = data_dict["g_test"] == target_group
    #     num_group_samples = int(group_mask.sum().item())
    #     print(f"Samples in target_group={target_group}: {num_group_samples}")

    #     if num_group_samples > 0:
    #         x_cont_g = data_dict["X_cont_test"][group_mask].to(device=device, dtype=torch.float32)
    #         x_cat_g = data_dict["X_cat_test"][group_mask].to(device=device, dtype=torch.long)
    #         g_ids_g = data_dict["g_test"][group_mask].to(device=device, dtype=torch.long)

    #         reset_flow_debug_stats(flow_model)

    #         with torch.no_grad():
    #             context_g = ctx_model(x_cat_g, g_ids_g)
    #             _ = flow_model.log_prob(x_cont_g, context_g)

    #         print(f"\n=== Flow debug summary for target_group={target_group} ===")
    #         print_flow_debug_summary(flow_model)

    # ranked = rank_test_examples_by_density(ar_model, ctx_model, flow_model, data_dict, device=device, top_k=None)
    # print("Lowest-density test indices:", ranked["order"][:10])
    # print("Highest-density test indices:", ranked["order"][-10:])

    # # Compare 5 highest with 5 lowest indices #
    # rank_comparison = inspect_test_examples(
    #     ar_model=ar_model,
    #     ctx_model=ctx_model,
    #     flow_model=flow_model,
    #     data_dict=data_dict,
    #     device=device,
    #     indices=list(ranked["order"][:5]) + list(ranked["order"][-5:]),
    # )
    # for r in rank_comparison:
    #     print(f"\nTest index {r['test_index']} - group_id: {r['group_id']}")
    #     print(f"Logp_cat: {r['logp_cat']:.4f}, Logp_cont: {r['logp_cont']:.4f}, Logp_total: {r['logp_total']:.4f}")
    #     # print("Row data:", r["row"])

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--experiment-folder", type=str, required=True)
    parser.add_argument("--batch-size-eval", type=int, default=1024)
    parser.add_argument("--target-group", type=int, default=None)
    args = parser.parse_args()

    main(args)
