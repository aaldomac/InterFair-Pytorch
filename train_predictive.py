import torch

from modules.pipelines.train_predictive_pipeline import (
    PipelineConfig,
    SplitConfig,
    LoaderConfig,
    ModelConfig,
    OptimizerConfig,
    run_predictive_pipeline,
)
from modules.predictive.trainer import TrainConfig

from modules.utils.saving_utils import save_pipeline_result

from modules.metrics.fairness_metrics import (
    evaluate_ensemble_fairness_from_loader,
    fairness_summary_to_frame,
)

from modules.metrics.distribution_decomposition_metrics import (
    group_distribution_analysis_from_ensemble_outputs,
    group_entropy_decompositions_to_frame,
)

if __name__ == "__main__":
    config = PipelineConfig(
            dataset_name="adult",
            dataset_kwargs={"drop_na": True},
            split=SplitConfig(test_size=0.25, val_size=0.15, seed=42),
            loader=LoaderConfig(batch_size=256, eval_batch_size=512),
            model=ModelConfig(hidden_dims=(256, 128), dropout=0.2),
            optimizer=OptimizerConfig(lr=1e-3, weight_decay=1e-4),
            train=TrainConfig(
                epochs=10,
                patience=2,
                binary=True,
                threshold=0.5,
                verbose=True,
            ),
            n_models=3,
            append_protected_to_predictor=True,
            build_px_loaders=False,
            device=None,
        )

    result = run_predictive_pipeline(config)

    print("\nTest metrics")
    print(result.test_metrics)

    print("\nWorst groups by test accuracy")
    print(result.group_metrics.head(10))

    if result.ensemble_metrics is not None:
        print("\nEnsemble output arrays")
        for key, value in result.ensemble_metrics.items():
            print(f"{key}: shape={value.shape}")

    fairness = evaluate_ensemble_fairness_from_loader(
        result.ensemble_metrics,
        result.data.test_loader,
        binary=result.data.binary,
        threshold=result.config.train.threshold,
        positive_class=1,
        alpha=1.0,
    )

    summary = fairness_summary_to_frame(fairness)
    print(summary)

    # Useful scalar examples:
    print(f"Statistical parity aggregate: {fairness['statistical_parity']['aggregate']}")
    print(f"Aleatoric uncertainty pairwise aggregate: {fairness['uncertainty']['aleatoric_uncertainty']['pairwise_aggregate']}")
    print(f"Epistemic uncertainty pairwise aggregate: {fairness['uncertainty']['epistemic_uncertainty']['pairwise_aggregate']}")

    # Save together with experiment outputs:
    result.test_metrics["statistical_parity"] = fairness["statistical_parity"]["aggregate"]

    # Evaluate entropy-like measures and comparisons
    group_ids = []
    for x, y, g in result.data.test_loader:
        group_ids.append(g)
    group_ids = torch.cat(group_ids)

    analysis = group_distribution_analysis_from_ensemble_outputs(
        result.ensemble_metrics,
        group_ids=group_ids,
    )

    decomp_df = group_entropy_decompositions_to_frame(
        analysis["group_entropy_decompositions"]
    )

    print(decomp_df)

    matrices = analysis["pairwise_group_mean_distribution_matrices"]

    print(matrices["kl"])               # KL(mean_i || mean_j)
    print(matrices["symmetric_kl"])     # KL(i||j) + KL(j||i)
    print(matrices["class_log_ratio"])  # [num_groups, num_groups, num_classes]

    # # Save the results and artifacts of the pipeline run
    # experiment_folder = save_pipeline_result(
    #     result,
    #     base_folder="saved_models",
    # )

    # print(f"Saved experiment to: {experiment_folder}")