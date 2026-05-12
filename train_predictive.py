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

    # Save the results and artifacts of the pipeline run
    experiment_folder = save_pipeline_result(
    result,
    base_folder="saved_models",
    )

    print(f"Saved experiment to: {experiment_folder}")