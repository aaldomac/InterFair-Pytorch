# import torch

# from modules.predictive.models import MLPClassifier
# from modules.predictive.losses import BinaryClassificationLoss, CompositeLoss
# from modules.predictive.trainer import PredictiveTrainer, TrainConfig
# from modules.utils.dataset_utils import load_dataset_by_name, split_df, fit_predictor_schema, transform_predictor_schema
# from modules.data.adult import ADULT_SPEC

# loaded = load_dataset_by_name("adult", drop_na=True)
# df = loaded.df

# train_df, val_df, test_df = split_df(
#     df,
#     test_size=0.25,
#     val_size=0.15,
#     seed=42,
#     stratify=ADULT_SPEC.group_col
# )

# predictor_schema = fit_predictor_schema(train_df, ADULT_SPEC)
# X_train, y_train, g_train = transform_predictor_schema(train_df, predictor_schema)
# X_val, y_val, g_val = transform_predictor_schema(val_df, predictor_schema)

# model = MLPClassifier(
#     input_dim=X_train.shape[1],
#     hidden_dims=(256, 128),
#     output_dim=1,
#     dropout=0.1,
# )

# optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4)
# loss_fn = CompositeLoss(
#     task_loss=BinaryClassificationLoss(),
#     regularizer=None,  # No regularizer for this example
#     regularizer_weight=0.0,
# )

# trainer = PredictiveTrainer(
#     model=model,
#     optimizer=optimizer,
#     loss_fn=loss_fn,
#     config=TrainConfig(epochs=50, patience=5, binary=True, threshold=0.5),
#     device="cuda" if torch.cuda.is_available() else "cpu",
#     fair_loader=None,
# )

# model, history = trainer.fit(train_loader, val_loader)
from modules.pipelines.train_predictive_pipeline import (
    PipelineConfig,
    SplitConfig,
    LoaderConfig,
    ModelConfig,
    OptimizerConfig,
    run_predictive_pipeline,
)
from modules.predictive.trainer import TrainConfig

if __name__ == "__main__":
    config = PipelineConfig(
            dataset_name="adult",
            dataset_kwargs={"drop_na": True},
            split=SplitConfig(test_size=0.25, val_size=0.15, seed=42),
            loader=LoaderConfig(batch_size=256, eval_batch_size=512),
            model=ModelConfig(hidden_dims=(256, 128), dropout=0.2),
            optimizer=OptimizerConfig(lr=1e-3, weight_decay=1e-4),
            train=TrainConfig(
                epochs=50,
                patience=5,
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