import numpy as np
import kagglehub
import torch
import os
import yaml
import sys
import argparse
from torch.utils.data import TensorDataset, DataLoader

sys.path.append('/home/aaldoma/Projects/InterFairPytorch')

from modules.utils.dataset_utils import (
    load_dataset, 
    split_df,
    split_df_with_indices,
    fit_schema_px, 
    transform_px,
    fit_predictor_schema,
    transform_predictor,
    make_loader,
    to_tensor_dataset_ar,
    to_tensor_dataset_flow,
    to_tensor_dataset_predictive,
)
from modules.models.predictive_models import (
    Classifier, 
    train_single_predictive_model,
    evaluate_ensemble,
    train_predictive_ensemble,
)
from modules.models.generative_models import (
    ARModel,
    ContextEncoder, 
    ConditionalRealNVPFlow, 
    train_ar_model,
    train_flow_model,
)
from modules.utils.saving_utils import (
    save_full_experiment,
    save_full_reproducibility_bundle
    # file_sha256
)

# REPRODUCIBILITY #
def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

# CONFIG #
def load_yaml_config(yaml_path: str):
    with open(yaml_path, "r") as f:
        cfg = yaml.safe_load(f)
    return cfg

def parse_arguments():
    parser = argparse.ArgumentParser(description='Train AR+Flow and predictive ensemble on Adult Income Dataset')

    parser.add_argument('--yaml_path', type=str, default=None, required=False, help='Path to the experiment YAML configuration.')
    parser.add_argument('--dataset_name', type=str, default='adult_income', required=False, help='Dataset to use: adult_income or compas')
    parser.add_argument('--experiment_name', type=str, required=False, help='Name of the experiment (used for saving)')

    parser.add_argument('--seed', type=int, default=42, required=False, help="Seed for reproducibility")

    parser.add_argument('--predictive_model_epochs', type=int, default=50, required=False, help='Number of epochs to train the model')
    parser.add_argument('--ar_epochs', type=int, default=50, required=False, help='Number of epochs to train the AR model')
    parser.add_argument('--flow_epochs', type=int, default=50, required=False, help='Number of epochs to train the flow model')

    parser.add_argument('--batch_size', type=int, default=128, required=False, help='Batch size for training')
    parser.add_argument('--num_models', type=int, default=5, required=False, help='Number of models in the ensemble')

    parser.add_argument('--test_split', type=float, default=0.25, required=False, help="Fraction of training data to use for testing")
    parser.add_argument('--validation_split', type=float, default=0.15, required=False, help="Fraction of training data to use for validation")
    # parser.add_argument('--downsample_frac', type=float, default=1.0, required=False, help='Whether to downsample majority groups')
    # parser.add_argument('--inject_noise_level', type=float, default=0.0, required=False, help='Whether to inject label noise to some groups')
    return parser.parse_args()

def apply_cli_overrides(cfg: dict, args):
    if args.seed is not None:
        cfg["experiment"]["seed"] = args.seed

    if args.experiment_name is not None:
        cfg["experiment"]["name"] = args.experiment_name

    # if args.save_dir is not None:
    #     cfg["experiment"]["save_dir"] = args.save_dir

    if args.ar_epochs is not None:
        cfg["ar_model"]["epochs"] = args.ar_epochs

    if args.flow_epochs is not None:
        cfg["flow_model"]["epochs"] = args.flow_epochs

    # if args.predictive_lr is not None:
    #     cfg["predictive_model"]["epochs"] = args.predictive_lr

    if args.predictive_model_epochs is not None:
        cfg["predictive_model"]["epochs"] = args.predictive_model_epochs

    return cfg


def build_predictive_models(cfg, input_dim, device):
    models = []
    optimizers = []

    ensemble_size = cfg["predictive_model"]["ensemble_size"]
    hidden = tuple(cfg["predictive_model"]["hidden"])
    dropout = cfg["predictive_model"]["dropout"]
    lr = cfg["predictive_model"]["lr"]

    for k in range(ensemble_size):
        model = Classifier(
            input_dim=input_dim,
            hidden=hidden,
            dropout=dropout,
        ).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        models.append(model)
        optimizers.append(optimizer)

    return models, optimizers

def main(cfg: dict):
    # ----------------------------
    # Load data
    # ----------------------------
    # csv_path = cfg["data"]["csv_path"]
    df, original_columns, pg_table, pg = load_dataset(cfg["data"]["dataset_name"])

    print(f"Loaded dataset with {len(df)} rows and {len(df.columns)} columns.")
    print(f"Dirichlet group proportions:\n{pg_table}")

    df_train, df_val, df_test, train_idx, val_idx, test_idx = split_df_with_indices(
        df,
        test_size=cfg["split"]["test_size"],
        val_size=cfg["split"]["val_size"],
        seed=cfg["experiment"]["seed"],
    )

    # ----------------------------
    # Generative preprocessing
    # ----------------------------
    schema_px = fit_schema_px(
        df_train=df_train,
        protected_cols=tuple(cfg["data"]["protected_attributes"]),
        label_col=cfg["data"]["label_col"],
        group_id_col=cfg["data"]["group_id_col"],
        log1p_cols=tuple(cfg["preprocessing"]["log1p_cols"]),
        drop_feature_cols=tuple(cfg["preprocessing"]["drop_feature_cols"]),
    )

    X_cat_train, X_cont_train, g_train, vocab_sizes = transform_px(
        df_train, schema_px, group_id_col=cfg["data"]["group_id_col"]
    )
    X_cat_val, X_cont_val, g_val, _ = transform_px(
        df_val, schema_px, group_id_col=cfg["data"]["group_id_col"]
    )
    X_cat_test, X_cont_test, g_test, _ = transform_px(
        df_test, schema_px, group_id_col=cfg["data"]["group_id_col"]
    )

    print("X_cat_train size:", len(X_cat_train))
    print("X_cont_train shape:", X_cont_train.shape)

    num_groups = int(df[cfg["data"]["group_id_col"]].nunique())
    cont_dim = int(X_cont_train.shape[1])

    # ----------------------------
    # Build generative models
    # ----------------------------
    ar_model = ARModel(
        vocab_sizes=vocab_sizes,
        num_groups=num_groups,
        g_emb_dim=cfg["ar_model"]["g_emb_dim"],
        x_emb_dim=cfg["ar_model"]["x_emb_dim"],
        hidden=cfg["ar_model"]["hidden"],
        dropout=cfg["ar_model"]["dropout"],
    ).to(device)

    ctx_model = ContextEncoder(
        vocab_sizes=vocab_sizes,
        num_groups=num_groups,
        g_emb_dim=cfg["context_model"]["g_emb_dim"],
        x_emb_dim=cfg["context_model"]["x_emb_dim"],
        hidden=cfg["context_model"]["hidden"],
        out_dim=cfg["context_model"]["out_dim"],
    ).to(device)

    flow_model = ConditionalRealNVPFlow(
        dim=cont_dim,
        context_dim=cfg["context_model"]["out_dim"],
        num_couplings=cfg["flow_model"]["num_couplings"],
        hidden=cfg["flow_model"]["hidden"],
        seed=cfg["flow_model"]["seed"],
    ).to(device)

    ar_optimizer = torch.optim.Adam(ar_model.parameters(), lr=cfg["ar_model"]["lr"])
    flow_optimizer = torch.optim.Adam(
        list(flow_model.parameters()) + list(ctx_model.parameters()),
        lr=cfg["flow_model"]["lr"]
    )
    # ----------------------------
    # DataLoaders for generative training
    # ----------------------------
    ar_train_loader = make_loader(
        to_tensor_dataset_ar(X_cat_train, g_train),
        batch_size=cfg["ar_model"]["batch_size"],
        shuffle=True,
    )
    ar_val_loader = make_loader(
        to_tensor_dataset_ar(X_cat_val, g_val),
        batch_size=cfg["experiment"]["batch_size"],
        shuffle=False,
    )

    flow_train_loader = make_loader(
        to_tensor_dataset_flow(X_cat_train, X_cont_train, g_train),
        batch_size=cfg["flow_model"]["batch_size"],
        shuffle=True,
    )
    flow_val_loader = make_loader(
        to_tensor_dataset_flow(X_cat_val, X_cont_val, g_val),
        batch_size=cfg["experiment"]["batch_size"],
        shuffle=False,
    )

    # ----------------------------
    # Train generative models
    # ----------------------------
    ar_model, ar_train_losses, ar_val_losses = train_ar_model(
        model=ar_model,
        optimizer=ar_optimizer,
        train_loader=ar_train_loader,
        val_loader=ar_val_loader,
        epochs=cfg["ar_model"]["epochs"],
        device=device,
    )

    flow_model, ctx_model, flow_train_losses, flow_val_losses = train_flow_model(
        flow_model=flow_model,
        ctx_model=ctx_model,
        optimizer=flow_optimizer,
        train_loader=flow_train_loader,
        val_loader=flow_val_loader,
        epochs=cfg["flow_model"]["epochs"],
        device=device,
    )

    # ----------------------------
    # Evaluate generative models on test
    # ----------------------------
    ar_model.eval()
    flow_model.eval()
    ctx_model.eval()

    with torch.no_grad():
        x_cat_test_t = torch.as_tensor(X_cat_test, dtype=torch.long, device=device)
        x_cont_test_t = torch.as_tensor(X_cont_test, dtype=torch.float32, device=device)
        g_test_t = torch.as_tensor(g_test, dtype=torch.long, device=device)

        logp_cat_test = ar_model.log_prob(x_cat_test_t, g_test_t).detach().cpu().numpy()
        context_test = ctx_model(x_cat_test_t, g_test_t)
        logp_cont_test = flow_model.log_prob(x_cont_test_t, context_test).detach().cpu().numpy()
        logp_x_given_g_test = logp_cat_test + logp_cont_test

    # ----------------------------
    # Predictive preprocessing
    # ----------------------------
    pred_schema = fit_predictor_schema(
        df_train=df_train,
        protected_cols=tuple(cfg["data"]["protected_attributes"]),
        label_col=cfg["data"]["label_col"],
        log1p_cols=tuple(cfg["preprocessing"]["log1p_cols"]),
    )

    X_pred_train, y_train = transform_predictor(
        df_train,
        pred_schema,
        protected_cols=tuple(cfg["data"]["protected_attributes"]),
        label_col=cfg["data"]["label_col"],
    )
    X_pred_val, y_val = transform_predictor(
        df_val,
        pred_schema,
        protected_cols=tuple(cfg["data"]["protected_attributes"]),
        label_col=cfg["data"]["label_col"],
    )
    X_pred_test, y_test = transform_predictor(
        df_test,
        pred_schema,
        protected_cols=tuple(cfg["data"]["protected_attributes"]),
        label_col=cfg["data"]["label_col"],
    )

    pred_train_loader = make_loader(
        to_tensor_dataset_predictive(X_pred_train, y_train),
        batch_size=cfg["experiment"]["batch_size"],
        shuffle=True,
    )
    pred_val_loader = make_loader(
        to_tensor_dataset_predictive(X_pred_val, y_val),
        batch_size=cfg["experiment"]["batch_size"],
        shuffle=False,
    )
    pred_test_loader = make_loader(
        to_tensor_dataset_predictive(X_pred_test, y_test),
        batch_size=cfg["experiment"]["batch_size"],
        shuffle=False,
    )

    # ----------------------------
    # Build and train predictive ensemble
    # ----------------------------
    predictive_models, predictive_optimizers = build_predictive_models(
        cfg=cfg,
        input_dim=X_pred_train.shape[1],
        device=device,
    )

    predictive_models, predictive_histories = train_predictive_ensemble(
        models=predictive_models,
        optimizers=predictive_optimizers,
        train_loader=pred_train_loader,
        val_loader=pred_val_loader,
        cfg=cfg,
        device=device,
    )

    # ----------------------------
    # Evaluate predictive ensemble
    # ----------------------------
    ensemble_predictions, entropy, aleatoric, epistemic = evaluate_ensemble(
        models=predictive_models,
        loader=pred_test_loader,
        device=device,
    )

    # ----------------------------
    # Metadata / histories / results
    # ----------------------------
    model_metadata = {
        "ar_model": {
            "class": "ARModel",
            "vocab_sizes": vocab_sizes,
            "num_groups": num_groups,
            "g_emb_dim": cfg["ar_model"]["g_emb_dim"],
            "x_emb_dim": cfg["ar_model"]["x_emb_dim"],
            "hidden": cfg["ar_model"]["hidden"],
            "dropout": cfg["ar_model"]["dropout"],
        },
        "ctx_model": {
            "class": "ContextEncoder",
            "vocab_sizes": vocab_sizes,
            "num_groups": num_groups,
            "g_emb_dim": cfg["context_model"]["g_emb_dim"],
            "x_emb_dim": cfg["context_model"]["x_emb_dim"],
            "hidden": cfg["context_model"]["hidden"],
            "out_dim": cfg["context_model"]["out_dim"],
        },
        "flow_model": {
            "class": "ConditionalRealNVPFlow",
            "dim": cont_dim,
            "context_dim": cfg["context_model"]["out_dim"],
            "num_couplings": cfg["flow_model"]["num_couplings"],
            "hidden": cfg["flow_model"]["hidden"],
            "seed": cfg["flow_model"]["seed"],
        },
        "predictive_model": {
            "class": "PredictiveMLP",
            "input_dim": int(X_pred_train.shape[1]),
            "hidden": cfg["predictive_model"]["hidden"],
            "dropout": cfg["predictive_model"]["dropout"],
            "ensemble_size": cfg["predictive_model"]["ensemble_size"],
        },
    }

    histories = {
        "ar_train_loss": ar_train_losses,
        "ar_val_loss": ar_val_losses,
        "flow_train_loss": flow_train_losses,
        "flow_val_loss": flow_val_losses,
    }

    for i, hist in enumerate(predictive_histories):
        for k, v in hist.items():
            histories[f"predictive_{i}_{k}"] = v

    data_info = {
        "dataset_name": cfg["data"]["dataset_name"],
        # "csv_path": csv_path,
        # "csv_sha256": file_sha256(csv_path),
        "num_rows": int(len(df)),
        "original_columns": original_columns,
        "processed_columns": list(df.columns),
    }

    results_dict = {
        "logp_cat_test": logp_cat_test,
        "logp_cont_test": logp_cont_test,
        "logp_x_given_g_test": logp_x_given_g_test,
        "ensemble_predictions": ensemble_predictions,
        "ensemble_entropy": entropy,
        # "ensemble_lower_bound": lower_bound,
        # "ensemble_upper_bound": upper_bound,
        "ensemble_aleatoric": aleatoric,
        "y_test": y_test,
        "g_test": g_test,
    }

    # ----------------------------
    # Save everything
    # ----------------------------
    exp_root = os.path.join(
        cfg["experiment"]["save_dir"],
        cfg["experiment"]["name"],
        f'seed_{cfg["experiment"]["seed"]}',
    )

    save_full_experiment(
        ar_model=ar_model,
        flow_model=flow_model,
        ctx_model=ctx_model,
        predictive_models=predictive_models,
        config=cfg,
        results_dict=results_dict if cfg["saving"]["save_results"] else None,
        exact_path=exp_root,
        ar_optimizer=ar_optimizer if cfg["saving"]["save_optimizers"] else None,
        flow_optimizer=flow_optimizer if cfg["saving"]["save_optimizers"] else None,
        predictive_optimizers=predictive_optimizers if cfg["saving"]["save_optimizers"] else None,
    )

    save_full_reproducibility_bundle(
        experiment_folder=exp_root,
        config=cfg,
        schema=schema_px,
        train_idx=train_idx,
        val_idx=val_idx,
        test_idx=test_idx,
        model_metadata=model_metadata,
        histories=histories if cfg["saving"]["save_histories"] else None,
        data_info=data_info,
    )

    print(f"Saved experiment to: {exp_root}")


if __name__ == "__main__":
    # Parse command-line arguments
    args = parse_arguments()
    parse_config = vars(args)

    yaml_config = load_yaml_config(args.yaml_path) if args.yaml_path is not None else {}
    cfg = apply_cli_overrides(yaml_config, args)

    save_dir = cfg["experiment"]["save_dir"]
    os.makedirs(save_dir, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # if config["yaml_path"] is not None:
    #     with open(config["yaml_path"], "r") as f:
    #         cfg = yaml.safe_load(f)
    #     config["intervention"] = cfg["intervention"]
    #     for key, label in cfg[config["dataset_name"]].items():
    #         config[key] = label
    # else:
    #     config["intervention"] = None

    main(cfg)
