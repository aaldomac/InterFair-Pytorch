"""Per-condition mean run-level ECDFs of binary ensemble probabilities."""
import argparse
import json
import warnings
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def read_json(path):
    return json.loads(path.read_text()) if path.exists() else {}


def ecdf_on_grid(values, grid):
    """Empty conditional samples are undefined, never zero-valued CDFs."""
    values = np.asarray(values, dtype=float)
    if not len(values):
        return np.full(len(grid), np.nan)
    return np.searchsorted(np.sort(values), grid, side="right") / len(values)


def summarize_ecdfs(curves, shading):
    """Equal weight per nonempty run; pointwise SD or Student-t CI."""
    curves = np.asarray(curves, dtype=float)
    valid = np.isfinite(curves).all(axis=1)
    curves = curves[valid]
    n = len(curves)
    if n == 0:
        empty = np.full(curves.shape[1], np.nan)
        return empty, empty.copy(), n
    mean = curves.mean(axis=0)
    if shading == "none":
        width = np.zeros_like(mean)
    elif n < 2:
        width = np.full_like(mean, np.nan)
    else:
        width = curves.std(axis=0, ddof=1)
        if shading == "ci95":
            from scipy.stats import t
            width = width * t.ppf(.975, n - 1) / np.sqrt(n)
    return mean, width, n


def load_run(run, threshold_override=None):
    from modules.utils.loading_utils import load_saved_outputs
    ensemble, y, groups = load_saved_outputs(run)
    p = np.asarray(ensemble["mean_probs"], dtype=float)
    y, groups = np.asarray(y).reshape(-1), np.asarray(groups).reshape(-1)
    if p.ndim != 2 or p.shape[1] != 2 or not len(p):
        raise ValueError("Require nonempty binary [N, 2] probabilities.")
    if len(y) != len(p) or len(groups) != len(p):
        raise ValueError("Misaligned predictions, labels and groups.")
    if (not np.isfinite(p).all() or (p < 0).any() or (p > 1).any()
            or not np.allclose(p.sum(1), 1) or not np.isin(y, [0, 1]).all()):
        raise ValueError("Invalid probabilities or labels.")
    context = read_json(run / "audit/audit_context.json")
    config = read_json(run / "config.json")
    threshold = threshold_override
    if threshold is None:
        threshold = context.get("threshold", config.get("train", {}).get("threshold", .5))
    threshold = float(threshold)
    if not np.isfinite(threshold) or not 0 <= threshold <= 1:
        raise ValueError("Invalid threshold.")
    names = context.get("group_names") or {}
    if not names:
        info = read_json(run / "data/data_info.json")
        names = {str(v): k for k, v in info.get("group_id", {}).items()}
    condition, seed = run.parent.name, run.name
    if (run / "summary.csv").exists():
        summary = pd.read_csv(run / "summary.csv")
        if len(summary) != 1:
            raise ValueError("Expected one summary row.")
        condition = str(summary.iloc[0]["condition"])
        seed = str(summary.iloc[0].get("data_seed", seed))
    return dict(run=str(run.resolve()), condition=condition, seed=seed,
                p=p, y=y, groups=groups, names=names, threshold=threshold)


def plot_runs(runs, grid, args, destination):
    thresholds = {r["threshold"] for r in runs}
    if len(thresholds) != 1:
        raise ValueError("Selected runs use different thresholds; select consistent runs.")
    threshold = thresholds.pop()
    ids = sorted(set().union(*(set(r["groups"]) for r in runs)))
    colors = plt.get_cmap("tab20")(np.linspace(0, 1, max(2, len(ids))))
    fig, axes = plt.subplots(1, 3, figsize=(16, 4.8), sharex=True, sharey=True)
    all_rows, summary_rows = [], []
    for ax, scope in zip(axes, ("overall", "predicted_class=0", "predicted_class=1")):
        for gid, color in zip(ids, colors):
            curves, names = [], set()
            for r in runs:
                names.add(str(r["names"].get(str(gid), gid)))
                p1 = r["p"][:, 1]
                group_mask = r["groups"] == gid
                pred = (p1 >= threshold).astype(int)
                mask = group_mask.copy()
                if scope != "overall":
                    mask &= pred == int(scope[-1])
                values = p1[mask]
                curves.append(ecdf_on_grid(values, grid))
                n, total = len(values), int(group_mask.sum())
                all_rows.append(dict(
                    run=r["run"], seed=r["seed"], condition=r["condition"],
                    group_id=gid, scope=scope, support=n, group_support=total,
                    threshold=threshold, fraction_of_group=n/total if total else np.nan,
                    mean_p1=values.mean() if n else np.nan,
                    true_class1_prevalence=r["y"][mask].mean() if n else np.nan,
                    accuracy=(pred[mask] == r["y"][mask]).mean() if n else np.nan,
                    near_boundary_fraction=(np.abs(values-threshold) <= args.band).mean() if n else np.nan,
                    mean_distance_to_threshold=np.abs(values-threshold).mean() if n else np.nan,
                    p1_q10=np.quantile(values, .1) if n else np.nan,
                    p1_median=np.median(values) if n else np.nan,
                    p1_q90=np.quantile(values, .9) if n else np.nan,
                ))
            if len(names) != 1:
                raise ValueError(f"Group {gid} changes name across runs: {names}")
            name = names.pop()
            mean, width, n_valid = summarize_ecdfs(curves, args.shading)
            if n_valid < len(runs):
                warnings.warn(f"{destination.name}: {scope}, {name}: "
                              f"{n_valid}/{len(runs)} nonempty runs.")
            if n_valid:
                ax.step(grid, mean, where="post", color=color,
                        label=f"{name} ({n_valid}/{len(runs)} runs)")
                if np.isfinite(width).all() and args.shading != "none":
                    ax.fill_between(grid, np.clip(mean-width, 0, 1),
                                    np.clip(mean+width, 0, 1),
                                    color=color, alpha=.17, step="post")
            else:
                ax.plot([], [], color=color, label=f"{name}: NA (0/{len(runs)})")
            for x, m, w in zip(grid, mean, width):
                summary_rows.append(dict(
                    group_id=gid, group=name, scope=scope, p1=x,
                    mean_ecdf=m, half_width=w, valid_runs=n_valid,
                    total_runs=len(runs), shading=args.shading))
        ax.axvline(threshold, color="black", linestyle="--", linewidth=1)
        ax.axvspan(max(0, threshold-args.band), min(1, threshold+args.band),
                   color="grey", alpha=.08)
        title = scope if scope == "overall" else f"Predicted: {args.class_names[int(scope[-1])]}"
        ax.set(title=title, xlabel=r"$\bar p_1$", xlim=(0, 1), ylim=(0, 1))
        ax.grid(alpha=.2)
        ax.legend(fontsize=6, loc="best")
    axes[0].set_ylabel("Mean within-run cumulative fraction")
    shade = {"sd": "±1 SD across runs", "ci95": "pointwise 95% t CI", "none": "no shading"}
    fig.suptitle(f"{runs[0]['condition']} | {len(runs)} runs | {shade[args.shading]}\n"
                 "Equal run weights; empty subsets omitted and coverage reported",
                 fontsize=10)
    fig.tight_layout()
    for fmt in args.formats:
        fig.savefig(str(destination) + "." + fmt, dpi=180)
    plt.close(fig)
    pd.DataFrame(all_rows).to_csv(str(destination)+"_runs.csv", index=False)
    pd.DataFrame(summary_rows).to_csv(str(destination)+"_ecdf.csv", index=False)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--conditions", nargs="+")
    parser.add_argument("--threshold", type=float)
    parser.add_argument("--band", type=float, default=.1)
    parser.add_argument("--grid-points", type=int, default=501)
    parser.add_argument("--shading", choices=["sd", "ci95", "none"], default="sd")
    parser.add_argument("--class-names", nargs=2, default=["Class 0", "Class 1"])
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"], default=["png", "pdf"])
    parser.add_argument("--per-run", action="store_true", help="Also save individual-run figures.")
    args = parser.parse_args()
    if not np.isfinite(args.band) or not 0 < args.band <= 1 or args.grid_points < 2:
        parser.error("Require 0 < band <= 1 and grid-points >= 2.")
    paths = ([args.input] if (args.input/"audit").is_dir() else
             sorted(r for r in (args.input/"runs").glob("*/seed_*") if (r/"COMPLETE").is_file()))
    runs = [load_run(r, args.threshold) for r in paths]
    runs = [r for r in runs if not args.conditions or r["condition"] in args.conditions]
    if not runs:
        parser.error("No selected completed runs.")
    identities = [(r["condition"], r["seed"]) for r in runs]
    if len(set(identities)) != len(identities):
        parser.error("Duplicate condition/seed identities.")
    args.out.mkdir(parents=True, exist_ok=True)
    grid = np.linspace(0, 1, args.grid_points)
    for i, condition in enumerate(dict.fromkeys(r["condition"] for r in runs), 1):
        selected = [r for r in runs if r["condition"] == condition]
        plot_runs(selected, grid, args, args.out/f"condition_{i:02d}_boundary")
        if args.per_run:
            for j, run in enumerate(selected, 1):
                plot_runs([run], grid, args, args.out/f"condition_{i:02d}_run_{j:03d}")
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings["condition_order"] = list(dict.fromkeys(r["condition"] for r in runs))
    settings["selected_runs"] = [r["run"] for r in runs]
    (args.out/"settings.json").write_text(json.dumps(settings, indent=2)+"\n")
    print(f"Saved averaged ECDFs to {args.out}")

if __name__ == "__main__":
    main()