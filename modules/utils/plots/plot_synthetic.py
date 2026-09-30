"""Plot saved synthetic data; optionally evaluate a saved deep ensemble.

Run from the project root:
  python -m scripts.plot_synthetic --run PATH_TO_SEED_FOLDER
  python -m scripts.plot_synthetic --run PATH_TO_SEED_FOLDER --ensemble

--run accepts a trained run (containing synthetic_data/) or a generated-only
folder (containing metadata.json and train.npz). No data are regenerated and
no training occurs. Ensemble mode requires this project's modules/ package,
PyTorch, pandas, scikit-learn and joblib, plus the original saved checkpoints
and fitted preprocessing. Load only your own trusted joblib artifacts.

Outputs: data_<split>.png/.pdf; optionally ensemble_maps.png/.pdf,
ensemble_maps.npz (unrounded plotted arrays) and plot_settings.json.
Default output is RUN/plots/<split>/. Re-running replaces these plot files only.

Maps cover the four supported squares only. Nuisance inputs are independent
N(0,1) draws, shared across grid points and groups to reduce visual MC noise.
Uncertainty is computed for each complete 6D input BEFORE averaging nuisance
samples. These maps visualize a fitted predictor, not confidence intervals.
"""
from __future__ import annotations

import argparse
import contextlib
import io
import json
from pathlib import Path
import sys

import numpy as np


def read_saved_data(run, split):
    """Read raw arrays and DGP metadata, without loading any model or torch."""
    folder = run / 'synthetic_data' if (run / 'synthetic_data').is_dir() else run
    meta = json.loads((folder / 'metadata.json').read_text())
    with np.load(folder / f'{split}.npz', allow_pickle=False) as archive:
        data = {key: archive[key] for key in archive.files}
    x, y, groups = data['X'], data['y'], data['group']
    if x.ndim != 2 or x.shape[1] != 6 or len(x) == 0:
        raise ValueError('Expected nonempty raw X with six features.')
    if y.shape != (len(x),) or groups.shape != y.shape:
        raise ValueError('Labels and groups must match the raw feature rows.')
    if not np.isfinite(x).all() or not np.isin(y, [0, 1]).all() or not np.isin(groups, range(4)).all():
        raise ValueError('Invalid features, binary labels, or group IDs.')
    if len(meta['flip_probability']) != 4:
        raise ValueError('Expected four group-specific flip probabilities.')
    expected = np.column_stack((groups // 2, groups % 2))
    if not np.all(np.abs(x[:, :2] - 3 * (2 * expected - 1)) <= 1.00001):
        raise ValueError('Coordinates do not match the four-square synthetic generator.')
    return folder, meta, data


def condition_title(meta):
    """Include the actual intervention strength, not just the scenario name."""
    scenario = meta["scenario"]
    if scenario == 'stripe':
        info = meta['stripe']
        return (f"stripe | w={info['half_width']:.4g} | slope={info['slope']:g}"
                f" | rho={info['expected_retention']:.4g} | group={info['group']}")
    keys = {"scarcity": ["rho"], "noise": ["eta"], "additive": ["strength"],
            "interaction": ["strength"], "cancellation": ["eta", "rho"]}.get(scenario, [])
    return scenario + "".join(f" | {key}={meta[key]:g}" for key in keys)


def make_grid(meta, resolution):
    """Create supported 2D grids and their exact oracle probabilities."""
    axis = np.linspace(-1, 1, resolution)
    z1, z2 = np.meshgrid(axis, axis)
    clean = (z1 + .5 * z2 + .35 * np.sin(np.pi * z2) >= 0).astype(float)
    xy, oracle = [], []
    for g, eta in enumerate(meta['flip_probability']):
        center = 3 * (2 * np.array([g // 2, g % 2]) - 1)
        xy.append(np.stack((z1 + center[0], z2 + center[1]), axis=-1))
        oracle.append(eta + (1 - 2 * eta) * clean)
    return np.asarray(xy), np.asarray(oracle)


def decorate(ax, counts=None):
    """Label raw coordinates, group support and the analytical clean boundary."""
    from matplotlib.patches import Rectangle
    for g in range(4):
        cx, cy = 3 * (2 * np.array([g // 2, g % 2]) - 1)
        ax.add_patch(Rectangle((cx - 1, cy - 1), 2, 2, fill=False, lw=.7, ec='0.4'))
        z2 = np.linspace(-1, 1, 300)
        z1 = -.5 * z2 - .35 * np.sin(np.pi * z2)
        ax.plot(cx + z1, cy + z2, '--', color='black', lw=1)
        label = f'{g//2}{g%2}' + (f'  (n={counts[g]:,})' if counts is not None else '')
        ax.text(cx, cy + 1.18, label, ha='center', fontsize=9)
    ax.set(xlim=(-4.5, 4.5), ylim=(-4.5, 4.7), xlabel=r'$X_1$ (proxy for $S_1$)', ylabel=r'$X_2$ (proxy for $S_2$)')
    ax.set_xticks([-3, 3]); ax.set_yticks([-3, 3])
    ax.set_aspect('equal')
    for spine in ax.spines.values():
        spine.set_visible(False)


def draw_stripe(ax, meta):
    """Magenta dotted edges mark withheld training support, even on audit maps."""
    if 'stripe' not in meta:
        return
    info = meta['stripe']
    if info['half_width'] == 0:
        return
    g = int(info['group'], 2)
    cx, cy = 3*(2*np.array([g//2, g%2])-1)
    z2 = np.linspace(-1., 1., 1000)
    for sign in (-1, 1):
        z1 = info['slope']*z2 + sign*info['half_width']
        z1 = np.where(np.abs(z1) <= 1., z1, np.nan)
        ax.plot(cx+z1, cy+z2, ':', color='#E600A9', lw=1.6)


def heatmap(ax, xy, values, *, vmax, cmap):
    """Draw disjoint group maps without interpolating through unsupported gaps."""
    import matplotlib.pyplot as plt
    norm = plt.Normalize(0, vmax)
    for g in range(4):
        ax.pcolormesh(xy[g, :, :, 0], xy[g, :, :, 1], values[g],
                      shading='nearest', cmap=cmap, norm=norm, rasterized=True)
    decorate(ax)
    return plt.cm.ScalarMappable(norm=norm, cmap=cmap)


def save_figure(fig, output, name, dpi):
    """Save a screen image and a publication PDF with rasterized dense layers."""
    for suffix in ('png', 'pdf'):
        path = output / f'{name}.{suffix}'
        fig.savefig(path, dpi=dpi, bbox_inches='tight')
        print(f'Saved {path}')


def plot_data(data, meta, xy, oracle, args, output):
    """Show observed labels and oracle probabilities; subsample uniformly overall."""
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, BoundaryNorm
    from matplotlib.lines import Line2D
    fig, axes = plt.subplots(1, 2, figsize=(10, 5), layout='constrained')
    rng = np.random.default_rng(args.plot_seed)
    n = len(data['y'])
    ids = rng.choice(n, size=min(n, args.max_points), replace=False)
    # One global sampling rate preserves scarcity visually in expectation.
    cmap = ListedColormap(['#0072B2', '#D55E00'])
    axes[0].scatter(data['X'][ids, 0], data['X'][ids, 1], c=data['y'][ids],
                    cmap=cmap, norm=BoundaryNorm([-.5, .5, 1.5], 2),
                    s=7, alpha=.6, linewidths=0, rasterized=True)
    counts = np.bincount(data['group'].astype(int), minlength=4)
    decorate(axes[0], counts)
    axes[0].set_title(f'Observed {args.split} labels ({len(ids):,}/{n:,} shown)')
    axes[0].legend(handles=[Line2D([], [], marker='o', ls='', color=cmap(i), label=f'Y = {i}') for i in range(2)],
                    loc='center', frameon=False)
    m = heatmap(axes[1], xy, oracle, vmax=1, cmap='viridis')
    axes[1].set_title('Oracle class-1 probability')
    fig.colorbar(m, ax=axes[1], fraction=.046, pad=.03, label=r'$P(Y=1\mid X)$')
    for ax in axes:
        draw_stripe(ax, meta)
    stripe_note = ' | magenta dotted: stripe edges' if 'stripe' in meta else ''
    fig.suptitle(f"{condition_title(meta)} | data seed {meta['seed']} | dashed: true clean boundary{stripe_note}")
    save_figure(fig, output, f'data_{args.split}', args.dpi)
    return fig


def load_ensemble(run, data_folder, device):
    """Load shared checkpoints and the saved train-fitted preprocessing."""
    from modules.utils.checkpoint_utils import load_ensemble as load_saved
    return load_saved(run, device=device)


def evaluate_grid(xy, meta, models, schema, binary, args):
    """Average existing evaluator outputs over nuisance draws, in bounded chunks."""
    import pandas as pd
    import torch
    from modules.utils.dataset_utils import transform_predictor_schema
    from modules.predictive.ensemble import evaluate_ensemble
    points = xy.reshape(-1, 2)
    groups = np.repeat(np.arange(4), args.grid_size ** 2)
    nuisance = np.random.default_rng(args.plot_seed).normal(size=(args.nuisance_draws, 4))
    names = ['probability', 'alea', 'epis', 'tot']
    result = {name: np.zeros(len(points), dtype=float) for name in names}
    # At most approximately batch_size raw rows per outer evaluation chunk.
    step = max(1, args.batch_size // args.nuisance_draws)
    for start in range(0, len(points), step):
        end = min(start + step, len(points))
        count = end - start
        raw = np.column_stack((np.repeat(points[start:end], args.nuisance_draws, axis=0),
                               np.tile(nuisance, (count, 1))))
        g = np.repeat(groups[start:end], args.nuisance_draws)
        df = pd.DataFrame(raw, columns=meta['feature_names'])
        # transform_predictor_schema requires target/group columns even at inference.
        df['S1'], df['S2'], df['group_id'], df['y'] = g // 2, g % 2, g, 0
        x, _, _ = transform_predictor_schema(df, schema)
        loader = torch.utils.data.DataLoader(torch.utils.data.TensorDataset(x), batch_size=args.batch_size)
        with contextlib.redirect_stdout(io.StringIO()):
            values = evaluate_ensemble(models, loader, binary=binary, device=args.device)
        arrays = [values['mean_probs'][:, 1], values['aleatoric_uncertainty'],
                  values['epistemic_uncertainty'], values['predictive_entropy']]
        for name, value in zip(names, arrays):
            if not np.isfinite(value).all():
                raise ValueError(f'Nonfinite grid output: {name}')
            result[name][start:end] = value.reshape(count, args.nuisance_draws).mean(axis=1)
        if start == 0 or end == len(points) or (start // step) % 25 == 0:
            print(f'Grid: {end:,}/{len(points):,} spatial points evaluated', flush=True)
    for name in result:
        result[name] = result[name].reshape(xy.shape[:-1])
    np.testing.assert_allclose(result['alea'] + result['epis'], result['tot'], atol=2e-6)
    return result


def plot_ensemble(xy, oracle, maps, meta, args, output, n_models):
    """Show oracle, learned mean probability, member entropy and disagreement."""
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(1, 4, figsize=(17, 5.1), layout='constrained')
    # Independent normalization: aleatoric retains its natural entropy range;
    # epistemic uses its own observed maximum unless explicitly overridden.
    # A positive fallback keeps an exactly-zero disagreement map well-defined.
    epis_max = getattr(args, 'epis_max', None)
    if epis_max is None:
        epis_max = max(float(np.max(maps['epis'])), 1e-8)
    panels = [('Oracle probability', oracle, 1, 'viridis'),
              ('Ensemble probability', maps['probability'], 1, 'viridis'),
              ('Aleatoric uncertainty', maps['alea'], np.log(2), 'magma'),
              ('Epistemic uncertainty', maps['epis'], epis_max, 'magma')]
    for i, (ax, (title, values, vmax, cmap)) in enumerate(zip(axes, panels)):
        m = heatmap(ax, xy, values, vmax=vmax, cmap=cmap)
        ax.set_title(title)
        draw_stripe(ax, meta)
        if i:
            ax.set_ylabel('')
        label = 'Class-1 probability' if i < 2 else 'Entropy (nats)'
        extend = 'max' if np.any(values > vmax) else 'neither'
        fig.colorbar(m, ax=ax, orientation='horizontal', pad=.08, fraction=.055, label=label, extend=extend)
    for g in range(4):
        p = maps['probability'][g]
        if p.min() < .5 < p.max():
            axes[1].contour(xy[g, :, :, 0], xy[g, :, :, 1], p, levels=[.5], colors='cyan', linewidths=1.2)
    fig.suptitle(f"{condition_title(meta)} | seed {meta['seed']} | {n_models} members | "
                 f"averaged over {args.nuisance_draws} nuisance draws\n"
                 'Dashed black: true clean boundary; cyan: ensemble probability = 0.5'
                 + ('; magenta dotted: stripe edges' if 'stripe' in meta else ''))
    save_figure(fig, output, 'ensemble_maps', args.dpi)
    np.savez_compressed(output / 'ensemble_maps.npz', xy=xy, oracle=oracle, **maps)
    print(f"Epistemic range: {maps['epis'].min():.6f} to {maps['epis'].max():.6f} nats; color maximum {epis_max:g}")
    return fig


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--run', required=True, type=Path, help='One saved seed directory, or its synthetic_data directory (data-only).')
    parser.add_argument('--split', choices=['train', 'validation', 'audit', 'reference'], default='train', help='Observed data to scatter; train reveals scarcity (default).')
    parser.add_argument('--ensemble', action='store_true', help='Also load and visualize the saved deep ensemble; requires the full run directory.')
    parser.add_argument('--out', type=Path, help='Output directory; default RUN/plots/SPLIT.')
    parser.add_argument('--grid-size', type=int, default=60, help='Grid points per axis per group (default 60; total 4*60^2).')
    parser.add_argument('--nuisance-draws', type=int, default=32, help='N(0,1) nuisance samples per spatial point in ensemble mode (default 32).')
    parser.add_argument('--batch-size', type=int, default=4096, help='Maximum inference batch size (default 4096).')
    parser.add_argument('--max-points', type=int, default=6000, help='Maximum scatter points, sampled uniformly across the whole split.')
    parser.add_argument('--plot-seed', type=int, default=123, help='Seed for scatter subsampling and nuisance draws; not the saved data seed.')
    parser.add_argument('--epis-max', type=float, default=None, help='Epistemic color maximum in nats; default log(2). Use the SAME value when comparing conditions; clipped values are marked on colorbar.')
    parser.add_argument('--device', default='cpu', help='Inference device, e.g. cpu or cuda; used only with --ensemble.')
    parser.add_argument('--dpi', type=int, default=220, help='PNG and rasterized PDF layer resolution.')
    parser.add_argument('--show', action='store_true', help='Also open interactive windows; default is save-only for servers.')
    args = parser.parse_args()
    for key in ['grid_size', 'nuisance_draws', 'batch_size', 'max_points', 'dpi']:
        if getattr(args, key) < 1:
            parser.error(f'--{key.replace("_", "-")} must be positive')
    if args.grid_size < 2 or args.plot_seed < 0:  # or not np.isfinite(args.epis_max) or args.epis_max <= 0:
        parser.error('Need grid-size >=2, plot-seed >=0 and a finite positive epis-max.')
    args.run = args.run.resolve()
    # Support both python -m and direct script execution from the repository.
    for parent in [Path.cwd(), *Path(__file__).resolve().parents]:
        if (parent / 'modules' / 'utils' / 'dataset_utils.py').is_file():
            sys.path.insert(0, str(parent)); break
    import matplotlib
    if not args.show:
        matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'pdf.fonttype': 42, 'ps.fonttype': 42})
    folder, meta, data = read_saved_data(args.run, args.split)
    output = (args.out or args.run / 'plots' / args.split).resolve()
    output.mkdir(parents=True, exist_ok=True)
    xy, oracle = make_grid(meta, args.grid_size)
    plot_data(data, meta, xy, oracle, args, output)
    if args.ensemble:
        models, schema, binary = load_ensemble(args.run, folder, args.device)
        maps = evaluate_grid(xy, meta, models, schema, binary, args)
        plot_ensemble(xy, oracle, maps, meta, args, output, len(models))
    settings = {k: str(v) if isinstance(v, Path) else v for k, v in vars(args).items()}
    settings.update(scenario=meta['scenario'], data_seed=meta['seed'],
                    raw_counts=np.bincount(data['group'].astype(int), minlength=4).tolist(),
                    averaging='uncertainty evaluated before averaging nuisance draws',
                    output=str(output))
    (output / 'plot_settings.json').write_text(json.dumps(settings, indent=2) + '\n')
    if args.show:
        plt.show()
    plt.close('all')


if __name__ == '__main__':
    main()
