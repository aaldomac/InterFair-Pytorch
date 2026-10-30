"""Plot run-level diagnostic summaries without recomputing statistics."""
import math
import textwrap
from functools import wraps
from pathlib import Path
import numpy as np
from modules.utils.plots.group_labels import add_group_key

# Palette from plotting_functions.py; no ICML/tueplots or seaborn dependency.
COLORS = {
    "green": "#009E73", "red": "#D55E00", "blue": "#0072B2", "orange": "#E69F00",
    "purple": "#CC79A7", "gray": "#7F7F7F",
    "black": "#000000",
}
PALETTE = tuple(COLORS.values())
HATCHES = ("", "//", "\\\\", "xx", "..", "++", "oo")
PAPER_STYLE = {
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 11, "axes.labelsize": 11, "axes.titlesize": 11,
    "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 9,
    "figure.titlesize": 11, "text.usetex": False,
    "lines.linewidth": 1.8, "lines.markersize": 3.5,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.linewidth": .6, "axes.grid": False,
    "axes.facecolor": "white", "figure.facecolor": "white",
    "savefig.facecolor": "white", "legend.frameon": False,
    "grid.color": "#7F7F7F", "grid.linestyle": "--",
    "grid.linewidth": .6, "grid.alpha": .4,
    "pdf.fonttype": 42, "ps.fonttype": 42,
}


def _paper_style(func):
    @wraps(func)
    def wrapped(*args, **kwargs):
        import matplotlib.pyplot as plt
        # Local context: importing/calling this module does not restyle other plots.
        with plt.rc_context(PAPER_STYLE):
            return func(*args, **kwargs)
    return wrapped


def _layout(count, ncols, figsize, row_height):
    import matplotlib.pyplot as plt
    if count < 1:
        raise ValueError("Supply at least one metric/group.")
    if not isinstance(ncols, int) or ncols < 1:
        raise ValueError("ncols must be a positive integer.")
    cols = min(count, ncols)
    rows = math.ceil(count / cols)
    # Convenient publication sizes, not ACM requirements. Override with the
    # actual width of your manuscript's column/text area (in inches).
    size = figsize if figsize is not None else (3.33 if cols == 1 else 7.0, row_height * rows + .65)
    return plt.subplots(rows, cols, figsize=size, squeeze=False, constrained_layout=True)


def _heading(fig, title, note, show_notes):
    fig.diagnostic_note = note  # Retained for captions when hidden in the plot.
    parts = [str(title)] if title else []
    if show_notes:
        parts.append(textwrap.fill(note, width=max(30, int(fig.get_figwidth() * 16))))
    if parts:
        fig.suptitle("\n".join(parts), fontsize=PAPER_STYLE["figure.titlesize"], y=1.0, x=.5, ha='center', va='top')


def _errors(data, errorbars):
    if errorbars not in {"ci95", "sd", "none"}:
        raise ValueError("errorbars must be 'ci95', 'sd', or 'none'.")
    if errorbars == "none":
        return np.full(len(data), np.nan)
    return data["std" if errorbars == "sd" else "ci95"].to_numpy(float)


def save_figure(fig, path, *, dpi=300):
    """Save a PDF/PNG/etc. at its configured size, creating parent directories.

    Prefer PDF for the paper. Unlike bbox_inches='tight', this preserves the
    requested physical dimensions. The plotting functions still return figures
    without automatically saving them, matching the original API.
    """
    import matplotlib.pyplot as plt
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with plt.rc_context({**PAPER_STYLE, "savefig.bbox": None}):
        fig.savefig(path, dpi=dpi, bbox_inches=None)
    return path


@_paper_style
def plot_panels(summary, metrics, *, scopes, scope_labels=None, errorbars='ci95', title='',
                ncols=2, figsize=(7.0, 5.5), show_notes=False, metric_labels=None, hatches=True, group_labels=None, show_overall_mean=False,):
    """Grouped means; existing CI/SD half-widths are plotted unchanged."""
    metrics, scopes = list(metrics), list(scopes)
    if not scopes:
        raise ValueError('Supply at least one scope.')
    groups = list(dict.fromkeys(summary.group))
    names = scope_labels or {}
    fig, axes = _layout(len(metrics), ncols, figsize, 2.7)
    width = .8/len(scopes)
    for ax, metric in zip(axes.flat, metrics):
        for j, scope in enumerate(scopes):
            part = summary[(summary.metric == metric) & (summary.scope == scope)].set_index('group').reindex(groups)
            # print(f"Metric: {metric}, Scope: {scope}, Groups: {groups}, Part:\n{part}")
            values = part['mean'].to_numpy(float)
            errors = _errors(part, errorbars)
            positions = np.arange(len(groups)) + (j-(len(scopes)-1)/2)*width
            finite = np.isfinite(values)
            ax.bar(positions[finite], values[finite], width*.92, color=PALETTE[j % len(PALETTE)], alpha=.85,
                   edgecolor=COLORS['black'], linewidth=.35, 
                #    hatch=HATCHES[j % len(HATCHES)] if hatches else None,
                    hatch=HATCHES[0] if hatches else None,
                   label=names.get(scope, scope))
            # # Set fixed y limits depending on the metric type
            # if metric == "U_alea":
            #     ax.set_ylim(0, 0.08)
            # elif metric == "U_epis":
            #     ax.set_ylim(0, 0.012)
            
            good = finite & np.isfinite(errors)
            if errorbars != 'none' and good.any():
                ax.errorbar(positions[good], values[good], yerr=errors[good], fmt='none', color='#333', capsize=2)
            for x in positions[~finite]:
                ax.text(x, .025, 'NA', transform=ax.get_xaxis_transform(), ha='center', fontsize=PAPER_STYLE["xtick.labelsize"])
        if show_overall_mean:
            overall = summary[(summary.metric == metric) & (summary.scope == 'overall')].set_index('group').reindex(groups)

            means = overall['mean'].to_numpy(float)
            centers = np.arange(len(groups))
            valid = np.isfinite(means)

            ax.hlines(y=means[valid], xmin=centers[valid] - 0.40, xmax=centers[valid] + 0.40,
                colors='black', linestyles='--', linewidth=1, zorder=5, label='Group mean',)
        ax.set_xticks(np.arange(len(groups)), groups, rotation=20, ha='center', fontsize=PAPER_STYLE["xtick.labelsize"]-2)
        ax.set_ylabel('Mean (nats)', fontsize=PAPER_STYLE["axes.labelsize"]-4)
        ax.set_xlim(-.6, len(groups)-.4)
        ax.set_title(textwrap.fill(str((metric_labels or {}).get(metric, metric)), 32), fontsize=PAPER_STYLE["axes.titlesize"])
        ax.axhline(0, color='#555', linewidth=.6)
        if metric.startswith('error_auroc'):
            ax.axhline(.5, color='#777', linestyle='--', linewidth=.8)
        ax.set_axisbelow(True)
        ax.grid(axis='y')
        ax.spines[['top','right']].set_visible(False)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    # Include the legend only in the second panel of condition 01: single axis 
    axes.flat[1].legend(fontsize=PAPER_STYLE["legend.fontsize"]-3, frameon=False, loc='outside right upper')
    # label = {'ci95':'pointwise 95% t CI across runs', 'sd':'sample SD across runs', 'none':'no error bars'}[errorbars]
    # _heading(fig, title, label+'; NA = incomplete/undefined; each panel has its own scale', show_notes)
    # add_group_key(fig, group_labels)
    return fig


@_paper_style
def plot_contrasts(summary, metrics, *, scope='overall', errorbars='ci95', title='',
                   ncols=2, figsize=(7.0, 5.5), show_notes=False, metric_labels=None, group_labels=None):
    """Signed within-run contrasts; no multiplicity adjustment is introduced."""
    metrics = list(metrics)
    import matplotlib.pyplot as plt
    part = summary[summary.scope == scope]
    contrasts = list(dict.fromkeys(part.contrast))
    fig, axes = _layout(len(metrics), ncols, figsize, max(2.3, .25*len(contrasts)+1))
    for ax, metric in zip(axes.flat, metrics):
        data = part[part.metric == metric].set_index('contrast').reindex(contrasts)
        values = data['mean'].to_numpy(float)
        errors = _errors(data, errorbars)
        positions = np.arange(len(contrasts))
        good = np.isfinite(values)
        ax.plot(values[good], positions[good], 'o', color=COLORS['blue'])
        usable = good & np.isfinite(errors)
        if errorbars != 'none' and usable.any():
            ax.errorbar(values[usable], positions[usable], xerr=errors[usable], fmt='none', color=COLORS['blue'], capsize=3)
        for y in positions[~good]:
            ax.text(.02, y, 'NA', transform=ax.get_yaxis_transform(), va='center')
        ax.axvline(0,color='#444',linewidth=.8)
        ax.set_yticks(positions, contrasts)
        ax.invert_yaxis()
        ax.set_title(textwrap.fill(str((metric_labels or {}).get(metric, metric)), 32))
        ax.set_axisbelow(True)
        ax.grid(axis='x')
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    # _heading(fig, f'{title}: {scope}' if title else str(scope),
    #          f'Signed within-run contrasts; error bars={errorbars}; no multiplicity adjustment', show_notes)
    # add_group_key(fig, group_labels)
    return fig


@_paper_style
def plot_reliability(frame, title='', *, ncols=2, figsize=(7.0, 5.5), show_notes=False, group_labels=None,):
    """Equal-run bin averages over occupied bins; descriptive, no pooled-person CI.

    group_labels: optional mapping from group identifiers to legend labels.
    ncols: retained for backward compatibility; unused.
    """
    import matplotlib.pyplot as plt

    groups = list(dict.fromkeys(frame.group))
    fig, ax = plt.subplots(figsize=figsize)

    # Reference line for perfect calibration.
    ax.plot([0, 1], [0, 1], '--', color=COLORS['gray'], linewidth=1, label='Perfect calibration', zorder=1,)

    linestyles = ('-', '--', '-.', ':')

    for i, group in enumerate(groups):
        part = (frame.loc[frame.group == group].groupby('bin', sort=True)[['confidence', 'accuracy']].mean())

        label = group_labels.get(group, str(group)) if group_labels is not None else str(group)

        ax.plot(part['confidence'], part['accuracy'], color=PALETTE[i % len(PALETTE)],
            linestyle=linestyles[(i // len(PALETTE)) % len(linestyles)], marker='o',
            markersize=4, label=label, zorder=2,
        )

    ax.set(xlim=(0, 1), ylim=(0, 1), xlabel='Mean assigned confidence', ylabel='Decision accuracy',)
    ax.set_aspect('equal', adjustable='box')
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.legend(loc='best')

    # _heading(fig, title, 'Equal-run means over occupied confidence bins; below diagonal = overconfidence', show_notes,)

    return fig


@_paper_style
def plot_accuracy_coverage(summary, title='', *, errorbars='ci95', figsize=(7.0, 4.5), show_notes=False, group_labels=None):
    """Overlay subgroup curves; full coverage left, declining coverage right."""
    import matplotlib.pyplot as plt
    if summary.empty:
        raise ValueError("Empty summary frame: no accuracy-coverage data.")
    if summary.condition.nunique() != 1:
        raise ValueError('Plot one condition at a time.')
    fig, ax = plt.subplots(figsize=figsize, constrained_layout=True)
    styles = ('-', '--', '-.', ':')
    y_min = 1
    for i, ((gid, group), part) in enumerate(summary.groupby(['group_id', 'group'], sort=False)):
        part = part.sort_values('coverage')
        x, y = part.coverage.to_numpy(float), part['mean'].to_numpy(float)
        if np.any(y < y_min):
            y_min = np.min(y[np.isfinite(y)])
        err = _errors(part, errorbars)
        color = PALETTE[i % len(PALETTE)]
        label = (group_labels or {}).get(group, str(group))
        ax.step(x, y, where='post', color=color, label=label,
                linestyle=styles[(i // len(PALETTE)) % len(styles)])
        if errorbars != 'none':
            ax.fill_between(x, y-err, y+err, step='post',
                where=np.isfinite(y) & np.isfinite(err), color=color,
                alpha=.18, linewidth=0)
    ax.set(xlim=(1, 0), ylim=(y_min, 1),
           xlabel='Coverage (fraction retained; most uncertain removed first)',
           ylabel='Accuracy of retained predictions')
    ax.set_axisbelow(True)
    ax.grid(True)
    ax.legend(loc='best')
    band = {'ci95': 'pointwise 95% t CI across runs',
            'sd': 'sample SD across runs', 'none': 'no uncertainty band'}[errorbars]
    # _heading(fig, title, 'Total predictive entropy ranking; equal-run means; '
    #          + band + '; zero-coverage accuracy undefined', show_notes)
    return fig