"""Plot run-level diagnostic summaries without recomputing statistics."""
import math
import textwrap
from functools import wraps
from pathlib import Path
import numpy as np


# Palette from plotting_functions.py; no ICML/tueplots or seaborn dependency.
COLORS = {
    "blue": "#0072B2", "orange": "#E69F00", "green": "#009E73",
    "red": "#D55E00", "purple": "#CC79A7", "gray": "#7F7F7F",
    "black": "#000000",
}
PALETTE = tuple(COLORS.values())
HATCHES = ("", "//", "\\\\", "xx", "..", "++", "oo")
PAPER_STYLE = {
    "figure.dpi": 150, "savefig.dpi": 300,
    "font.family": "serif", "font.serif": ["DejaVu Serif"],
    "font.size": 8, "axes.labelsize": 8, "axes.titlesize": 8,
    "legend.fontsize": 7, "xtick.labelsize": 7, "ytick.labelsize": 7,
    "figure.titlesize": 9, "text.usetex": False,
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
        fig.suptitle("\n".join(parts), fontsize=9)


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
                ncols=2, figsize=(7.0, 5.5), show_notes=True, metric_labels=None, hatches=True):
    """Grouped means; existing CI/SD half-widths are plotted unchanged."""
    metrics, scopes = list(metrics), list(scopes)
    if not scopes:
        raise ValueError('Supply at least one scope.')
    import matplotlib.pyplot as plt
    groups = list(dict.fromkeys(summary.group))
    names = scope_labels or {}
    fig, axes = _layout(len(metrics), ncols, figsize, 2.7)
    width = .8/len(scopes)
    for ax, metric in zip(axes.flat, metrics):
        for j, scope in enumerate(scopes):
            part = summary[(summary.metric == metric) & (summary.scope == scope)].set_index('group').reindex(groups)
            values = part['mean'].to_numpy(float)
            errors = _errors(part, errorbars)
            positions = np.arange(len(groups)) + (j-(len(scopes)-1)/2)*width
            finite = np.isfinite(values)
            ax.bar(positions[finite], values[finite], width*.92, color=PALETTE[j % len(PALETTE)], alpha=.85,
                   edgecolor=COLORS['black'], linewidth=.35,
                   hatch=HATCHES[j % len(HATCHES)] if hatches else None,
                   label=names.get(scope, scope))
            good = finite & np.isfinite(errors)
            if errorbars != 'none' and good.any():
                ax.errorbar(positions[good], values[good], yerr=errors[good], fmt='none', color='#333', capsize=2)
            for x in positions[~finite]:
                ax.text(x, .025, 'NA', transform=ax.get_xaxis_transform(), ha='center', fontsize=7)
        ax.set_xticks(np.arange(len(groups)), groups, rotation=40, ha='right')
        ax.set_xlim(-.6, len(groups)-.4)
        ax.set_title(textwrap.fill(str((metric_labels or {}).get(metric, metric)), 32))
        ax.axhline(0, color='#555', linewidth=.6)
        if metric.startswith('error_auroc'):
            ax.axhline(.5, color='#777', linestyle='--', linewidth=.8)
        ax.set_axisbelow(True)
        ax.grid(axis='y')
        ax.spines[['top','right']].set_visible(False)
    for ax in list(axes.flat)[len(metrics):]:
        ax.set_visible(False)
    axes.flat[0].legend(fontsize=7, frameon=False)
    label = {'ci95':'pointwise 95% t CI across runs', 'sd':'sample SD across runs', 'none':'no error bars'}[errorbars]
    _heading(fig, title, label+'; NA = incomplete/undefined; each panel has its own scale', show_notes)
    return fig


@_paper_style
def plot_contrasts(summary, metrics, *, scope='overall', errorbars='ci95', title='',
                   ncols=2, figsize=(7.0, 5.5), show_notes=True, metric_labels=None):
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
    _heading(fig, f'{title}: {scope}' if title else str(scope),
             f'Signed within-run contrasts; error bars={errorbars}; no multiplicity adjustment', show_notes)
    return fig


@_paper_style
def plot_reliability(frame, title='', *, ncols=2, figsize=(7.0, 5.5), show_notes=True):
    """Equal-run bin averages over occupied bins; descriptive, no pooled-person CI."""
    import matplotlib.pyplot as plt
    groups = list(dict.fromkeys(frame.group))
    fig, axes = _layout(len(groups), ncols, figsize, 2.5)
    for ax, group in zip(axes.flat, groups):
        part = frame[frame.group == group].groupby('bin',sort=True)[['confidence','accuracy']].mean()
        ax.plot([0,1],[0,1], '--', color='#777', linewidth=1)
        ax.plot(part.confidence, part.accuracy, 'o-',color=COLORS['blue'],markersize=4)
        ax.set(xlim=(0,1), ylim=(0,1), xlabel='Mean assigned confidence', ylabel='Decision accuracy', title=group)
        ax.set_aspect('equal', adjustable='box')
        ax.set_axisbelow(True)
        ax.grid()
    for ax in list(axes.flat)[len(groups):]:
        ax.set_visible(False)
    _heading(fig, title, 'Equal-run means over occupied confidence bins; below diagonal = overconfidence', show_notes)
    return fig