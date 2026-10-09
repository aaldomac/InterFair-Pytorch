"""Grouped horizontal bars for arbitrary conditions and numerical metrics.

Examples (from project root):
 python -m scripts.plot_uncertainty_bars --input experiments/kanubala --metrics F_tot F_U F_hidden F_U_int I_alea I_tot
 python plot_uncertainty_bars.py --input report/runs.csv --metrics F_alea F_epis --scale none

Input: experiment root (reads completed runs via sibling report.py), per-run CSV,
or long summary CSV with condition,metric,mean,std,n. No PyTorch required.
Measurement results: pass measurement_uncertainty_runs.csv to --input.
Group reports: use --input groups.csv --where group=11 --metrics alea epis tot.
Order: --conditions loan_no_bias loan_single loan_additive loan_intersectional loan_compounded.
Options: --scale none|minmax|maxabs, --errorbars sd|none, --annotate,
--formats png pdf svg, --title TEXT, --figsize WIDTH HEIGHT, --show.
Saves figures plus _values.csv (raw and plotted statistics) and _settings.json.
Nonlinear per-run metrics are averaged as already computed; maxima are not
reconstructed from aggregated group/component columns. No pooling of groups.
Without a seed column, each CSV row is assumed to be an independent repetition.
With a seed column, duplicate condition/seed rows require filtering first.
F_hidden is an explicit alias for the existing hidden_normalized column.
Raw values and signed interactions are preserved by default. Error bars are SD,
not confidence intervals. Min-max scaling is descriptive across selected condition
MEANS, with SD divided by that same range (normalization uncertainty not estimated).
"""
from pathlib import Path
import argparse
import json
import warnings
import numpy as np
import pandas as pd

ALIASES = {'F_hidden': 'hidden_normalized'}
LABELS = {
 'F_tot':r'$F_{\mathrm{tot}}$', 'F_alea':r'$F_{\mathrm{alea}}$',
 'F_epis':r'$F_{\mathrm{epis}}$', 'F_U':r'$F_U$',
 'hidden_normalized':r'$F_{\mathrm{hidden}}$', 'F_U_int':r'$F_{U,\mathrm{int}}$',
 'I_alea':r'$\mathcal{I}_{\mathrm{alea}}$', 'I_epis':r'$\mathcal{I}_{\mathrm{epis}}$',
 'I_tot':r'$\mathcal{I}_{\mathrm{tot}}$', 'I_oracle':r'$\mathcal{I}_{\mathrm{oracle}}$'}
NATS = {'F_tot','F_alea','F_epis','I_alea','I_epis','I_tot','I_oracle',
        'alea','epis','tot','oracle','alea_error','delta_alea','delta_epis','delta_tot'}
DIMENSIONLESS = {'F_U','hidden_normalized','F_U_int'}

def read_results(source):
    """Load a CSV or live completed artifacts, without retraining or rewriting reports."""
    source=Path(source)
    if source.is_file():
        return pd.read_csv(source,dtype={'condition':str,'group':str,'scenario':str})
    if (source/'runs').is_dir():
        files = sorted(p for p in (source/'runs').glob('*/*/summary.csv')
                       if (p.parent/'COMPLETE').exists())
        if not files:
            raise ValueError('No completed runs found')
        frame = pd.concat([pd.read_csv(p) for p in files], ignore_index=True)
        frame = frame.rename(columns={'data_seed':'seed'})
        if 'hidden_normalized' in frame:
            frame['F_hidden'] = frame['hidden_normalized']
        return frame
    if (source/'runs.csv').is_file():
        return pd.read_csv(source/'runs.csv',dtype={'condition':str})
    raise ValueError('Use an experiment root, report directory, or a CSV file.')

def summarize_metrics(frame, metrics, *, condition_col='condition', conditions=None, seed_col='seed', input_format='runs'):
    """Return condition/metric/std/n; each row must be an independent run.

    For precomputed long summaries select input_format='summary'.
    Missing metrics/conditions or nonfinite means are rejected, not plotted as zero.
    """
    metrics=[ALIASES.get(m,m) for m in metrics]
    if not metrics or len(set(metrics))!=len(metrics):
        raise ValueError('Select distinct metrics (including after alias resolution).')
    if condition_col not in frame:
        raise ValueError(f'Missing condition column: {condition_col}')
    frame=frame.copy()
    if frame[condition_col].isna().any():
        raise ValueError('Missing condition identifiers.')
    frame[condition_col]=frame[condition_col].astype(str)
    available=list(dict.fromkeys(frame[condition_col]))
    order=available if conditions is None else list(conditions)
    if not order or len(set(order))!=len(order) or set(order)-set(available):
        raise ValueError(f'Invalid condition selection; available: {available}')
    frame=frame[frame[condition_col].isin(order)]
    rows=[]
    if input_format=='summary':
        if not {'metric','mean','std','n'} <= set(frame):
            raise ValueError('Long summary requires metric, mean, std, n columns.')
        frame['metric']=frame.metric.map(lambda m:ALIASES.get(m,m))
        for c in order:
            for m in metrics:
                part=frame[(frame[condition_col]==c)&(frame.metric==m)]
                if len(part)!=1:raise ValueError(f'Need one summary row for {c}/{m}.')
                r=part.iloc[0]
                rows.append(dict(condition=c,metric=m,mean=float(r['mean']),std=float(r['std']),n=float(r['n'])))
    else:
        # Prefer a real user-defined F_hidden column if provided alone; the alias
        # is for this project's normalized hidden-disparity convention only.
        if 'F_hidden' in frame and 'hidden_normalized' not in frame:
            raise ValueError('Rename F_hidden explicitly: alias here means hidden_normalized; cannot infer your convention.')
        missing=set(metrics)-set(frame)
        if missing:
            raise ValueError(f'Missing metrics {sorted(missing)}. Available columns: {list(frame)}')
        if seed_col in frame and frame.duplicated([condition_col,seed_col]).any():
            raise ValueError('Multiple rows per condition/seed. Filter group/pair first with --where, or use another condition column.')
        for c in order:
            part=frame[frame[condition_col]==c]
            for m in metrics:
                x=pd.to_numeric(part[m],errors='raise').to_numpy(dtype=float)
                if not np.isfinite(x).all():raise ValueError(f'Nonfinite values in {c}/{m}.')
                rows.append(dict(condition=c,metric=m,mean=float(x.mean()),
                                 std=float(x.std(ddof=1)) if len(x)>1 else np.nan,n=len(x)))
    result=pd.DataFrame(rows)
    if not np.isfinite(result['mean']).all() or not np.isfinite(result.n).all() or (result.n<1).any() or (result.n%1!=0).any():
        raise ValueError('Invalid summary means or run counts.')
    if np.isinf(result['std']).any() or (result['std'].dropna()<0).any():
        raise ValueError('Invalid SD.')
    if ((result.n>1)&result['std'].isna()).any():
        raise ValueError('Missing SD for a multi-run summary.')
    result.loc[result.n==1,'std']=np.nan
    result['n']=result.n.astype(int)
    return result


def scale_summary(summary, scale='none'):
    """Retain raw columns; append plot_mean/plot_std and explicit scale parameters."""
    if scale not in ('none','minmax','maxabs'):
        raise ValueError('Unknown scale.')
    out=summary.copy()
    out['plot_mean']=out['mean']
    out['plot_std']=out['std']
    parameters={}
    for metric in dict.fromkeys(out.metric):
        mask=out.metric==metric
        means=out.loc[mask,'mean']
        offset=0.
        factor=1.
        if scale=='minmax':
            offset=float(means.min())
            factor=float(means.max()-means.min())
        elif scale=='maxabs':
            factor=float(means.abs().max()) or 1.
        constant=scale=='minmax' and factor==0
        if constant:
            warnings.warn(f'{metric}: constant condition means; min-max undefined. Drawn at zero without error bars; raw values saved.')
            out.loc[mask,'plot_mean']=0.
            out.loc[mask,'plot_std']=np.nan
        else:
            out.loc[mask,'plot_mean']=(means-offset)/factor
            out.loc[mask,'plot_std']=out.loc[mask,'std']/factor
        parameters[metric]=dict(offset=offset,factor=factor,constant_minmax=constant)
    return out,parameters


def plot_grouped_bars(summary, *, scale='none', errorbars='sd', title=None,
                      condition_labels=None, metric_labels=None, annotate=False,
                      figsize=None, cmap='tab10'):
    """N conditions on Y, M bars per condition, values on X. Returns fig,ax,plotted,scales.

    condition_labels/metric_labels optionally map raw IDs to display labels.
    Signed metrics extend left of zero. No absolute-value transformation is applied.
    """
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator
    if errorbars not in ('sd','none'):
        raise ValueError('errorbars must be sd or none.')
    conditions=list(dict.fromkeys(summary.condition))
    metrics=list(dict.fromkeys(summary.metric))
    if summary.duplicated(['condition','metric']).any() or len(summary)!=len(conditions)*len(metrics):
        raise ValueError('Summary must have one row per condition/metric pair.')
    plotted,parameters=scale_summary(summary,scale)
    n,m=len(conditions),len(metrics)
    if not n or not m:
        raise ValueError('Nothing to plot.')
    figsize=figsize or (11,max(3.8,n*max(.9,.23*m)+1.8))
    with plt.rc_context({'font.size':11,'pdf.fonttype':42,'ps.fonttype':42}):
        fig,ax=plt.subplots(figsize=figsize,layout='constrained')
        if cmap=='tab10' and m>10:
            cmap='tab20' if m<=20 else 'viridis'
        colors=plt.get_cmap(cmap)(np.linspace(0,1,m))
        h=.78/m
        y=np.arange(n)
        for i,c in enumerate(conditions):
            if i%2==0:ax.axhspan(i-.46,i+.46,color='#F3F5F7',zorder=0)
        for j,metric in enumerate(metrics):
            part=plotted[plotted.metric==metric].set_index('condition').loc[conditions]
            values=part.plot_mean.to_numpy()
            std=part.plot_std.to_numpy()
            pos=y+(j-(m-1)/2)*h
            label=(metric_labels or {}).get(metric,LABELS.get(metric,metric))
            if scale=='none':
                label+= ' [nats]' if metric in NATS else (' [unitless]' if metric in DIMENSIONLESS else '')
            ax.barh(pos,values,height=h*.9,color=colors[j],label=label,zorder=3)
            valid=np.isfinite(std)
            if errorbars=='sd' and valid.any():
                ax.errorbar(values[valid],pos[valid],xerr=std[valid],fmt='none',
                            ecolor='#303030',elinewidth=.8,capsize=2,zorder=4)
            if annotate:
                for value,yy in zip(values,pos):
                    ax.annotate(f'{value:.3g}',(value,yy),xytext=(4 if value>=0 else -4,0),
                                textcoords='offset points',ha='left' if value>=0 else 'right',va='center',fontsize=8)
        labels=[(condition_labels or {}).get(c,c.replace('loan_','',1).replace('_',' ')) for c in conditions]
        ax.set_yticks(y,labels)
        ax.invert_yaxis()
        ax.set_ylabel('Condition')
        xlabel={'none':'Metric value (units in legend)','minmax':'Per-metric min–max scaled value',
                'maxabs':'Per-metric value / maximum absolute condition mean'}[scale]
        ax.set_xlabel(xlabel);ax.xaxis.set_major_locator(MaxNLocator(7))
        ax.axvline(0,color='#555555',lw=.9,zorder=2)
        ax.grid(axis='x',alpha=.25,zorder=0)
        for spine in ['top','right','left']:
            ax.spines[spine].set_visible(False)
        ax.tick_params(axis='y',length=0)
        ax.margins(x=.16 if annotate else .06)
        ax.legend(loc='upper left',bbox_to_anchor=(1.01,1),frameon=False,title='Metric')
        notes='Mean across runs' + ('; error bars: ±1 sample SD' if errorbars=='sd' else '')
        if (summary.n==1).any():
            notes+='; SD unavailable for single runs'
        if any(v['constant_minmax'] for v in parameters.values()):
            notes+='; constant metrics mapped to 0'
        ax.set_title((title+'\n' if title else '')+notes,loc='left',fontsize=11,pad=14)
    return fig,ax,plotted,parameters


def main():
    p=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--metrics',nargs='+',default=['F_tot','F_U','F_hidden','F_U_int','I_alea','I_tot'])
    p.add_argument('--conditions',nargs='+',help='Exact condition IDs, in desired top-to-bottom order')
    p.add_argument('--condition-col',default='condition')
    p.add_argument('--seed-col',default='seed')
    p.add_argument('--input-format',choices=['runs','summary'],default='runs')
    p.add_argument('--where',action='append',default=[],metavar='COLUMN=VALUE',help='Filter group/pair/etc; repeatable')
    p.add_argument('--scale',choices=['none','minmax','maxabs'],default='none')
    p.add_argument('--errorbars',choices=['sd','none'],default='sd')
    p.add_argument('--title')
    p.add_argument('--annotate',action='store_true')
    p.add_argument('--out',type=Path,default=Path('figures/uncertainty_bars'),help='Output prefix')
    p.add_argument('--formats',nargs='+',choices=['png','pdf','svg'],default=['png','pdf'])
    p.add_argument('--figsize',nargs=2,type=float)
    p.add_argument('--cmap',default='tab10')
    p.add_argument('--dpi',type=int,default=250)
    p.add_argument('--show',action='store_true')
    a=p.parse_args()
    if a.dpi<1 or (a.figsize and any(not np.isfinite(v) or v<=0 for v in a.figsize)):
        p.error('Invalid dpi/figsize.')
    import matplotlib
    if not a.show:matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    try:
        frame=read_results(a.input)
        for selection in a.where:
            key,sep,value=selection.partition('=')
            if not sep or key not in frame:raise ValueError(f'Invalid filter {selection}')
            frame=frame[frame[key].astype(str)==value]
        summary=summarize_metrics(frame,a.metrics,condition_col=a.condition_col,
                    conditions=a.conditions,seed_col=a.seed_col,input_format=a.input_format)
        if len(set(summary.n))>1:
            warnings.warn('Conditions have different completed run counts; see saved CSV.')
        if a.scale=='minmax':
            warnings.warn('Min-max scaling changes the zero reference and can conceal absolute differences; use raw scale for signed interactions.')
        fig,ax,plotted,scales=plot_grouped_bars(summary,scale=a.scale,errorbars=a.errorbars,
                          title=a.title,annotate=a.annotate,figsize=a.figsize,cmap=a.cmap)
        a.out.parent.mkdir(parents=True,exist_ok=True)
        for fmt in a.formats:
            path=Path(str(a.out)+'.'+fmt);fig.savefig(path,dpi=a.dpi,bbox_inches='tight')
            print(path)
        plotted.to_csv(str(a.out)+'_values.csv',index=False)
        settings={k:str(v) if isinstance(v,Path) else v for k,v in vars(a).items()}
        settings['scales']=scales;settings['F_hidden_alias']='hidden_normalized'
        Path(str(a.out)+'_settings.json').write_text(json.dumps(settings,indent=2)+'\n')
        if a.show:
            plt.show()
        plt.close(fig)
    except (ValueError,KeyError,OSError,ImportError) as exc:
        p.exit(1,f'Error: {exc}\n')

if __name__=='__main__':
    main()
