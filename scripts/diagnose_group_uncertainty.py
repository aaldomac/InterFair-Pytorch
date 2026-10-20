"""Analyze saved uncertainty, errors, class composition and paired group contrasts."""
import argparse
import json
from pathlib import Path
import numpy as np
import pandas as pd
from modules.metrics.group_diagnostics import read_run_diagnostics, summarize_runs, compute_contrasts


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, required=True)
    parser.add_argument('--out', type=Path, required=True)
    parser.add_argument('--conditions', nargs='+')
    parser.add_argument('--seeds', nargs='+')
    parser.add_argument('--high-confidence', type=float, default=.9)
    parser.add_argument('--bins', type=int, default=10)
    parser.add_argument('--metrics', nargs='+', default=['U_alea','U_epis','U_tot','accuracy','high_conf_error_rate'])
    parser.add_argument('--contrasts', type=Path, help='JSON: contrast name -> exact group label -> weight')
    parser.add_argument('--class-names', nargs='+', help='Display names in numerical class order: class 0, class 1, ...')
    parser.add_argument('--errorbars', choices=['sd','ci95','none'], default='ci95')
    parser.add_argument('--formats', nargs='+', choices=['png','pdf','svg'], default=['png','pdf'])
    args = parser.parse_args()
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from modules.utils.plots.plot_group_diagnostics import plot_panels, plot_contrasts, plot_reliability
    print(f'Loading diagnostics from {args.input} and saving to {args.out}')
    try:
        if (args.input/'audit').is_dir():
            runs = [args.input]
            print(f'Analyzing saved diagnostics in {args.input}')
        else:
            from modules.utils.reporting_utils import completed_runs
            runs = completed_runs(args.input)
        if not runs:
            raise ValueError('No saved runs found.')
        tables, curves, metadata = [], [], []
        for run in runs:
            table, curve, meta = read_run_diagnostics(run, high_confidence=args.high_confidence, bins=args.bins)
            if args.conditions and meta['condition'] not in args.conditions:
                continue
            if args.seeds and meta['seed'] not in args.seeds:
                continue
            tables.append(table); curves.append(curve); metadata.append(meta)
        if not tables:
            raise ValueError('No runs match the selection.')
        data, reliability = pd.concat(tables,ignore_index=True), pd.concat(curves,ignore_index=True)
        classes = {m['classes'] for m in metadata}
        if len(classes) != 1:
            raise ValueError('Selected runs must share a class universe.')
        c = classes.pop()
        if args.class_names and len(args.class_names) != c:
            raise ValueError('--class-names must match the number of classes.')
        print(f"Data conditions= {data.condition.unique()}")
        for condition in data.condition.unique():
            policies = {(m['threshold'],m['positive_class']) for m in metadata if m['condition']==condition}
            if len(policies) != 1:
                raise ValueError(f'Mixed decision policies within {condition}; select consistent runs.')
        for gid, part in data.groupby('group_id'):
            if part.group.nunique() != 1:
                raise ValueError(f'Group ID {gid} changes meaning across runs.')
        identities = ['condition','run','seed','group_id','group','scope']
        metrics = [k for k in data if k not in identities]
        unknown = set(args.metrics)-set(metrics)
        if unknown:
            raise ValueError(f'Unknown metrics {unknown}. Available: {metrics}')
        counts = data.groupby('condition').run.nunique().to_dict()
        summary = summarize_runs(data, ['condition','group_id','group','scope'], metrics, counts)
        if args.out.resolve() == args.input.resolve() or args.out.resolve().is_relative_to(args.input.resolve()):
            raise ValueError('Choose a diagnostic output directory outside the source experiment/run.')
        args.out.mkdir(parents=True,exist_ok=True)
        data.to_csv(args.out/'group_diagnostic_runs.csv',index=False)
        summary.to_csv(args.out/'group_diagnostic_summary.csv',index=False)
        reliability.to_csv(args.out/'reliability_runs.csv',index=False)
        # Error-minus-correct differences are paired within each group and run.
        index = ['condition','run','seed','group_id','group']
        correct = data[data.scope=='correct'].set_index(index)
        wrong = data[data.scope=='incorrect'].set_index(index)
        gap_metrics = ['U_alea','U_epis','U_tot','mean_confidence']
        gaps = (wrong[gap_metrics]-correct[gap_metrics]).reset_index()
        gap_summary = summarize_runs(gaps, ['condition','group_id','group'], gap_metrics, counts)
        gaps.to_csv(args.out/'error_minus_correct_runs.csv',index=False)
        gap_summary.to_csv(args.out/'error_minus_correct_summary.csv',index=False)
        contrast_summary = None
        specifications = None
        if args.contrasts:
            specifications = json.loads(args.contrasts.read_text())
            contrasts = compute_contrasts(data, specifications, args.metrics)
            contrast_summary = summarize_runs(contrasts, ['condition','scope','contrast'], args.metrics, counts)
            contrasts.to_csv(args.out/'contrast_runs.csv',index=False)
            contrast_summary.to_csv(args.out/'contrast_summary.csv',index=False)
        label = {f'true_class={k}': f'True: {(args.class_names or list(range(c)))[k]}' for k in range(c)}
        label.update({f'predicted_class={k}': f'Predicted: {(args.class_names or list(range(c)))[k]}' for k in range(c)})
        def save(fig, stem):
            for fmt in args.formats:
                fig.savefig(args.out/(stem+'.'+fmt),dpi=180)
            plt.close(fig)
        for i, condition in enumerate(data.condition.unique(),1):
            part = summary[summary.condition==condition]
            stem = f'condition_{i:02d}'
            panels = [
                ('quality', ['accuracy','TPR','FPR','nll','brier','high_conf_coverage','high_conf_error_rate','high_conf_risk'], ['overall']),
                # Can uncertainty rank errors above correct predictions? In binary setting with threshold=0.5 error_auroc_U_tot = error_auroc_low_confidence
                ('error_detection', ['error_auroc_U_alea','error_auroc_U_epis','error_auroc_U_tot','error_auroc_low_confidence','ece'], ['overall']),
                # Does uncertainty increase with mistakes? (paired within-run differences)
                ('correctness', ['U_alea','U_epis','U_tot','mean_confidence'], ['correct','incorrect']),
                # How are actual members of each outcome class handled?
                ('true_class', ['U_alea','U_epis','U_tot','accuracy'], [f'true_class={k}' for k in range(c)]),
                # What are the predictions like among recipients of each decision?
                ('predicted_class', ['U_alea','U_epis','U_tot','accuracy'], [f'predicted_class={k}' for k in range(c)]),
                # prevalence_class_k is the percentage of samples in true class k; mean_prob_class_k is the average predicted probability for true class k.
                ('composition', [f'prevalence_class_{k}' for k in range(c)]+[f'mean_prob_class_{k}' for k in range(c)], ['overall']),
                # How much does class composition affect the group comparison?
                ('standardized', ['U_alea','U_epis','U_tot','accuracy','high_conf_error_rate'], ['overall','class_standardized']),
            ]
            for name, requested, scopes in panels:
                save(plot_panels(part, requested, scopes=scopes, scope_labels=label,
                    errorbars=args.errorbars, title=condition+': '+name), stem+'_'+name)
            gap_part = gap_summary[gap_summary.condition==condition].assign(scope='incorrect minus correct')
            save(plot_panels(gap_part, gap_metrics, scopes=['incorrect minus correct'],
                errorbars=args.errorbars, title=condition+': paired error-minus-correct gaps'), stem+'_error_gaps')
            # Realiability and ece tells if confidence is numerically trustworthy.
            save(plot_reliability(reliability[reliability.condition==condition], condition),stem+'_reliability')
            if contrast_summary is not None:
                for scope in ['overall','class_standardized']+[f'true_class={k}' for k in range(c)]:
                    save(plot_contrasts(contrast_summary[contrast_summary.condition==condition], args.metrics,
                        scope=scope, errorbars=args.errorbars, title=condition),stem+'_contrasts_'+scope.replace('=','_'))
        settings = {k:str(v) if isinstance(v,Path) else v for k,v in vars(args).items()}
        settings.update(runs=metadata, contrasts_resolved=specifications,
            condition_file_order=list(data.condition.unique()),
            confidence_definition='Probability assigned to chosen decision',
            standardization='Uniform true-class weights; absent classes give NA',
            interval='Pointwise Student-t CI across complete run estimates, not simultaneous/population CI',
            brier='Binary: class-1 squared error; multiclass: sum of squared class errors',
            reliability='Equal-run averages over occupied fixed-width bins; bin support exported')
        (args.out/'settings.json').write_text(json.dumps(settings,indent=2)+'\n')
        print(f'Diagnostics saved to {args.out}')
    except (ValueError,KeyError,OSError,ImportError) as exc:
        parser.exit(1,f'Error: {exc}\n')


if __name__ == '__main__':
    main()