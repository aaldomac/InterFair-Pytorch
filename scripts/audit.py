"""Recompute general uncertainty and predictive metrics from saved predictions."""
import argparse
import json
from pathlib import Path
import yaml
from modules.metrics.uncertainty_audit import audit_uncertainties
from modules.metrics.ensemble_quality import ensemble_quality
from modules.utils.loading_utils import load_saved_outputs


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run',type=Path,required=True)
    parser.add_argument('--config',type=Path,help='Experiment YAML containing audit settings')
    parser.add_argument('--out',type=Path,required=True)
    args=parser.parse_args()
    if args.out.exists():
        raise FileExistsError(args.out)
    cfg=yaml.safe_load((args.config or args.run/'experiment.yaml').read_text())
    opts=cfg.get('audit',{})
    ensemble,y,g=load_saved_outputs(args.run)
    audit=audit_uncertainties(ensemble,g,alpha=opts.get('alpha',.05),
                             interaction_weights=opts.get('interaction_weights'))
    metrics,groups=ensemble_quality(ensemble,y,g,
        threshold=cfg.get('pipeline',{}).get('train',{}).get('threshold',.5))
    args.out.mkdir(parents=True)
    (args.out/'uncertainty_audit.json').write_text(json.dumps(audit,indent=2)+'\n')
    (args.out/'predictive_metrics.json').write_text(json.dumps(metrics,indent=2)+'\n')
    groups.to_csv(args.out/'group_predictive_metrics.csv',index=False)

if __name__ == '__main__':
    main()
