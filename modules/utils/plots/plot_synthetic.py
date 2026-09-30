"""Plot observed loan rows and optionally their saved ensemble's uncertainties.

Coordinates show two selected raw features. Colors use all seven predictors;
these are projections of observations, not a two-dimensional decision surface.
"""
import argparse
import json
from pathlib import Path
import numpy as np


def read_saved_data(run, split):
    folder = run/'synthetic_data' if (run/'synthetic_data').is_dir() else run
    meta=json.loads((folder/'metadata.json').read_text())
    with np.load(folder/f'{split}.npz',allow_pickle=False) as archive:
        data={k:archive[k] for k in archive.files}
    if meta.get('generator')!='kanubala_2025_repository':
        raise ValueError('This plotter requires saved loan data; retain the old plotter for historical square data')
    return folder,meta,data


def evaluate_rows(run, data, meta, device='cpu', batch_size=512):
    import pandas as pd
    from torch.utils.data import DataLoader,TensorDataset
    from modules.utils.checkpoint_utils import load_ensemble
    from modules.utils.dataset_utils import transform_predictor_schema
    from modules.predictive.ensemble import evaluate_ensemble
    models,schema,binary=load_ensemble(run,device=device)
    frame=pd.DataFrame(data['X'],columns=meta['feature_names'])
    frame['S1'],frame['S2']=data['S'][:,0],data['S'][:,1]
    frame['y'],frame['group_id']=data['y'],data['group']
    tensors=transform_predictor_schema(frame,schema)
    return evaluate_ensemble(models,DataLoader(TensorDataset(*tensors),batch_size=batch_size,shuffle=False),
                             binary=binary,device=device)


def plot_rows(data, meta, values, *, features, title, vmax, indices):
    import matplotlib.pyplot as plt
    fig,axes=plt.subplots(2,2,figsize=(10,8),sharex=True,sharey=True,constrained_layout=True)
    xy=data['X'][:,[meta['feature_names'].index(f) for f in features]]
    for g,ax in enumerate(axes.flat):
        ids=indices[data['group'][indices]==g]
        scatter=ax.scatter(xy[ids,0],xy[ids,1],c=np.asarray(values)[ids],s=10,alpha=.7,
                           cmap='viridis',vmin=0,vmax=max(float(vmax),1e-12))
        ax.set_title(f"Group {meta['group_order'][g]} | n={int((data['group']==g).sum())}")
        ax.set_xlabel(features[0]); ax.set_ylabel(features[1])
    fig.colorbar(scatter,ax=axes.ravel().tolist(),label=title,shrink=.8)
    intervention=meta.get('uncertainty',{}).get('kind','none')
    fig.suptitle(f"{meta['scenario']} | uncertainty: {intervention}\n{title} (observed-row projection)")
    return fig


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--split',choices=['train','validation','audit','reference'],default='train')
    p.add_argument('--features',nargs=2,default=['Income','LoanAmount'])
    p.add_argument('--ensemble',action='store_true')
    p.add_argument('--out',type=Path)
    p.add_argument('--device',default='cpu')
    p.add_argument('--batch-size',type=int,default=512)
    p.add_argument('--max-points',type=int,default=12000)
    p.add_argument('--plot-seed',type=int,default=0)
    p.add_argument('--dpi',type=int,default=180)
    p.add_argument('--show',action='store_true')
    args=p.parse_args()
    if args.max_points<1 or args.batch_size<1 or args.dpi<1:
        p.error('Point count, batch size and dpi must be positive')
    import matplotlib
    if not args.show: matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    _,meta,data=read_saved_data(args.run,args.split)
    if len(set(args.features))!=2 or any(f not in meta['feature_names'] for f in args.features):
        p.error(f"Choose two distinct features from {meta['feature_names']}")
    output=args.out or args.run/'plots'/args.split
    output.mkdir(parents=True,exist_ok=True)
    ids=np.sort(np.random.default_rng(args.plot_seed).choice(len(data['y']),size=min(args.max_points,len(data['y'])),replace=False))
    panels={'labels':(data['y'],1),'oracle_probability':(data['p_true'],1)}
    arrays={}
    if args.ensemble:
        arrays=evaluate_rows(args.run,data,meta,args.device,args.batch_size)
        panels.update(predicted_probability=(arrays['mean_probs'][:,1],1),
                      aleatoric=(arrays['aleatoric_uncertainty'],np.log(2)),
                      epistemic=(arrays['epistemic_uncertainty'],max(float(arrays['epistemic_uncertainty'].max()),1e-12)),
                      total=(arrays['predictive_entropy'],np.log(2)))
        np.savez_compressed(output/'ensemble_rows.npz',**arrays,row_id=data['row_id'])
    for name,(values,vmax) in panels.items():
        fig=plot_rows(data,meta,values,features=args.features,title=name,vmax=vmax,indices=ids)
        for suffix in ['png','pdf']:
            fig.savefig(output/f'{name}.{suffix}',dpi=args.dpi)
        if not args.show: plt.close(fig)
    (output/'plot_settings.json').write_text(json.dumps(dict(split=args.split,features=args.features,
        plotted_indices=ids.tolist(),color_maxima={k:float(v[1]) for k,v in panels.items()},
        interpretation='Observed-row projections; model evaluated on full predictor vectors'),indent=2))
    if args.show: plt.show()
    print(f'Saved plots to {output}')

if __name__=='__main__':
    main()
