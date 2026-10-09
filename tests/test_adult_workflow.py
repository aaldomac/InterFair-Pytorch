import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

class AdultWorkflowTests(unittest.TestCase):
    def test_loader_labels_and_missing_values(self):
        from modules.data.adult import load_dataset
        with tempfile.TemporaryDirectory() as tmp:
            path=Path(tmp)/'adult.csv'
            pd.DataFrame({'sex':['Male','Female','Male'], 'race':['White']*3,
                'native-country':['United-States']*3,'income':['<=50K.','>50K.','<=50K'],
                'workclass':['Private',' Private',' ?'],'age':[30,40,50]}).to_csv(path,index=False)
            d=load_dataset(csv_path=path,compute_pg=False)
            self.assertEqual(len(d.df),2)
            self.assertEqual(d.df.income.tolist(),[1,0])
            d=load_dataset(csv_path=path,compute_pg=False,positive_label='>50K')
            self.assertEqual(d.df.income.tolist(),[0,1])

    def test_partitions_preserve_source_and_rare_groups(self):
        from modules.utils.dataset_utils import split_df
        frame=pd.DataFrame({'group':['a']*40+['b']*3+['c']*2,'x':np.arange(45)})
        train,val,test=split_df(frame,rare_group_policy='train_only',seed=10)
        ids=[set(d.index) for d in [train,val,test]]
        self.assertEqual(set.union(*ids),set(frame.index))
        self.assertFalse(ids[0]&ids[1] or ids[0]&ids[2] or ids[1]&ids[2])
        self.assertEqual((train.group=='c').sum(),2)
        self.assertNotIn('c',test.group.values)
        for d in [train,val,test]:
            np.testing.assert_array_equal(d.x,frame.loc[d.index,'x'])
            self.assertIn('b',d.group.values)
        for a,b in zip([train,val,test],split_df(frame,rare_group_policy='train_only',seed=10)):
            np.testing.assert_array_equal(a.index,b.index)

    def test_adult_training_and_general_report(self):
        from modules.pipelines.experiment_pipeline import run_experiment
        from modules.pipelines.experiment_config import read_config,pipeline_config,ROOT
        from modules.reporting.real_report import report
        with tempfile.TemporaryDirectory() as tmp:
            folder=Path(tmp); path=folder/'adult.csv'
            n=240
            df=pd.DataFrame({'age':np.arange(n)%50+18,'gender':['Male','Female']*(n//2),
                'race':['White']*(n//2)+['Black']*(n//2),'native-country':['United-States']*n,
                'income':['<=50K','<=50K','>50K','>50K']*(n//4),
                'capital-gain':[0]*n,'capital-loss':[0]*n,'workclass':['Private']*n})
            rare=df.iloc[:1].copy(); rare['race']='Other'
            pd.concat([df,rare],ignore_index=True).to_csv(path,index=False)
            cfg,_=read_config(ROOT/'configs/adult.yaml')
            cfg['pipeline']['train'].update(epochs=1,patience=1,verbose=False)
            cfg['pipeline']['n_models']=1
            cfg['pipeline']['model']['hidden_dims']=[8]
            kwargs=dict(cfg['dataset']['kwargs'],csv_path=str(path))
            config=pipeline_config(cfg,kwargs,1000,data_seed=0)
            dest=folder/'runs/baseline/seed_0'
            r=run_experiment(config,out=dest,audit_config=cfg['audit'])
            self.assertIsNone(r['uncertainty_audit']['F_U_int'])
            pd.DataFrame([dict(condition='baseline',data_seed=0,**r['predictive_metrics'],
                **r['uncertainty_audit']['maxima'])]).to_csv(dest/'summary.csv',index=False)
            (dest/'COMPLETE').touch()
            output=report(folder)
            self.assertTrue((output/'groups_stats.csv').exists())
            coverage=pd.read_csv(output/'group_coverage.csv')
            self.assertEqual((~coverage.evaluated).sum(),1)
            self.assertIn('DF_epsilon',pd.read_csv(output/'runs.csv'))
            with np.load(dest/'splits/split_indices.npz') as saved:
                train_idx=saved['train_idx']; test_idx=saved['test_idx']
            self.assertFalse(set(train_idx)&set(test_idx))
            np.testing.assert_array_equal(r['pipeline_result'].data.loaded.df.iloc[test_idx].income,
                                          r['pipeline_result'].data.test_df.income)

if __name__=='__main__': unittest.main()
