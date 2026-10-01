import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import pandas as pd
import torch
from modules.metrics.uncertainty_audit import audit_uncertainties
from modules.pipelines.train_predictive_pipeline import PipelineConfig, SplitConfig, ModelConfig, prepare_data
from modules.pipelines.experiment_pipeline import run_experiment
from modules.predictive.trainer import TrainConfig
from modules.utils.checkpoint_utils import load_ensemble
from modules.predictive.ensemble import evaluate_ensemble
from modules.data.synthetic_uncertainty import generate


class SharedTests(unittest.TestCase):
    def test_arbitrary_groups_and_multiclass_bound(self):
        e=dict(aleatoric_uncertainty=np.array([0.,.2,.3]),
               epistemic_uncertainty=np.array([0.,.1,.2]),
               predictive_entropy=np.array([0.,.3,.5]),mean_probs=np.ones((3,3))/3)
        a=audit_uncertainties(e,np.array([2,7,9]))
        self.assertEqual(a['group_ids'],[2,7,9])
        self.assertIsNone(a['F_U_int'])
        self.assertAlmostEqual(a['normalization'],np.log(3))
        with self.assertRaises(ValueError):
            audit_uncertainties(e,[2,7,9],interaction_weights={2:1,7:-1})

    def test_original_audit_parity(self):
        # Four-group cancellation retains the original binary definitions and bound.
        l=np.log(2)
        e=dict(aleatoric_uncertainty=np.array([l,l,l,0]),
               epistemic_uncertainty=np.array([0,0,0,l]),predictive_entropy=np.full(4,l))
        a=audit_uncertainties(e,np.arange(4),interaction_weights={0:1,1:-1,2:-1,3:1})
        self.assertAlmostEqual(a['F_U_int'],.5)
        np.testing.assert_allclose(a['hoeffding_component_radii'],l*np.sqrt(np.log(16/.05)/2))

    def test_fixed_splits_exclude_oracles_and_reference(self):
        c=PipelineConfig(dataset_name='synthetic_uncertainty',dataset_kwargs=dict(train=80,validation=40,audit=48,reference=56),append_protected_to_predictor=False)
        d=prepare_data(c)
        self.assertEqual((len(d.train_df),len(d.val_df),len(d.test_df)),(80,40,48))
        self.assertEqual(d.num_features,7)
        self.assertTrue(set(d.train_df.index).isdisjoint(d.test_df.index))
        c2=PipelineConfig(dataset_name=c.dataset_name,dataset_kwargs=c.dataset_kwargs,model_seed=99,append_protected_to_predictor=False)
        np.testing.assert_array_equal(prepare_data(c2).train_df.index,d.train_df.index)

    def test_loan_and_stripe(self):
        for scenario in ['loan_no_bias','loan_intersectional','stripe']:
            kwargs={'rho':.5} if scenario=='stripe' else {}
            splits,meta=generate(scenario=scenario,train=120,validation=80,audit=100,reference=100,**kwargs)
            self.assertEqual(len(splits['audit']['y']),100)
            self.assertEqual(meta['group_order'],['00','01','10','11'])

    def test_real_csv_multiclass_train_save_reload(self):
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            frame=pd.DataFrame({'x':np.tile(np.arange(3),60), 's':np.repeat([0,1,2],60),
                                'y':np.tile(np.arange(3),60)})
            frame.to_csv(root/'input.csv',index=False)
            cfg=PipelineConfig(dataset_name='csv_tabular',dataset_kwargs=dict(path=str(root/'input.csv'),label_col='y',protected_cols=['s']),
                split=SplitConfig(seed=4),model_seed=12,n_models=1,append_protected_to_predictor=False,
                model=ModelConfig(hidden_dims=(8,),dropout=0),train=TrainConfig(epochs=1,patience=1,binary=False,verbose=False),device='cpu')
            result=run_experiment(cfg,out=root/'run',audit_config={'fairness':True,'distribution':True})
            self.assertEqual(result['uncertainty_audit']['group_ids'],[0,1,2])
            self.assertIsNone(result['uncertainty_audit']['F_U_int'])
            models,schema,binary=load_ensemble(root/'run')
            e=evaluate_ensemble(models,result['pipeline_result'].data.test_loader,binary=binary,device='cpu')
            np.testing.assert_allclose(e['mean_probs'],result['pipeline_result'].ensemble_metrics['mean_probs'])
            self.assertTrue((root/'run/audit/distribution.json').exists())
            np.testing.assert_allclose(e['epistemic_uncertainty'],0,atol=1e-6)
            with self.assertRaises(FileExistsError):
                run_experiment(cfg,out=root/'run')

    def test_image_model_shared_trainer_contract(self):
        from modules.predictive.models import ImageClassifier
        from modules.predictive.losses import BinaryClassificationLoss, CompositeLoss
        from modules.predictive.trainer import PredictiveTrainer
        from torch.utils.data import DataLoader, TensorDataset
        model=ImageClassifier(dropout=0)
        loader=DataLoader(TensorDataset(torch.rand(8,3,16,16),torch.tensor([0,1]*4),torch.tensor([0,0,1,1]*2)),batch_size=4)
        trainer=PredictiveTrainer(model=model,optimizer=torch.optim.Adam(model.parameters()),
            loss_fn=CompositeLoss(task_loss=BinaryClassificationLoss()),
            config=TrainConfig(epochs=1,patience=1,binary=True,verbose=False),device='cpu')
        trained,_=trainer.fit(loader,loader)
        self.assertEqual(evaluate_ensemble([trained],loader,binary=True)['mean_probs'].shape,(8,2))


class DatasetAdapterTests(unittest.TestCase):
    def test_adult_local_csv(self):
        with tempfile.TemporaryDirectory() as tmp:
            p=Path(tmp)/'adult.csv'
            pd.DataFrame({'age':np.arange(160)+18, 'gender':['Male','Female']*80,
                'race':['White']*160, 'native-country':['United-States']*160,
                'income':['<=50K','<=50K','>50K','>50K']*40,
                'capital-gain':[0]*160,'capital-loss':[0]*160}).to_csv(p,index=False)
            c=PipelineConfig(dataset_name='adult',dataset_kwargs={'csv_path':str(p),'compute_pg':False},
                append_protected_to_predictor=False,n_models=1,device='cpu',model=ModelConfig(hidden_dims=(4,)),
                train=TrainConfig(epochs=1,patience=1,binary=True,verbose=False))
            r=run_experiment(c)
            self.assertEqual(len(r['group_predictive_metrics']),2)

    def test_celeba_adapter_train_save_reload(self):
        try:
            import torchvision
        except ImportError:
            self.skipTest('torchvision optional dependency unavailable')
        from PIL import Image
        from modules.data.CelebA import CELEBA_ATTR_TO_IDX
        class FakeCelebA:
            def __init__(self, **kwargs):
                self.transform=kwargs['transform']
                self.attr=torch.zeros(16,40,dtype=torch.long)
                self.attr[:,CELEBA_ATTR_TO_IDX['Male']]=torch.tensor([0,1]*8)
                self.attr[:,CELEBA_ATTR_TO_IDX['Young']]=torch.tensor([0,0,1,1]*4)
                self.attr[:,CELEBA_ATTR_TO_IDX['Smiling']]=torch.tensor([0]*4+[1]*4+[0]*4+[1]*4)
            def __len__(self): return 16
            def __getitem__(self,index):
                return self.transform(Image.fromarray(np.full((180,180,3),index*10,dtype=np.uint8))), self.attr[index]
        with patch('modules.data.CelebA.CelebA',FakeCelebA), tempfile.TemporaryDirectory() as tmp:
            c=PipelineConfig(dataset_name='celeba',dataset_kwargs={'root':tmp,'image_size':16},
                append_protected_to_predictor=False,n_models=1,device='cpu',
                train=TrainConfig(epochs=1,patience=1,binary=True,verbose=False))
            r=run_experiment(c,out=Path(tmp)/'run')
            models,schema,binary=load_ensemble(Path(tmp)/'run')
            self.assertEqual(schema['modality'],'image')
            e=evaluate_ensemble(models,r['pipeline_result'].data.test_loader,binary=binary)
            np.testing.assert_allclose(e['mean_probs'],r['pipeline_result'].ensemble_metrics['mean_probs'])

if __name__=='__main__':
    unittest.main()
