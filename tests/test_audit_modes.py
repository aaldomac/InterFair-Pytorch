"""Audit-mode contract, numerical reduction, and checkpoint reuse regressions."""
import json
import math
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch
import numpy as np
import torch
from modules.metrics.log_ratio import maximum_log_ratio, metric_log_ratio, subgroup_metric_values
from modules.metrics.fairness_metrics import differential_fairness, _class_probabilities, evaluate_ensemble_fairness
from modules.pipelines.audit_pipeline import AuditConfig, AuditInputs, run_audit
from scripts.audit import audit_saved_run, main


class RatioTests(unittest.TestCase):
    def test_scalar_and_ordering(self):
        for values, groups in [([.8,.4], [9,2]), ([.4,.8], [2,9])]:
            r = maximum_log_ratio(values, groups)
            self.assertAlmostEqual(r['epsilon'], math.log(2))
            self.assertEqual(r['maximizing_pair'], [9,2])
        self.assertEqual(maximum_log_ratio([.8,.8])['epsilon'], 0)
        self.assertEqual(maximum_log_ratio([[.8,.2],[.4,.6]])['component'], 1)

    def test_edge_cases(self):
        self.assertEqual(maximum_log_ratio([0.,0.])['epsilon'], 0)
        self.assertEqual(maximum_log_ratio([0.,.4])['epsilon'], math.inf)
        self.assertEqual(maximum_log_ratio([[0.,.4],[0.,.8]])['epsilon'], math.log(2))
        for x in [[], [.4], [np.nan,.4], [np.nan,np.nan]]:
            self.assertIsNone(maximum_log_ratio(x)['epsilon'])
        for x in [[-.1,.4], [np.inf,.4], [-np.inf,.4], [[.2],[.3,.4]]]:
            with self.assertRaises(ValueError):
                maximum_log_ratio(x)
        with self.assertRaises(ValueError):
            maximum_log_ratio([.4,.8], [1,1])

    def test_named_custom_missing_and_smoothing(self):
        p = np.array([[.2,.8],[.6,.4],[.6,.4],[.8,.2]])
        y = np.array([1,0,1,0]); g = np.array([0,0,1,1])
        for metric in ('tpr', 'fpr'):
            self.assertAlmostEqual(metric_log_ratio(p,y,g,metric)['epsilon'], math.log(2))
        r = metric_log_ratio(p,y,g,'accuracy')
        self.assertAlmostEqual(r['epsilon'], math.log(.7/.6))
        r = metric_log_ratio(p,y,g,lambda p,y,k: p[:,1].mean())
        self.assertAlmostEqual(r['epsilon'], math.log(2))
        r = metric_log_ratio(p,y,g,'tpr',offset=.1)
        self.assertAlmostEqual(r['epsilon'], math.log(.9/.5))
        self.assertIsNone(metric_log_ratio(p,y,g,'tpr',expected_groups=[2])['epsilon'])
        self.assertIsNone(metric_log_ratio(p,np.zeros(4,dtype=int),g,'tpr',offset=1)['epsilon'])
        for offset in [-1, np.nan, np.inf]:
            with self.assertRaises(ValueError):
                metric_log_ratio(p,y,g,offset=offset)

    def test_hand_calculation_and_df_regression(self):
        p = np.array([[.2,.8],[.6,.4],[.7,.3],[.8,.2]])
        groups = [0,0,1,1]
        soft = differential_fairness(p, groups, alpha=0, soft=True)
        hard = differential_fairness(p, groups, alpha=0, soft=False)
        np.testing.assert_allclose(soft[3], [[.4,.6],[.75,.25]])
        np.testing.assert_allclose(hard[3], [[.5,.5],[1.,0.]])
        self.assertAlmostEqual(soft[0], math.log(2.4), places=6)
        self.assertEqual(hard[0], math.inf)
        smooth = differential_fairness(p, groups, alpha=1, soft=False)
        np.testing.assert_allclose(smooth[3], [[.5,.5],[.75,.25]])
        self.assertAlmostEqual(smooth[0], math.log(2))
        multi = np.array([[.6,.3,.1],[.2,.3,.5]])
        self.assertAlmostEqual(differential_fairness(multi,[0,1],alpha=0,soft=True)[0], math.log(5))
        self.assertEqual(differential_fairness([],[],alpha=0)[0], 0.)
        self.assertEqual(differential_fairness([.4],[0],alpha=0)[0], 0.)

    def test_aggregate_before_hardening_and_ties(self):
        from modules.predictive.ensemble import evaluate_ensemble
        from torch.utils.data import DataLoader, TensorDataset
        class Fixed(torch.nn.Module):
            def __init__(self, p):
                super().__init__(); self.value = math.log(p/(1-p))
            def forward(self, x):
                return torch.full((len(x),1), self.value)
        # Two members vote positive, but the mean probability is below .5.
        ensemble = evaluate_ensemble([Fixed(.51),Fixed(.51),Fixed(.01)],
            DataLoader(TensorDataset(torch.ones(2,1)),batch_size=2),binary=True,device='cpu')
        hard = _class_probabilities(ensemble['mean_probs'], soft=False)
        np.testing.assert_array_equal(hard, [[1,0],[1,0]])
        np.testing.assert_array_equal(_class_probabilities([[.5,.5]],soft=False), [[0,1]])
        np.testing.assert_array_equal(_class_probabilities([[.4,.4,.2]],soft=False), [[1,0,0]])
        np.testing.assert_array_equal(_class_probabilities([[.3,.7]],threshold=.7,soft=False), [[0,1]])

    def test_multiclass_rates_performance_and_wrapper(self):
        from modules.metrics.fairness_metrics import performance_fairness, evaluate_ensemble_fairness_from_loader
        from torch.utils.data import DataLoader, TensorDataset
        p = np.array([[.1,.2,.7],[.6,.1,.3],[.1,.5,.4],[.3,.6,.1]])
        y = np.array([2,0,2,0]); g = np.array([0,0,1,1])
        r = metric_log_ratio(p,y,g,'tpr',positive_class=2)
        self.assertAlmostEqual(r['epsilon'], math.log(.7/.4))
        self.assertAlmostEqual(metric_log_ratio(p,y,g,'fpr',positive_class=2)['epsilon'], math.log(3))
        for soft in (False,True):
            f = performance_fairness({'mean_probs':p},y,g,binary=False,positive_class=2,soft=soft)
            self.assertEqual(f['soft'],soft)
            self.assertAlmostEqual(f['groups'][0]['TPR'], .7 if soft else 1.,places=6)
            self.assertAlmostEqual(f['groups'][0]['Accuracy'], .65 if soft else 1.,places=6)
            loader=DataLoader(TensorDataset(torch.ones(4,1),torch.tensor(y),torch.tensor(g)),batch_size=2)
            wrapped=evaluate_ensemble_fairness_from_loader({'mean_probs':p},loader,binary=False,soft=soft)
            expected=evaluate_ensemble_fairness({'mean_probs':p},y,g,binary=False,soft=soft)
            self.assertEqual(wrapped['differential_fairness']['aggregate'],expected['differential_fairness']['aggregate'])

    def test_python_modes_and_continuous_scores(self):
        p=np.array([[.2,.8],[.6,.4],[.7,.3],[.8,.2]])
        inputs=AuditInputs({'mean_probs':p},np.array([1,0,1,0]),np.array([0,0,1,1]))
        results=[]
        for soft in (True,False):
            result=run_audit(inputs,dict(enabled=False,fairness=True,soft=soft,smoothing_alpha=0))
            self.assertEqual(result['audit_settings']['soft'],soft)
            results.append(result)
        self.assertEqual(results[0]['predictive_metrics'],results[1]['predictive_metrics'])
        self.assertNotEqual(results[0]['audit_settings']['audit_id'],results[1]['audit_settings']['audit_id'])
        self.assertNotEqual(results[0]['additional_metrics']['fairness']['statistical_parity']['aggregate'],
                            results[1]['additional_metrics']['fairness']['statistical_parity']['aggregate'])
        self.assertFalse(AuditConfig.from_mapping({'soft':False}).df_soft)
        with self.assertRaises(ValueError):
            AuditConfig.from_mapping({'soft':False,'df_soft':True})
        # Ignore stale multiclass labels; decision remains ensemble argmax.
        f=evaluate_ensemble_fairness({'mean_probs':[[.1,.2,.7],[.6,.3,.1]],'predictions':[1,1]},
                                    [2,0],[0,1],binary=False,soft=False)
        np.testing.assert_allclose(f['differential_fairness']['group_rates'],[[.25,.25,.5],[.5,.25,.25]])


class SavedAuditTests(unittest.TestCase):
    def test_train_once_audit_modes_cli_and_inference(self):
        from modules.pipelines.experiment_pipeline import run_experiment
        from modules.pipelines.train_predictive_pipeline import PipelineConfig, ModelConfig
        from modules.predictive.trainer import TrainConfig, PredictiveTrainer
        from modules.utils.loading_utils import load_saved_outputs
        config=PipelineConfig(dataset_name='synthetic_data',dataset_kwargs=dict(
            seed=8,train=80,validation=40,audit=60,reference=40),n_models=2,
            model_seed=21,device='cpu',model=ModelConfig(hidden_dims=(4,),dropout=0),
            train=TrainConfig(epochs=1,verbose=False))
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); source=root/'source'
            run_experiment(config,out=source,audit_config={'fairness':True,'df_soft':True})
            checkpoints={p:p.read_bytes() for p in (source/'predictive_ensemble').glob('*.pt')}
            expected=load_saved_outputs(source)[0]['mean_probs']
            with patch.object(PredictiveTrainer,'fit',side_effect=AssertionError('must not train')):
                soft=audit_saved_run(source,root/'soft',overrides={'soft':True})
                hard=audit_saved_run(source,root/'hard',overrides={'soft':False})
                changed=audit_saved_run(source,root/'threshold',threshold=.7)
                self.assertEqual(len({r['audit_settings']['audit_id'] for r in (soft,hard,changed)}),3)
                for name,value in [('soft',True),('hard',False)]:
                    context=json.loads((root/name/'audit/audit_context.json').read_text())
                    self.assertEqual(context['soft'],value)
                    self.assertIn(context['audit_id'],(root/name/'summary.csv').read_text())
                    self.assertEqual(json.loads((root/name/'artifact_manifest.json').read_text())['soft'],value)
                with self.assertRaises(ValueError):
                    audit_saved_run(source,source,overrides={'soft':False},overwrite=True)
                with self.assertRaises(FileExistsError):
                    audit_saved_run(source,root/'soft',overrides={'soft':False})
                cfg=root/'override.yaml'; cfg.write_text('audit:\n  df_soft: false\n')
                for arg,value in [('True',True),('False',False),(None,False)]:
                    out=root/('cli'+str(arg))
                    argv=['audit','--run',str(source),'--out',str(out),'--config',str(cfg),'--disparity-metric','tpr']
                    if arg is not None: argv+=['--soft',arg]
                    with patch.object(sys,'argv',argv): main()
                    saved=json.loads((out/'audit/audit_config.json').read_text())
                    self.assertEqual(saved['df_soft'],value)
                    self.assertEqual(saved['disparity_metric'],'tpr')
                inherited = audit_saved_run(source, root/'inherited')
                self.assertTrue(inherited['audit_settings']['soft'])
                from modules.utils.plots.plot_group_metrics import read_group_run
                table = read_group_run(root/'soft')
                self.assertIn('TPR', table)
                self.assertIn('log_ratio_class_probability.value.0', table)
                from modules.metrics.group_diagnostics import read_run_diagnostics
                dsoft, _, meta = read_run_diagnostics(root/'soft')
                dhard, _, _ = read_run_diagnostics(root/'hard')
                self.assertTrue(meta['audit_soft'])
                key = 'audit_prediction_frequency_class_1'
                self.assertFalse(np.allclose(dsoft[key], dhard[key], equal_nan=True))
                np.testing.assert_allclose(dsoft['nll'], dhard['nll'], equal_nan=True)
                # Remove every saved prediction: now only checkpoint inference can work.
                for path in (source/'results').glob('ensemble_*.npy'): path.unlink()
                inferred=load_saved_outputs(source)[0]['mean_probs']
                np.testing.assert_allclose(inferred,expected,atol=1e-7)
                audit_saved_run(source,root/'inferred',overrides={'df_soft':False})
                # Legacy raw synthetic data + saved preprocessing fallback.
                (source/'data/audit_inputs.npz').unlink()
                np.testing.assert_allclose(load_saved_outputs(source)[0]['mean_probs'],expected,atol=1e-7)
            self.assertTrue(checkpoints)
            for path,content in checkpoints.items(): self.assertEqual(path.read_bytes(),content)

    def test_mixed_policy_reporting_rejected(self):
        from modules.utils.reporting_utils import completed_runs
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp)
            for i in range(2):
                run=root/'runs'/'baseline'/f'seed_{i}'
                (run/'audit').mkdir(parents=True); (run/'COMPLETE').touch()
                (run/'audit/audit_context.json').write_text(json.dumps({'audit_id':str(i)}))
            with self.assertRaisesRegex(ValueError,'Mixed audit'):
                completed_runs(root)

if __name__ == '__main__': unittest.main()
