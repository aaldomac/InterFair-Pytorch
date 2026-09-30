"""Run: python -m unittest tests.test_synthetic -v"""
import importlib.util
import tempfile
import unittest
from pathlib import Path
import numpy as np
from Projects.InterFairPytorch.modules.data.synthetic_data import generate, parameters, h, LOG2
from modules.metrics.uncertainty_audit import audit_uncertainties


class SyntheticTests(unittest.TestCase):
    def test_paired_scarcity(self):
        sizes = dict(train=160, validation=48, audit=64, reference=64)
        a, _ = generate(seed=9, uncertainty={'kind':'baseline'}, **sizes)
        b, _ = generate(seed=9, uncertainty={'kind':'scarcity','rho':.25}, **sizes)
        for split in ('validation', 'audit', 'reference'):
            for key in a[split]:
                np.testing.assert_array_equal(a[split][key], b[split][key])
        n=int((a['train']['group']==3).sum())
        self.assertEqual(len(b['train']['y']), len(a['train']['y'])-n+max(1,int(n*.25)))
        self.assertTrue(set(b['train']['row_id']) <= set(a['train']['row_id']))

    def test_entropy_interaction(self):
        for scenario, expected in [('additive', 0.), ('interaction', .2)]:
            noise, _ = parameters(scenario=scenario, strength=.2)
            self.assertAlmostEqual(float(np.dot([1,-1,-1,1], h(noise)/LOG2)), expected)

    def test_cancellation(self):
        ensemble = dict(aleatoric_uncertainty=np.array([LOG2,LOG2,LOG2,0]),
                        epistemic_uncertainty=np.array([0,0,0,LOG2]),
                        predictive_entropy=np.full(4,LOG2))
        audit = audit_uncertainties(ensemble, np.arange(4))
        self.assertAlmostEqual(audit['pairs']['0-3']['hidden_normalized'],1.)
        self.assertAlmostEqual(audit['maxima']['F_tot'],0.)


@unittest.skipUnless(importlib.util.find_spec('torch'), 'PyTorch is not installed')
class IntegrationTests(unittest.TestCase):
    def test_train_save_reload(self):
        from modules.pipelines.experiment_config import read_config, pipeline_config, ROOT
        from modules.pipelines.experiment_pipeline import run_experiment
        from Projects.InterFairPytorch.modules.data.synthetic_data import load_dataset
        cfg, _ = read_config(ROOT/'configs'/'synthetic'/'smoke.yaml')
        config = pipeline_config(cfg, dict(cfg['dataset']['kwargs'], uncertainty={'kind':'scarcity','rho':.25}, seed=0),1000)
        with tempfile.TemporaryDirectory() as tmp:
            out = Path(tmp)/'run'
            result = run_experiment(config, out=out)
            data = result['pipeline_result'].data
            self.assertEqual(data.num_features, 7)
            self.assertLess(len(data.train_df), cfg['dataset']['kwargs']['train'])
            self.assertEqual(len(next(iter(data.train_loader))), 3)
            self.assertEqual(len(result['pipeline_result'].models), 2)
            self.assertTrue((out/'predictive_ensemble'/'predictor_0.pt').exists())
            loaded = load_dataset(folder=out/'synthetic_data')
            self.assertEqual(len(loaded.df),len(data.loaded.df))
            np.testing.assert_array_equal(loaded.df['y'],data.loaded.df['y'])
            self.assertTrue(np.isfinite(result['predictive_metrics']['nll']))


if __name__ == '__main__':
    unittest.main()