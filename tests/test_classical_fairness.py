import math
import tempfile
import unittest
from pathlib import Path
import numpy as np
from modules.metrics.fairness_metrics import (
    performance_fairness,classical_fairness_summary,evaluate_ensemble_fairness,
    differential_fairness)


def outputs(p):
    p=np.asarray(p)
    return {'mean_probs':np.column_stack((1-p,p)), 'predictions':(p>=.5).astype(int)}


class ClassicalFairnessTests(unittest.TestCase):
    def test_rates_ratios_and_existing_df(self):
        # Group 0 TP=2,FN=0,FP=1,TN=1; group 1 TP=1,FN=1,FP=0,TN=2.
        e=outputs([.9,.8,.7,.1,.9,.1,.2,.1])
        y=np.array([1,1,0,0]*2); g=np.repeat([0,1],4)
        a=performance_fairness(e,y,g)
        self.assertEqual(a['groups'][0]['TPR'],1.)
        self.assertEqual(a['groups'][1]['TPR'],.5)
        self.assertAlmostEqual(a['summary']['TPR_epsilon'],math.log(.75/.5))
        self.assertAlmostEqual(a['summary']['TPR_max_ratio'],1.5)
        self.assertEqual(a['summary']['Accuracy_epsilon'],0.)
        f=evaluate_ensemble_fairness(e,y,g)
        s=classical_fairness_summary(f,a)
        self.assertEqual(s['DF_epsilon'],differential_fairness(e['mean_probs'][:,1],g)[0])
        self.assertEqual(s['EO_gap'],.5)
        self.assertEqual(s['EOdds_gap'],.5)

    def test_saved_argmax_is_not_used_for_different_threshold(self):
        e=outputs([.6,.2,.8,.2])
        y=np.array([1,0,1,0]); g=np.array([0,0,1,1])
        f=evaluate_ensemble_fairness(e,y,g,threshold=.7)
        p=performance_fairness(e,y,g,threshold=.7)
        self.assertEqual(f['equal_opportunity']['aggregate'],1.)
        self.assertEqual(p['summary']['TPR_gap'],1.)

    def test_undefined_tpr_remains_missing(self):
        e=outputs([.8,.2,.8,.2]); y=[0,0,1,0]; g=[0,0,1,1]
        p=performance_fairness(e,y,g)
        self.assertIsNone(p['groups'][0]['TPR'])
        self.assertIsNone(p['summary']['TPR_epsilon'])
        self.assertIsNone(p['pairs'][0]['TPR_gap'])
        self.assertIsNone(classical_fairness_summary(evaluate_ensemble_fairness(e,y,g),p)['EO_gap'])
        self.assertIsNotNone(p['summary']['Accuracy_epsilon'])

    def test_pipeline_save_and_report(self):
        from modules.pipelines.experiment_pipeline import run_experiment
        from modules.pipelines.experiment_config import ROOT,read_config,pipeline_config
        from modules.reporting.classical_report import render_report
        from modules.reporting.synthetic_report import load_results,build_report
        cfg,_=read_config(ROOT/'configs/synthetic/smoke.yaml')
        c=pipeline_config(cfg,dict(cfg['dataset']['kwargs'],seed=0),1000)
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); dest=root/'runs/baseline/seed_0'
            r=run_experiment(c,out=dest,audit_config=cfg['audit'])
            self.assertIn('DF_epsilon',r['additional_metrics']['classical_summary'])
            (dest/'COMPLETE').touch()
            report=render_report(root)
            self.assertTrue((report/'pairs.csv').exists())
            frames=load_results(root)
            _,exports=build_report(*frames[:3])
            self.assertIn('classical_stats.csv',exports)
            self.assertIn('Accuracy_epsilon',exports['runs.csv'])

if __name__=='__main__':
    unittest.main()
