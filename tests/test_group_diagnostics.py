import json
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path
import numpy as np
import pandas as pd
from modules.metrics.group_diagnostics import diagnose_predictions, error_auroc, compute_contrasts, summarize_runs


def ensemble(scores):
    p = np.column_stack([1-np.asarray(scores), scores])
    h = -(p*np.log(np.clip(p, 1e-300, 1))).sum(1)
    return dict(mean_probs=p, aleatoric_uncertainty=h*.8, epistemic_uncertainty=h*.2)


class DiagnosticsTests(unittest.TestCase):
    def test_confident_errors(self):
        frame, _ = diagnose_predictions(ensemble([.95,.9,.2,.1]), [0,1,0,0], [0]*4)
        row = frame.set_index('scope').loc['overall']
        self.assertAlmostEqual(row.high_conf_error_rate, .25)
        self.assertAlmostEqual(row.high_conf_coverage, .75)
        self.assertAlmostEqual(row.high_conf_risk, 1/3)
        self.assertAlmostEqual(row.TPR, 1)
        self.assertAlmostEqual(row.FPR, 1/3)
        self.assertAlmostEqual(error_auroc([0,1], [1,1]), .5)
        self.assertAlmostEqual(error_auroc([0,1], [0,1]), 1)
        self.assertTrue(np.isnan(error_auroc([1,1], [0,1])))

    def test_composition_missing_and_policy(self):
        frame, _ = diagnose_predictions(ensemble([.1,.2,.3,.8]), [0,0,0,1], [0]*4,
                                        threshold=.7, positive_class=0)
        f = frame.set_index('scope')
        self.assertAlmostEqual(f.loc['overall','U_tot'], .75*f.loc['true_class=0','U_tot']+.25*f.loc['true_class=1','U_tot'])
        self.assertAlmostEqual(f.loc['class_standardized','U_tot'], .5*(f.loc['true_class=0','U_tot']+f.loc['true_class=1','U_tot']))
        self.assertEqual(f.loc['overall','accuracy'], 1)
        missing, _ = diagnose_predictions(ensemble([.2,.3]), [0,0], [0,0])
        self.assertTrue(np.isnan(missing.set_index('scope').loc['class_standardized','U_tot']))

    def test_multiclass(self):
        p = np.array([[.7,.2,.1],[.1,.2,.7],[.2,.7,.1]])
        h = -(p*np.log(p)).sum(1)
        ens = dict(mean_probs=p, aleatoric_uncertainty=h, epistemic_uncertainty=np.zeros(3))
        f, _ = diagnose_predictions(ens, [0,1,1], [0,0,0])
        self.assertAlmostEqual(f.set_index('scope').loc['overall','accuracy'], 2/3)
        with self.assertRaises(ValueError):
            diagnose_predictions(ens, [0,1], [0,0])

    def test_paired_contrast(self):
        rows = [dict(condition='x',run=str(i),seed=str(i),scope='overall',group=g,m=i*100+v)
                for i in range(3) for g,v in [('A',2),('B',5)]]
        c = compute_contrasts(pd.DataFrame(rows), {'B-A':{'A':-1,'B':1}}, ['m'])
        self.assertTrue(np.all(c.m == 3))
        s = summarize_runs(c,['condition','scope','contrast'],['m'],{'x':3})
        self.assertEqual(s.iloc[0].ci95, 0)
        c.loc[0,'m'] = np.nan
        s = summarize_runs(c,['condition','scope','contrast'],['m'],{'x':3})
        self.assertTrue(np.isnan(s.iloc[0]['mean']))
        self.assertEqual(s.iloc[0].n_valid, 2)

    def test_cli_and_source_routing(self):
        from modules.metrics.group_diagnostics import read_run_diagnostics
        with tempfile.TemporaryDirectory() as temp:
            root = Path(temp)
            for seed in range(2):
                run = root/'exp/runs/baseline'/f'seed_{seed}'
                (run/'audit').mkdir(parents=True)
                (run/'results').mkdir()
                for key, value in ensemble(np.tile([.05,.3,.6,.95],8)).items():
                    np.save(run/'results'/f'ensemble_{key}.npy', value)
                np.savez(run/'audit/audit_rows.npz',y=np.tile([0,1,0,1],8),group_id=np.repeat(np.arange(8),4))
                (run/'config.json').write_text('{}')
                names = {str(i):'_'.join(f'{i:03b}') for i in range(8)}
                (run/'audit/audit_context.json').write_text(json.dumps({'group_names':names}))
                (run/'COMPLETE').touch()
            dest = root/'out'
            subprocess.run([sys.executable,'-m','scripts.diagnose_group_uncertainty',
                '--input',str(root/'exp'),'--out',str(dest),'--formats','png',
                '--contrasts','configs/diagnostics/adult_contrasts.json'],check=True,capture_output=True)
            self.assertTrue((dest/'condition_01_true_class.png').is_file())
            self.assertTrue((dest/'contrast_summary.csv').is_file())
            linked = root/'linked'
            (linked/'audit').mkdir(parents=True)
            (linked/'audit/audit_context.json').write_text(json.dumps({'source_run':str(run),'group_names':names}))
            data, _, _ = read_run_diagnostics(linked)
            self.assertEqual(len(data[data.scope=='overall']),8)

if __name__ == '__main__':
    unittest.main()
