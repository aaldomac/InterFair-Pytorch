"""Scientific regression checks for the single-generator migration."""
import unittest
from copy import deepcopy
import numpy as np
from modules.data.loan_data import generate_loan_splits, SCENARIOS
from modules.data.synthetic_uncertainty import generate
from modules.data.uncertainty_conditions import apply_uncertainty, h, stripe_region_mask

SIZES=dict(train=600,validation=200,audit=400,reference=400)


class LoanInterventionTests(unittest.TestCase):
    def test_all_original_scenarios_unchanged(self):
        for kind in SCENARIOS:
            with self.subTest(kind=kind):
                expected,em=generate_loan_splits(seed=7,scenario='loan_'+kind,**SIZES)
                actual,am=generate(seed=7,scenario='loan_'+kind,**SIZES)
                self.assertEqual(em,am)
                for split in expected:
                    for key in expected[split]:
                        np.testing.assert_array_equal(expected[split][key],actual[split][key])

    def test_zero_noise_is_identity_and_inputs_are_immutable(self):
        raw,meta=generate(seed=7,**SIZES)
        before=deepcopy(raw)
        out,_=apply_uncertainty(raw,meta,{'kind':'baseline','baseline_noise':0},seed=7)
        for split in raw:
            for key in raw[split]:
                np.testing.assert_array_equal(before[split][key],raw[split][key])
                np.testing.assert_array_equal(raw[split][key],out[split][key])
        self.assertNotIn('uncertainty',meta)

    def test_scarcity_nested_and_heldout_identical(self):
        base,_=generate(seed=7,uncertainty={'kind':'baseline'},**SIZES)
        previous=set(base['train']['row_id'])
        for rho in [1.,.5,.25]:
            out,_=generate(seed=7,uncertainty={'kind':'scarcity','rho':rho},**SIZES)
            self.assertTrue(set(out['train']['row_id'])<=previous)
            previous=set(out['train']['row_id'])
            n=(base['train']['group']==3).sum()
            self.assertEqual((out['train']['group']==3).sum(),max(1,int(np.floor(n*rho))))
            for split in ['validation','audit','reference']:
                for key in base[split]:
                    np.testing.assert_array_equal(base[split][key],out[split][key])
            for key in base['train']:
                np.testing.assert_array_equal(base['train'][key][base['train']['group']!=3],out['train'][key][out['train']['group']!=3])

    def test_noise_coupling_and_oracles(self):
        lo,lm=generate(seed=9,uncertainty={'kind':'noise','eta':.1,'baseline_noise':0},**SIZES)
        hi,hm=generate(seed=9,uncertainty={'kind':'noise','eta':.4,'baseline_noise':0},**SIZES)
        for split in lo:
            np.testing.assert_array_equal(lo[split]['X'],hi[split]['X'])
            np.testing.assert_array_equal(lo[split]['y_clean'],hi[split]['y_clean'])
            lf=lo[split]['y']!=lo[split]['y_clean']; hf=hi[split]['y']!=hi[split]['y_clean']
            self.assertTrue(np.all(~lf|hf))
            self.assertFalse(hf[hi[split]['group']!=3].any())
            rates=np.asarray(hm['flip_probability'])[hi[split]['group']]
            expected=rates+(1-2*rates)*hi[split]['y_clean']
            np.testing.assert_allclose(hi[split]['p_true'],expected)
            np.testing.assert_allclose(hi[split]['U_true'],h(expected))

    def test_cancellation_targets_and_entropy_patterns(self):
        _,m=generate(seed=1,uncertainty={'kind':'cancellation','eta':.3,'rho':.25},**SIZES)
        np.testing.assert_allclose(m['flip_probability'],[.3,.05,.05,.05])
        np.testing.assert_allclose(m['requested_retention'],[1,1,1,.25])
        for kind,expected in [('additive',0),('interaction',.2)]:
            _,meta=generate(seed=1,uncertainty={'kind':kind,'strength':.2},**SIZES)
            self.assertAlmostEqual(meta['oracle_interaction_bits'],expected)

    def test_combination_preserves_discrimination_covariates(self):
        original,_=generate(seed=3,scenario='loan_intersectional',**SIZES)
        out,_=generate(seed=3,scenario='loan_intersectional',uncertainty={
            'kind':'custom','baseline_noise':0,'flip_probability':{'11':.3},'retention':{'11':.5}},**SIZES)
        for split in out:
            lookup={rid:i for i,rid in enumerate(original[split]['row_id'])}
            ix=[lookup[rid] for rid in out[split]['row_id']]
            np.testing.assert_array_equal(out[split]['X'],original[split]['X'][ix])
            np.testing.assert_array_equal(out[split]['y_clean'],original[split]['y'][ix])

    def test_stripe_uses_training_geometry_only(self):
        base,meta=generate(seed=3,**SIZES)
        cfg={'kind':'stripe','rho':.5,'validation':False}
        out,m=apply_uncertainty(base,meta,cfg,seed=3)
        s=m['stripe']
        inside=stripe_region_mask(base['train']['X'],s)
        expected=~((base['train']['group']==3)&inside)
        np.testing.assert_array_equal(out['train']['row_id'],base['train']['row_id'][expected])
        self.assertAlmostEqual(s['realized_retention']['audit'],1.)
        changed=deepcopy(base)
        changed['audit']['X']*=1000
        changed['reference']['X']*=1000
        _,m2=apply_uncertainty(changed,meta,cfg,seed=3)
        self.assertEqual(s,m2['stripe'])
        for split in ['validation','audit','reference']:
            np.testing.assert_array_equal(out[split]['row_id'],base[split]['row_id'])

    def test_reject_invalid_or_ignored_parameters(self):
        for u in [
            {'kind':'baseline','eta':.3}, {'kind':'noise','eta':-.1},
            {'kind':'stripe','rho':.5,'half_width':1},
            {'kind':'stripe','rho':.5,'features':['Gender','Income']},
            {'kind':'stripe','half_width':1e9},
            {'kind':'custom','retention':{'11':0}},
            {'kind':'custom','flip_probability':{11:.2}},
            {'kind':'additive','strength':1},
        ]:
            with self.subTest(u=u),self.assertRaises(ValueError):
                generate(seed=3,uncertainty=u,**SIZES)

    def test_report_stripe_saved_and_reload(self):
        import tempfile,json
        from pathlib import Path
        from modules.pipelines.experiment_config import read_config,pipeline_config,ROOT
        from modules.pipelines.experiment_pipeline import run_experiment
        from modules.reporting.synthetic_report import load_results,stripe_region_report
        from modules.data.synthetic_uncertainty import load_dataset
        cfg,_=read_config(ROOT/'configs/synthetic/smoke.yaml')
        kwargs=dict(cfg['dataset']['kwargs'],seed=0,uncertainty={'kind':'stripe','rho':.5,'validation':False})
        with tempfile.TemporaryDirectory() as tmp:
            root=Path(tmp); dest=root/'runs/stripe/seed_0'
            result=run_experiment(pipeline_config(cfg,kwargs,1000),out=dest,audit_config=cfg['audit'])
            (dest/'COMPLETE').touch()
            self.assertEqual(len(load_results(root)[0]),1)
            text,exports=stripe_region_report(root)
            self.assertIn('STRIPE',text)
            self.assertIn('stripe_regions.csv',exports)
            loaded=load_dataset(folder=dest/'synthetic_data')
            pd=result['pipeline_result'].data
            np.testing.assert_array_equal(loaded.df['y'],pd.loaded.df['y'])
            self.assertEqual(loaded.metadata['stripe'],pd.loaded.metadata['stripe'])

if __name__=='__main__':
    unittest.main()
