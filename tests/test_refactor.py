"""Regression tests for audit identity, persistence and training correctness."""
import json
import sys
import tempfile
import unittest
from dataclasses import replace
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, TensorDataset

from modules.pipelines.audit_pipeline import AuditConfig, AuditInputs, run_audit
from modules.pipelines.train_predictive_pipeline import (
    PipelineConfig, LoaderConfig, ModelConfig, SplitConfig, run_predictive_pipeline)
from modules.pipelines.experiment_pipeline import run_experiment
from modules.predictive.ensemble import evaluate_ensemble
from modules.predictive.trainer import PredictiveTrainer, TrainConfig
from modules.predictive.losses import BinaryClassificationLoss, CompositeLoss, RegularizerContext
from modules.utils.loading_utils import load_audit_inputs, load_saved_outputs, load_split_indices
from modules.utils.saving_utils import _jsonable, save_run_summary, save_split_indices
from scripts.audit import audit_saved_run


class OffsetModel(torch.nn.Module):
    def __init__(self, offset=0.):
        super().__init__()
        self.offset = torch.nn.Parameter(torch.tensor(float(offset)))

    def forward(self, x):
        return x[:, :1] + self.offset


def fixture_loader(batch_size):
    return DataLoader(TensorDataset(
        torch.tensor([[4.], [-4.], [.7], [-.7], [1.2], [-1.2]]),
        torch.tensor([1, 0, 1, 0, 0, 1]), torch.tensor([7, 7, 2, 2, 7, 2])),
        batch_size=batch_size, shuffle=False)


class RefactorTests(unittest.TestCase):
    def test_batch_size_and_member_storage_invariance(self):
        models = [OffsetModel(-1), OffsetModel(1)]
        full = evaluate_ensemble(models, fixture_loader(1), binary=True, device='cpu')
        streaming = evaluate_ensemble(models, fixture_loader(4), binary=True, device='cpu', store_member_probs=False)
        self.assertNotIn('ensemble_probs', streaming)
        for key, value in streaming.items():
            np.testing.assert_allclose(value, full[key], atol=1e-7)
        y, g = fixture_loader(1).dataset.tensors[1:]
        a = run_audit(AuditInputs(full, y.numpy(), g.numpy(), group_names={7:'seven', 2:'two'}))['uncertainty_audit']
        self.assertEqual(a['group_ids'], [2, 7])
        for gid in (2, 7):
            p = full['mean_probs'][g.numpy() == gid].astype(float)
            h = lambda x: -(x*np.log(np.clip(x, 1e-10, None))).sum(axis=-1)
            self.assertAlmostEqual(a['group_mix_entropy'][gid], h(p.mean(0)), places=6)
            self.assertAlmostEqual(a['group_het_entropy'][gid], h(p.mean(0))-h(p).mean(), places=6)
            self.assertGreaterEqual(a['group_het_entropy'][gid], -1e-6)
        first = evaluate_ensemble(models[:1], fixture_loader(2), binary=True, device='cpu')
        self.assertFalse(np.allclose(first['mean_probs'], full['mean_probs']))

    def test_pair_omission_preserves_maxima(self):
        e = evaluate_ensemble([OffsetModel()], fixture_loader(3), binary=True, device='cpu')
        y, g = fixture_loader(1).dataset.tensors[1:]
        inputs = AuditInputs(e, y.numpy(), g.numpy())
        a = run_audit(inputs)['uncertainty_audit']
        b = run_audit(inputs, {'include_pairs':False})['uncertainty_audit']
        self.assertEqual(a['maxima'], b['maxima'])
        self.assertEqual(b['pairs'], {})
        self.assertFalse(b['pairs_included'])

    def test_ensemble_threshold_is_respected(self):
        outputs = evaluate_ensemble([OffsetModel()], fixture_loader(3), binary=True, device='cpu', threshold=.9)
        np.testing.assert_array_equal(outputs['predictions'], outputs['mean_probs'][:,1] >= .9)

    def test_trainer_accuracy_counts_integers(self):
        model = OffsetModel()
        loader = DataLoader(TensorDataset(torch.ones(10, 1), torch.tensor([1]*7+[0]*3)), batch_size=10)
        trainer = PredictiveTrainer(model, torch.optim.SGD(model.parameters(), lr=0),
            CompositeLoss(BinaryClassificationLoss()), TrainConfig(epochs=1, verbose=False), device='cpu')
        self.assertEqual(trainer.evaluate(loader)['accuracy'], .7)
        self.assertEqual(trainer.train_epoch(loader, 1)['accuracy'], .7)

    def test_no_regularizer_and_context_regularizer(self):
        from modules.regularizers.df_regularizer import DFRegularizerConfig, DifferentialFairnessRegularizer
        model = OffsetModel(.2)
        loader = fixture_loader(3)
        context = RegularizerContext(model, torch.device('cpu'), 1, 0, loader)
        self.assertEqual(zero.item(), 0)
        regularizer = DifferentialFairnessRegularizer(DFRegularizerConfig(burn_in_epochs=0))
        penalty, stats = regularizer(context)
        self.assertTrue(torch.isfinite(penalty))
        self.assertGreaterEqual(stats['df_epsilon'], 0)
        penalty.backward()
        self.assertIsNotNone(model.offset.grad)
        legacy, _ = regularizer(model=model, fair_loader=loader, device=torch.device('cpu'), epoch=1)
        self.assertAlmostEqual(penalty.item(), legacy.item())
        self.assertTrue(model.training)

    def test_ece_includes_confidence_one(self):
        from modules.metrics.calibration_metrics import expected_calibration_error
        self.assertEqual(expected_calibration_error(torch.tensor([[1., 0.]]), torch.tensor([1])), 1.)

    def test_split_round_trip(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            expected = ([10, 2], [5], [8, 9])
            save_split_indices(*expected, root/'splits')
            for a,b in zip(load_split_indices(root), expected):
                np.testing.assert_array_equal(a, b)

    def test_config_validation_does_not_generate_data(self):
        import yaml
        from modules.pipelines.experiment_config import ROOT, read_config, pipeline_config
        with patch('modules.data.synthetic_data.generate', autospec=True, side_effect=AssertionError('must not sample')):
            cfg,_ = read_config(ROOT/'configs/synthetic/smoke.yaml')
        with tempfile.TemporaryDirectory() as tmp:
            path = Path(tmp)/'config.yaml'
            cfg['pipeline']['train'].update(patience=0, threshold=1.)
            path.write_text(yaml.safe_dump(cfg))
            cfg,_ = read_config(path)
            pipeline_config(cfg, cfg['dataset']['kwargs'], 1).train.validate()
            cfg['dataset']['kwargs'] = {'folder':'saved/generated'}
            cfg['conditions'] = {'baseline':{}}
            path.write_text(yaml.safe_dump(cfg))
            read_config(path)
        self.assertEqual(AuditConfig.from_mapping({'alpha':.1}).confidence_alpha, .1)
        with self.assertRaises(ValueError):
            AuditConfig.from_mapping({'alpha':.1, 'confidence_alpha':.2})

    def test_audit_options_and_defaults(self):
        from modules.pipelines.experiment_config import ROOT, read_config, pipeline_config
        cfg,_ = read_config(ROOT/'configs/synthetic/smoke.yaml')
        cfg['pipeline'] = {}
        yaml_default = pipeline_config(cfg, cfg['dataset']['kwargs'], 0)
        self.assertEqual(yaml_default.n_models, PipelineConfig().n_models)
        self.assertEqual(yaml_default.append_protected_to_predictor, PipelineConfig().append_protected_to_predictor)
        with self.assertRaises(ValueError):
            AuditConfig.from_mapping({'fairness':'false'})

    def test_training_saved_audit_and_reports_round_trip(self):
        from modules.reporting.classical_report import main as report_main
        from modules.reporting.real_report import report
        config = PipelineConfig(dataset_name='synthetic_data', dataset_kwargs=dict(
            seed=3, train=120, validation=80, audit=100, reference=80,
            uncertainty={'kind':'stripe','rho':.7,'validation':False}),
            n_models=2, model_seed=17, device='cpu', loader=LoaderConfig(batch_size=17, eval_batch_size=19),
            model=ModelConfig(hidden_dims=(6,), dropout=0),
            train=TrainConfig(epochs=1, patience=0, verbose=False))
        options = dict(fairness=True, distribution=True, confidence_alpha=.1, smoothing_alpha=.7,
                       interaction_weights={0:1, 1:-1, 2:-1, 3:1})
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            source = root/'original/runs/stripe/seed_3'
            result = run_experiment(config, out=source, audit_config=options)
            save_run_summary(source, result, dict(condition='stripe', data_seed=3, model_seed_start=17))
            (source/'COMPLETE').touch()
            before = {str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            destination = root/'reaudited/runs/stripe/seed_3'
            recalculated = audit_saved_run(source, destination)
            for name in ('uncertainty_audit', 'predictive_metrics', 'additional_metrics'):
                self.assertEqual(_jsonable(result[name]), _jsonable(recalculated[name]))
            pd.testing.assert_frame_equal(result['group_predictive_metrics'], recalculated['group_predictive_metrics'])
            self.assertEqual(result['pipeline_result'].test_metrics['accuracy'], result['predictive_metrics']['accuracy'])
            self.assertEqual(result['pipeline_result'].test_metrics['loss'], result['predictive_metrics']['nll'])
            for a,b in zip(load_split_indices(source), (result['pipeline_result'].data.train_df.index,
                    result['pipeline_result'].data.val_df.index, result['pipeline_result'].data.test_df.index)):
                np.testing.assert_array_equal(a, b)
            second = audit_saved_run(destination, root/'second', overrides={'alpha':.2, 'fairness':False})
            self.assertEqual(second['uncertainty_audit']['alpha'], .2)
            self.assertNotIn('fairness', second['additional_metrics'])
            with patch.object(sys, 'argv', ['classical_report', str(root/'original')]):
                report_main()
            report(root/'reaudited')
            from modules.reporting.synthetic_report import load_results, build_report
            frames = load_results(root/'reaudited')
            build_report(*frames[:3])
            after = {str(p.relative_to(source)):p.read_bytes() for p in source.rglob('*') if p.is_file()}
            self.assertEqual(before, after)
            with self.assertRaises(FileExistsError):
                audit_saved_run(source, destination)
            self.assertFalse((source/'config.npy').exists())
            self.assertFalse((source/'results/group_metrics.pkl').exists())
            # Old bundles have enough information in data_info and synthetic_data.
            (source/'audit/audit_context.json').unlink()
            (source/'audit/diagnostic_rows.npz').unlink()
            old_inputs = load_audit_inputs(source)
            legacy = run_audit(old_inputs, options)
            self.assertEqual(_jsonable(result['uncertainty_audit']), _jsonable(legacy['uncertainty_audit']))

    def test_streaming_pipeline_can_be_reaudited(self):
        config = PipelineConfig(dataset_name='synthetic_data', dataset_kwargs=dict(
            seed=8, train=80, validation=40, audit=60, reference=40),
            n_models=2, store_member_probs=False, model_seed=21, device='cpu',
            model=ModelConfig(hidden_dims=(4,), dropout=0),
            train=TrainConfig(epochs=1, verbose=False))
        with tempfile.TemporaryDirectory() as tmp:
            source = Path(tmp)/'run'
            result = run_experiment(config, out=source)
            self.assertNotIn('ensemble_probs', result['pipeline_result'].ensemble_metrics)
            reaudited = run_audit(load_audit_inputs(source))
            self.assertEqual(_jsonable(result['uncertainty_audit']), _jsonable(reaudited['uncertainty_audit']))

    def test_command_line_training_and_batch_reaudit(self):
        from scripts.run_experiment import main as train_main
        from scripts.audit import main as audit_main
        from modules.pipelines.experiment_config import ROOT, read_config
        cfg,_ = read_config(ROOT/'configs/synthetic/smoke.yaml')
        cfg['conditions'] = {'baseline':{}}
        cfg['data_seeds'] = [0]
        cfg['pipeline']['n_models'] = 1
        cfg['pipeline']['train'].update(epochs=1, patience=0, verbose=False)
        cfg['pipeline']['store_member_probs'] = False
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            with patch('scripts.run_experiment.read_config', return_value=(cfg, root/'source')), \
                    patch.object(sys, 'argv', ['run_experiment', '--config', 'unused.yaml']):
                train_main()
            source = root/'source/runs/baseline/seed_0'
            self.assertTrue((source/'COMPLETE').exists())
            with patch.object(sys, 'argv', ['audit', '--experiment', str(root/'source'), '--out', str(root/'audited')]):
                audit_main()
            dest = root/'audited/runs/baseline/seed_0'
            self.assertTrue((dest/'COMPLETE').exists())
            self.assertEqual(json.loads((source/'audit/uncertainty_audit.json').read_text()),
                             json.loads((dest/'audit/uncertainty_audit.json').read_text()))



if __name__ == '__main__':
    unittest.main()
