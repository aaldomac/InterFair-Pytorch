"""One audit entry point for in-memory and saved, row-aligned predictions."""
from dataclasses import dataclass, field
from typing import Any, Mapping
import numpy as np

from modules.metrics.ensemble_quality import ensemble_quality
from modules.metrics.uncertainty_audit import audit_uncertainties, add_group_entropy_metrics

AUDIT_SCHEMA_VERSION = 2

@dataclass(frozen=True)
class AuditConfig:
    enabled: bool = True
    confidence_alpha: float = .05
    interaction_weights: Mapping | None = None
    fairness: bool = False
    distribution: bool = False
    smoothing_alpha: float = 1.
    positive_class: int = 1
    synthetic_diagnostics: bool = True
    include_pairs: bool = True
    df_soft: bool = False

    def __post_init__(self):
        for name in ('enabled', 'fairness', 'distribution', 'synthetic_diagnostics', 'include_pairs', 'df_soft'):
            if type(getattr(self, name)) is not bool:
                raise ValueError(f'audit.{name} must be boolean')
        if not 0 < self.confidence_alpha < 1:
            raise ValueError('confidence_alpha must be in (0, 1)')
        if not np.isfinite(self.smoothing_alpha) or self.smoothing_alpha <= 0:
            raise ValueError('smoothing_alpha must be finite and positive')
        if type(self.positive_class) is not int or self.positive_class < 0:
            raise ValueError('positive_class must be a nonnegative integer')

    @classmethod
    def from_mapping(cls, options=None):
        if isinstance(options, cls):
            return options
        values = dict(options or {})
        if 'alpha' in values:  # Existing YAML files remain supported.
            old = values.pop('alpha')
            if 'confidence_alpha' in values and old != values['confidence_alpha']:
                raise ValueError('alpha and confidence_alpha disagree')
            values['confidence_alpha'] = old
        unknown = set(values) - set(cls.__dataclass_fields__)
        if unknown:
            raise ValueError(f'Unknown audit settings: {sorted(unknown)}')
        return cls(**values)


@dataclass
class AuditInputs:
    ensemble: Mapping[str, np.ndarray]
    labels: np.ndarray
    groups: np.ndarray
    group_names: Mapping[int, str] = field(default_factory=dict)
    binary: bool = True
    threshold: float = .5
    num_classes: int = 2
    metadata: Mapping[str, Any] = field(default_factory=dict)
    provenance: Mapping[str, Any] = field(default_factory=dict)
    features: np.ndarray | None = None
    protected: np.ndarray | None = None 


def audit_inputs_from_result(result):
    data, config = result.data, result.config
    meta = data.loaded.metadata
    seed = config.model_seed if config.model_seed is not None else config.split.seed
    return AuditInputs(
        ensemble=result.ensemble_metrics,
        labels=data.test_df[data.spec.label_col].to_numpy(dtype=int),
        groups=data.test_df[data.spec.group_id_col].to_numpy(dtype=int),
        group_names={int(g): str(name) for name, g in data.loaded.group_id.items()},
        binary=data.binary, threshold=config.train.threshold, num_classes=data.num_classes,
        metadata={key: meta[key] for key in ('oracle_entropy_bits', 'stripe', 'feature_names') if key in meta},
        provenance=dict(data_seed=meta.get('seed', config.split.seed),
                        model_seeds=[seed+i for i in range(config.n_models)],
                        method='deep_ensemble' if config.n_models > 1 else 'single_model'),
        features=data.test_df[meta['feature_names']].to_numpy() if 'stripe' in meta else None,
        protected=data.test_df[['S1', 'S2']].to_numpy() if 'stripe' in meta else None,
    )


def run_audit(inputs: AuditInputs, config=None):
    """Compute metrics without inference, data loading, or writing files."""
    options = AuditConfig.from_mapping(config)
    print(f"options.df_soft in run_audit= {options.df_soft}")
    ensemble, labels, groups = inputs.ensemble, inputs.labels, inputs.groups
    probs = np.asarray(ensemble['mean_probs'])
    if probs.ndim != 2 or probs.shape[1] != inputs.num_classes:
        raise ValueError('num_classes does not match saved mean probabilities')
    if not 0 <= inputs.threshold <= 1 or options.positive_class >= inputs.num_classes:
        raise ValueError('Invalid threshold or positive class')
    metrics, by_group = ensemble_quality(ensemble, labels, groups,
        threshold=inputs.threshold, group_names=inputs.group_names, binary=inputs.binary)
    audit, additional = None, {}
    decompositions = None
    if options.enabled or options.distribution:
        from modules.metrics.distribution_decomposition_metrics import group_entropy_decompositions
        decompositions = group_entropy_decompositions(probs, groups)
    if options.enabled:
        audit = audit_uncertainties(ensemble, groups, alpha=options.confidence_alpha,
            group_names=inputs.group_names, interaction_weights=options.interaction_weights,
            num_classes=inputs.num_classes, include_pairs=options.include_pairs)
        audit['group_mix_entropy'] = {g: dec.entropy_of_mean for g, dec in decompositions.items()}
        audit['group_het_entropy'] = {g: dec.entropy_of_mean-dec.mean_entropy for g, dec in decompositions.items()}
        audit['group_het_definition'] = 'H(mean_x predictive probability) - mean_x H(predictive probability)'
        audit = add_group_entropy_metrics(audit,interaction_weights=options.interaction_weights,)
        if options.synthetic_diagnostics:
            meta = inputs.metadata
            if 'oracle_entropy_bits' in meta:
                oracle = np.asarray(meta['oracle_entropy_bits'], dtype=float)*np.log(2)
                ids = np.asarray(audit['group_ids'], dtype=int)
                if np.any(ids < 0) or np.any(ids >= len(oracle)):
                    raise ValueError('Oracle entropy metadata does not cover observed group IDs')
                selected = oracle[ids]
                audit['oracle_entropy_nats'] = selected.tolist()
                audit['alea_minus_oracle'] = (np.asarray(audit['means'])[:, 0]-selected).tolist()
            if 'stripe' in meta:
                if inputs.features is None:
                    raise ValueError('Stripe diagnostics require saved row-aligned features; disable synthetic_diagnostics to skip')
                from modules.metrics.synthetic_diagnostics import audit_stripe_regions
                audit['stripe_regions'] = audit_stripe_regions(ensemble, groups,
                    inputs.features, inputs.protected, meta)
        audit.update(inputs.provenance)
        audit.update(schema_version=AUDIT_SCHEMA_VERSION,
            brier_convention='class-1 squared error' if inputs.num_classes == 2 else 'sum of classwise squared errors')
    if options.fairness:
        from modules.metrics.fairness_metrics import (
            evaluate_ensemble_fairness, performance_fairness, classical_fairness_summary)
        kwargs = dict(binary=inputs.binary, threshold=inputs.threshold,
                      positive_class=options.positive_class, alpha=options.smoothing_alpha)
        fairness = evaluate_ensemble_fairness(ensemble, labels, groups, soft=options.df_soft, **kwargs)
        performance = performance_fairness(ensemble, labels, groups, **kwargs)
        additional.update(fairness=fairness, performance_fairness=performance,
                          classical_summary=classical_fairness_summary(fairness, performance))
    if options.distribution:
        from modules.metrics.distribution_decomposition_metrics import pairwise_group_mean_distribution_matrices
        additional['distribution'] = dict(group_entropy_decompositions=decompositions,
            pairwise_group_mean_distribution_matrices=(
                pairwise_group_mean_distribution_matrices(probs, groups) if options.include_pairs else {}))
    return dict(uncertainty_audit=audit, predictive_metrics=metrics,
                group_predictive_metrics=by_group, additional_metrics=additional)