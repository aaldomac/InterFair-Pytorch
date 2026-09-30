"""Code-faithful Kanubala & Valera (AIES 2025) loan data, plus a UQ adapter.

Adapted from KANUBALAD/On-the-Misalignment-Between-Legal-Notions-and-Statistical-
Metrics-of-Intersectional-Fairness, commit 642d228485b51efc25ec127990ed3160b6a81e98.
Upstream MIT license: see LICENSE-kanubala.txt in the delivered package.

Do not interpret the upstream sigmoid score as a Bernoulli label probability.
Y is a deterministic threshold decision given all seven observed predictors.
"""
from copy import deepcopy
import argparse
import json
from pathlib import Path
import numpy as np

UPSTREAM_COMMIT = '642d228485b51efc25ec127990ed3160b6a81e98'
UPSTREAM_URL = ('https://github.com/KANUBALAD/'
                'On-the-Misalignment-Between-Legal-Notions-and-Statistical-Metrics-of-Intersectional-Fairness')
SCENARIOS = ('no_bias', 'single', 'additive', 'intersectional', 'compounded')
FEATURES = ['Gender', 'Race', 'Education', 'Income', 'Savings', 'LoanAmount', 'Duration']
GROUPS = ['00', '01', '10', '11']

def scenario_config(scenario='no_bias', *, seed=42, sample_size=10000, overrides=None):
    """Load the complete upstream preset; allow explicit, recorded code extensions.

    Coefficient dictionaries are patched by key. Overrides cannot silently
    change sample_size/random_seed, which are controlled by the explicit arguments.
    """
    scenario = scenario.removeprefix('loan_')
    if scenario not in SCENARIOS:
        raise ValueError(f'Unknown scenario {scenario!r}; must be one of {SCENARIOS}')
    with Path(__file__).with_name('loan_presets.json').open() as f:
        cfg = json.load(f)[scenario]
    if overrides is not None:
        if not isinstance(overrides, dict):
            raise TypeError(f'overrides must be a dict, not {type(overrides).__name__}')
        for key, value in overrides.items():
            if key not in cfg or key in ('sample_size', 'random_seed'):
                raise ValueError(f'Cannot override {key!r}; only {list(cfg.keys())} are allowed')
            if isinstance(cfg[key], dict):
                if not isinstance(value, dict) or set(value)-set(cfg[key]):
                    raise ValueError(f'Cannot override {key!r} with {value!r}; must be a dict with keys {list(cfg[key].keys())}')
                cfg[key].update(deepcopy(value))
            else:
                cfg[key] = deepcopy(value)
    if type(seed) is not int or not 0 <= seed < 2**32:
        raise ValueError(f'seed must be a 32-bit unsigned integer, not {seed!r}')
    if type(sample_size) is not int or sample_size <= 0:
        raise ValueError(f'sample_size must be a positive integer, not {sample_size!r}')
    cfg.update(random_seed=seed, sample_size=sample_size)
    for key in ('prob_gender', 'prob_race', 'beta', 'beta_I', 'beta_S', 'gamma', 'eta', 'delta'):
        if not isinstance(cfg[key], (int, float)) or not np.isfinite(cfg[key]):
            raise ValueError(f'{key} must be finite.')
    if not 0 < cfg['prob_gender'] < 1 or not 0 < cfg['prob_race'] < 1 or cfg['eta'] < 0:
        raise ValueError('Both group probabilities must be in (0,1); eta must be nonnegative.')
    # Discrimation variables
    for key in ('thetas', 'beta_coef', 'rhos', 'kappas', 'nus', 'lambdas'):
        if not all(isinstance(v, (int, float)) and np.isfinite(v) for v in cfg[key].values()):
            raise ValueError(f'{key} values must be finite numbers.')
    return cfg


def generate_loan_dataset(scenario='no_bias', *, seed=42, sample_size=10000, overrides=None):
    """Reproduce all upstream generator outputs without changing global RNG state.

    RandomState intentionally matches upstream np.random.seed + legacy draws.
    Both outcome-noise draws are intentionally preserved for RNG/output parity,
    although neither contributes to Y. beta is also unused in upstream code.
    """
    import pandas as pd
    c = scenario_config(scenario, seed=seed, sample_size=sample_size, overrides=overrides)
    rng = np.random.RandomState(seed)
    G = rng.binomial(1, c['prob_gender'], sample_size)
    R = rng.binomial(1, c['prob_race'], sample_size)
    UE = rng.normal(0, .25, sample_size)
    UI = rng.normal(0, 4, sample_size)
    US = rng.normal(0, 5, sample_size)
    UL = rng.normal(0, 10, sample_size)
    UD = rng.normal(0, 9, sample_size)
    rng.normal(0, c['eta'], sample_size)  # First unused upstream outcome-noise draw.
    def effect(key):
        d = c[key]
        return d['G']*G + d['R']*R + d['GR']*(G*R)
    E = -.5 + c['lambdas']['E']*effect('thetas') + UE
    I = -4 + 3*E + c['lambdas']['I']*effect('beta_coef') + UI
    S = -4 + 1.5*np.where(I > 0, I, 0) + US
    L = 1 - c['beta_I']*I - c['beta_S']*S + c['lambdas']['L']*effect('rhos') + UL
    D = -1 - c['beta_I']*I + L + c['lambdas']['D']*effect('kappas') + UD
    alpha = np.where((I > 0) & (S > 0), 1, -1)
    UY = rng.normal(0, c['eta'], sample_size)  # Second unused draw is returned upstream.
    bias = c['lambdas']['Y']*effect('nus')
    odds = 15.0 + c['delta']*(-L-D) + .3*(I+S+alpha*I*S)
    with np.errstate(over='ignore'):
        score = 1/(1+np.exp(-odds))
    threshold = .5+c['gamma']*bias
    Y = (score <= threshold).astype(int)
    frame = pd.DataFrame(dict(Gender=G, Race=R, Education=E, Income=I,
                              Savings=S, LoanAmount=L, Duration=D, Y=Y))
    return dict(df_array=np.vstack([G,R,E,I,S,L,D,Y]).T,
                noises_U=np.vstack([UE,UI,US,UL,UD,UY]).T,
                data_df=frame, thresholds=threshold, probabilities=score,
                config=c)


def _as_split(frame, indices, row_offset=0):
    """Map upstream rows to the existing synthetic NPZ contract."""
    part = frame.iloc[indices]
    sensitive = part[['Gender', 'Race']].to_numpy(dtype=np.int8)
    y = part.Y.to_numpy(dtype=np.int64)
    return dict(X=part[FEATURES].to_numpy(dtype=np.float64), y=y, S=sensitive,
                group=(2*sensitive[:, 0]+sensitive[:, 1]).astype(np.int64),
                y_clean=y.copy(), p_true=y.astype(float), U_true=np.zeros(len(y)),
                row_id=np.asarray(indices, dtype=np.int64)+row_offset)


def generate_loan_splits(*, seed=42, scenario='loan_no_bias', train=6000,
                         validation=1500, audit=2500, reference=10000,
                         loan_overrides=None):
    """Original sampled population + fixed splits for the existing UQ pipeline.

    Counts are TOTALS, not per-group. train+validation+audit=10000 reproduces
    the upstream default population. Audit indices match sklearn's unstratified
    train_test_split(..., test_size=audit/N, random_state=seed) when that fraction
    produces the same integer count. Validation is an additional split, absent
    from the original logistic-regression experiment. Reference is independent.
    """
    sizes = dict(train=train, validation=validation, audit=audit, reference=reference)
    if any(type(n) is not int or n < 1 for n in sizes.values()):
        raise ValueError('All loan split sizes must be positive integer TOTAL counts.')
    n = train+validation+audit
    generated = generate_loan_dataset(scenario, seed=seed, sample_size=n, overrides=loan_overrides)
    frame = generated['data_df']
    perm = np.random.RandomState(seed).permutation(n)
    audit_ids, remaining = perm[:audit], perm[audit:]
    # Reuse the upstream 75% development pool; reserve validation inside it.
    dev_perm = np.random.RandomState(seed ^ 0xA5A5A5A5).permutation(len(remaining))
    val_ids = remaining[dev_perm[:validation]]
    train_ids = remaining[dev_perm[validation:]]
    reference_seed = seed ^ 0x9E3779B9
    ref = generate_loan_dataset(scenario, seed=reference_seed,
                               sample_size=reference, overrides=loan_overrides)['data_df']
    splits = {name: _as_split(frame, ids) for name,ids in
              [('train',train_ids),('validation',val_ids),('audit',audit_ids)]}
    splits['reference'] = _as_split(ref, np.arange(reference), row_offset=n)
    counts = {name: np.bincount(d['group'], minlength=4).tolist() for name,d in splits.items()}
    if any(min(c) == 0 for c in counts.values()):
        raise ValueError('A loan split has an empty intersectional group; increase its size.')
    pg, pr = generated['config']['prob_gender'], generated['config']['prob_race']
    meta = dict(seed=seed, scenario='loan_'+scenario.removeprefix('loan_'),
                generator='kanubala_2025_repository', source_url=UPSTREAM_URL,
                source_commit=UPSTREAM_COMMIT, replication_target='repository code, not paper Eq.10',
                loan_config=generated['config'], original_sample_size=n,
                reference_seed=reference_seed, requested_total=sizes,
                group_order=GROUPS, group_meaning={'S1':'Gender','S2':'Race'},
                feature_names=FEATURES, includes_protected_in_X=True,
                oracle_conditioning='all seven original predictors, including Gender and Race',
                oracle_probability='deterministic threshold indicator, NOT upstream sigmoid score',
                oracle_entropy_bits=[0.]*4, oracle_interaction_bits=0.,
                retention=[1.]*4, counts=counts, entropy_units='nats',
                population_pg=dict(zip(GROUPS,[(1-pg)*(1-pr),(1-pg)*pr,pg*(1-pr),pg*pr])),
                split_protocol='unstratified held-out audit; extra validation within development pool',
                label_semantics='Y copied unchanged from repository; no approve/deny relabeling',
                numpy_version=np.__version__)
    return splits, meta


def main():
    """Export the unsplit upstream population CSV for direct replication checks."""
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--scenario', choices=SCENARIOS, default='no_bias')
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--sample-size', type=int, default=10000)
    p.add_argument('--out', type=Path, required=True)
    a = p.parse_args()
    if a.out.exists():
        p.error(f'Refusing to overwrite {a.out}')
    result = generate_loan_dataset(a.scenario, seed=a.seed, sample_size=a.sample_size)
    a.out.parent.mkdir(parents=True, exist_ok=True)
    result['data_df'].to_csv(a.out, index=False)
    print(f'Saved {a.out}; source commit {UPSTREAM_COMMIT}')


if __name__ == '__main__':
    main()
