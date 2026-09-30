"""Row-aligned interventions on loan data; no covariate/outcome generator here.

Noise is symmetric label flipping in every split. Scarcity changes training only.
All random streams are independent of the base generator and paired across conditions.
"""
from copy import deepcopy
import numpy as np
from Projects.InterFairPytorch.modules.data.synthetic_generators.loan_data import GROUPS, FEATURES

LOG2 = float(np.log(2.0))


def h(p):
    p = np.asarray(p, dtype=float)
    if np.any(~np.isfinite(p)) or np.any((p < 0) | (p > 1)):
        raise ValueError('Probabilities must be finite and in [0,1].')
    return -p*np.log(np.clip(p, np.finfo(float).tiny, 1)) - (1-p)*np.log(np.clip(1-p, np.finfo(float).tiny, 1))


def inverse_h_bits(target):
    target = np.asarray(target, dtype=float)
    if np.any(~np.isfinite(target)) or np.any((target < 0) | (target > 1)):
        raise ValueError('Entropy targets must lie in [0,1] bits.')
    lo, hi = np.zeros_like(target), np.full_like(target, .5)
    for _ in range(60):
        mid = (lo+hi)/2
        low = h(mid)/LOG2 < target
        lo, hi = np.where(low, mid, lo), np.where(low, hi, mid)
    return (lo+hi)/2


def _group(value, key):
    if value not in GROUPS:
        raise ValueError(f'{key} must be a quoted group: {GROUPS}')
    return GROUPS.index(value)


def parameters(scenario='baseline', rho=1., eta=.2, strength=.15, *, baseline_noise=.05,
               target_group='11', noise_group='00', entropy_base=.30):
    """Preserve former uncertainty patterns, now applied to deterministic loan labels."""
    values = np.array([rho, eta, strength, baseline_noise, entropy_base], dtype=float)
    if not np.isfinite(values).all() or not 0 < rho <= 1 or not 0 <= eta <= .5 or strength < 0 or not 0 <= baseline_noise <= .5 or not 0 <= entropy_base <= 1:
        raise ValueError('Invalid uncertainty probability, retention or entropy parameter.')
    g, ng = _group(target_group, 'target_group'), _group(noise_group, 'noise_group')
    noise, retain = np.full(4, baseline_noise, dtype=float), np.ones(4)
    if scenario == 'scarcity':
        retain[g] = rho
    elif scenario == 'noise':
        noise[g] = eta
    elif scenario in ('additive','interaction'):
        pattern = np.array([0,1,1,2] if scenario == 'additive' else [0,0,0,1])
        noise = inverse_h_bits(entropy_base + strength*pattern)
    elif scenario == 'cancellation':
        noise[ng], retain[g] = eta, rho
    elif scenario not in ('baseline','stripe','custom'):
        raise ValueError(f'Unknown uncertainty kind: {scenario}')
    return noise, retain


def normalize_condition(condition):
    if not isinstance(condition, dict):
        raise ValueError('uncertainty must be a mapping or null')
    cfg = deepcopy(condition)
    kind = cfg.setdefault('kind','baseline')
    common = {'kind','baseline_noise'}
    allowed = {
        'baseline':set(), 'scarcity':{'rho','target_group'}, 'noise':{'eta','target_group'},
        'cancellation':{'rho','eta','target_group','noise_group'},
        'additive':{'strength','entropy_base'}, 'interaction':{'strength','entropy_base'},
        'stripe':{'rho','half_width','slope','center','features','target_group','validation'},
        'custom':{'flip_probability','retention'},
    }
    if kind not in allowed or set(cfg)-common-allowed[kind]:
        raise ValueError(f'Unknown or inapplicable uncertainty settings: {cfg}')
    if kind in ('additive','interaction') and 'baseline_noise' in cfg:
        raise ValueError('Entropy-pattern interventions use entropy_base, not baseline_noise')
    args = {k:cfg[k] for k in ('rho','eta','strength','baseline_noise','target_group','noise_group','entropy_base') if k in cfg}
    noise, retain = parameters(kind, **args)
    if kind == 'custom':
        for key, array in [('flip_probability',noise),('retention',retain)]:
            if key in cfg:
                if not isinstance(cfg[key],dict):
                    raise ValueError(f'{key} must be a mapping of quoted group IDs')
                for group, value in cfg[key].items():
                    i = _group(group,key)
                    if not np.isfinite(value) or not (0 <= value <= .5 if key == 'flip_probability' else 0 < value <= 1):
                        raise ValueError(f'Invalid {key} for {group}')
                    array[i] = value
    if kind == 'stripe':
        if ('rho' in cfg) == ('half_width' in cfg):
            raise ValueError('Stripe requires exactly one of rho or half_width')
        features = cfg.get('features',['Income','LoanAmount'])
        if not isinstance(features,(list,tuple)) or len(features)!=2 or len(set(features))!=2 or any(f not in FEATURES[2:] for f in features):
            raise ValueError('Stripe features must be two distinct continuous loan features')
        if not np.isfinite(cfg.get('slope',-.5)) or not np.isfinite(cfg.get('center',0.)):
            raise ValueError('Stripe slope and center must be finite')
        if 'half_width' in cfg and (not np.isfinite(cfg['half_width']) or cfg['half_width']<0):
            raise ValueError('Stripe half_width must be finite and nonnegative')
        if type(cfg.get('validation',True)) is not bool:
            raise ValueError('Stripe validation must be boolean')
    return cfg, noise, retain


def stripe_region_mask(X, stripe):
    """Use saved training-only geometry on any split, without refitting."""
    z = (np.asarray(X)[:, stripe['feature_indices']] - np.asarray(stripe['mean']))/np.asarray(stripe['scale'])
    return np.abs(z[:,0]-stripe['slope']*z[:,1]-stripe['center']) < stripe['half_width']


def fit_stripe(train, cfg):
    features = cfg.get('features',['Income','LoanAmount'])
    indices = [FEATURES.index(f) for f in features]
    x = train['X'][:,indices]
    mean, scale = x.mean(axis=0), x.std(axis=0)
    if np.any(scale <= 0):
        raise ValueError('Stripe features must vary in training data')
    info = dict(features=list(features),feature_indices=indices,mean=mean.tolist(),scale=scale.tolist(),
                slope=float(cfg.get('slope',-.5)),center=float(cfg.get('center',0.)),
                group=cfg.get('target_group','11'),restrict_validation=cfg.get('validation',True),
                fitted_on='unmodified training covariates',expected_retention=None,
                width_convention='abs(z_feature_1 - slope*z_feature_2 - center) < half_width')
    target = train['group'] == _group(info['group'],'target_group')
    z = (x-mean)/scale
    distance = np.abs(z[:,0]-info['slope']*z[:,1]-info['center'])[target]
    if 'rho' in cfg:
        # Order statistic calibrated only on target-group training observations.
        keep = max(1, int(np.floor(len(distance)*cfg['rho'])))
        info['half_width'] = 0. if keep == len(distance) else float(np.sort(distance)[len(distance)-keep])
        info['requested_rho'] = float(cfg['rho'])
        info['input_mode'] = 'training_retention'
    else:
        info['half_width'] = float(cfg['half_width'])
        info['requested_rho'] = None
        info['input_mode'] = 'half_width'
    info['count_policy'] = 'strict geometric filter; boundary ties retained; validation uses frozen train geometry'
    return info


def apply_uncertainty(splits, metadata, condition, *, seed):
    """Return new arrays; leave the original loan generation untouched."""
    cfg, noise, retain = normalize_condition(condition)
    meta = deepcopy(metadata)
    stripe = fit_stripe(splits['train'],cfg) if cfg['kind']=='stripe' else None
    result = {}
    for stream,name in enumerate(('train','validation','audit','reference')):
        raw = splits[name]
        d = {k:v.copy() for k,v in raw.items()}
        # Shared uniforms make label flips nested as the group noise rate increases.
        rng = np.random.default_rng(np.random.SeedSequence([seed, 731, stream]))
        rates = noise[d['group']]
        d['y'] = np.bitwise_xor(d['y_clean'], (rng.random(len(d['y'])) < rates).astype(np.int64))
        d['p_true'] = rates+(1-2*rates)*d['y_clean']
        d['U_true'] = h(d['p_true'])
        keep = np.ones(len(d['y']),dtype=bool)
        if name == 'train':
            for g in range(4):
                ids = np.flatnonzero(d['group']==g)
                perm = np.random.default_rng(np.random.SeedSequence([seed, 947, g])).permutation(ids)
                count = max(1,int(np.floor(len(ids)*retain[g])))
                keep[perm[count:]] = False
        if stripe is not None and (name=='train' or name=='validation' and stripe['restrict_validation']):
            keep &= ~((d['group']==GROUPS.index(stripe['group'])) & stripe_region_mask(d['X'],stripe))
        if any(not np.any(keep & (d['group']==g)) for g in range(4)):
            raise ValueError(f'Intervention removed an entire group in {name}; reduce its strength or increase data size')
        result[name] = {k:v[keep] for k,v in d.items()}
    meta.update(base_scenario=metadata['scenario'],uncertainty=cfg,
                intervention_version=1,flip_probability=noise.tolist(),
                requested_retention=retain.tolist(),
                counts={name:np.bincount(d['group'],minlength=4).tolist() for name,d in result.items()},
                oracle_entropy_bits=(h(noise)/LOG2).tolist(),
                oracle_interaction_bits=float(np.dot([1,-1,-1,1],h(noise)/LOG2)),
                oracle_probability='P(Y=1 | all original loan predictors and protected group) after symmetric label flips',
                label_semantics='Y is the upstream deterministic loan decision XOR a group-specific Bernoulli flip',
                intervention_order='paired label flips on original rows, then training retention / geometric filtering')
    meta['retention'] = (np.asarray(meta['counts']['train'])/np.asarray(metadata['counts']['train'])).tolist()
    if stripe is not None:
        g=GROUPS.index(stripe['group'])
        stripe['realized_retention']={name:meta['counts'][name][g]/metadata['counts'][name][g] for name in result}
        meta['stripe']=stripe
    return result, meta
