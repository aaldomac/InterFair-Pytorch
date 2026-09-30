import numpy as np
from modules.data.synthetic_uncertainty import GROUPS, LOG2

def audit_stripe_regions(ensemble, groups, X, S, metadata):
    """Descriptive regional means; empty regions use null, never zero.
    These are not simultaneous intervals or additional fairness maxima.
    Each condition's region follows its own width; compare group-wide metrics
    on the fixed audit population when comparing different stripe widths.
    """
    from modules.data.synthetic_uncertainty import stripe_region_mask
    if 'stripe' not in metadata:
        return []
    info = metadata['stripe']
    g = GROUPS.index(info['group'])
    inside = stripe_region_mask(X, S, info['half_width'], info['slope'])
    groups = np.asarray(groups)
    oracle = metadata['oracle_entropy_bits'][g]*LOG2
    rows = []
    for region, mask in [('inside', inside), ('outside', ~inside)]:
        mask = mask & (groups == g)
        count = int(mask.sum())
        row = dict(group=info['group'], region=region, support=count,
                   half_width=info['half_width'], slope=info['slope'],
                   expected_retention=info['expected_retention'], oracle_alea=oracle)
        for name, key in [('alea','aleatoric_uncertainty'), ('epis','epistemic_uncertainty'),
                          ('tot','predictive_entropy')]:
            if count > 0:
                row[name] = float(np.asarray(ensemble[key][mask].mean()))
            else:
                row[name] = None
        row['alea_error'] = row['alea']-oracle if count else None
        rows.append(row)
    return rows
