"""Reusable subgroup metric computation and componentwise log-ratio reduction."""
import numpy as np


def maximum_log_ratio(values, groups=None):
    """Reduce [G] or [G,C] nonnegative values without clipping or dropping groups.

    Negative/infinite inputs raise. NaN means undefined and invalidates the
    entire result. Fewer than two groups is undefined. Shared zeros contribute
    zero (including an all-zero component); positive/zero contributes infinity.
    The maximizing pair is ordered numerator, denominator, then component.
    """
    x = np.asarray(values, dtype=float)
    if x.ndim == 1:
        x = x[:, None]
    if x.ndim != 2 or not x.shape[1]:
        raise ValueError('Expected scalar or matching vector values per group')
    ids = list(range(len(x))) if groups is None else list(groups)
    if len(ids) != len(x) or len(set(ids)) != len(ids):
        raise ValueError('Group IDs must be distinct and match values')
    if np.isinf(x).any() or (x < 0).any():
        raise ValueError('Metric values must be nonnegative and finite or NaN (undefined)')
    result = dict(epsilon=None, aggregate=None, groups=ids, group_values=np.where(np.isnan(x), None, x).tolist(),
                  maximizing_pair=None, component=None, status='undefined')
    if len(x) < 2 or np.isnan(x).any():
        result['reason'] = 'fewer_than_two_groups' if len(x) < 2 else 'undefined_subgroup_value'
        return result
    lo, hi = x.min(0), x.max(0)
    gaps = np.zeros(x.shape[1])
    positive = lo > 0
    gaps[positive] = np.log(hi[positive]) - np.log(lo[positive])
    gaps[(lo == 0) & (hi > 0)] = np.inf
    k = int(np.argmax(gaps))
    # Deterministic ties independent of input ordering, including all-equal data.
    order = sorted(range(len(ids)), key=lambda i: (type(ids[i]).__name__, repr(ids[i])))
    a = next(i for i in order if x[i,k] == hi[k])
    b = next(i for i in order if i != a and x[i,k] == lo[k])
    result.update(epsilon=float(gaps[k]), aggregate=float(gaps[k]),
                  maximizing_pair=[ids[a], ids[b]], component=k, status='ok')
    return result


def class_probability(p, y, positive_class):
    return p.mean(0)


def tpr(p, y, positive_class):
    selected = p[y == positive_class, positive_class]
    return selected.mean() if len(selected) else np.nan


def fpr(p, y, positive_class):
    selected = p[y != positive_class, positive_class]
    return selected.mean() if len(selected) else np.nan


def accuracy(p, y, positive_class):
    return p[np.arange(len(y)), y].mean()


SUBGROUP_METRICS = dict(class_probability=class_probability, tpr=tpr, fpr=fpr, accuracy=accuracy)


def subgroup_metric_values(probabilities, labels, groups, metric='class_probability', *,
                          positive_class=1, expected_groups=None):
    """Compute scalar/vector values, accepting a registered name or callable(p,y,k).

    p is already the ensemble mean or hardened ensemble mean. Empty declared
    groups and unsupported conditional rates produce NaN, never pseudo-data.
    Multiclass TPR/FPR are one-vs-rest for positive_class. Soft accuracy is the
    expected accuracy of a categorical draw from p, not argmax accuracy.
    """
    p, y, g = np.asarray(probabilities), np.asarray(labels), np.asarray(groups)
    if p.ndim != 2 or y.shape != (len(p),) or g.shape != y.shape:
        raise ValueError('Inputs must be row aligned')
    if p.shape[1] < 2 or not np.isfinite(p).all() or (p < 0).any() or (p > 1).any() or not np.allclose(p.sum(1), 1):
        raise ValueError('Expected finite probability rows summing to one')
    if not np.issubdtype(y.dtype, np.integer) or (y < 0).any() or (y >= p.shape[1]).any():
        raise ValueError('Invalid class labels')
    if not 0 <= positive_class < p.shape[1]:
        raise ValueError('Invalid positive_class')
    fn = SUBGROUP_METRICS[metric] if isinstance(metric, str) else metric
    ids = sorted(set(g.tolist()) | set([] if expected_groups is None else expected_groups))
    observed = {gid: np.asarray(fn(p[g == gid], y[g == gid], positive_class), dtype=float)
                for gid in ids if np.any(g == gid)}
    shape = next(iter(observed.values())).shape if observed else (() if metric != 'class_probability' else (p.shape[1],))
    values = [observed.get(gid, np.full(shape, np.nan)) for gid in ids]
    return np.asarray(values).reshape((len(ids),) + shape), ids


def metric_log_ratio(probabilities, labels, groups, metric='class_probability', *,
                     positive_class=1, offset=0., expected_groups=None):
    """Add explicit offset delta to M(g), then compare (M(g)+delta)/(M(h)+delta).

    This is not a count prior. Missing rates remain undefined even with offset.
    """
    if not np.isfinite(offset) or offset < 0:
        raise ValueError('offset must be finite and nonnegative')
    values, ids = subgroup_metric_values(probabilities, labels, groups, metric,
        positive_class=positive_class, expected_groups=expected_groups)
    maximum_log_ratio(values, ids)  # Validate raw values before smoothing.
    result = maximum_log_ratio(values + offset, ids)
    result.update(metric=metric if isinstance(metric, str) else getattr(metric, '__name__', 'callable'),
                  raw_group_values=np.where(np.isnan(values), None, values).tolist(), offset=float(offset),
                  convention='generalized_metric_log_ratio')
    return result
