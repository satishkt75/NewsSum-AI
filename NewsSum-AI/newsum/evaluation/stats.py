"""Paired significance and multiple-testing corrections."""
import numpy as np


def paired_bootstrap(a, b, iterations=1000, seed=42):
    """One-sided paired bootstrap probability that A does not improve over B."""
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    if a.shape != b.shape or a.size == 0:
        raise ValueError('Paired vectors must be equal-sized and nonempty')
    differences = a - b
    rng = np.random.default_rng(seed)
    draws = rng.integers(0, a.size, size=(iterations, a.size))
    means = differences[draws].mean(axis=1)
    return {'mean_difference': float(differences.mean()),
            'ci95': [float(x) for x in np.percentile(means, [2.5, 97.5])],
            'p_one_sided': float((np.count_nonzero(means <= 0)+1)/(iterations+1))}


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    order = np.argsort(p)
    adjusted = np.empty(len(p))
    running = 0.0
    for rank, idx in enumerate(order):
        running = max(running, min(1.0, float(p[idx])*(len(p)-rank)))
        adjusted[idx] = running
    return adjusted.tolist()


def seeds_summary(numbers):
    numbers = np.asarray(numbers, dtype=float)
    return {'mean': float(numbers.mean()), 'sd': float(numbers.std(ddof=1)) if len(numbers)>1 else 0.0}
