"""Shared statistics helpers for profilers."""

from __future__ import annotations


def percentile(data: list[float], p: float) -> float:
    """Linear-interpolation percentile (numpy-style).

    Avoids the biased ``int(n * p / 100)`` nearest-rank that returns the max
    for p50 when n=2.
    """
    if not data:
        return 0.0
    s = sorted(float(x) for x in data)
    if len(s) == 1:
        return s[0]
    p = max(0.0, min(100.0, float(p)))
    k = (len(s) - 1) * (p / 100.0)
    f = int(k)
    c = min(f + 1, len(s) - 1)
    if f == c:
        return s[f]
    return s[f] + (s[c] - s[f]) * (k - f)
