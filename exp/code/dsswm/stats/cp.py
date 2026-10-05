"""Clopper-Pearson exact binomial interval."""
from scipy.stats import beta


def clopper_pearson(k: int, n: int, alpha: float = 0.05):
    if n == 0:
        return 0.0, 1.0
    lo = 0.0 if k == 0 else float(beta.ppf(alpha / 2, k, n - k + 1))
    hi = 1.0 if k == n else float(beta.ppf(1 - alpha / 2, k + 1, n - k))
    return lo, hi


# ----------------------------------------------------------------------------------------------- r4 additions
# One-sided exact Clopper-Pearson bounds (round 4, methodology §2). The two-sided `clopper_pearson` above is kept
# unchanged so that every r3 number stays reproducible.

def cp_upper_one_sided(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) exact upper bound: Beta^{-1}(1 - alpha; k + 1, n - k). Returns 1 when k == n."""
    k, n = int(k), int(n)
    if n <= 0:
        return 1.0
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if k == n:
        return 1.0
    return float(beta.ppf(1.0 - alpha, k + 1, n - k))


def cp_lower_one_sided(k: int, n: int, alpha: float = 0.05) -> float:
    """One-sided (1 - alpha) exact lower bound: Beta^{-1}(alpha; k, n - k + 1). Returns 0 when k == 0."""
    k, n = int(k), int(n)
    if n <= 0:
        return 0.0
    if not 0 <= k <= n:
        raise ValueError(f"need 0 <= k <= n, got k={k}, n={n}")
    if k == 0:
        return 0.0
    return float(beta.ppf(alpha, k, n - k + 1))
