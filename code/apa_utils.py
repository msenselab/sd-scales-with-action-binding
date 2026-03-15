"""Utilities for APA-formatted statistical reporting."""

import numpy as np
from typing import Dict, Tuple, Optional


def format_pvalue(p: float, threshold: float = 0.001) -> str:
    """Format p-value in APA style.

    Args:
        p: p-value to format
        threshold: threshold for reporting as '< threshold'

    Returns:
        Formatted p-value string (e.g., 'p = .023' or 'p < .001')
    """
    if p < threshold:
        return f"p < {threshold:.3f}".replace('0.', '.')
    else:
        return f"p = {p:.3f}".replace('0.', '.')


def format_ttest(t: float, df: int, p: float, d: Optional[float] = None,
                 two_tailed: bool = True) -> str:
    """Format t-test results in APA style.

    Args:
        t: t-statistic
        df: degrees of freedom
        p: p-value
        d: Cohen's d effect size (optional)
        two_tailed: whether test is two-tailed

    Returns:
        APA-formatted string (e.g., 't(23) = 4.56, p = .001, d = 0.93')
    """
    result = f"t({df}) = {t:.2f}, {format_pvalue(p)}"
    if d is not None:
        result += f", d = {d:.2f}"
    return result


def format_anova(f: float, df1: int, df2: int, p: float,
                 eta_sq: Optional[float] = None,
                 partial_eta_sq: Optional[float] = None) -> str:
    """Format ANOVA results in APA style.

    Args:
        f: F-statistic
        df1: numerator degrees of freedom
        df2: denominator degrees of freedom
        p: p-value
        eta_sq: eta-squared effect size
        partial_eta_sq: partial eta-squared effect size

    Returns:
        APA-formatted string (e.g., 'F(3, 69) = 6.74, p < .001, ηp² = .17')
    """
    result = f"F({df1}, {df2}) = {f:.2f}, {format_pvalue(p)}"

    if partial_eta_sq is not None:
        result += f", ηp² = {partial_eta_sq:.2f}"
    elif eta_sq is not None:
        result += f", η² = {eta_sq:.2f}"

    return result


def format_correlation(r: float, n: int, p: float) -> str:
    """Format correlation in APA style.

    Args:
        r: correlation coefficient
        n: sample size
        p: p-value

    Returns:
        APA-formatted string (e.g., 'r(22) = .45, p = .032')
    """
    df = n - 2
    return f"r({df}) = {r:.2f}, {format_pvalue(p)}"


def format_regression(b: float, se: float, t: float, p: float,
                     beta: Optional[float] = None) -> str:
    """Format regression coefficient in APA style.

    Args:
        b: unstandardized coefficient
        se: standard error
        t: t-statistic
        p: p-value
        beta: standardized coefficient (optional)

    Returns:
        APA-formatted string
    """
    result = f"b = {b:.3f}, SE = {se:.3f}, t = {t:.2f}, {format_pvalue(p)}"
    if beta is not None:
        result += f", β = {beta:.2f}"
    return result


def format_lmm_fixed_effect(name: str, coef: float, se: float,
                            z: float, p: float) -> str:
    """Format linear mixed model fixed effect in APA style.

    Args:
        name: effect name
        coef: coefficient
        se: standard error
        z: z-statistic
        p: p-value

    Returns:
        APA-formatted string
    """
    return f"{name}: b = {coef:.3f}, SE = {se:.3f}, z = {z:.2f}, {format_pvalue(p)}"


def format_mean_sd(mean: float, sd: float, unit: str = "") -> str:
    """Format mean and SD in APA style.

    Args:
        mean: mean value
        sd: standard deviation
        unit: unit of measurement (e.g., 's' for seconds)

    Returns:
        APA-formatted string (e.g., 'M = 1.05 s, SD = 0.15 s')
    """
    unit_str = f" {unit}" if unit else ""
    return f"M = {mean:.2f}{unit_str}, SD = {sd:.2f}{unit_str}"


def format_ci(lower: float, upper: float, ci_level: int = 95) -> str:
    """Format confidence interval in APA style.

    Args:
        lower: lower bound
        upper: upper bound
        ci_level: confidence level (default 95)

    Returns:
        APA-formatted string (e.g., '95% CI [0.23, 0.45]')
    """
    return f"{ci_level}% CI [{lower:.2f}, {upper:.2f}]"


def cohens_d_paired(diff: np.ndarray) -> float:
    """Calculate Cohen's d for paired samples.

    Args:
        diff: array of difference scores

    Returns:
        Cohen's d effect size
    """
    return np.mean(diff) / np.std(diff, ddof=1)


def cohens_d_independent(group1: np.ndarray, group2: np.ndarray) -> float:
    """Calculate Cohen's d for independent samples (pooled SD).

    Args:
        group1: first group values
        group2: second group values

    Returns:
        Cohen's d effect size
    """
    n1, n2 = len(group1), len(group2)
    var1, var2 = np.var(group1, ddof=1), np.var(group2, ddof=1)
    pooled_sd = np.sqrt(((n1 - 1) * var1 + (n2 - 1) * var2) / (n1 + n2 - 2))
    return (np.mean(group1) - np.mean(group2)) / pooled_sd
