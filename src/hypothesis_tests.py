"""Statistical hypothesis testing utilities for insurance risk analytics."""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.stats import chi2_contingency, ttest_ind, mannwhitneyu
from typing import Tuple, Dict, Any


# ── KPI computation ──────────────────────────────────────────────────────────

def claim_frequency(df: pd.DataFrame) -> float:
    """Return proportion of policies with at least one claim."""
    return (df["TotalClaims"] > 0).mean()


def claim_severity(df: pd.DataFrame) -> pd.Series:
    """Return claim amounts for policies that had a claim (severity values)."""
    return df.loc[df["TotalClaims"] > 0, "TotalClaims"]


def margin(df: pd.DataFrame) -> pd.Series:
    """Return per-policy margin: TotalPremium − TotalClaims."""
    return df["TotalPremium"] - df["TotalClaims"]


# ── Group selection ───────────────────────────────────────────────────────────

def select_groups(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """Filter df to two named groups; return (group_a_df, group_b_df).

    Args:
        df: Full dataset.
        group_col: Column name to split on.
        group_a: Label for the control group.
        group_b: Label for the test group.

    Returns:
        Tuple of (control DataFrame, test DataFrame).
    """
    a = df[df[group_col] == group_a].copy()
    b = df[df[group_col] == group_b].copy()
    if a.empty or b.empty:
        raise ValueError(
            f"One or both groups not found in '{group_col}': {group_a!r}, {group_b!r}"
        )
    return a, b


def group_summary(
    df: pd.DataFrame,
    group_col: str,
    group_a: str,
    group_b: str,
) -> pd.DataFrame:
    """Compute KPI summary table for two groups.

    Args:
        df: Full dataset.
        group_col: Column to split on.
        group_a: Control group label.
        group_b: Test group label.

    Returns:
        DataFrame with one row per group: n, claim_freq, mean_severity, mean_margin.
    """
    rows = []
    for label in [group_a, group_b]:
        sub = df[df[group_col] == label]
        sev = claim_severity(sub)
        rows.append({
            "group": label,
            "n": len(sub),
            "claim_freq": round(claim_frequency(sub), 6),
            "mean_severity": round(sev.mean(), 2) if len(sev) > 0 else 0.0,
            "mean_margin": round(margin(sub).mean(), 4),
            "n_claims": int((sub["TotalClaims"] > 0).sum()),
        })
    return pd.DataFrame(rows).set_index("group")


# ── Statistical tests ────────────────────────────────────────────────────────

def chi_squared_frequency_test(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Chi-squared test of independence for claim frequency.

    Builds a 2×2 contingency table (group × claimed/not-claimed) and tests
    whether the proportion of policies with a claim differs between groups.

    Args:
        df_a: Control group DataFrame.
        df_b: Test group DataFrame.
        alpha: Significance level (default 0.05).

    Returns:
        Dict with chi2, p_value, dof, decision, effect_size (Cramér's V).
    """
    claimed_a = (df_a["TotalClaims"] > 0).sum()
    not_claimed_a = len(df_a) - claimed_a
    claimed_b = (df_b["TotalClaims"] > 0).sum()
    not_claimed_b = len(df_b) - claimed_b

    table = np.array([[claimed_a, not_claimed_a], [claimed_b, not_claimed_b]])
    chi2, p_value, dof, _ = chi2_contingency(table, correction=False)

    # Cramér's V as effect size
    n = table.sum()
    cramers_v = np.sqrt(chi2 / (n * (min(table.shape) - 1)))

    n_a = table[0].sum()
    n_b = table[1].sum()

    return {
        "test": "Chi-squared",
        "kpi": "Claim Frequency",
        "statistic": round(chi2, 4),
        "p_value": float(p_value),
        "dof": dof,
        "effect_size": round(cramers_v, 6),
        "effect_label": "Cramér's V",
        "decision": "Reject H₀" if p_value < alpha else "Fail to reject H₀",
        "significant": p_value < alpha,
        "freq_a": round(claimed_a / n_a, 6),
        "freq_b": round(claimed_b / n_b, 6),
    }


def t_test_severity(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Welch's t-test on claim severity (policies with claims only).

    Args:
        df_a: Control group DataFrame.
        df_b: Test group DataFrame.
        alpha: Significance level (default 0.05).

    Returns:
        Dict with t-statistic, p_value, decision, and Cohen's d effect size.
    """
    sev_a = claim_severity(df_a).values
    sev_b = claim_severity(df_b).values

    if len(sev_a) < 2 or len(sev_b) < 2:
        raise ValueError("Insufficient claims data for severity t-test.")

    t_stat, p_value = ttest_ind(sev_a, sev_b, equal_var=False)

    # Cohen's d
    pooled_std = np.sqrt((sev_a.std() ** 2 + sev_b.std() ** 2) / 2)
    cohens_d = (sev_a.mean() - sev_b.mean()) / pooled_std if pooled_std > 0 else 0.0

    return {
        "test": "Welch's t-test",
        "kpi": "Claim Severity",
        "statistic": round(t_stat, 4),
        "p_value": float(p_value),
        "dof": len(sev_a) + len(sev_b) - 2,
        "effect_size": round(cohens_d, 6),
        "effect_label": "Cohen's d",
        "decision": "Reject H₀" if p_value < alpha else "Fail to reject H₀",
        "significant": p_value < alpha,
        "mean_a": round(sev_a.mean(), 2),
        "mean_b": round(sev_b.mean(), 2),
    }


def t_test_margin(
    df_a: pd.DataFrame,
    df_b: pd.DataFrame,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Welch's t-test on per-policy margin (TotalPremium − TotalClaims).

    Args:
        df_a: Control group DataFrame.
        df_b: Test group DataFrame.
        alpha: Significance level (default 0.05).

    Returns:
        Dict with t-statistic, p_value, decision, and Cohen's d effect size.
    """
    m_a = margin(df_a).values
    m_b = margin(df_b).values

    t_stat, p_value = ttest_ind(m_a, m_b, equal_var=False)

    pooled_std = np.sqrt((m_a.std() ** 2 + m_b.std() ** 2) / 2)
    cohens_d = (m_a.mean() - m_b.mean()) / pooled_std if pooled_std > 0 else 0.0

    return {
        "test": "Welch's t-test",
        "kpi": "Margin",
        "statistic": round(t_stat, 4),
        "p_value": float(p_value),
        "dof": len(m_a) + len(m_b) - 2,
        "effect_size": round(cohens_d, 6),
        "effect_label": "Cohen's d",
        "decision": "Reject H₀" if p_value < alpha else "Fail to reject H₀",
        "significant": p_value < alpha,
        "mean_a": round(m_a.mean(), 4),
        "mean_b": round(m_b.mean(), 4),
    }


def z_test_proportions(
    n_a: int,
    k_a: int,
    n_b: int,
    k_b: int,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Two-proportion z-test for claim frequency on large samples.

    Args:
        n_a: Total policies in group A.
        k_a: Policies with a claim in group A.
        n_b: Total policies in group B.
        k_b: Policies with a claim in group B.
        alpha: Significance level (default 0.05).

    Returns:
        Dict with z-statistic, p_value, and decision.
    """
    p_a = k_a / n_a
    p_b = k_b / n_b
    p_pool = (k_a + k_b) / (n_a + n_b)
    se = np.sqrt(p_pool * (1 - p_pool) * (1 / n_a + 1 / n_b))

    if se == 0:
        raise ValueError("Pooled standard error is zero; groups may be identical.")

    z_stat = (p_a - p_b) / se
    p_value = 2 * (1 - stats.norm.cdf(abs(z_stat)))

    return {
        "test": "Two-proportion z-test",
        "kpi": "Claim Frequency",
        "statistic": round(z_stat, 4),
        "p_value": float(p_value),
        "dof": None,
        "effect_size": round(p_a - p_b, 6),
        "effect_label": "Difference in proportions",
        "decision": "Reject H₀" if p_value < alpha else "Fail to reject H₀",
        "significant": p_value < alpha,
        "freq_a": round(p_a, 6),
        "freq_b": round(p_b, 6),
    }


# ── Results formatting ────────────────────────────────────────────────────────

def build_results_table(results: list[Dict[str, Any]]) -> pd.DataFrame:
    """Compile a list of test result dicts into a summary DataFrame.

    Args:
        results: List of dicts, each from a test function, augmented with
                 'hypothesis', 'group_a', and 'group_b' keys.

    Returns:
        Summary DataFrame with one row per hypothesis test.
    """
    rows = []
    for r in results:
        rows.append({
            "Hypothesis": r.get("hypothesis", ""),
            "Group A (Control)": r.get("group_a", ""),
            "Group B (Test)": r.get("group_b", ""),
            "KPI": r.get("kpi", ""),
            "Test": r.get("test", ""),
            "Statistic": r.get("statistic", ""),
            "p-value": f"{r['p_value']:.2e}" if r.get("p_value") is not None else "",
            "Decision": r.get("decision", ""),
        })
    return pd.DataFrame(rows)
