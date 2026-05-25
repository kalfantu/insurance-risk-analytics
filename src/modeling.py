"""Predictive modelling utilities for insurance risk analytics."""

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    mean_squared_error, r2_score,
    accuracy_score, precision_score, recall_score, f1_score, roc_auc_score,
)
from typing import Tuple, Dict, Any, List


# ── Constants ────────────────────────────────────────────────────────────────

# Columns that carry no signal (single value or are IDs)
_DROP_COLS = [
    "UnderwrittenCoverID", "PolicyID",
    "Language", "Country", "ItemType",
    "StatutoryClass", "StatutoryRiskType",
    "VehicleIntroDate",   # superseded by vehicle_age
    "Model",              # 411 unique values — too high cardinality
    "TransactionMonth",   # decomposed into year/month features
]

# Low-cardinality categoricals to one-hot encode
_OHE_COLS = [
    "Citizenship", "LegalType", "Title", "Bank", "AccountType",
    "MaritalStatus", "Gender", "Province", "VehicleType",
    "bodytype", "AlarmImmobiliser", "TrackingDevice",
    "TermFrequency", "CoverCategory", "CoverType", "CoverGroup",
    "Section", "Product", "ExcessSelected",
]

# Label-encode higher-cardinality categoricals
_LABEL_COLS = [
    "MainCrestaZone", "SubCrestaZone", "make",
]


# ── Feature engineering ───────────────────────────────────────────────────────

def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add derived features and drop low-signal columns.

    Args:
        df: Cleaned insurance DataFrame.

    Returns:
        DataFrame with engineered features, ready for encoding.
    """
    out = df.copy()

    # Temporal features
    out["transaction_month_dt"] = pd.to_datetime(out["TransactionMonth"], errors="coerce")
    out["transaction_year"] = out["transaction_month_dt"].dt.year.astype("Int64")
    out["transaction_month_num"] = out["transaction_month_dt"].dt.month.astype("Int64")

    # Vehicle age at transaction date (dataset ends Aug 2015)
    out["vehicle_age"] = out["transaction_year"] - out["RegistrationYear"].astype("Int64")
    out["vehicle_age"] = out["vehicle_age"].clip(lower=0)

    # Premium efficiency ratio — how much premium per unit of sum insured
    out["premium_rate"] = np.where(
        out["SumInsured"] > 0,
        out["CalculatedPremiumPerTerm"] / out["SumInsured"],
        0.0,
    )

    # Capital utilisation — outstanding capital as fraction of sum insured
    out["capital_utilisation"] = np.where(
        out["SumInsured"] > 0,
        out["CapitalOutstanding"] / out["SumInsured"],
        0.0,
    )

    out = out.drop(columns=["transaction_month_dt"], errors="ignore")
    return out


def encode_features(df: pd.DataFrame) -> pd.DataFrame:
    """Drop low-signal columns and encode categoricals.

    One-hot encodes low-cardinality columns; label-encodes higher-cardinality
    ones; drops ID and constant columns.

    Args:
        df: DataFrame after engineer_features().

    Returns:
        Fully numeric DataFrame ready for modelling.
    """
    out = df.copy()

    # Drop columns
    cols_to_drop = [c for c in _DROP_COLS if c in out.columns]
    out = out.drop(columns=cols_to_drop)

    # Label encode
    for col in _LABEL_COLS:
        if col in out.columns:
            out[col] = out[col].astype("category").cat.codes

    # One-hot encode
    ohe_present = [c for c in _OHE_COLS if c in out.columns]
    out = pd.get_dummies(out, columns=ohe_present, drop_first=True, dtype=int)

    # Cast remaining object columns (safety net)
    for col in out.select_dtypes("object").columns:
        out[col] = out[col].astype("category").cat.codes

    # Cast all columns to float32 — halves memory vs float64; XGBoost/sklearn both accept it
    for col in out.select_dtypes(exclude="float32").columns:
        out[col] = out[col].astype("float32")

    # Sanitize column names — XGBoost 3.x rejects special characters
    import re
    out.columns = [re.sub(r"[^A-Za-z0-9_]", "_", str(c)) for c in out.columns]

    return out


def prepare_severity_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare train/test split for severity regression (claims > 0 only).

    Target is log1p(TotalClaims) to reduce right-skew.

    Args:
        df: Encoded DataFrame.
        test_size: Fraction of data held out for testing.
        random_state: Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test (log1p-transformed target).
    """
    claims_df = df[df["TotalClaims"] > 0].copy()
    y = np.log1p(claims_df["TotalClaims"])
    X = claims_df.drop(columns=["TotalClaims", "TotalPremium"], errors="ignore")
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def prepare_classification_data(
    df: pd.DataFrame,
    test_size: float = 0.2,
    random_state: int = 42,
) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
    """Prepare train/test split for claim probability classification.

    Args:
        df: Encoded DataFrame.
        test_size: Fraction held out for testing.
        random_state: Random seed for reproducibility.

    Returns:
        X_train, X_test, y_train, y_test (binary 0/1 target).
    """
    y = (df["TotalClaims"] > 0).astype(int)
    X = df.drop(columns=["TotalClaims", "TotalPremium"], errors="ignore")
    return train_test_split(X, y, test_size=test_size, random_state=random_state,
                            stratify=y)


# ── Evaluation ────────────────────────────────────────────────────────────────

def evaluate_regression(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    log_transformed: bool = True,
) -> Dict[str, float]:
    """Compute RMSE and R² for regression models.

    Args:
        y_true: True target values.
        y_pred: Predicted values.
        log_transformed: If True, inverse-transforms via expm1 before scoring.

    Returns:
        Dict with rmse and r2 in original (Rand) scale.
    """
    if log_transformed:
        y_true = np.expm1(y_true)
        y_pred = np.expm1(y_pred)

    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2 = r2_score(y_true, y_pred)
    return {"rmse": round(rmse, 2), "r2": round(r2, 4)}


def evaluate_classification(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray,
) -> Dict[str, float]:
    """Compute classification metrics.

    Args:
        y_true: True binary labels.
        y_pred: Predicted binary labels.
        y_prob: Predicted probabilities for the positive class.

    Returns:
        Dict with accuracy, precision, recall, f1, roc_auc.
    """
    return {
        "accuracy": round(accuracy_score(y_true, y_pred), 4),
        "precision": round(precision_score(y_true, y_pred, zero_division=0), 4),
        "recall": round(recall_score(y_true, y_pred, zero_division=0), 4),
        "f1": round(f1_score(y_true, y_pred, zero_division=0), 4),
        "roc_auc": round(roc_auc_score(y_true, y_prob), 4),
    }


# ── Results formatting ────────────────────────────────────────────────────────

def build_regression_comparison(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compile regression model results into a comparison table.

    Args:
        results: List of dicts, each with 'model', 'rmse', 'r2'.

    Returns:
        Sorted DataFrame (best RMSE first).
    """
    df = pd.DataFrame(results)[["model", "rmse", "r2"]]
    return df.sort_values("rmse").reset_index(drop=True)


def build_classification_comparison(results: List[Dict[str, Any]]) -> pd.DataFrame:
    """Compile classification model results into a comparison table.

    Args:
        results: List of dicts, each with 'model' plus metric keys.

    Returns:
        Sorted DataFrame (best ROC-AUC first).
    """
    cols = ["model", "accuracy", "precision", "recall", "f1", "roc_auc"]
    df = pd.DataFrame(results)[cols]
    return df.sort_values("roc_auc", ascending=False).reset_index(drop=True)


# ── Premium pricing framework ─────────────────────────────────────────────────

def compute_risk_premium(
    p_claim: np.ndarray,
    predicted_severity: np.ndarray,
    expense_loading: float = 0.15,
    profit_margin: float = 0.05,
) -> np.ndarray:
    """Compute risk-based premium using the two-part pricing formula.

    Premium = (P(claim) × Predicted Severity) + Expense Loading + Profit Margin

    Expense loading and profit margin are applied as fractions of the
    pure risk premium.

    Args:
        p_claim: Predicted claim probability per policy.
        predicted_severity: Predicted claim severity per policy (Rand).
        expense_loading: Fraction added for expenses (default 15%).
        profit_margin: Fraction added for profit (default 5%).

    Returns:
        Array of recommended premiums per policy.
    """
    pure_premium = p_claim * predicted_severity
    return pure_premium * (1 + expense_loading + profit_margin)
