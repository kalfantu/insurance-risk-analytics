"""Data loading and ingestion utilities for insurance risk analytics."""

import pandas as pd


DTYPE_MAP = {
    "UnderwrittenCoverID": "Int64",
    "PolicyID": "Int64",
    "TransactionMonth": str,
    "IsVATRegistered": "boolean",
    "Citizenship": str,
    "LegalType": str,
    "Title": str,
    "Language": str,
    "Bank": str,
    "AccountType": str,
    "MaritalStatus": str,
    "Gender": str,
    "Country": str,
    "Province": str,
    "PostalCode": str,
    "MainCrestaZone": str,
    "SubCrestaZone": str,
    "ItemType": str,
    "Mmcode": str,
    "VehicleType": str,
    "RegistrationYear": "Int64",
    "Make": str,
    "Model": str,
    "Cylinders": "Int64",
    "Cubiccapacity": "Int64",
    "Kilowatts": "Int64",
    "Bodytype": str,
    "NumberOfDoors": "Int64",
    "VehicleIntroDate": str,
    "CustomValueEstimate": float,
    "AlarmImmobiliser": str,
    "TrackingDevice": str,
    "CapitalOutstanding": float,
    "NewVehicle": "boolean",
    "WrittenOff": "boolean",
    "Rebuilt": "boolean",
    "Converted": "boolean",
    "CrossBorder": "boolean",
    "NumberOfVehiclesInFleet": "Int64",
    "SumInsured": float,
    "TermFrequency": str,
    "CalculatedPremiumPerTerm": float,
    "ExcessSelected": str,
    "CoverCategory": str,
    "CoverType": str,
    "CoverGroup": str,
    "Section": str,
    "Product": str,
    "StatutoryClass": str,
    "StatutoryRiskType": str,
    "TotalPremium": float,
    "TotalClaims": float,
}


def load_data(filepath: str, sep: str = "|") -> pd.DataFrame:
    """Load the raw insurance dataset from a delimited text file.

    Args:
        filepath: Path to the source data file.
        sep: Column delimiter used in the file (default ``|``).

    Returns:
        DataFrame with columns cast to the types defined in ``DTYPE_MAP``.
    """
    df = pd.read_csv(
        filepath,
        sep=sep,
        dtype={k: v for k, v in DTYPE_MAP.items() if v not in ("boolean", "Int64")},
        low_memory=False,
    )

    for col, dtype in DTYPE_MAP.items():
        if col not in df.columns:
            continue
        if dtype == "boolean":
            df[col] = df[col].map({"Y": True, "N": False, True: True, False: False})
        elif dtype == "Int64":
            df[col] = pd.to_numeric(df[col], errors="coerce").astype("Int64")

    if "TransactionMonth" in df.columns:
        df["TransactionMonth"] = pd.to_datetime(
            df["TransactionMonth"], errors="coerce"
        )

    return df


def summarize_data(df: pd.DataFrame) -> dict:
    """Return a summary dict describing shape, dtypes, nulls, and duplicates.

    Args:
        df: Input DataFrame.

    Returns:
        Dictionary with keys: ``shape``, ``dtypes``, ``null_counts``,
        ``null_pct``, ``duplicate_rows``.
    """
    null_counts = df.isnull().sum()
    return {
        "shape": df.shape,
        "dtypes": df.dtypes.to_dict(),
        "null_counts": null_counts.to_dict(),
        "null_pct": (null_counts / len(df) * 100).round(2).to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
    }


def compute_loss_ratio(df: pd.DataFrame) -> pd.Series:
    """Compute the loss ratio (TotalClaims / TotalPremium) per row.

    A loss ratio > 1 means claims exceed premiums (unprofitable policy).

    Args:
        df: DataFrame containing ``TotalClaims`` and ``TotalPremium`` columns.

    Returns:
        Series of loss ratios with ``inf`` replaced by ``NaN``.
    """
    ratio = df["TotalClaims"] / df["TotalPremium"]
    return ratio.replace([float("inf"), float("-inf")], float("nan"))


def flag_high_missing(df: pd.DataFrame, threshold: float = 0.5) -> list:
    """Return column names where the fraction of missing values exceeds threshold.

    Args:
        df: Input DataFrame.
        threshold: Fraction above which a column is considered high-missing.

    Returns:
        List of column names.
    """
    missing_frac = df.isnull().mean()
    return list(missing_frac[missing_frac > threshold].index)
