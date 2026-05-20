"""Unit tests for src.eda_utils."""

import pandas as pd
import pytest

from src.eda_utils import (
    loss_ratio_by_group,
    missing_summary_table,
    monthly_trend,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

@pytest.fixture
def policy_df():
    return pd.DataFrame(
        {
            "Province": ["Gauteng", "Gauteng", "Western Cape", "KwaZulu-Natal"],
            "TotalPremium": [1000.0, 500.0, 800.0, 600.0],
            "TotalClaims": [200.0, 100.0, 400.0, 50.0],
            "TransactionMonth": pd.to_datetime(
                ["2014-02-01", "2014-03-01", "2014-02-01", "2014-04-01"]
            ),
        }
    )


# ---------------------------------------------------------------------------
# missing_summary_table
# ---------------------------------------------------------------------------

def test_missing_summary_only_incomplete_columns():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    table = missing_summary_table(df)
    assert "a" in table.index
    assert "b" not in table.index


def test_missing_summary_sorted_descending():
    df = pd.DataFrame(
        {
            "half": [None, None, 1, 1],
            "all": [None, None, None, None],
        }
    )
    table = missing_summary_table(df)
    assert table.index[0] == "all"


def test_missing_summary_empty_when_no_nulls():
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    table = missing_summary_table(df)
    assert len(table) == 0


# ---------------------------------------------------------------------------
# loss_ratio_by_group
# ---------------------------------------------------------------------------

def test_loss_ratio_by_group_returns_dataframe(policy_df):
    result = loss_ratio_by_group(policy_df, "Province")
    assert isinstance(result, pd.DataFrame)
    assert "LossRatio" in result.columns


def test_loss_ratio_by_group_correct_ratio(policy_df):
    result = loss_ratio_by_group(policy_df, "Province")
    gauteng = result[result["Province"] == "Gauteng"].iloc[0]
    expected_ratio = 300.0 / 1500.0
    assert pytest.approx(gauteng["LossRatio"]) == expected_ratio


def test_loss_ratio_by_group_sorted_descending(policy_df):
    result = loss_ratio_by_group(policy_df, "Province")
    ratios = result["LossRatio"].tolist()
    assert ratios == sorted(ratios, reverse=True)


# ---------------------------------------------------------------------------
# monthly_trend
# ---------------------------------------------------------------------------

def test_monthly_trend_sum(policy_df):
    trend = monthly_trend(policy_df, "TotalPremium", agg="sum")
    assert "Month" in trend.columns
    assert "TotalPremium" in trend.columns
    assert trend["TotalPremium"].sum() == pytest.approx(policy_df["TotalPremium"].sum())


def test_monthly_trend_count(policy_df):
    trend = monthly_trend(policy_df, "TotalPremium", agg="count")
    assert trend["TotalPremium"].sum() == len(policy_df)
