"""Unit tests for src.data_loader."""

import io
import pandas as pd
import pytest

from src.data_loader import (
    compute_loss_ratio,
    flag_high_missing,
    load_data,
    summarize_data,
)

# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------

SAMPLE_CSV = """UnderwrittenCoverID|PolicyID|TransactionMonth|TotalPremium|TotalClaims|Province|Gender
1|101|2014-02-01|1200.0|300.0|Gauteng|Male
2|102|2014-03-01|800.0|0.0|Western Cape|Female
3|103|2014-04-01|0.0|500.0|KwaZulu-Natal|Male
"""


@pytest.fixture
def sample_df():
    return pd.read_csv(io.StringIO(SAMPLE_CSV), sep="|")


# ---------------------------------------------------------------------------
# load_data
# ---------------------------------------------------------------------------

def test_load_data_returns_dataframe(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text(SAMPLE_CSV)
    df = load_data(str(path))
    assert isinstance(df, pd.DataFrame)
    assert len(df) == 3


def test_load_data_transaction_month_parsed(tmp_path):
    path = tmp_path / "test.txt"
    path.write_text(SAMPLE_CSV)
    df = load_data(str(path))
    assert pd.api.types.is_datetime64_any_dtype(df["TransactionMonth"])


# ---------------------------------------------------------------------------
# summarize_data
# ---------------------------------------------------------------------------

def test_summarize_data_shape(sample_df):
    summary = summarize_data(sample_df)
    assert summary["shape"] == (3, 7)


def test_summarize_data_no_duplicates(sample_df):
    summary = summarize_data(sample_df)
    assert summary["duplicate_rows"] == 0


def test_summarize_data_with_duplicates():
    df = pd.DataFrame({"a": [1, 1, 2]})
    summary = summarize_data(df)
    assert summary["duplicate_rows"] == 1


def test_summarize_data_null_counts():
    df = pd.DataFrame({"a": [1, None, 3], "b": [4, 5, 6]})
    summary = summarize_data(df)
    assert summary["null_counts"]["a"] == 1
    assert summary["null_counts"]["b"] == 0


# ---------------------------------------------------------------------------
# compute_loss_ratio
# ---------------------------------------------------------------------------

def test_loss_ratio_basic(sample_df):
    ratios = compute_loss_ratio(sample_df)
    assert pytest.approx(ratios.iloc[0]) == 300.0 / 1200.0


def test_loss_ratio_zero_premium_is_nan(sample_df):
    ratios = compute_loss_ratio(sample_df)
    assert pd.isna(ratios.iloc[2])


def test_loss_ratio_zero_claims():
    df = pd.DataFrame({"TotalPremium": [500.0], "TotalClaims": [0.0]})
    ratios = compute_loss_ratio(df)
    assert ratios.iloc[0] == 0.0


# ---------------------------------------------------------------------------
# flag_high_missing
# ---------------------------------------------------------------------------

def test_flag_high_missing_identifies_column():
    df = pd.DataFrame(
        {
            "good": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10],
            "bad": [None, None, None, None, None, None, 1, 2, 3, 4],
        }
    )
    flagged = flag_high_missing(df, threshold=0.5)
    assert "bad" in flagged
    assert "good" not in flagged


def test_flag_high_missing_empty_result():
    df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
    flagged = flag_high_missing(df)
    assert flagged == []
