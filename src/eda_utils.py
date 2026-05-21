"""EDA helper functions for insurance risk analytics."""

import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns


# ---------------------------------------------------------------------------
# Data summarization
# ---------------------------------------------------------------------------

def descriptive_stats(df: pd.DataFrame) -> pd.DataFrame:
    """Compute extended descriptive statistics for all numeric columns.

    Supplements the standard ``describe()`` output with skewness, excess
    kurtosis, zero-value count, and zero percentage — useful for spotting
    heavy-tailed or zero-inflated distributions common in claims data.

    Args:
        df: Source DataFrame.

    Returns:
        DataFrame with one column per numeric feature and rows:
        count, mean, std, min, 25%, 50%, 75%, max, skewness, kurtosis,
        zeros, zeros_pct.
    """
    num_df = df.select_dtypes(include="number")
    stats = num_df.describe().T
    stats["skewness"] = num_df.skew()
    stats["kurtosis"] = num_df.kurt()
    stats["zeros"] = (num_df == 0).sum()
    stats["zeros_pct"] = (stats["zeros"] / len(df) * 100).round(2)
    return stats.T


def dtype_review(df: pd.DataFrame) -> pd.DataFrame:
    """Classify every column by its inferred semantic type and flag mismatches.

    Columns are classified as one of: ``datetime``, ``boolean``,
    ``numerical``, ``categorical``, or ``unknown``.  A mismatch is flagged
    when the stored dtype does not match the expected semantic type — for
    example a column stored as ``object`` that looks like it should be
    numeric, or a date column still stored as a string.

    Args:
        df: Source DataFrame.

    Returns:
        DataFrame indexed by column name with columns ``stored_dtype``,
        ``semantic_type``, and ``flag`` (empty string or a warning message).
    """
    records = []
    for col in df.columns:
        dtype = df[col].dtype
        dtype_str = str(dtype)
        flag = ""

        if pd.api.types.is_datetime64_any_dtype(dtype):
            semantic = "datetime"
        elif pd.api.types.is_bool_dtype(dtype) or dtype_str == "boolean":
            semantic = "boolean"
        elif pd.api.types.is_numeric_dtype(dtype):
            semantic = "numerical"
            # boolean-like numeric (only 0/1 values)
            unique = df[col].dropna().unique()
            if set(unique).issubset({0, 1}):
                flag = "looks boolean — consider casting"
        elif dtype_str == "object":
            sample = df[col].dropna().head(500)
            semantic = "categorical"
            if len(sample) == 0:
                pass
            else:
                # try numeric first (higher confidence)
                numeric_converted = pd.to_numeric(sample, errors="coerce")
                if numeric_converted.notna().mean() > 0.9:
                    flag = ">90% values numeric but stored as object"
                else:
                    # try datetime only when there are real non-null values
                    try:
                        parsed = pd.to_datetime(sample, errors="coerce")
                        if parsed.notna().mean() > 0.8:
                            flag = "may be datetime — consider pd.to_datetime"
                    except Exception:
                        pass
        else:
            semantic = "unknown"

        records.append(
            {"column": col, "stored_dtype": dtype_str,
             "semantic_type": semantic, "flag": flag}
        )

    return pd.DataFrame(records).set_index("column")


# ---------------------------------------------------------------------------
# Univariate helpers
# ---------------------------------------------------------------------------

def plot_numeric_distributions(df: pd.DataFrame, cols: list, figsize: tuple = (16, 4)):
    """Plot histograms with KDE for a list of numeric columns.

    Args:
        df: Source DataFrame.
        cols: Column names to plot.
        figsize: Figure size per row of subplots.

    Returns:
        matplotlib Figure.
    """
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]))
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        sns.histplot(df[col].dropna(), kde=True, ax=ax, color="steelblue")
        ax.set_title(col)
        ax.set_xlabel("")
    fig.tight_layout()
    return fig


def plot_categorical_counts(
    df: pd.DataFrame, col: str, top_n: int = 20, figsize: tuple = (10, 5)
):
    """Bar chart of value counts for a categorical column.

    Args:
        df: Source DataFrame.
        col: Column to plot.
        top_n: Maximum categories shown (sorted by frequency).
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    counts = df[col].value_counts().head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    sns.barplot(x=counts.values, y=counts.index, ax=ax, palette="Blues_r")
    ax.set_title(f"Top {top_n} values – {col}")
    ax.set_xlabel("Count")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Missing data helpers
# ---------------------------------------------------------------------------

def missing_heatmap(df: pd.DataFrame, figsize: tuple = (14, 6)):
    """Render a heatmap of missing values across columns.

    Args:
        df: Source DataFrame.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(df.isnull(), yticklabels=False, cbar=False, cmap="viridis", ax=ax)
    ax.set_title("Missing Value Heatmap")
    fig.tight_layout()
    return fig


def missing_summary_table(df: pd.DataFrame) -> pd.DataFrame:
    """Return a tidy table of missing-value counts and percentages.

    Args:
        df: Source DataFrame.

    Returns:
        DataFrame indexed by column name with columns ``missing_count`` and
        ``missing_pct``, sorted descending by ``missing_pct``.
    """
    missing = df.isnull().sum()
    pct = (missing / len(df) * 100).round(2)
    table = pd.DataFrame({"missing_count": missing, "missing_pct": pct})
    return table[table["missing_count"] > 0].sort_values("missing_pct", ascending=False)


# ---------------------------------------------------------------------------
# Bivariate / risk helpers
# ---------------------------------------------------------------------------

def loss_ratio_by_group(df: pd.DataFrame, group_col: str) -> pd.DataFrame:
    """Aggregate loss ratio statistics grouped by a categorical column.

    Args:
        df: DataFrame with ``TotalClaims`` and ``TotalPremium`` columns.
        group_col: Column to group by (e.g. ``Province``, ``VehicleType``).

    Returns:
        DataFrame with columns ``TotalPremium``, ``TotalClaims``,
        ``LossRatio``, and ``PolicyCount``, sorted by ``LossRatio``.
    """
    grp = df.groupby(group_col, observed=True).agg(
        TotalPremium=("TotalPremium", "sum"),
        TotalClaims=("TotalClaims", "sum"),
        PolicyCount=("TotalPremium", "count"),
    )
    grp["LossRatio"] = grp["TotalClaims"] / grp["TotalPremium"]
    return grp.sort_values("LossRatio", ascending=False).reset_index()


def plot_loss_ratio_by_group(
    df: pd.DataFrame,
    group_col: str,
    top_n: int = 15,
    figsize: tuple = (10, 6),
):
    """Bar chart of loss ratio by a categorical grouping.

    Args:
        df: DataFrame with ``TotalClaims`` and ``TotalPremium`` columns.
        group_col: Column to group by.
        top_n: Limit to the top N groups by loss ratio.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    summary = loss_ratio_by_group(df, group_col).head(top_n)
    fig, ax = plt.subplots(figsize=figsize)
    colors = ["firebrick" if r > 1 else "steelblue" for r in summary["LossRatio"]]
    ax.barh(summary[group_col].astype(str), summary["LossRatio"], color=colors)
    ax.axvline(1.0, color="black", linestyle="--", linewidth=1, label="Break-even")
    ax.set_xlabel("Loss Ratio (Claims / Premium)")
    ax.set_title(f"Loss Ratio by {group_col}")
    ax.legend()
    fig.tight_layout()
    return fig


def correlation_heatmap(df: pd.DataFrame, cols: list, figsize: tuple = (10, 8)):
    """Pearson correlation heatmap for a subset of numeric columns.

    Args:
        df: Source DataFrame.
        cols: Numeric columns to include.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    corr = df[cols].corr()
    fig, ax = plt.subplots(figsize=figsize)
    sns.heatmap(
        corr,
        annot=True,
        fmt=".2f",
        cmap="RdBu_r",
        center=0,
        square=True,
        linewidths=0.5,
        ax=ax,
    )
    ax.set_title("Pearson Correlation Matrix")
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Outlier detection
# ---------------------------------------------------------------------------

def outlier_summary(df: pd.DataFrame, cols: list) -> pd.DataFrame:
    """IQR-based outlier counts for a list of numeric columns.

    Args:
        df: Source DataFrame.
        cols: Numeric columns to analyse.

    Returns:
        DataFrame indexed by column with Q1, Q3, IQR, lower/upper fences,
        outlier count, and outlier percentage.
    """
    records = []
    for col in cols:
        s = df[col].dropna()
        q1, q3 = s.quantile(0.25), s.quantile(0.75)
        iqr = q3 - q1
        lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        n_out = int(((s < lower) | (s > upper)).sum())
        records.append({
            "column": col,
            "Q1": round(q1, 4),
            "Q3": round(q3, 4),
            "IQR": round(iqr, 4),
            "lower_fence": round(lower, 4),
            "upper_fence": round(upper, 4),
            "outlier_count": n_out,
            "outlier_pct": round(n_out / len(s) * 100, 2),
        })
    return pd.DataFrame(records).set_index("column")


def plot_boxplots(
    df: pd.DataFrame, cols: list, log_scale: bool = False, figsize: tuple = (16, 5)
):
    """Side-by-side box plots for a list of numeric columns.

    Args:
        df: Source DataFrame.
        cols: Numeric columns to plot.
        log_scale: Apply log1p transform before plotting (useful for skewed data).
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    n = len(cols)
    fig, axes = plt.subplots(1, n, figsize=figsize)
    if n == 1:
        axes = [axes]
    for ax, col in zip(axes, cols):
        vals = df[col].dropna()
        if log_scale:
            import math
            data = vals[vals > 0].apply(math.log1p)
        else:
            data = vals
        ax.boxplot(data, vert=True, patch_artist=True,
                   boxprops=dict(facecolor="steelblue", alpha=0.6))
        label = f"log1p({col})" if log_scale else col
        ax.set_title(label, fontsize=9)
        ax.set_xticks([])
    fig.suptitle("Box Plots" + (" (log scale)" if log_scale else ""), y=1.01)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Bivariate helpers
# ---------------------------------------------------------------------------

def scatter_premium_vs_claims(
    df: pd.DataFrame,
    group_col: str = "PostalCode",
    top_n: int = 8,
    sample: int = 20_000,
    figsize: tuple = (10, 6),
):
    """Scatter plot of TotalPremium vs TotalClaims coloured by a group column.

    Subsets to rows where both columns are positive, draws a random sample
    for readability, and colours points by the top N groups.

    Args:
        df: Source DataFrame.
        group_col: Column used for colour grouping (e.g. ``PostalCode``, ``Province``).
        top_n: Number of top groups to colour distinctly; rest labelled 'Other'.
        sample: Maximum rows to plot (random sample if df is larger).
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    plot_df = df[(df["TotalPremium"] > 0) & (df["TotalClaims"] > 0)].copy()
    if len(plot_df) > sample:
        plot_df = plot_df.sample(sample, random_state=42)

    top_groups = plot_df[group_col].value_counts().head(top_n).index
    plot_df["_group"] = plot_df[group_col].where(
        plot_df[group_col].isin(top_groups), other="Other"
    ).astype(str)

    fig, ax = plt.subplots(figsize=figsize)
    palette = plt.cm.tab10.colors
    for i, grp in enumerate(list(top_groups.astype(str)) + ["Other"]):
        sub = plot_df[plot_df["_group"] == grp]
        color = palette[i % len(palette)] if grp != "Other" else "lightgrey"
        ax.scatter(
            sub["TotalPremium"], sub["TotalClaims"],
            s=15, alpha=0.5, color=color, label=grp
        )
    ax.set_xlabel("TotalPremium")
    ax.set_ylabel("TotalClaims")
    ax.set_title(f"TotalPremium vs TotalClaims — coloured by {group_col}")
    ax.legend(title=group_col, bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Geographic helpers
# ---------------------------------------------------------------------------

def premium_by_province(df: pd.DataFrame, figsize: tuple = (12, 5)):
    """Bar chart of total and mean TotalPremium by Province.

    Args:
        df: Source DataFrame.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    grp = df.groupby("Province", observed=True)["TotalPremium"].agg(
        Total="sum", Mean="mean", Count="count"
    ).sort_values("Total", ascending=False).reset_index()

    fig, axes = plt.subplots(1, 2, figsize=figsize)
    axes[0].barh(grp["Province"], grp["Total"] / 1e6, color="steelblue")
    axes[0].set_xlabel("Total Premium (millions)")
    axes[0].set_title("Total Premium by Province")

    axes[1].barh(grp["Province"], grp["Mean"], color="darkorange")
    axes[1].set_xlabel("Mean Premium per Policy")
    axes[1].set_title("Mean Premium by Province")

    fig.tight_layout()
    return fig


def cover_type_by_province(df: pd.DataFrame, top_n: int = 5, figsize: tuple = (14, 6)):
    """Stacked bar chart showing cover type mix across provinces.

    Args:
        df: Source DataFrame.
        top_n: Number of top cover types to show; rest grouped as 'Other'.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    top_covers = df["CoverType"].value_counts().head(top_n).index
    df2 = df.copy()
    df2["CoverTypeBucket"] = df2["CoverType"].where(
        df2["CoverType"].isin(top_covers), other="Other"
    )
    pivot = (
        df2.groupby(["Province", "CoverTypeBucket"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    pivot_pct = pivot.div(pivot.sum(axis=1), axis=0) * 100
    pivot_pct = pivot_pct.loc[
        df["Province"].value_counts().index  # sort by policy count
    ]

    fig, ax = plt.subplots(figsize=figsize)
    pivot_pct.plot(kind="bar", stacked=True, ax=ax, colormap="tab10")
    ax.set_xlabel("Province")
    ax.set_ylabel("% of Policies")
    ax.set_title("Cover Type Mix by Province")
    ax.legend(title="Cover Type", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


def top_makes_by_province(df: pd.DataFrame, top_n: int = 5, figsize: tuple = (14, 6)):
    """Plot a grouped bar chart of top vehicle makes by province (policy count).

    Args:
        df: Source DataFrame.
        top_n: Number of top makes to show.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    top_makes = df["make"].value_counts().head(top_n).index
    sub = df[df["make"].isin(top_makes)]
    pivot = (
        sub.groupby(["Province", "make"], observed=True)
        .size()
        .unstack(fill_value=0)
    )
    pivot = pivot.loc[df["Province"].value_counts().index]

    fig, ax = plt.subplots(figsize=figsize)
    pivot.plot(kind="bar", ax=ax, colormap="tab20")
    ax.set_xlabel("Province")
    ax.set_ylabel("Policy Count")
    ax.set_title(f"Top {top_n} Vehicle Makes by Province")
    ax.legend(title="Make", bbox_to_anchor=(1.01, 1), loc="upper left", fontsize=8)
    ax.tick_params(axis="x", rotation=45)
    fig.tight_layout()
    return fig


# ---------------------------------------------------------------------------
# Temporal helpers
# ---------------------------------------------------------------------------

def monthly_trend(df: pd.DataFrame, value_col: str, agg: str = "sum") -> pd.DataFrame:
    """Aggregate a numeric column by transaction month.

    Args:
        df: DataFrame with a ``TransactionMonth`` datetime column.
        value_col: Column to aggregate.
        agg: Aggregation function name (``sum``, ``mean``, ``count``).

    Returns:
        DataFrame indexed by month with the aggregated value.
    """
    monthly = (
        df.set_index("TransactionMonth")[value_col]
        .resample("ME")
        .agg(agg)
        .reset_index()
    )
    monthly.columns = ["Month", value_col]
    return monthly


def plot_monthly_trend(
    df: pd.DataFrame, value_col: str, agg: str = "sum", figsize: tuple = (12, 4)
):
    """Line chart of a monthly aggregated metric.

    Args:
        df: DataFrame with a ``TransactionMonth`` datetime column.
        value_col: Column to aggregate.
        agg: Aggregation function name.
        figsize: Figure size.

    Returns:
        matplotlib Figure.
    """
    trend = monthly_trend(df, value_col, agg)
    fig, ax = plt.subplots(figsize=figsize)
    ax.plot(trend["Month"], trend[value_col], marker="o", color="steelblue")
    ax.set_title(f"Monthly {agg.capitalize()} of {value_col}")
    ax.set_xlabel("Month")
    ax.set_ylabel(value_col)
    fig.autofmt_xdate()
    fig.tight_layout()
    return fig
