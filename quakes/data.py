"""Load the earthquake catalogue and gas production, and build the training table."""

import numpy as np
import pandas as pd


def load_events(path):
    """KNMI catalogue (pipe-separated text) -> DataFrame of induced events."""
    df = pd.read_csv(path, sep="|")
    df.columns = [c.lstrip("#") for c in df.columns]
    df = df[df["EventType"] == "induced or triggered event"].copy()
    df["Time"] = pd.to_datetime(df["Time"], format="ISO8601")
    return df


def monthly_counts(events, min_mag, start, end):
    """Number of events with magnitude >= min_mag in each month (empty months = 0)."""
    sel = events[events["Magnitude"] >= min_mag]
    counts = sel.groupby(sel["Time"].dt.to_period("M")).size()
    months = pd.period_range(start, end, freq="M")
    return counts.reindex(months, fill_value=0).astype(float)


def load_production(path):
    """Groningen production per gas year (Oct-Sep) -> monthly series in bcm per month.

    One row per gas year, e.g. "'12 - '13" = Oct 2012 to Sep 2013. The values are
    in thousands of Nm3 (53 million = 53 bcm in the record year 2012-13). Only
    yearly totals exist, so each year is spread evenly over its 12 months.
    """
    df = pd.read_csv(path)
    monthly = {}
    for label, volume in zip(df.iloc[:, 0], df.iloc[:, 1]):
        yy = int(label.strip("'").split("-")[0].strip(" '"))
        year = 1900 + yy if yy >= 50 else 2000 + yy
        for month in pd.period_range(f"{year}-10", f"{year + 1}-09", freq="M"):
            monthly[month] = volume * 1000 / 1e9 / 12
    return pd.Series(monthly).sort_index()


def make_table(counts, n_lags, production=None):
    """Inputs and target for every month t.

    Inputs: counts of months t-1 .. t-n_lags, plus (optionally) production over
    the past 12 months and its change from the 12 months before.
    Target: the count of month t.
    """
    X = pd.concat({f"lag{k}": counts.shift(k) for k in range(1, n_lags + 1)}, axis=1)
    if production is not None:
        prod = production.reindex(counts.index).fillna(0.0)   # 0 after the field closed
        X["prod_12m"] = prod.shift(1).rolling(12).sum()
        X["prod_change"] = X["prod_12m"] - X["prod_12m"].shift(12)
    table = pd.concat([X, counts.rename("target")], axis=1).dropna()
    return table[X.columns], table["target"]


def split(index, train_end, val_end):
    """Boolean masks for train (<= train_end), validation (<= val_end) and test (rest)."""
    train = index <= pd.Period(train_end, "M")
    val = ~train & (index <= pd.Period(val_end, "M"))
    test = ~train & ~val
    return train, val, test
