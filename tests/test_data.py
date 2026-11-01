"""Checks on loading the data and building the inputs."""

import numpy as np
import pandas as pd
import pytest

from quakes import data


# ---------------------------------------------------------------- counts

def test_monthly_counts_respects_magnitude_cutoff(toy_events):
    counts = data.monthly_counts(toy_events, 1.5, "2020-01", "2020-03")
    # Jan: 1.5 and 2.8 count, 1.4 does not. Feb: none. Mar: both.
    assert counts.tolist() == [2.0, 0.0, 2.0]


def test_monthly_counts_fills_empty_months_with_zero(toy_events):
    counts = data.monthly_counts(toy_events, 1.5, "2019-11", "2020-06")
    assert len(counts) == 8
    assert counts["2019-11"] == 0 and counts["2020-06"] == 0
    assert not counts.isna().any()


def test_monthly_counts_total_matches_catalogue(events):
    counts = data.monthly_counts(events, 1.5, "1995-01", "2026-09")
    in_window = events[(events["Magnitude"] >= 1.5)
                       & (events["Time"] >= "1995-01-01")
                       & (events["Time"] < "2026-10-01")]
    assert counts.sum() == len(in_window)


def test_catalogue_only_has_induced_events(events):
    assert (events["EventType"] == "induced or triggered event").all()
    assert len(events) > 1000


# ---------------------------------------------------------------- production units

def test_production_record_year_is_about_53_bcm(production):
    # Gas year 2012-13 (Oct 2012 - Sep 2013), the modern peak: about 53 bcm.
    year = production[pd.Period("2012-10", "M"):pd.Period("2013-09", "M")]
    assert len(year) == 12
    assert 50 < year.sum() < 56


def test_production_is_spread_evenly_over_the_gas_year(production):
    year = production[pd.Period("2012-10", "M"):pd.Period("2013-09", "M")]
    assert np.allclose(year, year.iloc[0])


def test_production_is_monthly_and_non_negative(production):
    assert production.index.freqstr.startswith("M")
    assert production.index.is_monotonic_increasing
    assert not production.index.duplicated().any()
    assert (production >= 0).all()
    # The all-time peak was gas year 1976 at about 88 bcm, i.e. ~7.3 bcm a month.
    # Anything far above that means the thousands-of-Nm3 conversion broke.
    assert 6 < production.max() < 8


# ---------------------------------------------------------------- no future data

def test_lags_point_to_past_months(toy_counts):
    X, y = data.make_table(toy_counts, n_lags=3)
    for t in X.index[:10]:
        for k in (1, 2, 3):
            assert X.loc[t, f"lag{k}"] == toy_counts[t - k]
        assert y[t] == toy_counts[t]


def test_changing_this_months_count_does_not_change_its_inputs(toy_counts):
    X_before, _ = data.make_table(toy_counts, n_lags=12)
    t = X_before.index[20]
    changed = toy_counts.copy()
    changed[t] += 100
    X_after, y_after = data.make_table(changed, n_lags=12)
    pd.testing.assert_series_equal(X_before.loc[t], X_after.loc[t])
    assert y_after[t] == toy_counts[t] + 100


def test_changing_this_months_production_does_not_change_its_inputs(toy_counts):
    prod = pd.Series(1.0, index=toy_counts.index)
    X_before, _ = data.make_table(toy_counts, 12, prod)
    t = X_before.index[-1]
    changed = prod.copy()
    changed[t] = 999.0
    X_after, _ = data.make_table(toy_counts, 12, changed)
    pd.testing.assert_series_equal(X_before.loc[t], X_after.loc[t])


def test_production_inputs_use_the_previous_12_months(toy_counts):
    prod = pd.Series(np.arange(len(toy_counts), dtype=float), index=toy_counts.index)
    X, _ = data.make_table(toy_counts, 12, prod)
    t = X.index[5]
    expected = prod[t - 12:t - 1].sum()           # months t-12 .. t-1
    assert X.loc[t, "prod_12m"] == pytest.approx(expected)
    prev = prod[t - 24:t - 13].sum()              # months t-24 .. t-13
    assert X.loc[t, "prod_change"] == pytest.approx(expected - prev)


def test_production_after_closure_counts_as_zero(toy_counts):
    prod = pd.Series(2.0, index=toy_counts.index[:30])   # field closes after month 30
    X, _ = data.make_table(toy_counts, 12, prod)
    assert X["prod_12m"].iloc[-1] == 0.0
    assert not X.isna().any().any()


# ---------------------------------------------------------------- split

def test_split_is_chronological_and_complete(toy_counts):
    X, _ = data.make_table(toy_counts, 12)
    train, val, test = data.split(X.index, "2002-12", "2003-12")
    assert (train.astype(int) + val.astype(int) + test.astype(int) == 1).all()
    assert X.index[train].max() < X.index[val].min()
    assert X.index[val].max() < X.index[test].min()
    assert X.index[train].max() == pd.Period("2002-12", "M")
    assert X.index[val].max() == pd.Period("2003-12", "M")
