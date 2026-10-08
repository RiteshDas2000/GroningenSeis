import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from quakes import data, models

ROOT = Path(__file__).resolve().parents[1]


def test_monthly_counts_fill_empty_months():
    events = pd.DataFrame({"Time": pd.to_datetime(["2020-01-05", "2020-01-20", "2020-03-02"]),
                           "Magnitude": [1.6, 0.8, 2.0]})
    c = data.monthly_counts(events, 1.5, "2020-01", "2020-04")
    assert c.tolist() == [1, 0, 1, 0]


def test_inputs_only_use_the_past():
    counts = pd.Series(np.arange(30.0), index=pd.period_range("2000-01", periods=30, freq="M"))
    X, y = data.make_table(counts, n_lags=3)
    t = pd.Period("2001-06", "M")
    assert y[t] == counts[t]
    assert [X.loc[t, f"lag{k}"] for k in (1, 2, 3)] == [counts[t - 1], counts[t - 2], counts[t - 3]]


def test_production_units():
    prod = data.load_production(ROOT / "data" / "groningen_production_gasyear.csv")
    record_year = prod["2012-10":"2013-09"].sum()
    assert abs(record_year - 53.3) < 0.1          # bcm in gas year 2012-13
    assert prod["2024-01"] == 0


def test_split_covers_every_month_once():
    idx = pd.period_range("2010-01", "2020-12", freq="M")
    train, val, test = data.split(idx, "2014-12", "2017-12")
    assert (train.astype(int) + val + test == 1).all()


def test_network_stops_early_and_predicts_positive():
    rng = np.random.default_rng(0)
    Xs = rng.normal(size=(120, 3))
    y = rng.poisson(1.0, 120).astype(float)
    train = np.arange(120) < 80
    pred, history = models.neural_network(Xs, y, train, ~train, epochs=200, patience=5)
    assert len(history["train"]) < 200
    assert (pred >= 0).all()
