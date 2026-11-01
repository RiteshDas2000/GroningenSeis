"""Checks on the three forecasts."""

import numpy as np
import pandas as pd
import pytest

from quakes import data, models


@pytest.fixture
def toy_problem():
    """Poisson counts with a slow trend, split into train / validation / test."""
    rng = np.random.default_rng(1)
    months = pd.period_range("1995-01", "2014-12", freq="M")
    rate = np.linspace(0.5, 2.0, len(months))
    counts = pd.Series(rng.poisson(rate), index=months, dtype=float)
    X, y = data.make_table(counts, 12)
    train, val, test = data.split(X.index, "2008-12", "2011-12")
    Xs = ((X - X[train].mean()) / X[train].std()).to_numpy()
    return X, y, Xs, train, val, test


def test_mse():
    assert models.mse([1, 2, 3], [1, 2, 3]) == 0.0
    assert models.mse([0, 0], [1, 3]) == pytest.approx(5.0)


def test_baseline_recovers_a_known_line():
    months = pd.period_range("2000-01", "2004-12", freq="M")
    X = pd.DataFrame({f"lag{k}": np.linspace(0, 5, len(months)) for k in (1, 2, 3)},
                     index=months)
    y = 0.5 + 2.0 * X["lag1"]
    train = np.ones(len(months), dtype=bool)
    pred, (intercept, slope) = models.baseline(X, y, train, n_lags=3)
    assert intercept == pytest.approx(0.5)
    assert slope == pytest.approx(2.0)
    assert np.allclose(pred, y)


def test_baseline_is_fitted_on_training_months_only(toy_problem):
    X, y, _, train, _, _ = toy_problem
    _, coef = models.baseline(X, y, train, 12)
    y_spoiled = y.copy()
    y_spoiled[~train] += 50                       # wreck validation and test targets
    _, coef_spoiled = models.baseline(X, y_spoiled, train, 12)
    assert coef == pytest.approx(coef_spoiled)


def test_poisson_predictions_are_positive(toy_problem):
    _, y, Xs, train, _, _ = toy_problem
    pred = models.poisson_regression(Xs, y, train)
    assert pred.shape == y.shape
    assert (pred > 0).all()


# ---------------------------------------------------------------- early stopping

def test_neural_network_stops_early(toy_problem, capsys):
    _, y, Xs, train, val, _ = toy_problem
    patience, epochs = 5, 300
    pred, hist = models.neural_network(Xs, y, train, val, epochs=epochs,
                                       patience=patience, min_delta=0.01, seed=0)
    n_run = len(hist["val"])
    assert n_run < epochs                                   # it stopped before the cap
    assert n_run - hist["best_epoch"] == patience           # exactly `patience` epochs without gain
    assert "stopping" in capsys.readouterr().out


def test_neural_network_returns_best_epoch_weights(toy_problem, capsys):
    _, y, Xs, train, val, _ = toy_problem
    pred, hist = models.neural_network(Xs, y, train, val, epochs=300,
                                       patience=5, min_delta=0.0, seed=0)
    best = hist["best_epoch"]
    assert hist["val"][best - 1] == pytest.approx(min(hist["val"]))
    # The returned predictions come from the best epoch's network, clipped at 0.
    # Clipping can only move a prediction closer to a non-negative count, so the
    # validation error can match the best epoch's or be lower, never higher.
    assert models.mse(y[val], pred[val]) <= hist["val"][best - 1] + 1e-9


def test_neural_network_predictions_are_non_negative(toy_problem, capsys):
    _, y, Xs, train, val, _ = toy_problem
    pred, _ = models.neural_network(Xs, y, train, val, epochs=20, patience=5, seed=0)
    assert (pred >= 0).all()
    assert pred.shape == y.shape


def test_neural_network_is_reproducible(toy_problem, capsys):
    _, y, Xs, train, val, _ = toy_problem
    a, _ = models.neural_network(Xs, y, train, val, epochs=20, patience=5, seed=3)
    b, _ = models.neural_network(Xs, y, train, val, epochs=20, patience=5, seed=3)
    assert np.allclose(a, b)
