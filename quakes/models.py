"""The three forecasts: a simple baseline, Poisson regression and a neural network."""

import copy

import numpy as np
from sklearn.linear_model import PoissonRegressor
from sklearn.neural_network import MLPRegressor


def mse(a, b):
    return float(np.mean((np.asarray(a) - np.asarray(b)) ** 2))


def baseline(X, y, train, n_lags):
    """Black line: straight-line fit of next month's count vs the past 12-month average.

    Fitted on the training months only. Returns the prediction for every month
    and the fitted (intercept, slope).
    """
    past_avg = X[[f"lag{k}" for k in range(1, n_lags + 1)]].mean(axis=1)
    slope, intercept = np.polyfit(past_avg[train], y[train], 1)
    return (intercept + slope * past_avg).to_numpy(), (intercept, slope)


def poisson_regression(Xs, y, train, alpha=1.0):
    model = PoissonRegressor(alpha=alpha, max_iter=1000).fit(Xs[train], y[train])
    return model.predict(Xs)


def neural_network(Xs, y, train, val, hidden=(8,), activation="relu", l2=2.0,
                   learning_rate=0.001, batch_size=16, epochs=300, patience=20,
                   min_delta=0.01, seed=0):
    """Train epoch by epoch, print the loss, stop when validation stops improving.

    Returns the predictions of the best epoch and the loss history.
    """
    y = np.asarray(y)
    net = MLPRegressor(hidden_layer_sizes=hidden, activation=activation, alpha=l2,
                       learning_rate_init=learning_rate, batch_size=batch_size,
                       random_state=seed)
    rng = np.random.default_rng(seed)
    history = {"train": [], "val": [], "best_epoch": 0}
    best_val, best_net = np.inf, None

    for epoch in range(1, epochs + 1):
        order = rng.permutation(np.flatnonzero(train))   # shuffle the training months
        net.partial_fit(Xs[order], y[order])             # one pass over the training data
        history["train"].append(mse(y[train], net.predict(Xs[train])))
        history["val"].append(mse(y[val], net.predict(Xs[val])))
        print(f"epoch {epoch:4d}   train loss {history['train'][-1]:.3f}   "
              f"val loss {history['val'][-1]:.3f}")

        if history["val"][-1] < best_val - min_delta:    # new best: keep these weights
            best_val, best_net = history["val"][-1], copy.deepcopy(net)
            history["best_epoch"] = epoch
        elif epoch - history["best_epoch"] >= patience:
            print(f"no improvement for {patience} epochs, stopping")
            break

    print(f"using the weights from epoch {history['best_epoch']} (val loss {best_val:.3f})")
    return np.maximum(best_net.predict(Xs), 0), history   # counts can't be negative
