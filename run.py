"""Forecast next month's number of Groningen earthquakes.

Figure 1: all events on a map
Figure 2: monthly counts with the predictions (train / validation / test)
Figure 3: neural network loss per epoch

Run:  python run.py
"""
from pathlib import Path

import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler

from quakes import data, models, plots

# ---------------------------------------------------------------- settings
DATA = Path(__file__).resolve().parent / "data"
EVENTS_FILE = DATA / "knmi_groningen_events.txt"
PRODUCTION_FILE = DATA / "groningen_production_gasyear.csv"   # dashboardgroningen.nl
USE_PRODUCTION = True    # False = inputs are only the past counts

MIN_MAG = 1.5            # only count events with magnitude >= this
START, END = "1995-01", "2026-09"
N_LAGS = 12              # inputs: counts of the last N_LAGS months
TRAIN_END = "2022-12"    # train on START .. TRAIN_END
VAL_END = "2024-12"      # validate on TRAIN_END .. VAL_END, test on the rest

NETWORK = dict(
    hidden=(8,),         # nodes per hidden layer, e.g. (16, 8) for two layers
    activation="relu",   # relu, tanh or logistic
    l2=2.0,              # weight decay (try 0.01 to see overfitting)
    learning_rate=0.001,
    batch_size=16,
    epochs=300,          # maximum; stops earlier when validation stops improving
    patience=20,
    min_delta=0.01,
    seed=0,
)

# ---------------------------------------------------------------- data
events = data.load_events(EVENTS_FILE)
counts = data.monthly_counts(events, MIN_MAG, START, END)
production = data.load_production(PRODUCTION_FILE) if USE_PRODUCTION else None
X, y = data.make_table(counts, N_LAGS, production)
train, val, test = data.split(X.index, TRAIN_END, VAL_END)

print(f"{len(events)} induced events, {int(counts.sum())} with M >= {MIN_MAG}")
print(f"inputs: {', '.join(X.columns)}")
print(f"months: train {train.sum()}, validation {val.sum()}, test {test.sum()}\n")

Xs = StandardScaler().fit(X[train]).transform(X)     # scale using training data only

# ---------------------------------------------------------------- models
pred_base, (a, b) = models.baseline(X, y, train, N_LAGS)
pred_poisson = models.poisson_regression(Xs, y, train)
pred_net, history = models.neural_network(Xs, y, train, val, **NETWORK)

print(f"\nbaseline: next month = {a:.2f} + {b:.2f} x (past 12-month average)")
print("mean squared error")
for name, p in [("baseline", pred_base), ("Poisson regression", pred_poisson),
                ("neural network", pred_net)]:
    print(f"  {name:20s} train {models.mse(y[train], p[train]):.3f}   "
          f"val {models.mse(y[val], p[val]):.3f}   test {models.mse(y[test], p[test]):.3f}")

# ---------------------------------------------------------------- figures
plots.event_map(events)
plots.predictions(X.index, y, (train, val, test), {
    "expected count given the past 12 months": (pred_base, "black"),
    "Poisson regression": (pred_poisson, "#2f6db3"),
    "neural network": (pred_net, "#d9622b"),
}, MIN_MAG, production)
plots.loss(history)
plt.show()
