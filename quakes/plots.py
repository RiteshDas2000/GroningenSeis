"""The three figures."""

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

SPLITS = [("train", "#e8f0fa"), ("validation", "#fdf1e3"), ("test", "#eaf5ea")]


def event_map(events):
    """Figure 1: all events, coloured by year, size by magnitude."""
    fig, ax = plt.subplots(figsize=(7, 5.5))
    sc = ax.scatter(events["Longitude"], events["Latitude"], c=events["Time"].dt.year,
                    cmap="viridis", s=4 * 3.0 ** events["Magnitude"].clip(lower=0),
                    alpha=0.7, linewidths=0)
    fig.colorbar(sc, ax=ax, label="year")
    ax.set_aspect(1 / np.cos(np.radians(53.3)))
    ax.set_xlabel("longitude (°E)")
    ax.set_ylabel("latitude (°N)")
    ax.set_title(f"All {len(events)} induced events (size ~ magnitude)")
    fig.tight_layout()


def predictions(index, y, masks, lines, min_mag, production=None, top=5.5):
    """Figure 2: monthly counts and predictions over time.

    masks:  (train, val, test) boolean arrays
    lines:  {label: (values, colour)} for the black line and the models
    """
    t = index.to_timestamp()
    fig, ax = plt.subplots(figsize=(11, 4.5))
    for mask, (label, colour) in zip(masks, SPLITS):
        tt = t[mask]
        ax.axvspan(tt[0], tt[-1] + pd.offsets.MonthEnd(1), color=colour, zorder=0)
        ax.text(tt[0] + (tt[-1] - tt[0]) / 2, 1.0, label, transform=ax.get_xaxis_transform(),
                ha="center", va="bottom")

    ax.bar(t, y, width=25, color="#c4c4c4", label="observed count")
    for label, (values, colour) in lines.items():
        ax.plot(t, values, color=colour, lw=2.2 if colour == "black" else 1.8, label=label)

    for tt, v in zip(t[y > top], y[y > top]):            # label bars that go off the top
        ax.text(tt, top, f"{v:.0f}↑", ha="center", va="top", fontsize=8)
    ax.set_ylim(0, top)
    ax.set_ylabel(f"events M ≥ {min_mag} per month")

    handles, labels = ax.get_legend_handles_labels()
    if production is not None:                           # production on a second y-axis
        ax2 = ax.twinx()
        prod = production.reindex(index).fillna(0.0) * 12
        ax2.plot(t, prod, color="#3a8f3a", lw=1.5, ls="--")
        ax2.set_ylabel("gas production (bcm per year)", color="#3a8f3a")
        ax2.set_ylim(0, prod.max() * 1.15)
        handles += ax2.get_lines()
        labels += ["gas production (right axis)"]
    ax.legend(handles, labels, frameon=False, loc="upper left")
    fig.tight_layout()


def loss(history):
    """Figure 3: neural network loss per epoch."""
    fig, ax = plt.subplots(figsize=(7, 4))
    epochs = range(1, len(history["train"]) + 1)
    ax.plot(epochs, history["train"], label="train")
    ax.plot(epochs, history["val"], label="validation")
    ax.axvline(history["best_epoch"], color="grey", ls="--", lw=1,
               label=f"best epoch ({history['best_epoch']})")
    ax.set_xlabel("epoch")
    ax.set_ylabel("mean squared error")
    ax.set_title("Neural network loss")
    ax.legend(frameon=False)
    fig.tight_layout()
