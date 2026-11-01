"""Shared fixtures. Makes `quakes` importable when pytest runs from the repo folder."""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from quakes import data  # noqa: E402

DATA = ROOT / "data"


@pytest.fixture(scope="session")
def events():
    return data.load_events(DATA / "knmi_groningen_events.txt")


@pytest.fixture(scope="session")
def production():
    return data.load_production(DATA / "groningen_production_gasyear.csv")


@pytest.fixture
def toy_counts():
    """Five years of monthly counts with a known pattern (0, 1, 2, ... repeating)."""
    months = pd.period_range("2000-01", "2004-12", freq="M")
    return pd.Series(np.arange(len(months)) % 5, index=months, dtype=float)


@pytest.fixture
def toy_events():
    """A handful of events with known times and magnitudes."""
    return pd.DataFrame({
        "Time": pd.to_datetime(["2020-01-05 00:00", "2020-01-20 00:00", "2020-01-25 00:00",
                                "2020-03-10 00:00", "2020-03-31 23:59"]),
        "Magnitude": [1.4, 1.5, 2.8, 1.6, 3.0],
    })
