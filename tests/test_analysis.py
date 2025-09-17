import os
import sys
import pandas as pd
import numpy as np
import pytest

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from src.analysis_utils import (
    load_data,
    validate_columns,
    filter_high_gld,
    yearly_stats,
    train_linear_model,
)


def _tiny_df():
    dates = pd.date_range("2024-01-01", periods=12, freq="D")
    df = pd.DataFrame({
        "Date": dates,
        "SPX": np.linspace(3500, 3560, 12),
        "GLD": np.linspace(150, 200, 12),
        "USO": np.linspace(70, 75, 12),
        "SLV": np.linspace(18, 19, 12),
        "EUR/USD": np.linspace(1.10, 1.12, 12),
    }).set_index("Date")
    return df


def test_load_data_from_repo_root():
    root = os.path.dirname(os.path.dirname(__file__))
    csv_path = os.path.join(root, "gold_data_2015_25.csv")
    df = load_data(csv_path)
    assert len(df) > 0
    assert "GLD" in df.columns


def test_validate_columns_raises_on_missing():
    df = pd.DataFrame({"GLD": [1, 2], "SPX": [3, 4]})
    with pytest.raises(ValueError):
        validate_columns(df)


def test_filter_high_gld_top_quartile():
    df = _tiny_df()
    top = filter_high_gld(df, q=0.75)
    assert len(top) > 0
    assert top["GLD"].min() >= df["GLD"].quantile(0.75) - 1e-9


def test_yearly_stats_has_expected_cols():
    df = _tiny_df()
    ys = yearly_stats(df)
    assert {"mean", "std", "min", "max", "count"}.issubset(ys.columns)


def test_train_linear_model_returns_metrics():
    df = _tiny_df()
    _, (_, y_test, y_pred), metrics = train_linear_model(df, test_size=0.25, random_state=0)
    assert len(y_test) == len(y_pred)
    assert "r2" in metrics and "mae" in metrics
