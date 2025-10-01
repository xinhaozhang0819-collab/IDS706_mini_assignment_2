import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score, mean_absolute_error

REQUIRED_COLS = ["SPX", "GLD", "USO", "SLV", "EUR/USD"]


def load_data(path: str) -> pd.DataFrame:
    """Read CSV, parse Date, set Date index."""
    df = pd.read_csv(path)
    if "Date" in df.columns:
        df["Date"] = pd.to_datetime(df["Date"], errors="coerce")
        df = df.set_index("Date").sort_index()
    return df


def validate_columns(df: pd.DataFrame, cols=REQUIRED_COLS):
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"Missing required columns: {missing}")


def filter_high_gld(df: pd.DataFrame, q: float = 0.75) -> pd.DataFrame:
    validate_columns(df)
    gld_threshold = df["GLD"].quantile(q)
    return df[df["GLD"] > gld_threshold]


def _resolve_years_and_gld(df: pd.DataFrame):
    """Extract (years, gld_series) from df regardless of whether Date is index or column."""
    if df.index.name == "Date" and pd.api.types.is_datetime64_any_dtype(df.index):
        years = df.index.year
        gld = df["GLD"]
    elif "Date" in df.columns and pd.api.types.is_datetime64_any_dtype(df["Date"]):
        years = pd.to_datetime(df["Date"]).dt.year
        gld = df["GLD"]
    else:
        raise ValueError(
            "No datetime index/column named 'Date' to compute yearly stats."
        )
    return years, gld


def yearly_stats(df: pd.DataFrame) -> pd.DataFrame:
    years, gld = _resolve_years_and_gld(df)
    return gld.groupby(years).agg(["mean", "std", "min", "max", "count"])


def train_linear_model(df: pd.DataFrame, test_size=0.2, random_state=42):
    validate_columns(df)
    X = df[["SPX", "USO", "SLV", "EUR/USD"]]
    y = df["GLD"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state
    )
    model = LinearRegression().fit(X_train, y_train)
    y_pred = model.predict(X_test)
    metrics = {
        "r2": r2_score(y_test, y_pred),
        "mae": mean_absolute_error(y_test, y_pred),
        "coef": dict(zip(X.columns, model.coef_)),
        "intercept": float(model.intercept_),
    }
    return model, (X_test, y_test, y_pred), metrics