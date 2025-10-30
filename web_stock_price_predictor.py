import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from typing import Optional
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error

"""
Beginner-friendly Streamlit app (no deep learning)
- Downloads daily data with yfinance
- Creates simple lag features (previous N closes)
- Trains a Linear Regression model
- Shows test metrics (RMSE, MAE, MAPE) and a simple next-day prediction
"""

st.title("Beginner Stock Price Predictor (Regression)")

# Sidebar controls for beginner-friendly choices
with st.sidebar:
    st.header("Settings")
    stock = st.text_input("Stock ticker", "GOOG")
    period = st.selectbox("History period", ["6mo", "1y", "2y", "5y"], index=1)
    lags = st.slider("Lag features (days)", 1, 30, 5, help="How many previous days to use as input")
    test_size_pct = st.slider("Test size % (most recent)", 10, 50, 20)

# No auto-refresh; keep the app static for clarity

@st.cache_data(show_spinner=False, ttl=60)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    """Download daily OHLCV data from yfinance and flatten columns if needed."""
    df = yf.download(ticker, period=period, interval="1d")
    if df is None or df.empty:
        return pd.DataFrame()
    # If MultiIndex columns (can happen depending on yfinance settings), flatten them
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [" ".join([str(x) for x in col if x is not None]).strip() for col in df.columns.values]
    return df
data = load_data(stock, period)

if data.empty:
    st.error("No data found. Please check the ticker symbol or try a different date range.")
    st.stop()

def resolve_price_column(df: pd.DataFrame) -> Optional[str]:
    """Choose a reasonable price column with robust matching.

    Preference order:
      1) 'Adj Close' (any case/spacing variant)
      2) 'Close' (any case/spacing variant)
      3) any column containing 'close'
      4) first numeric column
    """
    cols = [str(c) for c in df.columns]
    lower_map = {str(c).lower(): str(c) for c in cols}

    # helpers
    def find_exact(name: str) -> Optional[str]:
        key = name.lower()
        if key in lower_map:
            return lower_map[key]
        return None

    def find_contains(piece: str) -> Optional[str]:
        piece_l = piece.lower()
        for lc, orig in lower_map.items():
            if piece_l in lc:
                return orig
        return None

    # 1) Adj Close variants
    for candidate in ["adj close", "adj_close", "adjclose"]:
        exact = find_exact(candidate)
        if exact:
            return exact
    contains_adj_close = None
    for piece in ["adj close", "adj_close", "adjclose"]:
        contains_adj_close = find_contains(piece)
        if contains_adj_close:
            return contains_adj_close

    # 2) Close variants
    for candidate in ["close"]:
        exact = find_exact(candidate)
        if exact:
            return exact
    contains_close = find_contains("close")
    if contains_close:
        return contains_close

    # 3) first numeric column fallback
    for c in df.columns:
        s = df[c]
        if pd.api.types.is_numeric_dtype(s):
            return str(c)
    return None

price_col = resolve_price_column(data)
if price_col is None:
    st.error("Could not find a usable price column in the downloaded data.")
    st.write({"columns": list(map(str, data.columns))})
    st.dataframe(data.head())
    st.stop()
elif price_col not in ("Adj Close", "Close"):
    st.info(f"Using column '{price_col}' as price (fallback)")

st.caption("Method: use previous N closing prices as inputs to predict the next close using Linear Regression.")

st.subheader(f"{stock} {price_col}")
st.line_chart(data[[price_col]])

# --- Model helpers ---
def make_supervised(series: pd.Series, lags: int) -> pd.DataFrame:
    df = pd.DataFrame({"target": series})
    for i in range(1, lags + 1):
        df[f"lag_{i}"] = series.shift(i)
    df = df.dropna()
    return df

series = data[price_col].astype(float)
supervised = make_supervised(series, lags)
if supervised.empty or len(supervised) < 10:
    st.warning("Not enough data after creating lag features. Increase period or reduce lags.")
    st.stop()

test_size = max(1, int(len(supervised) * (test_size_pct / 100.0)))
train = supervised.iloc[:-test_size]
test = supervised.iloc[-test_size:]

X_train = train.drop(columns=["target"]).values
y_train = train["target"].values
X_test = test.drop(columns=["target"]).values
y_test = test["target"].values

model = LinearRegression()
model.fit(X_train, y_train)

# Predict and inverse transform
pred = model.predict(X_test)

# Compute metrics: RMSE, MAE, MAPE
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
mae = float(mean_absolute_error(y_test, pred))
with np.errstate(divide='ignore', invalid='ignore'):
    mape_vals = np.abs((pred - y_test) / np.where(y_test == 0, np.nan, y_test)) * 100.0
    mape = float(np.nanmean(mape_vals))
cols = st.columns(3)
cols[0].metric("RMSE", f"{rmse:,.2f}")
cols[1].metric("MAE", f"{mae:,.2f}")
cols[2].metric("MAPE %", f"{mape:,.2f}")

# Build a DataFrame aligned to original index for plotting
plot_df = pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred,
}, index=test.index)

st.subheader("Actual vs Predicted (test set)")
st.line_chart(plot_df)

# Simple next-day prediction using latest lags
st.subheader("Next day prediction (naive)")
latest_lags = []
for i in range(1, lags + 1):
    latest_lags.append(series.iloc[-i])
latest_X = np.array(latest_lags).reshape(1, -1)
next_day_pred = float(model.predict(latest_X)[0])
st.metric("Predicted next close", f"{next_day_pred:,.2f}")