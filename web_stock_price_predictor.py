import streamlit as st
import pandas as pd
import numpy as np
from keras.models import load_model, Sequential
from keras.layers import Dense, LSTM, Dropout
from keras.callbacks import EarlyStopping
import matplotlib.pyplot as plt
import yfinance as yf
from datetime import datetime
from typing import List, Optional
from pathlib import Path
from sklearn.preprocessing import MinMaxScaler

# ------------------------------------------------------------
# Beginner-friendly Streamlit app for Stock Price Prediction
# - Downloads data from yfinance (supports period/interval for near real-time)
# - Shows simple moving averages
# - Loads a trained LSTM model (from the notebook)
# - Plots original vs predicted values and shows RMSE
# - Optionally forecasts next N business days
# ------------------------------------------------------------

st.title("Stock Price Predictor App")

# Sidebar controls for beginner-friendly choices
with st.sidebar:
    st.header("Settings")
    stock = st.text_input("Enter stock ticker (e.g., GOOG, AAPL, MSFT)", "GOOG")
    st.caption("Use Period + Interval to control how much data and at what granularity to download.")

    period_options = ["1d","5d","1mo","3mo","6mo","1y","2y","5y","10y","ytd","max"]
    interval_options = ["1m","2m","5m","15m","30m","60m","90m","1h","1d","5d","1wk","1mo","3mo"]
    period = st.selectbox("Period", options=period_options, index=5, help="How much history to fetch")
    interval = st.selectbox("Interval", options=interval_options, index=8, help="Bar size")
    include_prepost = st.checkbox("Include pre/post market (where available)", value=False)

    window = st.slider("Lookback window (bars)", min_value=30, max_value=200, value=100, step=10, help="Number of past bars used to predict the next bar")
    forecast_days = st.slider("Forecast next N business days", min_value=0, max_value=30, value=5, help="Naive iterative forecast using the trained model")

    st.divider()
    st.subheader("Training")
    train_epochs = st.slider("Training epochs (if training in app)", min_value=1, max_value=50, value=10, step=1)
    train_now = st.checkbox("Train/Update model now (use these epochs)", value=False, help="If checked, trains a small LSTM now; otherwise loads saved model if available")

# No auto-refresh; keep the app static for clarity

@st.cache_data(show_spinner=False, ttl=60)
def load_data(ticker: str, period: str, interval: str, prepost: bool) -> pd.DataFrame:
    """Download OHLCV data from yfinance with simple column handling.

    Uses period/interval for near real-time updates. TTL ensures cache refreshes periodically.
    """
    df = yf.download(ticker, period=period, interval=interval, prepost=prepost)
    if df is None or df.empty:
        return pd.DataFrame()
    # Handle MultiIndex columns (when multiple tickers are used)
    if isinstance(df.columns, pd.MultiIndex):
        # Try to select by locating the level that contains the ticker
        found = False
        try:
            mi = df.columns
            for lvl in range(mi.nlevels):
                if ticker in mi.get_level_values(lvl):
                    df = df.xs(ticker, axis=1, level=lvl)
                    found = True
                    break
        except Exception:
            found = False
        if not found:
            # Flatten columns like ('Close','AAPL') -> 'Close AAPL'
            df.columns = [" ".join([str(x) for x in col if x is not None]).strip() for col in df.columns.values]
    return df

# Validate period/interval combination for yfinance
minute_intervals = {"1m","2m","5m","15m","30m","60m","90m"}
effective_period = period
if interval in minute_intervals:
    # yfinance typically limits minute data to recent days; adjust if too long
    allowed_for_minute = {"1d","5d","7d","1mo"}
    if period not in allowed_for_minute:
        st.warning(f"Interval {interval} may not support period {period}. Using 5d instead.")
        effective_period = "5d"

data = load_data(stock, effective_period, interval, include_prepost)

if data.empty:
    st.error("No data found. Please check the ticker symbol or try a different date range.")
    st.stop()

# Determine price column robustly (prefer Adj Close, else Close)
def resolve_price_column(df: pd.DataFrame, ticker: str) -> Optional[str]:
    cols = [str(c) for c in df.columns]
    # Exact matches first
    if 'Adj Close' in cols:
        return 'Adj Close'
    if 'Close' in cols:
        return 'Close'
    # Look for variants like 'Adj Close AAPL' or 'AAPL Adj Close'
    lc_cols = [c.lower() for c in cols]
    ticker_lc = ticker.lower()
    # Prefer columns that include both the price name and ticker
    for key in ['adj close', 'adj_close']:
        for i, c in enumerate(lc_cols):
            if key in c and ticker_lc in c:
                return cols[i]
    # Then any column with adj close
    for key in ['adj close', 'adj_close']:
        for i, c in enumerate(lc_cols):
            if key in c:
                return cols[i]
    # Then try 'close' with ticker
    for i, c in enumerate(lc_cols):
        if 'close' in c and 'adj' not in c and ticker_lc in c:
            return cols[i]
    # Fallback: any 'close' without 'adj'
    for i, c in enumerate(lc_cols):
        if 'close' in c and 'adj' not in c:
            return cols[i]
    return None

price_col = resolve_price_column(data, stock)
if price_col is None:
    st.error("Could not find 'Adj Close' or 'Close' column in the downloaded data.")
    st.dataframe(data.tail())
    st.stop()

# Allow user to override price column if both exist
candidate_price_cols = [c for c in data.columns if str(c).lower().endswith("adj close") or str(c).lower().endswith("close") or str(c).lower()=="adj close" or str(c).lower()=="close"]
if len(candidate_price_cols) > 1:
    with st.sidebar:
        price_col = st.selectbox("Price column", options=candidate_price_cols, index=(candidate_price_cols.index(price_col) if price_col in candidate_price_cols else 0))

st.subheader("Raw Data")
st.caption("Quick peek at the latest rows")
st.dataframe(data.tail())

# Always show a basic price chart so users see a graph even if later steps fail
try:
    fig0, ax0 = plt.subplots(figsize=(15, 5))
    ax0.plot(data.index, data[price_col], label=price_col)
    ax0.set_title(f"{stock} {price_col}")
    ax0.set_xlabel("Date")
    ax0.set_ylabel("Price")
    ax0.legend()
    fig0.tight_layout()
    st.pyplot(fig0)
except Exception as e:
    st.warning(f"Could not render price chart: {e}")

# --- Model helpers ---
def make_sequences(scaled: np.ndarray, window: int):
    X, y = [], []
    for i in range(window, len(scaled)):
        X.append(scaled[i-window:i])
        y.append(scaled[i])
    return np.array(X), np.array(y)

def load_or_train_model(model_path: Path, x_train: np.ndarray, y_train: np.ndarray, window: int, epochs: int, force_train: bool = False):
    # Try to load an existing model
    if not force_train:
        try:
            if model_path.exists():
                model = load_model(str(model_path))
                return model, True
        except Exception:
            pass

    # Train a small fallback model quickly
    model = Sequential([
        LSTM(32, return_sequences=True, input_shape=(window, 1)),
        Dropout(0.1),
        LSTM(16, return_sequences=False),
        Dense(8, activation='relu'),
        Dense(1)
    ])
    model.compile(optimizer='adam', loss='mean_squared_error')
    callbacks = [EarlyStopping(monitor='loss', patience=3, restore_best_weights=True)]
    try:
        model.fit(x_train, y_train, batch_size=32, epochs=epochs, verbose=0, validation_split=0.1, callbacks=callbacks)
    except Exception:
        model.fit(x_train, y_train, batch_size=1, epochs=max(2, epochs//2), verbose=0)

    # Try to save for next time
    try:
        model.save(str(model_path))
        st.success(f"Trained a quick model and saved to {model_path.name}")
    except Exception:
        st.info("Trained a quick model in memory (saving skipped).")
    return model, False

# Helper: plot moving averages alongside price
def plot_ma(df: pd.DataFrame, col: str, ma_days: List[int]):
    fig, ax = plt.subplots(figsize=(15, 6))
    ax.plot(df.index, df[col], label=col, color='tab:blue')
    for d in ma_days:
        ax.plot(df.index, df[col].rolling(d).mean(), label=f"MA {d}")
    ax.set_title(f"{stock} {col} with Moving Averages")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    fig.tight_layout()
    return fig

st.subheader('Price with Moving Averages')
st.caption("Note: For intraday intervals, MA windows represent bars (e.g., 100 = 100 minutes if interval=1m).")
try:
    st.pyplot(plot_ma(data.copy(), price_col, [100, 200, 250]))
except Exception as e:
    st.info(f"Skipping MA plot due to: {e}")

"""
Prepare data and model
- Fit scaler on the full price series
- Build sequences from the full series
- Split 70/30 into train/test
- Load existing model if present; otherwise train/update a small model now
"""
full_series = pd.DataFrame(data[price_col]).copy()
scaler = MinMaxScaler(feature_range=(0, 1))
full_scaled = scaler.fit_transform(full_series[[price_col]])

if len(full_scaled) < window + 1:
    st.warning("Not enough data for the selected window. Reduce the window, increase period, or use a larger interval.")
    st.stop()

X, Y = make_sequences(full_scaled, window)
split_idx = int(len(X) * 0.7)
x_train, y_train = X[:split_idx], Y[:split_idx]
x_test, y_test = X[split_idx:], Y[split_idx:]

model_path = Path("Latest_stock_price_model.keras")
model, loaded_from_disk = load_or_train_model(model_path, x_train, y_train, window, epochs=train_epochs, force_train=train_now)

# Predict and inverse transform
pred_scaled = model.predict(x_test)
inv_pred = scaler.inverse_transform(pred_scaled)
inv_y = scaler.inverse_transform(y_test)

# Compute metrics: RMSE, MAE, MAPE
rmse = float(np.sqrt(np.mean((inv_pred - inv_y) ** 2)))
mae = float(np.mean(np.abs(inv_pred - inv_y)))
with np.errstate(divide='ignore', invalid='ignore'):
    mape_vals = np.abs((inv_pred - inv_y) / np.where(inv_y==0, np.nan, inv_y)) * 100.0
    mape = float(np.nanmean(mape_vals))
cols = st.columns(3)
cols[0].metric("RMSE", f"{rmse:,.2f}")
cols[1].metric("MAE", f"{mae:,.2f}")
cols[2].metric("MAPE %", f"{mape:,.2f}")

# Build a DataFrame aligned to original index for plotting
plot_index = data.index[window + split_idx:]
plot_df = pd.DataFrame({
    'original_test_data': inv_y.reshape(-1),
    'predictions': inv_pred.reshape(-1)
}, index=plot_index)

st.subheader("Original vs Predicted (Test segment)")
st.dataframe(plot_df.tail())

# Plot full series with predictions overlaid
fig, ax = plt.subplots(figsize=(15, 6))
ax.plot(data.index, data[price_col], label='Full Price Series', color='lightgray')
ax.plot(plot_df.index, plot_df['original_test_data'], label='Original Test Data', color='tab:blue')
ax.plot(plot_df.index, plot_df['predictions'], label='Predicted', color='tab:orange')
ax.set_title(f"{stock} {price_col}: Original vs Predicted")
ax.set_xlabel("Date")
ax.set_ylabel("Price")
ax.legend()
fig.tight_layout()
st.pyplot(fig)

# Optional: Forecast next N business days (naive iterative)
if forecast_days and forecast_days > 0:
    st.subheader(f"Next {forecast_days} business-day forecast")
    # Use the already-fitted scaler and full_scaled
    if len(full_scaled) >= window:
        last_window = full_scaled[-window:].copy()
        future_scaled = []
        for _ in range(forecast_days):
            x_input = last_window.reshape(1, last_window.shape[0], 1)
            next_scaled = model.predict(x_input, verbose=0)[0, 0]
            future_scaled.append(next_scaled)
            last_window = np.vstack([last_window[1:], [[next_scaled]]])
        future_prices = scaler.inverse_transform(np.array(future_scaled).reshape(-1, 1)).flatten()
        future_index = pd.bdate_range(start=data.index[-1] + pd.Timedelta(days=1), periods=forecast_days)
        forecast_df = pd.DataFrame({"forecast": future_prices}, index=future_index)

        # Plot last 100 points + forecast
        tail_plot = pd.concat([full_series.tail(100), forecast_df], axis=0)
        fig2, ax2 = plt.subplots(figsize=(15, 5))
        ax2.plot(tail_plot.index, tail_plot.iloc[:, 0], label=price_col)
        ax2.plot(forecast_df.index, forecast_df['forecast'], label='Forecast', color='tab:orange')
        ax2.set_title(f"{stock} {price_col}: Next {forecast_days} business-day forecast")
        ax2.set_xlabel("Date")
        ax2.set_ylabel("Price")
        ax2.legend()
        fig2.tight_layout()
        st.pyplot(fig2)
        st.dataframe(forecast_df)
    else:
        st.info("Not enough data to generate a forecast with the selected window.")