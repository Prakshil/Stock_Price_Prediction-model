import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, HistGradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import matplotlib.pyplot as plt

# ======================
# Helpers (define FIRST)
# ======================
def _get_price_series(df: pd.DataFrame) -> pd.Series:
    """Return a 1-D price series from a yfinance DataFrame (robust to MultiIndex/1-col)."""
    # MultiIndex (e.g., ('Close','AAPL'))
    if isinstance(df.columns, pd.MultiIndex):
        for name in ["Adj Close", "Close"]:
            candidates = [c for c in df.columns if isinstance(c, tuple) and name in c]
            if candidates:
                s = df[candidates[0]]
                if isinstance(s, pd.DataFrame):
                    s = s.iloc[:, 0]
                s = pd.to_numeric(s.squeeze(), errors="coerce")
                s.name = "Price"
                return s

    # Normal single index
    for name in ["Adj Close", "Close", "AdjClose", "close", "adjclose"]:
        if name in df.columns:
            s = df[name]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = pd.to_numeric(s.squeeze(), errors="coerce")
            s.name = "Price"
            return s

    # Fallback
    s = df.select_dtypes(include=[np.number]).iloc[:, 0]
    s = pd.to_numeric(s.squeeze(), errors="coerce")
    s.name = "Price"
    return s

def _ema(series: pd.Series, span: int) -> pd.Series:
    return series.ewm(span=span, adjust=False).mean()

def _macd_features(series: pd.Series) -> pd.DataFrame:
    """MACD, signal, hist — lagged by 1 to avoid look-ahead."""
    ema12 = _ema(series, 12)
    ema26 = _ema(series, 26)
    macd = ema12 - ema26
    signal = _ema(macd, 9)
    hist = macd - signal
    return pd.DataFrame({
        "macd_1": macd.shift(1),
        "macd_signal_1": signal.shift(1),
        "macd_hist_1": hist.shift(1),
    }, index=series.index)

def build_features(df: pd.DataFrame, lags: int = 3) -> pd.DataFrame:
    """Create lag features + MACD features + target."""
    series = _get_price_series(df)
    out = pd.DataFrame(index=series.index)
    # price lags
    for i in range(1, lags + 1):
        out[f"lag_{i}"] = series.shift(i)
    # MACD block
    out = out.join(_macd_features(series))
    # target
    out["target"] = series
    out.dropna(inplace=True)
    return out

# ======================
# Streamlit setup
# ======================
st.set_page_config(page_title="Enhanced Stock Price Predictor", layout="wide")
st.title("📈 Enhanced Stock Price Prediction App")

# Sidebar controls
st.sidebar.header("Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE.NS)", "AAPL")
period = st.sidebar.selectbox("Select Period", ["1y", "2y", "5y", "10y"], index=0)
lag_days = st.sidebar.slider("Number of Lag Days", 1, 10, 3)
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting"],
    index=0
)
compare_all = st.sidebar.checkbox("Compare All Models", value=False)

# ======================
# Data loader
# ======================
@st.cache_data(show_spinner=False)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period)
    return df.dropna()

data = load_data(ticker, period)
if data.empty:
    st.error("Failed to fetch data. Please check the ticker symbol.")
    st.stop()

# ======================
# Chart
# ======================
price_series = _get_price_series(data)
st.subheader(f"Stock Data for {ticker}")
st.line_chart(pd.DataFrame({"Price": price_series}))

# ======================
# Features & split
# ======================
df_feat = build_features(data, lag_days)

split_idx = int(len(df_feat) * (1 - test_size / 100))
X_train = df_feat.drop("target", axis=1).iloc[:split_idx]
X_test  = df_feat.drop("target", axis=1).iloc[split_idx:]
y_train = df_feat["target"].iloc[:split_idx]
y_test  = df_feat["target"].iloc[split_idx:]

# ======================
# Models
# ======================
MODELS = {
    "Linear Regression": LinearRegression(),
    "Random Forest": RandomForestRegressor(n_estimators=300, min_samples_leaf=3, random_state=42, n_jobs=-1),
    "Gradient Boosting": HistGradientBoostingRegressor(max_depth=3, learning_rate=0.05, max_iter=500),
}

def evaluate_model(name, model):
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae  = float(mean_absolute_error(y_test, preds))
    mape = float(np.mean(np.abs((y_test - preds) / y_test)) * 100)
    accuracy = 100 - mape  # accuracy-like score derived from MAPE
    return {
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Accuracy (%)": accuracy,
        "Predictions": preds
    }

results = []
if compare_all:
    for n, m in MODELS.items():
        results.append(evaluate_model(n, m))
else:
    results.append(evaluate_model(model_choice, MODELS[model_choice]))

# Leaderboard
st.subheader("📊 Model Performance Comparison")
metrics_df = pd.DataFrame([{k: v for k, v in r.items() if k != "Predictions"} for r in results])
st.dataframe(metrics_df.style.highlight_min(subset=["RMSE", "MAE", "MAPE"], color="lightgreen"))

# Best model & plots
best = min(results, key=lambda x: x["RMSE"])
best_preds = best["Predictions"]

fig, ax = plt.subplots(figsize=(10, 5))
ax.plot(y_test.index, y_test, label="Actual", linewidth=2)
ax.plot(y_test.index, best_preds, label=f"Predicted ({best['Model']})", linestyle="--")
ax.set_title(f"Actual vs Predicted ({ticker})")
ax.legend()
st.pyplot(fig)

# Residuals
residuals = y_test - best_preds
fig2, ax2 = plt.subplots(figsize=(10, 4))
ax2.plot(y_test.index, residuals, label="Residuals")
ax2.axhline(0, color="red", linestyle="--")
ax2.set_title("Residual Plot")
ax2.legend()
st.pyplot(fig2)

# Next-day prediction & accuracy display
latest_X = df_feat.drop("target", axis=1).iloc[-1:].values
next_day_pred = MODELS[best["Model"]].predict(latest_X)[0]
st.metric(label="Next Day Predicted Price", value=f"${next_day_pred:.2f}")
st.metric(label="Model Accuracy", value=f"{best['Accuracy (%)']:.2f}%")

# Export CSV
pred_df = pd.DataFrame({"Date": y_test.index, "Actual": y_test.values, "Predicted": best_preds})
csv = pred_df.to_csv(index=False).encode("utf-8")
st.download_button("⬇️ Download Predictions (CSV)", csv, f"{ticker}_predictions.csv", "text/csv")
