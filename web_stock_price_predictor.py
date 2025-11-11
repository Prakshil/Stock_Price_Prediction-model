import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import matplotlib.pyplot as plt


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

    for name in ["Adj Close", "Close", "AdjClose", "close", "adjclose"]:
        if name in df.columns:
            s = df[name]
            if isinstance(s, pd.DataFrame):
                s = s.iloc[:, 0]
            s = pd.to_numeric(s.squeeze(), errors="coerce")
            s.name = "Price"
            return s

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


st.set_page_config(page_title="Stock Price Predictor", layout="wide")
st.title("📈 Stock Price Prediction App")


st.sidebar.header("⚙️ Configuration")
ticker = st.sidebar.text_input("Enter Stock Ticker (e.g., AAPL, RELIANCE.NS)", "AAPL")
period = st.sidebar.selectbox("Select Period", ["1y", "2y", "5y", "10y"], index=0)
lag_days = st.sidebar.slider("Number of Lag Days", 1, 10, 3)
test_size = st.sidebar.slider("Test Size (%)", 10, 50, 20)

st.sidebar.header("🤖 Model Selection")
model_choice = st.sidebar.selectbox(
    "Choose Model",
    ["Linear Regression", "Random Forest", "Gradient Boosting", "Compare All Models"]
)


@st.cache_data(show_spinner=False)
def load_data(ticker: str, period: str) -> pd.DataFrame:
    df = yf.download(ticker, period=period)
    return df.dropna()

data = load_data(ticker, period)
if data.empty:
    st.error("Failed to fetch data. Please check the ticker symbol.")
    st.stop()

price_series = _get_price_series(data)

# Display current and previous day prices
st.subheader(f"📊 Current Stock Information for {ticker}")
col1, col2, col3 = st.columns(3)
current_price = float(price_series.iloc[-1])
previous_price = float(price_series.iloc[-2]) if len(price_series) > 1 else current_price
price_change = current_price - previous_price
price_change_pct = (price_change / previous_price) * 100 if previous_price != 0 else 0

col1.metric("Current Price", f"${current_price:.2f}", f"{price_change:+.2f} ({price_change_pct:+.2f}%)")
col2.metric("Previous Day Close", f"${previous_price:.2f}")
col3.metric("Last Updated", data.index[-1].strftime("%Y-%m-%d"))

st.subheader(f"📈 Historical Price Chart")
st.line_chart(pd.DataFrame({"Price": price_series}))


df_feat = build_features(data, lag_days)

split_idx = int(len(df_feat) * (1 - test_size / 100))
X_train = df_feat.drop("target", axis=1).iloc[:split_idx]
X_test  = df_feat.drop("target", axis=1).iloc[split_idx:]
y_train = df_feat["target"].iloc[:split_idx]
y_test  = df_feat["target"].iloc[split_idx:]

# Scale features for tree-based models
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


def train_and_evaluate_model(model_name: str, X_train, X_test, y_train, y_test, use_scaling=False):
    """Train a model and return predictions and metrics."""
    if model_name == "Linear Regression":
        model = LinearRegression()
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
    elif model_name == "Random Forest":
        model = RandomForestRegressor(
            n_estimators=300,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            bootstrap=True,
            random_state=42,
            n_jobs=-1,
            oob_score=True
        )
        X_tr = X_train_scaled if use_scaling else X_train
        X_ts = X_test_scaled if use_scaling else X_test
        model.fit(X_tr, y_train)
        preds = model.predict(X_ts)
    elif model_name == "Gradient Boosting":
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=4,
            min_samples_split=5,
            min_samples_leaf=3,
            subsample=0.8,
            max_features='sqrt',
            random_state=42,
            validation_fraction=0.1,
            n_iter_no_change=10,
            tol=1e-4
        )
        X_tr = X_train_scaled if use_scaling else X_train
        X_ts = X_test_scaled if use_scaling else X_test
        model.fit(X_tr, y_train)
        preds = model.predict(X_ts)
    
    # Calculate metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, preds)))
    mae = float(mean_absolute_error(y_test, preds))
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_vals = np.abs((preds - y_test) / np.where(y_test == 0, np.nan, y_test)) * 100.0
        mape = float(np.nanmean(mape_vals))
    accuracy = 100 - mape
    r2 = r2_score(y_test, preds)
    
    return model, preds, {
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Accuracy": accuracy,
        "R² Score": r2
    }


if model_choice == "Compare All Models":
    st.subheader("🔬 Model Comparison")
    
    models_data = {}
    all_results = []
    
    with st.spinner("Training all models..."):
        for name in ["Linear Regression", "Random Forest", "Gradient Boosting"]:
            use_scaling = name != "Linear Regression"
            model, preds, metrics = train_and_evaluate_model(
                name, X_train, X_test, y_train, y_test, use_scaling
            )
            models_data[name] = {"model": model, "predictions": preds, "metrics": metrics}
            all_results.append({"Model": name, **metrics})
    
    # Display comparison table
    st.subheader("📊 Performance Comparison")
    comparison_df = pd.DataFrame(all_results)
    st.dataframe(comparison_df.style.highlight_min(subset=["RMSE", "MAE", "MAPE"], color="lightgreen")
                                    .highlight_max(subset=["Accuracy", "R² Score"], color="lightgreen"))
    
    # Find best model
    best_model_name = min(all_results, key=lambda x: x["RMSE"])["Model"]
    st.success(f"🏆 Best Model: **{best_model_name}** (Lowest RMSE)")
    
    # Plot all predictions
    st.subheader("📈 Visual Comparison: Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(14, 6))
    ax.plot(y_test.index, y_test, label="Actual", linewidth=2.5, color='black', alpha=0.8)
    
    colors = ['blue', 'green', 'orange']
    linestyles = ['--', '-.', ':']
    
    for i, (name, data) in enumerate(models_data.items()):
        ax.plot(y_test.index, data["predictions"], 
                label=f"{name} (RMSE: {data['metrics']['RMSE']:.2f})",
                linestyle=linestyles[i], 
                linewidth=1.5,
                color=colors[i],
                alpha=0.7)
    
    ax.set_title(f"Model Comparison for {ticker}", fontsize=14, fontweight='bold')
    ax.set_xlabel("Date", fontsize=12)
    ax.set_ylabel("Price", fontsize=12)
    ax.legend(loc='best', fontsize=10)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Next-day predictions from all models
    st.subheader("🔮 Next-Day Predictions")
    latest_X = df_feat.drop("target", axis=1).iloc[-1:].values
    latest_X_scaled = scaler.transform(latest_X)
    
    pred_cols = st.columns(3)
    for i, (name, data) in enumerate(models_data.items()):
        if name == "Linear Regression":
            next_pred = data["model"].predict(latest_X)[0]
        else:
            next_pred = data["model"].predict(latest_X_scaled)[0]
        pred_cols[i].metric(f"{name}", f"${next_pred:.2f}")
    
    # Recommended prediction
    best_model = models_data[best_model_name]["model"]
    if best_model_name == "Linear Regression":
        recommended = best_model.predict(latest_X)[0]
    else:
        recommended = best_model.predict(latest_X_scaled)[0]
    st.success(f"🎯 **Recommended Prediction (Best Model):** ${recommended:.2f}")

else:
    # Single model mode
    st.subheader(f"🤖 Model: {model_choice}")
    
    use_scaling = model_choice != "Linear Regression"
    with st.spinner(f"Training {model_choice}..."):
        model, preds, metrics = train_and_evaluate_model(
            model_choice, X_train, X_test, y_train, y_test, use_scaling
        )
    
    # Display metrics
    st.subheader("📊 Model Performance")
    col1, col2, col3, col4, col5 = st.columns(5)
    col1.metric("RMSE", f"{metrics['RMSE']:.4f}")
    col2.metric("MAE", f"{metrics['MAE']:.4f}")
    col3.metric("MAPE", f"{metrics['MAPE']:.2f}%")
    col4.metric("Accuracy", f"{metrics['Accuracy']:.2f}%")
    col5.metric("R² Score", f"{metrics['R² Score']:.4f}")
    
    # Actual vs Predicted plot
    st.subheader("📈 Actual vs Predicted")
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(y_test.index, y_test, label="Actual", linewidth=2, color='black')
    ax.plot(y_test.index, preds, label=f"Predicted ({model_choice})", linestyle="--", linewidth=2)
    ax.set_title(f"Actual vs Predicted - {ticker} ({model_choice})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price")
    ax.legend()
    ax.grid(True, alpha=0.3)
    plt.tight_layout()
    st.pyplot(fig)
    
    # Residuals plot
    st.subheader("📉 Residual Analysis")
    residuals = y_test - preds
    fig2, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    axes[0].plot(y_test.index, residuals, color='red', alpha=0.6)
    axes[0].axhline(0, color='black', linestyle='--', linewidth=1.5)
    axes[0].set_title("Residuals Over Time")
    axes[0].set_xlabel("Date")
    axes[0].set_ylabel("Residual")
    axes[0].grid(True, alpha=0.3)
    
    axes[1].hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
    axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
    axes[1].set_title("Residual Distribution")
    axes[1].set_xlabel("Residual")
    axes[1].set_ylabel("Frequency")
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    st.pyplot(fig2)
    
    # Next-day prediction
    st.subheader("🔮 Next-Day Price Prediction")
    latest_X = df_feat.drop("target", axis=1).iloc[-1:].values
    if use_scaling:
        latest_X_scaled = scaler.transform(latest_X)
        next_day_pred = model.predict(latest_X_scaled)[0]
    else:
        next_day_pred = model.predict(latest_X)[0]
    
    pred_change = next_day_pred - current_price
    pred_change_pct = (pred_change / current_price) * 100
    
    st.metric(
        label=f"Predicted Next Day Price ({model_choice})", 
        value=f"${next_day_pred:.2f}",
        delta=f"{pred_change:+.2f} ({pred_change_pct:+.2f}%)"
    )
    
    # Export CSV
    pred_df = pd.DataFrame({
        "Date": y_test.index, 
        "Actual": y_test.values, 
        "Predicted": preds,
        "Residual": residuals
    })
    csv = pred_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        "⬇️ Download Predictions (CSV)", 
        csv, 
        f"{ticker}_{model_choice.replace(' ', '_')}_predictions.csv", 
        "text/csv"
    )