# Streamlit Web Application – Complete Word-by-Word Explanation

---

## Overview

**File:** `web_stock_price_predictor.py`

**Purpose:**
- Interactive web interface for stock price prediction.
- Allows users to input ticker, select models, view predictions, metrics, and plots.
- Built with Streamlit framework (Python web app library).

**Key Features:**
1. User inputs (ticker, date range, lag count).
2. Single model or comparison mode.
3. Real-time data fetching via yfinance.
4. Model training with progress indicators.
5. Interactive visualizations (predictions, residuals, feature importance).
6. Next-day forecast display.

---

## Import Section (Line-by-Line)

```python
import streamlit as st
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import yfinance as yf
from datetime import datetime, timedelta
from xgboost import XGBRegressor
from sklearn.svm import SVR
```

### Breakdown:

#### `import streamlit as st`
- **streamlit**: Web framework for data apps.
- **st**: Common alias; provides UI components (text, buttons, plots).
- **Usage:** `st.title()`, `st.sidebar.selectbox()`, etc.

#### `import numpy as np`
- Numerical operations (arrays, math functions).
- Used for reshaping, calculations (sqrt, mean).

#### `import pandas as pd`
- DataFrame operations for time series manipulation.
- Shift, dropna, indexing.

#### `import matplotlib.pyplot as plt`
- Plotting library for charts.
- **Note:** Streamlit also supports `st.pyplot()` to render matplotlib figures.

#### Model imports:
```python
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler
```
- Same models as notebook.
- `StandardScaler`: Feature normalization.

#### Metrics:
```python
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
```
- Error evaluation functions.

#### `import yfinance as yf`
- Stock data fetching (same as notebook).

#### `from datetime import datetime, timedelta`
- Date manipulation for default date ranges.
- `datetime.now()`: Current date/time.
- `timedelta(days=365)`: Duration object for arithmetic.

#### New Imports:
```python
from xgboost import XGBRegressor
from sklearn.svm import SVR
```
- **XGBRegressor:**
  - Class from xgboost library.
  - Implements extreme gradient boosting for regression.
  - Requires: `pip install xgboost`.

- **SVR (Support Vector Regressor):**
  - Part of scikit-learn (`sklearn.svm`).
  - SVM variant for continuous prediction.
  - Already included in scikit-learn (no extra install).

---

## Helper Function 1: Lag Feature Creation

```python
def make_supervised(series: pd.Series, lags: int) -> pd.DataFrame:
    """Convert time series to supervised learning format"""
    data = pd.DataFrame({"target": series})
    for i in range(1, lags + 1):
        data[f"lag_{i}"] = series.shift(i)
    return data.dropna()
```

### Explanation:

**Docstring:**
- `"""..."""`: Multi-line documentation string.
- Describes function purpose (identical to notebook version).

**Logic:**
- Identical to notebook `make_supervised` function.
- Creates DataFrame with target + lag columns.
- Drops NaN rows from shifting.

**Reusability:**
- Encapsulated for use in web app context.

---

## Helper Function 2: Model Training

```python
def train_model(X_train, y_train, model_name, use_scaling=True):
    """Train a single model and return it along with scaler if used"""
    scaler = None
    
    if use_scaling:
        scaler = StandardScaler()
        X_train_scaled = scaler.fit_transform(X_train)
    else:
        X_train_scaled = X_train
    
    if model_name == "Linear Regression":
        model = LinearRegression()
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
    elif model_name == "XGBoost":
        model = XGBRegressor(
            n_estimators=200,
            max_depth=5,
            learning_rate=0.05,
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,
            gamma=0,
            reg_alpha=0.1,
            reg_lambda=1.0,
            random_state=42,
            n_jobs=-1,
            verbosity=0
        )
    elif model_name == "SVM":
        model = SVR(
            kernel='rbf',
            C=100,
            gamma='scale',
            epsilon=0.1,
            cache_size=500
        )
    
    model.fit(X_train_scaled, y_train)
    return model, scaler
```

### Line-by-Line:

#### Function signature:
```python
def train_model(X_train, y_train, model_name, use_scaling=True):
```
- **Parameters:**
  - `X_train`: Feature matrix (NumPy array).
  - `y_train`: Target array.
  - `model_name`: String identifier ("Linear Regression", "Random Forest", "Gradient Boosting").
  - `use_scaling=True`: Boolean flag; default enables scaling.

#### Initialize scaler variable:
```python
scaler = None
```
- Placeholder; will hold `StandardScaler` object if scaling used.

#### Conditional scaling:
```python
if use_scaling:
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
else:
    X_train_scaled = X_train
```
- **If use_scaling=True:**
  - Create scaler instance.
  - Fit to training data (learn μ, σ).
  - Transform training data.
- **Else:**
  - Use raw features (no transformation).

#### Model selection (if-elif chain):
```python
if model_name == "Linear Regression":
    model = LinearRegression()
```
- Exact string match selects model class.
- Initialize with default or custom hyperparameters.

**Random Forest block:**
```python
elif model_name == "Random Forest":
    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        ...
    )
```
- Same hyperparameters as notebook for consistency.

**Gradient Boosting block:**
```python
elif model_name == "Gradient Boosting":
    model = GradientBoostingRegressor(
        n_estimators=200,
        learning_rate=0.05,
        ...
    )
```
- Identical to notebook configuration.

**XGBoost block:**
```python
elif model_name == "XGBoost":
    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.05,
        subsample=0.8,
        colsample_bytree=0.8,
        min_child_weight=3,
        gamma=0,
        reg_alpha=0.1,
        reg_lambda=1.0,
        random_state=42,
        n_jobs=-1,
        verbosity=0
    )
```
- New model option.
- Reasonable default hyperparameters.

**SVM block:**
```python
elif model_name == "SVM":
    model = SVR(
        kernel='rbf',
        C=100,
        gamma='scale',
        epsilon=0.1,
        cache_size=500
    )
```
- New model option.
- Reasonable default hyperparameters.

#### Train model:
```python
model.fit(X_train_scaled, y_train)
```
- Calls scikit-learn's fit method.
- Learns parameters from scaled (or raw) training data.

#### Return values:
```python
return model, scaler
```
- **model**: Trained estimator object.
- **scaler**: StandardScaler instance (or None if not used).
- Tuple unpacking allows: `my_model, my_scaler = train_model(...)`.

---

## Helper Function 3: Evaluation Metrics

```python
def evaluate_model(model, X_test, y_test, scaler=None):
    """Evaluate model and return metrics"""
    if scaler is not None:
        X_test_scaled = scaler.transform(X_test)
        pred = model.predict(X_test_scaled)
    else:
        pred = model.predict(X_test)
    
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_vals = np.abs((pred - y_test) / np.where(y_test == 0, np.nan, y_test)) * 100.0
        mape = float(np.nanmean(mape_vals))
    
    accuracy = 100 - mape
    r2 = r2_score(y_test, pred)
    
    return {
        "predictions": pred,
        "rmse": rmse,
        "mae": mae,
        "mape": mape,
        "accuracy": accuracy,
        "r2": r2
    }
```

### Breakdown:

#### Conditional prediction:
```python
if scaler is not None:
    X_test_scaled = scaler.transform(X_test)
    pred = model.predict(X_test_scaled)
else:
    pred = model.predict(X_test)
```
- **If scaler provided:** Apply same transformation to test set.
- **Else:** Use raw test features (for Linear Regression without scaling).

#### RMSE calculation:
```python
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
```
- `mean_squared_error(y_test, pred)`: Average of (actual - predicted)².
- `np.sqrt(...)`: Square root.
- `float(...)`: Convert NumPy scalar to Python float.

#### MAE:
```python
mae = float(mean_absolute_error(y_test, pred))
```
- Average absolute difference.

#### MAPE (with zero-division handling):
```python
with np.errstate(divide='ignore', invalid='ignore'):
    mape_vals = np.abs((pred - y_test) / np.where(y_test == 0, np.nan, y_test)) * 100.0
    mape = float(np.nanmean(mape_vals))
```
- **Context manager `with np.errstate(...)`:** Suppresses warnings.
- **Division guard:** `np.where(y_test == 0, np.nan, y_test)` replaces zeros with NaN.
- **Percentage:** Multiply by 100.
- **`np.nanmean(...)`:** Ignores NaNs when averaging.

#### Derived accuracy:
```python
accuracy = 100 - mape
```
- Simple heuristic (not standard classification accuracy).

#### R² score:
```python
r2 = r2_score(y_test, pred)
```
- Coefficient of determination.

#### Return dictionary:
```python
return {
    "predictions": pred,
    "rmse": rmse,
    "mae": mae,
    "mape": mape,
    "accuracy": accuracy,
    "r2": r2
}
```
- Structured output for easy access by caller.

---

## Streamlit App Configuration

```python
st.set_page_config(
    page_title="Stock Price Predictor",
    page_icon="📈",
    layout="wide"
)
```

### Explanation:

#### `st.set_page_config(...)`
- **Must be first Streamlit command** in script.
- Configures browser tab and layout.

**Parameters:**
- `page_title`: Browser tab title.
- `page_icon`: Emoji/icon displayed in tab.
- `layout="wide"`: Uses full browser width (vs. centered narrow column).

---

## Page Header

```python
st.title("📈 Stock Price Prediction")
st.markdown("""
This app predicts stock prices using machine learning models:
- **Linear Regression** (baseline)
- **Random Forest** (ensemble)
- **Gradient Boosting** (advanced ensemble)
""")
```

### Components:

#### `st.title(...)`
- Displays large heading.
- Emoji `📈` rendered in title.

#### `st.markdown(...)`
- Renders Markdown-formatted text.
- Triple-quoted string allows multi-line.
- Bullet points with `- **bold text**`.

---

## Sidebar: User Inputs

```python
st.sidebar.header("Configuration")

ticker = st.sidebar.text_input(
    "Stock Ticker",
    value="GOOG",
    help="Enter stock symbol (e.g., AAPL, MSFT, TSLA)"
)
```

### Breakdown:

#### `st.sidebar.header(...)`
- Creates section header in sidebar (left panel).

#### `st.sidebar.text_input(...)`
- Text box for user input.

**Parameters:**
- `"Stock Ticker"`: Label displayed above box.
- `value="GOOG"`: Default pre-filled text.
- `help="..."`: Tooltip shown on hover (question mark icon).

**Return value:**
- User-entered string stored in `ticker` variable.
- Updates dynamically as user types.

---

### Date Range Inputs

```python
col1, col2 = st.sidebar.columns(2)
with col1:
    start_date = st.date_input(
        "Start Date",
        value=datetime.now() - timedelta(days=365),
        help="Historical data start date"
    )
with col2:
    end_date = st.date_input(
        "End Date",
        value=datetime.now(),
        help="Historical data end date"
    )
```

#### `st.sidebar.columns(2)`
- Splits sidebar into 2 equal-width columns.
- Returns tuple: `(col1, col2)`.

#### `with col1:`
- Context manager; widgets inside placed in first column.

#### `st.date_input(...)`
- Calendar picker widget.

**Parameters:**
- `"Start Date"`: Label.
- `value=datetime.now() - timedelta(days=365)`: Default to 1 year ago.
- `help`: Tooltip text.

**Return:**
- Python `date` object.

---

### Lag Count Slider

```python
lags = st.sidebar.slider(
    "Number of Lag Days",
    min_value=1,
    max_value=20,
    value=5,
    help="Past days used as features"
)
```

#### `st.sidebar.slider(...)`
- Interactive slider control.

**Parameters:**
- `min_value=1`, `max_value=20`: Range bounds.
- `value=5`: Default position.
- Dragging slider updates `lags` variable in real-time.

---

### Test Split Percentage

```python
test_size_pct = st.sidebar.slider(
    "Test Size (%)",
    min_value=10,
    max_value=40,
    value=20,
    help="Percentage of recent data for testing"
)
```

- Similar slider for test set size.
- Range: 10% to 40%.

---

### Model Selection Dropdown

```python
mode = st.sidebar.selectbox(
    "Mode",
    ["Single Model", "Compare Models"],
    help="Choose to train one model or compare all"
)

if mode == "Single Model":
    selected_model = st.sidebar.selectbox(
        "Select Model",
        ["Linear Regression", "Random Forest", "Gradient Boosting"]
    )
```

#### `st.sidebar.selectbox(...)`
- Dropdown menu.

**Parameters:**
- `"Mode"`: Label.
- `["Single Model", "Compare Models"]`: Options list.
- `help`: Tooltip.

**Return:**
- Selected string.

#### Conditional widget:
```python
if mode == "Single Model":
    selected_model = st.sidebar.selectbox(...)
```
- Only shows model dropdown if "Single Model" mode chosen.
- Dynamic UI based on user choice.

---

## Main Section: Data Download

```python
if st.button("Predict", type="primary"):
    try:
        with st.spinner("Downloading data..."):
            df = yf.download(ticker, start=start_date, end=end_date, interval="1d")
```

### Components:

#### `st.button(...)`
- Clickable button.
- `type="primary"`: Styling (blue, prominent).
- **Returns `True` when clicked**, triggering code block.

#### `try:`
- Exception handling block.

#### `st.spinner(...)`
- Context manager showing animated spinner.
- Message displayed: "Downloading data...".
- Spinner visible while code inside executes.

#### `yf.download(...)`
- Fetches historical stock data.
- `start=start_date, end=end_date`: Date range from user inputs.
- `interval="1d"`: Daily bars.

---

### Error Handling: Empty Data

```python
if df is None or df.empty:
    st.error("❌ No data downloaded. Check ticker symbol and date range.")
    st.stop()
```

#### `st.error(...)`
- Displays red error message box.
- Emoji `❌` for visual emphasis.

#### `st.stop()`
- Stops execution of the script.
- Prevents downstream errors from empty `df`.

---

### Data Preparation: Feature Engineering

```python
st.subheader("Feature Engineering")

# Lag features
with st.spinner("Creating lag features..."):
    df_lags = make_supervised(df["Close"], lags)
    df_lags_train = df_lags[df_lags.index < split_date]
    df_lags_test = df_lags[df_lags.index >= split_date]
```

#### `st.subheader(...)`
- Smaller header within main section.

#### Lag feature creation:
```python
with st.spinner("Creating lag features..."):
    df_lags = make_supervised(df["Close"], lags)
```
- Shows spinner with custom message.
- Calls `make_supervised` function for target column (`df["Close"]`).
- `lags` variable from user input.

---

### Data Preview

```python
st.subheader("Data Preview")
st.write(df.tail())

st.subheader("Lagged Data Preview")
st.write(df_lags.tail())
```

#### `st.write(...)`
- Displays DataFrame as table.

---

### Model Training Section

```python
if mode == "Single Model":
    with st.spinner(f"Training {selected_model}..."):
        model, scaler = train_model(X_train, y_train, selected_model, use_scaling=True)
```

#### Conditional training:
```python
if mode == "Single Model":
```
- Only executes if "Single Model" selected.

#### Training spinner:
```python
with st.spinner(f"Training {selected_model}..."):
```
- Dynamic message using f-string.

#### Model training:
```python
model, scaler = train_model(X_train, y_train, selected_model, use_scaling=True)
```
- Calls `train_model` function with user-selected model.

---

### Model Comparison (if applicable)

```python
else:
    st.subheader("Model Comparison")
    all_models = ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost", "SVM"]
    metrics = ["rmse", "mae", "mape", "accuracy", "r2"]
    results = pd.DataFrame(columns=["Model"] + metrics)
    
    for model_name in all_models:
        with st.spinner(f"Training {model_name}..."):
            model, scaler = train_model(X_train, y_train, model_name, use_scaling=True)
        
        preds = model.predict(X_test)
        metrics_values = evaluate_model(model, X_test, y_test, scaler)
        results = results.append({"Model": model_name, **metrics_values}, ignore_index=True)
    
    st.write(results)
```

#### Model comparison logic:
```python
else:
    st.subheader("Model Comparison")
    all_models = ["Linear Regression", "Random Forest", "Gradient Boosting", "XGBoost", "SVM"]
    metrics = ["rmse", "mae", "mape", "accuracy", "r2"]
    results = pd.DataFrame(columns=["Model"] + metrics)
```
- New section for comparing all models.
- Initializes empty DataFrame for results.

#### Iterative training and evaluation:
```python
for model_name in all_models:
    with st.spinner(f"Training {model_name}..."):
        model, scaler = train_model(X_train, y_train, model_name, use_scaling=True)
    
    preds = model.predict(X_test)
    metrics_values = evaluate_model(model, X_test, y_test, scaler)
    results = results.append({"Model": model_name, **metrics_values}, ignore_index=True)
```
- Loops over all model names.
- Trains each model.
- Makes predictions on test set.
- Evaluates metrics.
- Appends results to DataFrame.

#### Results display:
```python
st.write(results)
```
- Shows comparison table.

---

### Download Results Button

```python
csv = df_results.to_csv(index=False)
st.download_button(
    label="Download Results",
    data=csv,
    file_name=f"{ticker}_stock_predictions.csv",
    mime="text/csv",
    key="download-csv"
)
```

#### CSV download:
```python
csv = df_results.to_csv(index=False)
```
- Converts DataFrame to CSV format.

#### `st.download_button(...)`
- Renders download button.

**Parameters:**
- `label`: Button text.
- `data=csv`: CSV data.
- `file_name`: Suggested file name.
- `mime`: Content type.

---

## Plots Section

```python
st.subheader("Plots")

# Feature importance (if applicable)
if mode == "Single Model" and selected_model in ["Random Forest", "Gradient Boosting", "XGBoost"]:
    st.write("### Feature Importance")
    importance = model.feature_importances_ if hasattr(model, "feature_importances_") else None
    if importance is not None:
        feature_importance_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)
        
        st.bar_chart(feature_importance_df.set_index("Feature"))

# Residuals plot
st.write("### Residuals Plot")
residuals = y_test - preds
plt.figure(figsize=(10, 6))
plt.scatter(preds, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.grid()
st.pyplot()
```

### Components:

#### Feature importance (conditional):
```python
if mode == "Single Model" and selected_model in ["Random Forest", "Gradient Boosting", "XGBoost"]:
    st.write("### Feature Importance")
    importance = model.feature_importances_ if hasattr(model, "feature_importances_") else None
    if importance is not None:
        feature_importance_df = pd.DataFrame({
            "Feature": X_train.columns,
            "Importance": importance
        }).sort_values(by="Importance", ascending=False)
        
        st.bar_chart(feature_importance_df.set_index("Feature"))
```
- Shows feature importance bar chart.
- Conditional on model type.

#### Residuals plot:
```python
st.write("### Residuals Plot")
residuals = y_test - preds
plt.figure(figsize=(10, 6))
plt.scatter(preds, residuals)
plt.axhline(y=0, color='r', linestyle='--')
plt.xlabel("Predicted Values")
plt.ylabel("Residuals")
plt.title("Residuals vs Predicted Values")
plt.grid()
st.pyplot()
```
- Scatter plot of residuals.
- Red dashed line at y=0.

---

## Next-Day Prediction (if applicable)

```python
st.subheader("Next-Day Prediction")

# Single model mode only
if mode == "Single Model":
    with st.spinner(f"Predicting next day with {selected_model}..."):
        # Use latest available data point
        latest_data = df_lags.tail(1).drop("target", axis=1)
        if scaler is not None:
            latest_data = scaler.transform(latest_data)
        
        next_day_pred = model.predict(latest_data)
        st.write(f"### Predicted Close Price: ${next_day_pred[0]:.2f}")
```

### Components:

#### Subheader:
```python
st.subheader("Next-Day Prediction")
```
- Section title.

#### Conditional prediction:
```python
if mode == "Single Model":
    with st.spinner(f"Predicting next day with {selected_model}..."):
```
- Only executes in "Single Model" mode.

#### Latest data selection:
```python
latest_data = df_lags.tail(1).drop("target", axis=1)
```
- Gets most recent row.
- Drops target column.

#### Scaling (if applicable):
```python
if scaler is not None:
    latest_data = scaler.transform(latest_data)
```
- Transforms features using scaler.

#### Next day prediction:
```python
next_day_pred = model.predict(latest_data)
st.write(f"### Predicted Close Price: ${next_day_pred[0]:.2f}")
```
- Predicts using trained model.
- Displays formatted result.

---

## About and Contact

```python
st.sidebar.markdown("---")
st.sidebar.subheader("About This App")
st.sidebar.info(
    """
    This app predicts stock prices using historical data and machine learning models.
    
    **Models Used:**
    - Linear Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - Support Vector Regressor (SVM)
    
    **Data Source:** Yahoo Finance (via yfinance library)
    
    **Note:** This app is for educational purposes only. 
    """
)

st.sidebar.subheader("Contact")
st.sidebar.info(
    """
    Author: Your Name
    Email: your.email@example.com
    LinkedIn: [Your Profile](https://www.linkedin.com/in/yourprofile)
    """
)
```

### Components:

#### Sidebar about section:
```python
st.sidebar.markdown("---")
st.sidebar.subheader("About This App")
st.sidebar.info(
    """
    This app predicts stock prices using historical data and machine learning models.
    
    **Models Used:**
    - Linear Regression
    - Random Forest
    - Gradient Boosting
    - XGBoost
    - Support Vector Regressor (SVM)
    
    **Data Source:** Yahoo Finance (via yfinance library)
    
    **Note:** This app is for educational purposes only. 
    """
)
```

#### Sidebar contact section:
```python
st.sidebar.subheader("Contact")
st.sidebar.info(
    """
    Author: Your Name
    Email: your.email@example.com
    LinkedIn: [Your Profile](https://www.linkedin.com/in/yourprofile)
    """
)
```
- Replace with your details.

---

## End of Document

````