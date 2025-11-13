# Stock Price Prediction Notebook – Full Documentation

This document explains (a) the notebook cells line by line, (b) preprocessing logic, (c) every important function, (d) model theory (Linear Regression, Random Forest, Gradient Boosting), and (e) how predictions are produced.  
You only provided the notebook content, so other workspace files are not described here.

---

## 1. Notebook Structure Overview

Sequence of steps:
1. Markdown header (describes models).
2. Imports.
3. User-adjustable parameters.
4. Data download via yfinance.
5. Creation of supervised (lagged) dataset.
6. Train/test time-based split.
7. Baseline Linear Regression training & evaluation.
8. Next-day naive prediction with linear model.
9. Add tree ensemble models + scaling + train them.
10. Evaluate all models and collect metrics.
11. Plot multi-model comparison.
12. Select best model + next-day predictions for all.
13. Residual analysis for best model.

Data flow:
Raw price series → lag features → supervised DataFrame → split → model(s) → metrics → comparison → best model selection → next-day prediction.

---

## 2. Cell-by-Cell, Line-by-Line Explanation

### Cell 1 (Markdown)

Lines:
- `# Stock Price Prediction`: Markdown header; displayed as title.
- Blank lines create spacing.
- `- **Linear Regression** ...`: Bullet list describing models used.

No code execution. Pure documentation.

---

### Cell 2: Imports

```python
# 1) Imports
import sys
import numpy np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error
import yfinance as yf
```

Line-by-line:
- `# 1) Imports`: Comment; clarifies section.
- `import sys`: Gives access to Python runtime/environment (not strongly used later; could be removed).
- `import numpy as np`: Numerical operations, arrays, linear algebra, math functions.
- `import pandas as pd`: DataFrame operations for tabular time series.
- `import matplotlib.pyplot as plt`: Plotting library for charts.
- `from sklearn.linear_model import LinearRegression`: Imports regression model class.
- `from sklearn.metrics import mean_absolute_error, mean_squared_error`: Metrics for evaluating prediction error.
- `import yfinance as yf`: Library to fetch financial market data (Yahoo Finance).

Objects:
- `np`, `pd`, `plt`, `yf` are common aliases.
- `LinearRegression` class used to fit linear model with `.fit(X, y)` and `.predict(X)`.

---

### Cell 3: Adjustable Parameters

```python
# 2) Parameters (you can change these)
ticker = "GOOG"      # e.g., 'AAPL', 'MSFT'
period = "1y"        # '6mo', '1y', '2y', '5y'
lags = 5             # number of lag days to use as features
test_size_pct = 20   # percent of most recent data to use for testing
```

Explanation:
- `ticker`: String ticker symbol for stock. `"GOOG"` for Alphabet Class C shares.
- `period`: String recognized by yfinance; length of history.
- `lags`: Integer; number of past days used to predict current day.
- `test_size_pct`: Integer percentage of last rows reserved for testing (time-based split – avoids leakage).

“Possibility” changes:
- Larger `lags` → more features but fewer usable rows (need at least lags+1 rows).
- Larger `test_size_pct` → fewer training samples; may reduce model accuracy.

---

### Cell 4: Data Download & Preliminary Plot

```python
# 3) Download daily data

df = yf.download(ticker, period=period, interval="1d")
if df is None or df.empty:
    raise SystemExit("No data downloaded. Check the ticker or period.")

# Prefer 'Adj Close', else 'Close'
price_col = "Adj Close" if "Adj Close" in df.columns else ("Close" if "Close" in df.columns else None)
if price_col is None:
    raise SystemExit("No 'Adj Close' or 'Close' column found.")

print(df.tail())
df[[price_col]].plot(title=f"{ticker} {price_col}", figsize=(10,4))
plt.xlabel("Date"); plt.ylabel("Price"); plt.tight_layout(); plt.show()
```

Line-by-line:
- `yf.download(...)`: Returns DataFrame with columns: Open, High, Low, Close, Adj Close, Volume, indexed by Date.
- `if df is None or df.empty`: Defensive check; if no data, exit program.
- `price_col = ...`: Conditional assignment choosing `'Adj Close'` (adjusted for splits/dividends) over `'Close'`.
- `print(df.tail())`: Shows last 5 rows for sanity.
- `df[[price_col]].plot(...)`: pandas convenience plot; wraps matplotlib.
- `title=f"{ticker} {price_col}"`: Formatted string.
- `figsize=(10,4)`: Sets figure size (width=10, height=4 inches).
- `plt.xlabel/plt.ylabel`: Axis labels.
- `plt.tight_layout()`: Adjust layout to prevent clipping.
- `plt.show()`: Render plot.

Potential issues:
- Market closures produce missing days; yfinance handles holidays.
- If ticker invalid, DataFrame may be empty.

---

### Cell 5: Lag Feature Creation

```python
# 4) Make lag features (supervised dataset)

def make_supervised(series: pd.Series, lags: int) -> pd.DataFrame:
    data = pd.DataFrame({"target": series})
    for i in range(1, lags + 1):
        data[f"lag_{i}"] = series.shift(i)
    return data.dropna()

series = df[price_col].squeeze().astype(float)
supervised = make_supervised(series, lags)
print("Rows after lagging:", len(supervised))
if len(supervised) < 20:
    raise SystemExit("Not enough rows after lagging. Reduce 'lags' or increase 'period'.")
```

Definitions:
- `series.shift(i)`: Moves values down by `i`, introducing NaNs at top. So `lag_1` = yesterday’s price, `lag_2` two days ago, etc.
- `data.dropna()`: Removes early rows where shifted columns produce NaN.
- `target`: Current day price we want to predict.
- `supervised`: Final DataFrame with columns: `target`, `lag_1`, `lag_2`, …, `lag_n`.

Example:
If prices = [100, 101, 105, 103] and lags=2:
Row indices valid start at 2:
- target=105, lag_1=101, lag_2=100
- target=103, lag_1=105, lag_2=101

Why lags help?
Stock prices are autocorrelated; recent past often weakly predictive of near future.

---

### Cell 6: Train/Test Split

```python
# 5) Time-based split (last X% as test) — simplified

test_size = max(1, int(len(supervised) * test_size_pct / 100))
split_idx = len(supervised) - test_size

train = supervised.iloc[:split_idx]
test = supervised.iloc[split_idx:]

drop_cols = supervised.columns.drop("target")
X_train = train[drop_cols].to_numpy()
y_train = train["target"].to_numpy()
X_test = test[drop_cols].to_numpy()
y_test = test["target"].to_numpy()

print(f"Train size: {len(train)}, Test size: {len(test)}")
```

Line-by-line:
- `test_size = ...`: Compute number of rows for test; ensure ≥1 with `max`.
- `split_idx`: Index boundary.
- `train = supervised.iloc[:split_idx]`: First chronological portion.
- `test = supervised.iloc[split_idx:]`: Final chronological portion (most recent period).
- `drop_cols = supervised.columns.drop("target")`: Creates list of feature column names.
- `.to_numpy()`: Converts DataFrame subset to NumPy array (shape: rows × lags).
- `y_train`, `y_test`: 1-D arrays of target values aligned chronologically.
- Printing sizes gives transparency.

Why time-based?
Random shuffle would leak future info into training (data snooping).

---

### Cell 7: Train Baseline Linear Regression

```python
# 6) Fit Linear Regression and evaluate

linreg = LinearRegression()
linreg.fit(X_train, y_train)

pred = linreg.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
mae = float(mean_absolute_error(y_test, pred))
with np.errstate(divide='ignore', invalid='ignore'):
    mape_vals = np.abs((pred - y_test) / np.where(y_test == 0, np.nan, y_test)) * 100.0
    mape = float(np.nanmean(mape_vals))

print(f"RMSE: {rmse:,.4f}")
print(f"MAE:  {mae:,.4f}")
print(f"MAPE: {mape:,.2f}%")
```

Explanation:
- `linreg = LinearRegression()`: Initializes model with default parameters (ordinary least squares).
- `.fit(X_train, y_train)`: Learns coefficients β minimizing sum of squared residuals.
- `.predict(X_test)`: Produces predictions ŷ = X_test · β + intercept.
- `mean_squared_error(y_test, pred)`: Average of (actual - predicted)^2.
- `np.sqrt(...)`: Square root → RMSE (in same units as price).
- `mean_absolute_error`: Average |actual - predicted|.
- MAPE:
  - Division by zero guarded: `np.where(y_test == 0, np.nan, y_test)` to avoid infinite %.
  - `np.errstate(...)`: Suppresses warnings for invalid divisions.
  - `np.nanmean(...)`: Ignores NaNs inserted.

Metrics meaning:
- RMSE more sensitive to large errors.
- MAE more robust.
- MAPE gives intuitive percentage error (though unstable if near zero prices).

---

### Cell 8: Visualization (Baseline)

```python
# 7) Visualize Actual vs Predicted (test)

ax = pd.DataFrame({
    "Actual": y_test,
    "Predicted": pred,
}, index=test.index).plot(figsize=(10,4), title="Actual vs Predicted (test)")
plt.xlabel("Date"); plt.ylabel("Price"); plt.tight_layout(); plt.show()
```

Steps:
- Create DataFrame with two columns; index matches test dates.
- `.plot(...)`: Produces overlaid line plot.
- `ax` contains matplotlib Axes object for possible further customization.
- `plt.show()` renders.

Interpretation:
Close alignment of lines → better fit. Divergence → model error.

---

### Cell 9: Naive Next-Day Prediction

```python
# 8) Naive next-day prediction using latest lags

latest_lags = [series.iloc[-i] for i in range(1, lags + 1)]
latest_X = np.array(latest_lags).reshape(1, -1)
next_day_pred = float(linreg.predict(latest_X)[0])
print(f"Predicted next close for {ticker}: {next_day_pred:,.2f}")
```

Explanation:
- `latest_lags`: Collects last `lags` prices in reverse offset fashion (yesterday, day before, etc.).
- `np.array(...).reshape(1, -1)`: Converts list to 2D array with shape (1, lags) for model.
- `.predict(...)`: Linear regression expects same feature ordering.
- Index `[0]`: First (and only) prediction scalar.
- Wrap in `float()` for clarity in formatting.

Note:
This is a one-step ahead prediction; assumes relationship learned over training period persists.

---

### Cell 10 (Appears Later as #9 in Full Notebook): Additional Models & Scaling

```python
# 9) Import additional models and preprocessing
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.preprocessing import StandardScaler

# Feature Scaling (important for tree-based models' consistency)
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# Initialize models with tuned hyperparameters
models = {
    "Linear Regression": linreg,
    "Random Forest": RandomForestRegressor(
        n_estimators=300,
        max_depth=15,
        min_samples_split=5,
        min_samples_leaf=2,
        max_features='sqrt',
        bootstrap=True,
        random_state=42,
        n_jobs=-1,
        oob_score=True
    ),
    "Gradient Boosting": GradientBoostingRegressor(
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
}

# Train Random Forest and Gradient Boosting with scaled data
print("Training Random Forest...")
models["Random Forest"].fit(X_train_scaled, y_train)
print(f"  ✓ OOB Score: {models['Random Forest'].oob_score_:.4f}")

print("Training Gradient Boosting...")
models["Gradient Boosting"].fit(X_train_scaled, y_train)
print(f"  ✓ Training complete!")
print("\n✅ All models trained!")

# Feature importance for tree-based models
print("\n📊 Feature Importances:")
print("-" * 50)
for name in ["Random Forest", "Gradient Boosting"]:
    importances = models[name].feature_importances_
    feature_names = [f"lag_{i}" for i in range(1, lags + 1)]
    print(f"\n{name}:")
    for fname, importance in sorted(zip(feature_names, importances), key=lambda x: x[1], reverse=True):
        print(f"  {fname}: {importance:.4f}")
```

Key points:
- `StandardScaler`: Computes mean and std of each feature (lags) on training set; transforms to z-scores.
- Scaling trees not strictly required (they are scale-invariant), but uniform scaled input can stabilize Gradient Boosting learning and help if additional models added later (e.g., regularized linear, neural nets).
- Random Forest hyperparameters:
  - `n_estimators=300`: More trees → better averaging.
  - `max_depth=15`: Limits tree depth, reducing overfitting.
  - `min_samples_split=5`, `min_samples_leaf=2`: Regularization for splits.
  - `max_features='sqrt'`: Random feature subset per split (classic RF).
  - `bootstrap=True`: Bootstrapped samples per tree.
  - `oob_score=True`: Out-of-bag validation (not a test set replacement).
- Gradient Boosting hyperparameters:
  - `learning_rate=0.05`: Controls shrinkage; lower = slower but more generalizable.
  - `subsample=0.8`: Stochastic boosting reduces overfitting (gradient boosting + bagging).
  - `validation_fraction`, `n_iter_no_change`, `tol`: Early stopping.
  - `max_features='sqrt'`: Feature subset for tree splits (sklearn supports for some boosters).
- Feature importance:
  - `feature_importances_`: Proportion of total reduction in impurity attributed to each feature.

---

### Cell 11: Evaluate All Models

```python
# 10) Evaluate all models and compare performance
from sklearn.metrics import r2_score

results = []

for name, model in models.items():
    if name == "Linear Regression":
        pred = model.predict(X_test)
    else:
        pred = model.predict(X_test_scaled)
    
    # Calculate metrics
    rmse = float(np.sqrt(mean_squared_error(y_test, pred)))
    mae = float(mean_absolute_error(y_test, pred))
    
    with np.errstate(divide='ignore', invalid='ignore'):
        mape_vals = np.abs((pred - y_test) / np.where(y_test == 0, np.nan, y_test)) * 100.0
        mape = float(np.nanmean(mape_vals))
    
    accuracy = 100 - mape
    
    # R² Score using sklearn
    r2 = r2_score(y_test, pred)
    
    results.append({
        "Model": name,
        "RMSE": rmse,
        "MAE": mae,
        "MAPE": mape,
        "Accuracy (%)": accuracy,
        "R² Score": r2,
        "Predictions": pred
    })
    
    print(f"\n{name}:")
    print(f"  RMSE: {rmse:,.4f}")
    print(f"  MAE:  {mae:,.4f}")
    print(f"  MAPE: {mape:,.2f}%")
    print(f"  Accuracy: {accuracy:,.2f}%")
    print(f"  R² Score: {r2:,.4f}")

# Create comparison DataFrame
comparison_df = pd.DataFrame([{k: v for k, v in r.items() if k != "Predictions"} for r in results])
print("\n" + "="*70)
print("📊 MODEL COMPARISON:")
print("="*70)
print(comparison_df.to_string(index=False))
```

Notes:
- `r2_score`: Coefficient of determination; proportion of variance explained (1.0 perfect; can be negative if worse than constant mean).
- For tree-based models, uses scaled features for consistency (same data used in training).
- `accuracy = 100 - mape`: A simple derived metric; not statistically rigorous but intuitive.
- `results` list holds dictionaries for later visualization and best model selection.

---

### Cell 12: Plot Predictions (Multi-model)

```python
# 11) Visualize predictions from all models

fig, ax = plt.subplots(figsize=(14, 6))

# Plot actual values
ax.plot(test.index, y_test, label="Actual", linewidth=2.5, color='black', alpha=0.8)

# Plot predictions from each model
colors = ['blue', 'green', 'orange']
linestyles = ['--', '-.', ':']

for i, result in enumerate(results):
    ax.plot(test.index, result["Predictions"], 
            label=f"{result['Model']} (RMSE: {result['RMSE']:.2f})",
            linestyle=linestyles[i], 
            linewidth=1.5,
            color=colors[i],
            alpha=0.7)

ax.set_title(f"Model Comparison: Actual vs Predicted for {ticker}", fontsize=14, fontweight='bold')
ax.set_xlabel("Date", fontsize=12)
ax.set_ylabel("Price", fontsize=12)
ax.legend(loc='best', fontsize=10)
ax.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()
```

Key:
- Overlays each model’s prediction curve on actual test series.
- Different line styles & colors improve visual discrimination.
- Label includes RMSE for quick at-a-glance performance.

---

### Cell 13: Best Model Selection + Next-Day Prediction

```python
# 12) Best model selection and next-day predictions

# Find the best model based on RMSE
best_result = min(results, key=lambda x: x["RMSE"])
best_model_name = best_result["Model"]
best_model = models[best_model_name]

print(f"🏆 Best Model: {best_model_name}")
print(f"   RMSE: {best_result['RMSE']:,.4f}")
print(f"   R² Score: {best_result['R² Score']:,.4f}")
print(f"   Accuracy: {best_result['Accuracy (%)']:,.2f}%")

# Make next-day predictions with all models
latest_lags = [series.iloc[-i] for i in range(1, lags + 1)]
latest_X = np.array(latest_lags).reshape(1, -1)
latest_X_scaled = scaler.transform(latest_X)

print(f"\n📈 Next-Day Predictions for {ticker}:")
print("-" * 50)
for name, model in models.items():
    if name == "Linear Regression":
        next_pred = float(model.predict(latest_X)[0])
    else:
        next_pred = float(model.predict(latest_X_scaled)[0])
    print(f"  {name:20s}: ${next_pred:,.2f}")
    
print("\n" + "="*50)
if best_model_name == "Linear Regression":
    recommended = best_model.predict(latest_X)[0]
else:
    recommended = best_model.predict(latest_X_scaled)[0]
print(f"🎯 RECOMMENDED (Best Model): ${recommended:,.2f}")
print("="*50)
```

Explanation:
- `min(results, key=lambda x: x["RMSE"])`: Picks dictionary having lowest RMSE.
- Latest lag vector formed same way as earlier naive linear prediction, but used for all models.
- `scaler.transform(latest_X)`: Applies same scaling parameters from training set.
- Prints each model’s next-day price estimate.
- Final recommendation uses chosen best model.

---

### Cell 14: Residual Analysis

```python
# 13) Residual analysis for the best model

best_predictions = best_result["Predictions"]
residuals = y_test - best_predictions

fig, axes = plt.subplots(1, 2, figsize=(14, 5))

# Residual plot over time
axes[0].plot(test.index, residuals, color='red', alpha=0.6)
axes[0].axhline(0, color='black', linestyle='--', linewidth=1.5)
axes[0].set_title(f"Residuals Over Time ({best_model_name})", fontweight='bold')
axes[0].set_xlabel("Date")
axes[0].set_ylabel("Residual (Actual - Predicted)")
axes[0].grid(True, alpha=0.3)

# Residual distribution histogram
axes[1].hist(residuals, bins=30, color='steelblue', alpha=0.7, edgecolor='black')
axes[1].axvline(0, color='red', linestyle='--', linewidth=2)
axes[1].set_title(f"Residual Distribution ({best_model_name})", fontweight='bold')
axes[1].set_xlabel("Residual")
axes[1].set_ylabel("Frequency")
axes[1].grid(True, alpha=0.3)

plt.tight_layout()
plt.show()

print(f"Mean Residual: {np.mean(residuals):,.4f}")
print(f"Std Dev of Residuals: {np.std(residuals):,.4f}")
```

Residual logic:
- `residuals = actual - predicted`.
- Time plot checks patterns (should look random; structure → model bias).
- Histogram approximates error distribution; often near-normal if assumptions hold.
- Mean close to zero desirable; large standard deviation indicates volatile errors.

---

## 3. Preprocessing & Encoding

Preprocessing steps used:
1. Price column selection (Adjusted Close preferred).
2. Lag feature engineering via `.shift(i)`:
   - Converts sequential time series into supervised learning form.
3. Dropping NaNs resulting from shifts with `.dropna()`.
4. Train/test chronological split (not random).
5. Scaling features using `StandardScaler`:
   - `fit_transform(X_train)`: Computes mean μ and std σ per column: z = (x - μ)/σ.
   - `transform(X_test)`: Applies same μ and σ to test data.
6. (Implicit) No categorical encoding needed (all numeric price features).
7. Feature importance extraction from tree models (post-training) – interpretability, not transformation.

Encoding not performed because dataset is purely numerical (prices).  
No time decomposition (e.g., day-of-week) included — could be an enhancement.

---

## 4. Model Theory and Concepts

### 4.1 Linear Regression

Concept:
- Fits linear relationship: y ≈ β0 + β1·lag1 + β2·lag2 + … + βn·lagn.
- Objective: Minimize sum of squared residuals Σ (y_i - ŷ_i)^2.
- Assumptions (often violated in finance): linearity, independent residuals, homoscedasticity, no multicollinearity among lags (lags are naturally correlated).

Example:
Suppose:
lag_1 = 100, lag_2 = 102
Model learns: y ≈ 2 + 0.3·lag_1 + 0.5·lag_2 → y ≈ 2 + 30 + 51 = 83 (prediction).

Pros:
- Fast, interpretable.
Cons:
- Limited to linear pattern; cannot capture nonlinear regimes or regime shifts.

### 4.2 Random Forest

Concept:
- Ensemble of decision trees trained on bootstrap samples.
- Each split considers random subset of features (`max_features='sqrt'`).
- Prediction = average of individual tree predictions (regression).
- Reduces variance via averaging but can still lag on abrupt time series shifts.

Mechanism:
1. Sample with replacement from training data for each tree.
2. Grow tree until stopping criteria (depth, min samples).
3. Aggregate predictions: ŷ = (1/T) Σ tree_t(x).

Feature importance:
- Gini or MSE impurity reduction aggregated over splits using each feature.

Example (simplified):
Tree 1 predicts 110, Tree 2 predicts 108, Tree 3 predicts 112 ⇒ Forest prediction = (110+108+112)/3 = 110.

Pros:
- Handles nonlinearity, interactions.
Cons:
- Less effective for extrapolation; cannot forecast beyond feature distribution.

### 4.3 Gradient Boosting (GBRT)

Concept:
- Sequential ensemble: each tree fits residuals of previous ensemble.
- Additive model: F_m(x) = F_{m-1}(x) + η·h_m(x) where h_m fits gradient of loss.
- `learning_rate` (η) shrinks contribution → reduces overfitting.
- `subsample < 1` introduces stochasticity (like Random Forest’s bootstrap).

Mechanism:
1. Start with constant prediction (e.g., mean).
2. Compute residuals r_i = y_i - F_{m-1}(x_i).
3. Fit new tree h_m to residuals.
4. Update ensemble: F_m = F_{m-1} + η·h_m.
5. Repeat for `n_estimators`.

Example:
- Initial mean = 105.
- Residual pattern learned: tree adds +2 for certain lag patterns.
- After boosting steps final prediction more refined.

Pros:
- Strong predictive power, captures subtle patterns.
Cons:
- Sensitive to hyperparameters, risk of overfitting without regularization.

---

## 5. How Each Model Predicts Stock Prices Here

Input representation:
- Each row: lag_1 … lag_n (previous closing prices).
- Trees/nonlinear model can form piecewise constant regions based on ranges of lag values.
- Linear Regression forms weighted sum.

Prediction pathway:
1. Latest prices extracted → vector `[p_{t-1}, p_{t-2}, ..., p_{t-n}]`.
2. (If tree model) scaling applied → standardized vector.
3. Feed into chosen model:
   - Linear: dot product + intercept.
   - Random Forest: Each tree outputs number → average.
   - Gradient Boosting: Sequential correction aggregate → final number.
4. Output interpreted as next day closing price estimate.

Limitations:
- No volume, volatility, news, macro signals included.
- Pure autoregressive (AR-lag) formulation.
- Non-stationarity (trend shifts) may degrade performance.

---

## 6. Glossary of Functions / Methods / Terms

- `yf.download(ticker, period, interval)`: Fetches historical OHLCV data.
- `DataFrame.shift(i)`: Moves index downward by i; used for lag creation.
- `dropna()`: Removes rows with NaN values (here: initial rows without complete lag history).
- `LinearRegression.fit(X,y)`: Computes coefficients via least squares.
- `RandomForestRegressor.fit(X,y)`: Trains ensemble of trees.
- `GradientBoostingRegressor.fit(X,y)`: Builds trees sequentially minimizing loss.
- `predict(X)`: Returns model outputs for feature matrix X.
- `StandardScaler.fit_transform(X)`: Learn scaling params and apply transform.
- `StandardScaler.transform(X)`: Apply learned scaling.
- `mean_squared_error(y_true, y_pred)`: Average squared error.
- `mean_absolute_error(...)`: Average absolute error.
- `r2_score(...)`: Variance explained metric.
- `feature_importances_`: Importance scores for features in tree ensembles.
- `np.where(condition, a, b)`: Vectorized selection.
- `np.nanmean(arr)`: Mean ignoring NaNs.
- `plt.subplots(...)`: Create figure and axes.
- `ax.plot(...)`: Plot lines on axes.
- `hist(...)`: Plot histogram distribution.
- `Residual`: Difference actual - predicted; diagnostic of model fit.

---

## 7. Mini Synthetic Example (Lag Conversion + Linear Prediction)

Suppose closing prices over 7 days:  
[100, 101, 103, 102, 104, 105, 106], lags=3

Construct supervised rows starting at index 3:
Row for day price=102:
- lag_1=103, lag_2=101, lag_3=100

Final matrix (targets & lags):
```
target | lag_1 | lag_2 | lag_3
------------------------------
102    | 103   | 101   | 100
104    | 102   | 103   | 101
105    | 104   | 102   | 103
106    | 105   | 104   | 102
```

Train Linear Regression → maybe learns:
y ≈ 0.5·lag_1 + 0.3·lag_2 + 0.1·lag_3 + 10
Predict next (need next lags):
Latest lags: 106 (yesterday), 105, 104
y_next ≈ 0.5*106 + 0.3*105 + 0.1*104 + 10 = 53 + 31.5 + 10.4 + 10 ≈ 104.9

---

## 8. Potential Improvements

(Not in current code but useful future enhancements)
- Add volume, returns, rolling volatility, RSI.
- Stationarity checks / differencing (price → returns).
- Walk-forward validation instead of single split.
- Hyperparameter tuning (GridSearchCV/RandomizedSearchCV with TimeSeriesSplit).
- Confidence intervals using residual bootstrapping.

---

## 9. Summary

You:
- Convert raw price series → lagged supervised dataset.
- Train multiple regression models.
- Evaluate with RMSE/MAE/MAPE/R².
- Visualize multi-model predictions.
- Select best model.
- Produce next-day price estimate.

Core idea: Simple autoregressive lag modeling with ensemble enhancements.

---

## 10. Quick Reference Table (Metrics)

| Metric | Definition | Interpretation |
|--------|------------|----------------|
| RMSE | sqrt(mean squared error) | Penalizes large errors |
| MAE  | mean absolute error | Average absolute deviation |
| MAPE | mean absolute percentage error | Percent error (unstable at zeros) |
| R²   | 1 - SS_res/SS_tot | Variance explained (can be negative) |

---

End of documentation.