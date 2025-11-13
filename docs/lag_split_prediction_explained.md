# Detailed Explanation: Lag Features, Train/Test Split, and Next-Day Prediction

---

## Part 1: Cell 4 – Lag Feature Creation (Word-by-Word)

### Full Code:
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

### Line-by-Line Breakdown:

#### Line 1: `def make_supervised(series: pd.Series, lags: int) -> pd.DataFrame:`
- `def`: Keyword to define a function.
- `make_supervised`: Function name (converts time series to supervised learning format).
- `series: pd.Series`: Input parameter; expects pandas Series (1-D column of prices).
- `lags: int`: Number of previous days to use as features.
- `-> pd.DataFrame`: Type hint indicating return type is a DataFrame.

**Purpose:** Transform sequential price data into tabular format with multiple lag columns.

---

#### Line 2: `data = pd.DataFrame({"target": series})`
- `pd.DataFrame(...)`: Creates new DataFrame.
- `{"target": series}`: Dictionary initializes one column named "target" containing original series values.
- `target`: Current day's price we want to predict.

**State after this line:**
```
Date       target
2023-01-01  100.5
2023-01-02  101.2
2023-01-03  102.8
...
```

---

#### Line 3-4: `for i in range(1, lags + 1):`
- `for`: Loop keyword.
- `i`: Loop variable (takes values 1, 2, 3, ..., lags).
- `range(1, lags + 1)`: Generates sequence starting at 1, ending at lags (inclusive).
  - If `lags=5`: range produces [1, 2, 3, 4, 5].

**Example:** If lags=3, loop runs 3 times (i=1, i=2, i=3).

---

#### Line 4: `data[f"lag_{i}"] = series.shift(i)`
- `f"lag_{i}"`: f-string creates column name dynamically.
  - When i=1: `"lag_1"`
  - When i=2: `"lag_2"`
  - When i=3: `"lag_3"`
- `series.shift(i)`: **Critical operation**
  - Shifts entire series DOWN by `i` positions.
  - Introduces `i` NaN values at the **top**.
  - Effectively moves each price value `i` rows later.

**Visual Example (shift mechanics):**

Original series:
```
Index  Price
0      100
1      101
2      103
3      102
4      104
```

After `series.shift(1)` (lag_1):
```
Index  lag_1
0      NaN    ← shifted, no previous value
1      100    ← yesterday's price
2      101
3      103
4      102
```

After `series.shift(2)` (lag_2):
```
Index  lag_2
0      NaN
1      NaN    ← need 2 previous values
2      100    ← price from 2 days ago
3      101
4      103
```

After `series.shift(3)` (lag_3):
```
Index  lag_3
0      NaN
1      NaN
2      NaN
3      100    ← price from 3 days ago
4      101
```

**Combined DataFrame after loop (lags=3):**
```
Index  target  lag_1  lag_2  lag_3
0      100     NaN    NaN    NaN
1      101     100    NaN    NaN
2      103     101    100    NaN
3      102     103    101    100
4      104     102    103    101
```

---

#### Line 5: `return data.dropna()`
- `data.dropna()`: Removes ALL rows containing ANY NaN value.
- Returns cleaned DataFrame.

**After dropna() (continuing example):**
```
Index  target  lag_1  lag_2  lag_3
3      102     103    101    100
4      104     102    103    101
```
Only rows 3 and 4 remain (first valid complete lag set starts at row index = lags).

**Why dropna?**
- Machine learning models cannot handle NaN.
- First `lags` rows lack complete history.
- Removing them ensures every row has valid features.

---

#### Line 7: `series = df[price_col].squeeze().astype(float)`
- `df[price_col]`: Extracts single column (e.g., "Adj Close").
- `.squeeze()`: Converts 1-column DataFrame to Series (if needed).
- `.astype(float)`: Ensures numeric float type (handles any string/object dtype edge cases).

**Result:** Clean 1-D pandas Series of closing prices.

---

#### Line 8: `supervised = make_supervised(series, lags)`
- Calls function with price series and lag count.
- Returns DataFrame with target + lag_1 ... lag_n columns.

---

#### Line 9: `print("Rows after lagging:", len(supervised))`
- Displays number of usable rows.
- Useful sanity check (should be original_rows - lags).

**Example:**
- Original data: 252 trading days (1 year).
- lags=5.
- Rows lost to NaN: 5.
- Remaining: 252 - 5 = 247 rows.

---

#### Lines 10-11: Safety check
```python
if len(supervised) < 20:
    raise SystemExit("Not enough rows...")
```
- `if len(supervised) < 20:`: Guards against tiny datasets.
- `raise SystemExit(...)`: Stops execution with error message.

**When triggers:**
- Very short period (e.g., "1mo" with large lags).
- Invalid ticker returning minimal data.

---

### Full Example with Real Numbers

**Input prices (5 days):**
```
Date        Price
2024-01-01  100.0
2024-01-02  101.5
2024-01-03  103.2
2024-01-04  102.8
2024-01-05  104.1
```

**Set lags=2, call `make_supervised(series, 2)`:**

Step 1: Initialize target column:
```
Date        target
2024-01-01  100.0
2024-01-02  101.5
2024-01-03  103.2
2024-01-04  102.8
2024-01-05  104.1
```

Step 2: Add lag_1 (shift by 1):
```
Date        target  lag_1
2024-01-01  100.0   NaN
2024-01-02  101.5   100.0
2024-01-03  103.2   101.5
2024-01-04  102.8   103.2
2024-01-05  104.1   102.8
```

Step 3: Add lag_2 (shift by 2):
```
Date        target  lag_1  lag_2
2024-01-01  100.0   NaN    NaN
2024-01-02  101.5   100.0  NaN
2024-01-03  103.2   101.5  100.0
2024-01-04  102.8   103.2  101.5
2024-01-05  104.1   102.8  103.2
```

Step 4: Drop NaN rows:
```
Date        target  lag_1  lag_2
2024-01-03  103.2   101.5  100.0
2024-01-04  102.8   103.2  101.5
2024-01-05  104.1   102.8  103.2
```

**Final supervised dataset:**
- Row 1: Predict 103.2 using [101.5, 100.0]
- Row 2: Predict 102.8 using [103.2, 101.5]
- Row 3: Predict 104.1 using [102.8, 103.2]

**Interpretation:** Each row says "given these past 2 prices, the next price was X".

---

## Part 2: Cell 5 – Time-Based Train/Test Split (Word-by-Word)

### Full Code:
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

---

### Line 1: `test_size = max(1, int(len(supervised) * test_size_pct / 100))`

Breaking down parts:
- `len(supervised)`: Total number of rows after lagging (e.g., 247).
- `test_size_pct`: User parameter (e.g., 20 = 20%).
- `test_size_pct / 100`: Converts percent to decimal (20/100 = 0.2).
- `len(supervised) * 0.2`: Calculates number of rows for test (247 * 0.2 = 49.4).
- `int(...)`: Converts to integer (49.4 → 49).
- `max(1, ...)`: Ensures at least 1 test sample (guards against zero).

**Example:**
- supervised has 247 rows.
- test_size_pct = 20.
- Calculation: max(1, int(247 * 0.2)) = max(1, 49) = 49.
- Result: `test_size = 49`.

---

### Line 2: `split_idx = len(supervised) - test_size`
- `len(supervised)`: Total rows (247).
- `test_size`: Rows reserved for test (49).
- Subtraction: 247 - 49 = 198.

**Result:** `split_idx = 198`.

**Meaning:** First 198 rows → training, last 49 rows → testing.

---

### Line 3-4: Slicing DataFrames
```python
train = supervised.iloc[:split_idx]
test = supervised.iloc[split_idx:]
```

- `.iloc[]`: Integer-location based indexing.
- `[:split_idx]`: Slice from start to index 198 (exclusive) → rows 0-197.
- `[split_idx:]`: Slice from index 198 to end → rows 198-246.

**Visual Split (using small example, 10 rows, test_size=3):**
```
supervised (10 rows):
Index  target  lag_1  lag_2
0      102     101    100
1      103     102    101
2      104     103    102
3      105     104    103
4      106     105    104
5      107     106    105
6      108     107    106
7      109     108    107   ← split_idx = 7
8      110     109    108   ← test starts
9      111     110    109
```

- `train = rows 0-6` (7 rows).
- `test = rows 7-9` (3 rows).

**Why chronological?**
- Time series order must be preserved.
- Training on future data = **data leakage** (invalid).
- Test set represents "most recent" unseen period.

---

### Line 5: `drop_cols = supervised.columns.drop("target")`
- `supervised.columns`: List of all column names ['target', 'lag_1', 'lag_2', ...].
- `.drop("target")`: Removes "target" from list.
- Result: `drop_cols = ['lag_1', 'lag_2', 'lag_3', ...]`.

**Purpose:** Separate features (lags) from target (what we predict).

---

### Lines 6-7: Extract Training Features and Labels
```python
X_train = train[drop_cols].to_numpy()
y_train = train["target"].to_numpy()
```

- `train[drop_cols]`: Selects only lag columns from training set.
- `.to_numpy()`: Converts DataFrame subset to NumPy array (scikit-learn requirement).
- `train["target"]`: Selects target column.

**Shapes (example with 198 train rows, 5 lags):**
- `X_train`: (198, 5) → 198 samples, 5 features.
- `y_train`: (198,) → 198 target values (1-D array).

**Example row:**
```
X_train[0] = [101.5, 100.2, 99.8, 100.0, 98.5]  # lag_1 to lag_5
y_train[0] = 103.2                               # target price
```

---

### Lines 8-9: Extract Test Features and Labels
```python
X_test = test[drop_cols].to_numpy()
y_test = test["target"].to_numpy()
```

Same logic applied to test subset.

**Shapes (49 test rows, 5 lags):**
- `X_test`: (49, 5).
- `y_test`: (49,).

---

### Line 10: Print Summary
```python
print(f"Train size: {len(train)}, Test size: {len(test)}")
```
- f-string interpolation.
- Example output: `Train size: 198, Test size: 49`.

---

## General Train/Test Split Concepts

### Standard Practice (Random Split):
**Used for:** Non-sequential data (e.g., image classification, general tabular data).

**Method:**
```python
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
```

**Why shuffle?**
- Ensures both sets have representative distribution of all classes/patterns.

**Example (random):**
```
Original indices: [0,1,2,3,4,5,6,7,8,9]
After shuffle: [3,7,1,9,0,5,2,8,4,6]
Train: [3,7,1,9,0,5,2]
Test: [8,4,6]
```

---

### Time Series Split (Our Approach):
**Why different?**
- **Temporal dependency:** Future shouldn't inform past.
- **Real-world simulation:** Model trained on historical data, tested on future.

**Method:** Sequential slicing.

**Example (10 samples, 20% test):**
```
Chronological order: [0,1,2,3,4,5,6,7,8,9]
Train: [0,1,2,3,4,5,6,7]  ← older data
Test: [8,9]                ← recent data
```

**Key Principle:** Train on past, evaluate on future.

---

### Why This Matters for Stocks:

**Scenario with random split (WRONG):**
- Training includes prices from December.
- Testing includes prices from June.
- Model "learns" future patterns → unrealistic accuracy.

**Scenario with time split (CORRECT):**
- Training: January-October prices.
- Testing: November-December prices.
- Model evaluated on genuinely unseen future.

---

### Visual Comparison:

**Timeline (1 year daily prices):**
```
Jan ──────────────────────────────────────── Dec
[════════════════════════════════════════════]
     Training (80%)              Test (20%)
[══════════════════════════════][═══════════]
                                 ↑
                          Most recent data
```

**Random split would scatter:**
```
Jan ──────────────────────────────────────── Dec
[T][T][E][T][E][T][T][E][T][T][E][T][E][T]...
 ↑   ↑  ↑
Train Test (mixed) ← Invalid for time series
```

---

## Part 3: Cell 8 – Next-Day Prediction (Word-by-Word)

### Full Code:
```python
# 8) Predict next day's close price

latest_lags = [series.iloc[-i] for i in range(1, lags + 1)]
latest_X = np.array(latest_lags).reshape(1, -1)
next_day_pred = float(linreg.predict(latest_X)[0])
print(f"Predicted next close for {ticker}: {next_day_pred:,.2f}")
```

---

### Line 1: `latest_lags = [series.iloc[-i] for i in range(1, lags + 1)]`

**Components:**
- List comprehension: `[... for i in ...]`.
- `range(1, lags + 1)`: Loop variable (e.g., if lags=5: [1,2,3,4,5]).
- `series.iloc[-i]`: Negative indexing.

**Negative Indexing Explained:**
- `-1`: Last element.
- `-2`: Second-to-last.
- `-3`: Third-to-last, etc.

**Example (series with 10 prices):**
```
Index  Price
0      100.0
1      101.2
2      102.5
...
7      107.8
8      108.3  ← series.iloc[-2]
9      109.1  ← series.iloc[-1]
```

- `series.iloc[-1]` = 109.1 (yesterday's close).
- `series.iloc[-2]` = 108.3 (2 days ago).
- `series.iloc[-3]` = 107.8 (3 days ago).

**Loop execution (lags=5):**
```python
i=1: series.iloc[-1] = 109.1  # yesterday
i=2: series.iloc[-2] = 108.3  # 2 days ago
i=3: series.iloc[-3] = 107.8  # 3 days ago
i=4: series.iloc[-4] = 106.5  # 4 days ago
i=5: series.iloc[-5] = 105.2  # 5 days ago
```

**Result:**
```python
latest_lags = [109.1, 108.3, 107.8, 106.5, 105.2]
```

**Interpretation:** Most recent 5 closing prices in **reverse chronological order** (yesterday first).

---

### Line 2: `latest_X = np.array(latest_lags).reshape(1, -1)`

**Step 1: `np.array(latest_lags)`**
- Converts Python list to NumPy array.
- Result: 1-D array `[109.1, 108.3, 107.8, 106.5, 105.2]`.

**Step 2: `.reshape(1, -1)`**
- `reshape`: Changes array dimensions.
- `1`: Number of rows (1 sample).
- `-1`: Auto-calculate columns (infers 5 from data length).

**Shape transformation:**
```
Before: (5,)         1-D array
After:  (1, 5)       2-D array (1 row, 5 columns)
```

**Why reshape?**
- Scikit-learn `.predict()` expects 2-D input: (n_samples, n_features).
- Even for single prediction, must be shaped as `(1, n_features)`.

**Visual:**
```python
# Wrong (1-D):
[109.1, 108.3, 107.8, 106.5, 105.2]

# Correct (2-D):
[[109.1, 108.3, 107.8, 106.5, 105.2]]
```

---

### Line 3: `next_day_pred = float(linreg.predict(latest_X)[0])`

**Breakdown:**

**Part 1: `linreg.predict(latest_X)`**
- `linreg`: Trained LinearRegression model (from earlier cell).
- `.predict(X)`: Generates predictions for input features.
- Input: `latest_X` with shape (1, 5).
- Output: NumPy array with shape (1,) containing single prediction.

**Example output:**
```python
array([110.45])
```

**Part 2: `[0]`**
- Extracts first (and only) element from prediction array.
- Converts from `array([110.45])` to scalar `110.45`.

**Part 3: `float(...)`**
- Ensures Python float type (for formatting compatibility).

**Final result:**
```python
next_day_pred = 110.45
```

---

### Line 4: Print Statement
```python
print(f"Predicted next close for {ticker}: {next_day_pred:,.2f}")
```

**f-string formatting:**
- `{ticker}`: Inserts stock symbol (e.g., "GOOG").
- `{next_day_pred:,.2f}`:
  - `:,`: Thousands separator (e.g., 1,234.56).
  - `.2f`: Two decimal places.

**Example output:**
```
Predicted next close for GOOG: 110.45
```

---

## Complete End-to-End Example

### Scenario Setup:
- Stock: AAPL
- Last 5 closing prices:
  ```
  Date        Price
  2024-11-08  180.5
  2024-11-09  181.2
  2024-11-10  179.8
  2024-11-11  182.1
  2024-11-12  183.4  ← Most recent (yesterday)
  ```

### Step-by-Step Prediction:

**1. Extract latest lags (lags=5):**
```python
series.iloc[-1] = 183.4  # yesterday
series.iloc[-2] = 182.1
series.iloc[-3] = 179.8
series.iloc[-4] = 181.2
series.iloc[-5] = 180.5

latest_lags = [183.4, 182.1, 179.8, 181.2, 180.5]
```

**2. Reshape for model:**
```python
latest_X = [[183.4, 182.1, 179.8, 181.2, 180.5]]  # Shape: (1, 5)
```

**3. Model prediction (hypothetical):**
- Assume trained coefficients:
  - β₀ (intercept) = 5.0
  - β₁ (lag_1) = 0.4
  - β₂ (lag_2) = 0.3
  - β₃ (lag_3) = 0.15
  - β₄ (lag_4) = 0.1
  - β₅ (lag_5) = 0.05

**Calculation:**
```
prediction = 5.0 + (0.4 × 183.4) + (0.3 × 182.1) + (0.15 × 179.8) + (0.1 × 181.2) + (0.05 × 180.5)
           = 5.0 + 73.36 + 54.63 + 26.97 + 18.12 + 9.025
           = 187.105
```

**4. Output:**
```python
next_day_pred = 187.11  # Rounded to 2 decimals
print(f"Predicted next close for AAPL: 187.11")
```

---

## Summary Table

| Concept | Code | Explanation |
|---------|------|-------------|
| **Lag Creation** | `series.shift(i)` | Moves prices down by i rows; creates historical features |
| **Remove NaNs** | `.dropna()` | Eliminates rows without complete lag history |
| **Time Split** | `iloc[:split_idx]` / `iloc[split_idx:]` | Preserves chronological order for valid evaluation |
| **Feature Extraction** | `train[drop_cols]` | Separates lag columns from target |
| **Latest Lags** | `series.iloc[-i]` | Accesses most recent prices backward |
| **Reshape** | `.reshape(1, -1)` | Formats single sample for model input |
| **Predict** | `model.predict(X)` | Generates forecast using learned coefficients |

---

## Key Takeaways

1. **Lag features convert time series into supervised learning problem.**
2. **Time-based split prevents data leakage; tests on future data.**
3. **Next-day prediction uses most recent historical prices as input.**
4. **Models cannot predict beyond patterns seen during training.**
5. **Single-step forecasts are more reliable than multi-step.**

End of detailed explanation.