# Model Explanations for Stock Price Prediction

## 1. Linear Regression (Baseline)

Concept:
- Approximates relationship: target ≈ intercept + Σ (coefficient_i * lag_i).
- Objective: Minimize sum of squared errors (ordinary least squares).
- Assumes linearity, additive effects, and that past lags have linear predictive power.

Theory (plain):
- If X is matrix of lag features and y is vector of current prices:
  Solve (Xᵀ X) β = Xᵀ y → β are coefficients.
- Prediction: y_hat = X_new · β + intercept.

Working in project:
- Uses lag_1 … lag_n as predictors.
- Fast to train; provides baseline error metrics (RMSE, MAE, MAPE).
- Gives quick “sanity check” against more complex models.

Example:
- Suppose lags: lag_1=105, lag_2=104, lag_3=103.
- Learned coefficients: intercept=2, b1=0.5, b2=0.2, b3=0.1.
- Prediction: 2 + 0.5*105 + 0.2*104 + 0.1*103 = 2 + 52.5 + 20.8 + 10.3 = 85.6 (illustrative only).

Strengths:
- Simple, interpretable coefficients.
- Low variance, quick training.

Limitations (for stocks):
- Prices are non-stationary; linear relationship may drift.
- Cannot model nonlinear regime shifts (e.g., sudden volatility spikes).

Usefulness here:
- Establishes baseline error to justify ensembles.
- Coefficients can indicate rough past price influence weights.

---

## 2. Random Forest Regressor

Concept:
- Ensemble of decision trees using bootstrap sampling + random feature subsets.
- Each tree learns piecewise constant mappings; final prediction = average of tree outputs.
- Reduces variance compared to a single tree.

Theory (process):
1. For each tree:
   - Sample training rows with replacement (bootstrap).
   - At each split, evaluate random subset of features (max_features='sqrt').
   - Grow tree until stopping criteria (max_depth, min_samples_leaf).
2. Aggregate predictions: y_hat = (1 / T) Σ tree_t(x).

Key hyperparameters in project:
- n_estimators=300: Number of trees.
- max_depth=15: Controls complexity.
- min_samples_split=5, min_samples_leaf=2: Prevent overly granular splits.
- max_features='sqrt': Encourages decorrelated trees.
- oob_score=True: Out-of-bag performance estimate (internal validation).

Feature importance:
- Based on total impurity reduction (MSE decrease) attributed to splits using each lag.

Example (simplified):
- Tree predictions for same lag vector: [110.2, 109.8, 111.0, 110.5].
- Final prediction ≈ (110.2 + 109.8 + 111.0 + 110.5)/4 = 110.375.

Strengths:
- Handles nonlinear relationships among lagged prices.
- Robust to noise; less prone to overfitting than single deep tree.
- Provides feature importances for interpretability.

Limitations (stocks):
- Does not extrapolate outside seen range (cannot foresee trend accelerations).
- Treats each sample independently (no inherent temporal continuity modeling).

Usefulness here:
- Improves predictive accuracy over linear when nonlinearity exists.
- Helps identify most influential lag days via importance scores.

---

## 3. Gradient Boosting Regressor

Concept:
- Sequential additive model: each new tree corrects residual errors of prior ensemble.
- Focuses learning on “hard to predict” patterns by fitting residuals.
- Controlled by learning_rate (shrinkage) + regularization parameters.

Theory (simplified):
- Start with constant prediction F₀(x) = mean(y).
- For m in 1..M:
  - Compute residuals r_m = y - F_{m-1}(x).
  - Fit small tree h_m(x) to r_m.
  - Update: F_m(x) = F_{m-1}(x) + η * h_m(x), where η=learning_rate.
- Final prediction: F_M(x).

Project hyperparameters:
- n_estimators=200: Number of boosting stages.
- learning_rate=0.05: Small step size for generalization.
- max_depth=4: Shallow trees (weak learners).
- subsample=0.8: Stochastic boosting reduces overfitting.
- min_samples_split=5, min_samples_leaf=3: Regularization.
- validation_fraction=0.1 + n_iter_no_change=10 + tol=1e-4: Early stopping on internal validation set.
- max_features='sqrt': Feature subsampling (adds randomness).

Example (high-level):
- Initial mean price: 100.
- Tree 1 adds pattern: +2 for certain lag shapes → F₁.
- Tree 2 corrects remaining +1 errors → F₂.
- After many trees, residuals shrink → final forecast stabilizes.

Strengths:
- High predictive power when tuned.
- Captures subtle nonlinear interactions.
- Early stopping reduces overfit.

Limitations (stocks):
- Sensitive to noisy targets (market randomness).
- Slower training than Random Forest for large hyperparameter searches.

Usefulness here:
- Often lowest RMSE among tested models.
- Balances bias and variance via staged improvements.
- Provides feature importance (similar method to Random Forest but influenced by residual correction sequence).

---

## 4. Comparative Summary

| Aspect | Linear Regression | Random Forest | Gradient Boosting |
|--------|-------------------|---------------|-------------------|
| Pattern Type | Linear additive | Nonlinear, averaged trees | Nonlinear residual correction |
| Overfitting Risk | Low (underfit risk) | Moderate (controlled by depth & leaf size) | Higher (needs regularization) |
| Interpretability | High (coefficients) | Medium (feature importance) | Medium (importance; sequence harder to parse) |
| Training Speed | Fast | Moderate | Slower |
| Extrapolation | None | None | None |
| Typical Accuracy (relative) | Lowest | Medium–High | High (if tuned) |
| Strength | Simplicity | Robust variance reduction | Fine-grained pattern capture |

---

## 5. How They Predict Prices in This Project

Data representation:
- Features = lagged closing prices: lag_1 … lag_n (yesterday, 2 days ago, etc.).
- Target = current day closing price at each row.

Pipeline per model:
1. Build supervised table with lags.
2. Split chronologically (training vs recent test).
3. (For tree models) apply StandardScaler to feature matrix.
4. Train model (fit).
5. Generate predictions on test set.
6. Compute metrics (RMSE, MAE, MAPE, R²).
7. Rank models by RMSE.
8. Extract latest lags for next-day prediction.

Interpretation:
- Linear model estimates a weighted sum of prior prices.
- Random Forest partitions lag value space into regions; average region outputs.
- Gradient Boosting incrementally sharpens prediction by correcting previous errors.

---

## 6. Numeric Micro Example (Unified)

Assume last 3 closing prices: [101, 103, 102]; goal: predict next price.

Synthetic learned behaviors:
- Linear Regression: ŷ = 0.5*lag_1 + 0.3*lag_2 + 0.1*lag_3 + 2  
  → 0.5*102 + 0.3*103 + 0.1*101 + 2 ≈ 51 + 30.9 + 10.1 + 2 ≈ 94 (illustrative).
- Random Forest: Trees vote: [95, 97, 96, 95] → average ≈ 95.75.
- Gradient Boosting:
  - Base: mean training price ≈ 98
  - Tree 1 residual adjustment: +1
  - Tree 2 small correction: -0.2
  - Final ≈ 98 + 1 - 0.2 = 98.8.

These differ because:
- Linear enforces global linear form.
- RF creates discrete region averages.
- GB layers incremental local refinements.

---

## 7. Why Multiple Models

- Baseline establishes minimum acceptable quality.
- Random Forest checks if nonlinear averaging helps.
- Gradient Boosting tests if sequential error correction yields further gains.
- Comparison prevents over-reliance on a single assumption type.

---

## 8. When Each May Be Preferred

| Scenario | Preferred |
|----------|-----------|
| Very small dataset | Linear Regression |
| Moderate size, noisy | Random Forest |
| Need best accuracy, can tune | Gradient Boosting |
| Need transparency | Linear Regression |
| Need feature ranking with nonlinear capture | Random Forest / Gradient Boosting |

---

## 9. Practical Notes for Stock Prediction

- All three are autoregressive (use only past prices).
- None inherently models seasonality or external drivers (news, macro).
- Improvement path: add returns, volatility, technical indicators (RSI, SMA, EMA), volume.
- For longer horizon forecasts: consider walk-forward validation and differencing.

---

## 10. Key Strength Contribution in This Project

| Model | Contribution |
|-------|--------------|
| Linear Regression | Quick baseline; exposes linear dependence magnitude. |
| Random Forest | Captures nonlinear influence of certain lag combinations. |
| Gradient Boosting | Refines prediction by targeting residual structures; often best RMSE. |

---

## 11. Potential Enhancements

- Transform price to log-return to stabilize variance.
- Add rolling features (moving averages, std dev).
- Use TimeSeriesSplit cross-validation for robust model selection.
- Hyperparameter tuning (RandomizedSearchCV with custom time split).
- Model stacking: combine predictions (e.g., weighted average based on inverse RMSE).

---

## 12. Summary

The three models offer a progression:
1. Linear Regression: simple, interpretable benchmark.
2. Random Forest: robust nonlinear learner through ensemble averaging.
3. Gradient Boosting: powerful residual-focused sequential learner.

In combination they allow evaluation of linear vs. nonlinear vs. sequential additive approaches to short-term price movement estimation based solely on lagged close values.

---

# XGBoost (eXtreme Gradient Boosting) - Advanced Theory

## Mathematical Foundation

### Objective Function

XGBoost minimizes a regularized objective:

```
L(φ) = Σᵢ l(ŷᵢ, yᵢ) + Σₖ Ω(fₖ)
```

Where:
- **l(ŷᵢ, yᵢ)**: Loss function (MSE for regression)
- **Ω(fₖ)**: Regularization term for tree k

### Regularization Term (Key Innovation)

```
Ω(f) = γT + (λ/2)Σⱼ₌₁ᵀ wⱼ²
```

Components:
- **T**: Number of leaves in tree
- **γ** (gamma): Complexity penalty per leaf
- **wⱼ**: Weight of leaf j
- **λ** (lambda): L2 regularization coefficient

This prevents overfitting by:
1. Penalizing trees with many leaves (γT term)
2. Shrinking leaf weights (L2 penalty)

### Additive Training

Build model sequentially:

```
ŷ⁽⁰⁾ = 0
ŷ⁽¹⁾ = f₁(x) = ŷ⁽⁰⁾ + f₁(x)
ŷ⁽²⁾ = f₁(x) + f₂(x) = ŷ⁽¹⁾ + f₂(x)
...
ŷ⁽ᵗ⁾ = Σₖ₌₁ᵗ fₖ(x) = ŷ⁽ᵗ⁻¹⁾ + fₜ(x)
```

At step t, minimize:

```
L⁽ᵗ⁾ = Σᵢ₌₁ⁿ l(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾ + fₜ(xᵢ)) + Ω(fₜ)
```

### Taylor Expansion Approximation

XGBoost uses second-order Taylor expansion:

```
L⁽ᵗ⁾ ≈ Σᵢ₌₁ⁿ [l(yᵢ, ŷᵢ⁽ᵗ⁻¹⁾) + gᵢfₜ(xᵢ) + (1/2)hᵢfₜ²(xᵢ)] + Ω(fₜ)
```

Where:
- **gᵢ = ∂l/∂ŷ⁽ᵗ⁻¹⁾**: First derivative (gradient)
- **hᵢ = ∂²l/∂ŷ⁽ᵗ⁻¹⁾²**: Second derivative (Hessian)

For MSE loss:
```
l(y, ŷ) = (y - ŷ)²/2
gᵢ = ŷᵢ⁽ᵗ⁻¹⁾ - yᵢ  (residual)
hᵢ = 1
```

### Optimal Leaf Weight

For a fixed tree structure, optimal weight for leaf j:

```
wⱼ* = -Gⱼ / (Hⱼ + λ)
```

Where:
- **Gⱼ = Σᵢ∈Iⱼ gᵢ**: Sum of gradients in leaf j
- **Hⱼ = Σᵢ∈Iⱼ hᵢ**: Sum of Hessians in leaf j
- **Iⱼ**: Set of samples in leaf j

### Split Finding Algorithm

**Gain from split:**

```
Gain = (1/2)[G_L²/(H_L + λ) + G_R²/(H_R + λ) - (G_L + G_R)²/(H_L + H_R + λ)] - γ
```

Where:
- **G_L, H_L**: Left child gradient/Hessian sums
- **G_R, H_R**: Right child gradient/Hessian sums
- **γ**: Complexity penalty

**Split only if Gain > 0** (regularization enforces this)

## Concrete Example (Stock Prediction)

### Setup
```
Training data (10 samples):
Index  lag_1  lag_2  lag_3  target  
0      100    99     98     101
1      101    100    99     102
2      102    101    100    103
3      103    102    101    102
4      102    103    102    104
5      104    102    103    105
6      105    104    102    106
7      106    105    104    105
8      105    106    105    107
9      107    105    106    108

Hyperparameters:
- n_estimators = 3 (for illustration)
- learning_rate = 0.1
- max_depth = 2
- lambda = 1.0
- gamma = 0
```

### Iteration 0: Initialize

```
ŷ⁽⁰⁾ = mean(targets) = 104.3
```

### Iteration 1: Build Tree T₁

**Step 1: Calculate gradients (residuals for MSE)**
```
g₀ = ŷ₀⁽⁰⁾ - y₀ = 104.3 - 101 = 3.3
g₁ = 104.3 - 102 = 2.3
g₂ = 104.3 - 103 = 1.3
g₃ = 104.3 - 102 = 2.3
g₄ = 104.3 - 104 = 0.3
g₅ = 104.3 - 105 = -0.7
g₆ = 104.3 - 106 = -1.7
g₇ = 104.3 - 105 = -0.7
g₈ = 104.3 - 107 = -2.7
g₉ = 104.3 - 108 = -3.7

h₀...h₉ = 1 (for MSE)
```

**Step 2: Find best split on lag_1**

Try split: lag_1 < 104
```
Left samples (I_L): [0,1,2,3,4] with lag_1 < 104
G_L = 3.3+2.3+1.3+2.3+0.3 = 9.5
H_L = 5

Right samples (I_R): [5,6,7,8,9] with lag_1 >= 104
G_R = -0.7-1.7-0.7-2.7-3.7 = -9.5
H_R = 5

Gain = 0.5[9.5²/(5+1) + (-9.5)²/(5+1) - 0²/(10+1)] - 0
     = 0.5[15.04 + 15.04 - 0]
     = 15.04
```

**Step 3: Calculate leaf weights**
```
Left leaf weight:
w_L = -G_L/(H_L + λ) = -9.5/(5+1) = -1.58

Right leaf weight:
w_R = -G_R/(H_R + λ) = -(-9.5)/(5+1) = 1.58
```

**Step 4: Update predictions**
```
For samples 0-4 (lag_1 < 104):
ŷ⁽¹⁾ = ŷ⁽⁰⁾ + η×w_L = 104.3 + 0.1×(-1.58) = 104.14

For samples 5-9 (lag_1 >= 104):
ŷ⁽¹⁾ = ŷ⁽⁰⁾ + η×w_R = 104.3 + 0.1×1.58 = 104.46
```

**New predictions:**
```
Sample 0: 104.14 (actual: 101, error: 3.14)
Sample 5: 104.46 (actual: 105, error: -0.54)
...
```

### Iteration 2: Build Tree T₂

**Recalculate residuals:**
```
g₀⁽²⁾ = 104.14 - 101 = 3.14
g₅⁽²⁾ = 104.46 - 105 = -0.54
...
```

**Find new split** (similar process)

**Continue for n_estimators rounds...**

### Final Prediction

```
ŷ_final = ŷ⁽⁰⁾ + 0.1×f₁(x) + 0.1×f₂(x) + 0.1×f₃(x)
```

For a new sample [lag_1=105, lag_2=104, lag_3=103]:
1. Tree 1: Falls in right leaf → +1.58
2. Tree 2: Falls in left leaf → -0.42
3. Tree 3: Falls in right leaf → +0.31

```
ŷ = 104.3 + 0.1×(1.58 - 0.42 + 0.31)
  = 104.3 + 0.147
  = 104.45
```

## Advanced Features Used in Project

### 1. Column Subsampling (`colsample_bytree=0.8`)

For each tree, randomly select 80% of features:
- Tree 1 might use: [lag_1, lag_2, lag_4, lag_5]
- Tree 2 might use: [lag_1, lag_3, lag_4, lag_5]
- Tree 3 might use: [lag_2, lag_3, lag_4, lag_5]

**Effect:** Decorrelates trees, reduces overfitting

### 2. Row Subsampling (`subsample=0.8`)

For each tree, train on random 80% of samples:
- Tree 1: samples [0,1,2,3,4,5,6,7]
- Tree 2: samples [1,2,3,4,5,6,8,9]
- Tree 3: samples [0,1,3,4,5,7,8,9]

**Effect:** Stochastic gradient boosting, improves generalization

### 3. L1 Regularization (`reg_alpha=0.1`)

Modified regularization:
```
Ω(f) = γT + (λ/2)Σwⱼ² + α·Σ|wⱼ|
```

**Effect:** 
- Encourages sparse leaf weights (some become exactly 0)
- Feature selection (some splits never used)

### 4. Min Child Weight (`min_child_weight=3`)

Only split if:
```
Σᵢ∈I_L hᵢ >= 3  AND  Σᵢ∈I_R hᵢ >= 3
```

**Effect:** 
- Prevents tiny leaves (noise)
- Conservative partitioning

## Why XGBoost Outperforms Gradient Boosting

| Aspect | Gradient Boosting | XGBoost |
|--------|-------------------|---------|
| Loss optimization | First derivative only | **Second derivative** (Newton's method) |
| Regularization | Manual pruning | **Built-in L1/L2** |
| Split finding | Exact greedy | **Approximate + parallel** |
| Missing values | Requires imputation | **Automatic handling** |
| Column sampling | No | **Yes** (like Random Forest) |
| Speed | Sequential | **Parallel tree building** |

## Feature Importance Calculation

XGBoost uses **gain-based importance**:

```
Importance(feature_j) = Σₜ Σₙ∈splits_on_j Gain(n,t)
```

Where:
- t: Tree index
- n: Node that splits on feature j
- Gain(n,t): Information gain from that split

**Normalized:**
```
Importance_norm(j) = Importance(j) / Σₖ Importance(k)
```

**Example output:**
```
lag_1: 0.4856  # Used in 48.56% of total gain
lag_2: 0.2201
lag_3: 0.1523
lag_4: 0.0820
lag_5: 0.0600
```

**Interpretation:**
- Yesterday's price (lag_1) is most predictive
- Importance decays with older lags
- Confirms temporal locality in stock prices

---

# SVM (Support Vector Machine) - Advanced Theory

## Mathematical Foundation (ε-SVR)

### Primal Problem

Minimize:
```
(1/2)||w||² + C·Σᵢ(ξᵢ + ξᵢ*)
```

Subject to:
```
yᵢ - (w·φ(xᵢ) + b) ≤ ε + ξᵢ
(w·φ(xᵢ) + b) - yᵢ ≤ ε + ξᵢ*
ξᵢ, ξᵢ* ≥ 0
```

Where:
- **w**: Weight vector in feature space
- **b**: Bias term
- **φ(x)**: Kernel feature mapping
- **ε**: Epsilon (tube width)
- **ξᵢ, ξᵢ***: Slack variables (upper/lower violations)
- **C**: Penalty for violations

### Dual Problem (Solved in Practice)

Maximize:
```
W(α, α*) = -ε·Σᵢ(αᵢ + αᵢ*) + Σᵢ yᵢ(αᵢ - αᵢ*) - (1/2)ΣᵢΣⱼ(αᵢ - αᵢ*)(αⱼ - αⱼ*)K(xᵢ, xⱼ)
```

Subject to:
```
Σᵢ(αᵢ - αᵢ*) = 0
0 ≤ αᵢ, αᵢ* ≤ C
```

**Solution:**
```
w = Σᵢ(αᵢ - αᵢ*)φ(xᵢ)
```

**Prediction:**
```
f(x) = Σᵢ(αᵢ - αᵢ*)K(xᵢ, x) + b
```

**Support vectors:** Samples where αᵢ ≠ 0 or αᵢ* ≠ 0

## RBF Kernel Deep Dive

### Definition

```
K(x, x') = exp(-γ·||x - x'||²)
```

Where:
- **γ** (gamma): Kernel bandwidth parameter
- **||x - x'||²**: Squared Euclidean distance

### Expansion (Infinite Dimensional)

RBF kernel implicitly maps to infinite dimensions:

```
φ(x) = [φ₁(x), φ₂(x), φ₃(x), ...]  (infinite)
```

**Taylor expansion:**
```
exp(-γ·||x - x'||²) = Σₙ₌₀^∞ [(-γ)ⁿ/n!] · ||x - x'||^(2n)
```

Each term corresponds to polynomial features of degree 2n.

### Gamma Interpretation

**High γ (e.g., 10):**
- Narrow Gaussian bell
- Only nearby points influence prediction
- Complex decision boundary
- Risk: Overfitting

**Low γ (e.g., 0.01):**
- Wide Gaussian bell
- Distant points also influence
- Smooth decision boundary
- Risk: Underfitting

**Auto-scaling (gamma='scale'):**
```
γ = 1 / (n_features × var(X))
```

For standardized data (var ≈ 1):
```
γ = 1 / n_features = 1 / 5 = 0.2
```

## Epsilon-Tube Concept (Detailed)

### Visual Representation

```
Price
  │
106─┤                    ●  ← Violates upper bound (ξᵢ*)
    │                   /│
105─┤   ┌─────────────┘ │  ← Upper bound: f(x) + ε
    │   │ ε-tube       /
104─┤───●─────────────●──  ← SVM fit: f(x) = w·x + b
    │   │            /
103─┤   └──────────●────  ← Lower bound: f(x) - ε
    │              │ │
102─┤              │ └──── Inside ε-tube (no penalty)
    │              ●  ← Violates lower bound (ξᵢ)
    └───────────────────
      lag_1, lag_2, ...
```

### Penalty Calculation

For sample i with prediction f(xᵢ):

**Case 1:** |yᵢ - f(xᵢ)| ≤ ε
```
Loss = 0  (inside tube)
ξᵢ = 0, ξᵢ* = 0
```

**Case 2:** yᵢ - f(xᵢ) > ε (above upper bound)
```
Loss = C·(yᵢ - f(xᵢ) - ε)
ξᵢ* = yᵢ - f(xᵢ) - ε
ξᵢ = 0
```

**Case 3:** f(xᵢ) - yᵢ > ε (below lower bound)
```
Loss = C·(f(xᵢ) - yᵢ - ε)
ξᵢ = f(xᵢ) - yᵢ - ε
ξᵢ* = 0
```

### C Parameter Effect

**High C (e.g., 100):**
- Strong penalty for violations
- Tight fit to data
- More support vectors
- Risk: Overfitting

**Low C (e.g., 1):**
- Weak penalty
- Allows more violations
- Fewer support vectors
- Risk: Underfitting

## Concrete Example (Stock Prediction)

### Setup
```
Training data (7 samples, 3 features):
Index  lag_1  lag_2  lag_3  →  target
0      1.0    0.8    0.6       102
1      1.2    1.0    0.8       103
2      0.9    1.2    1.0       104
3      1.1    0.9    1.2       103
4      1.3    1.1    0.9       105
5      1.0    1.3    1.1       104
6      1.2    1.0    1.3       106

Hyperparameters:
- C = 100
- gamma = 0.33 (≈ 1/3 features)
- epsilon = 0.5
```

### Step 1: Compute Kernel Matrix

```
K[i,j] = exp(-0.33·||xᵢ - xⱼ||²)

K = 
[1.00  0.94  0.87  0.91  0.83  0.91  0.87]
[0.94  1.00  0.91  0.94  0.87  0.87  0.91]
[0.87  0.91  1.00  0.94  0.91  0.83  0.87]
[0.91  0.94  0.94  1.00  0.87  0.87  0.83]
[0.83  0.87  0.91  0.87  1.00  0.91  0.87]
[0.91  0.87  0.83  0.87  0.91  1.00  0.91]
[0.87  0.91  0.87  0.83  0.87  0.91  1.00]
```

### Step 2: Solve Dual Problem

Quadratic programming yields:

```
α = [0, 15.2, 0, 8.7, 0, 12.3, 0]
α* = [10.5, 0, 6.8, 0, 18.9, 0, 0]

Support vectors: samples 0,1,2,3,4,5 (6 out of 7)
```

### Step 3: Calculate Bias

Using support vector i=1 (on margin):
```
b = y₁ - ε - Σⱼ(αⱼ - αⱼ*)K(xⱼ, x₁)
  = 103 - 0.5 - (15.2×1.0 + ...)
  = 102.5 - 14.8
  = 87.7
```

### Step 4: Prediction Function

```
f(x) = Σᵢ(αᵢ - αᵢ*)K(xᵢ, x) + b
     = (-10.5)K(x₀,x) + 15.2K(x₁,x) + (-6.8)K(x₂,x) 
       + 8.7K(x₃,x) + (-18.9)K(x₄,x) + 12.3K(x₅,x) + 87.7
```

### Step 5: Predict New Sample

For x_new = [1.1, 1.0, 1.0]:

```
K(x₀, x_new) = exp(-0.33·||[1.0,0.8,0.6] - [1.1,1.0,1.0]||²)
             = exp(-0.33·0.21)
             = 0.933

K(x₁, x_new) = exp(-0.33·||[1.2,1.0,0.8] - [1.1,1.0,1.0]||²)
             = exp(-0.33·0.05)
             = 0.984

... (calculate for all support vectors)

f(x_new) = (-10.5)×0.933 + 15.2×0.984 + (-6.8)×0.956 
           + 8.7×0.967 + (-18.9)×0.945 + 12.3×0.978 + 87.7
         = -9.80 + 14.96 - 6.50 + 8.41 - 17.86 + 12.03 + 87.7
         = 88.94

Predicted price: $88.94
```

## Why SVM Works for Noisy Stock Data

### 1. Epsilon Insensitivity

Daily price fluctuations often have noise ±$0.10-$0.50:
- Epsilon tube (ε=0.1) ignores these small errors
- Model focuses on significant trends
- Reduces overfitting to market noise

### 2. Support Vector Sparsity

Only ~30-40% of training samples become support vectors:
- Most "typical" prices are ignored
- Model defined by boundary cases (unusual patterns)
- Naturally filters out redundant information

### 3. Kernel Nonlinearity

RBF kernel captures:
- Complex lag interactions
- Local price regimes (bull/bear markets)
- Nonlinear momentum effects

**Example pattern captured:**
```
If (lag_1 ≈ 150 AND lag_2 < 148):
    Predict reversal (upward correction)
Else if (lag_1 > 155 AND lag_2 > 154):
    Predict continuation (momentum)
```

Linear model can't capture these interactions; SVM with RBF can.

## Comparison with Tree Models

| Aspect | Decision Trees | SVM (RBF) |
|--------|----------------|-----------|
| Decision boundary | Axis-aligned rectangles | Smooth curved surface |
| Feature interactions | Limited (depth constraint) | **Unlimited (kernel)** |
| Outlier sensitivity | High (split threshold affected) | **Low (ε-tube + support vectors)** |
| Extrapolation | Constant outside range | **Smooth decay** |
| Training time | O(n log n) | O(n² to n³) |
| Prediction time | O(log n) | **O(#support_vectors)** |

**Visual comparison (2D simplified):**

```
Decision Tree:
    │         │
────┼─────────┼──────
    │  $102  │ $105
────┼─────────┼──────
    │  $101  │
────┴─────────┴──────

SVM (RBF):
    ╱╲       ╱╲
───╱  ╲─────╱  ╲─────
  $101 $103 $105
──────────────────────
(Smooth curved regions)
```

## Hyperparameter Tuning Strategy

### Grid Search Example

```python
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [10, 50, 100, 200],
    'gamma': [0.1, 0.2, 0.5, 'scale'],
    'epsilon': [0.05, 0.1, 0.2]
}

grid_search = GridSearchCV(
    SVR(kernel='rbf'),
    param_grid,
    cv=5,  # 5-fold cross-validation
    scoring='neg_mean_squared_error'
)

grid_search.fit(X_train_scaled, y_train)
best_params = grid_search.best_params_
```

### Parameter Relationships

**If RMSE too high:**
1. Increase C (100 → 200) - tighter fit
2. Increase gamma (0.2 → 0.5) - more complex boundary
3. Decrease epsilon (0.1 → 0.05) - stricter tube

**If overfitting (train RMSE << test RMSE):**
1. Decrease C (100 → 50) - allow more violations
2. Decrease gamma (0.2 → 0.1) - smoother boundary
3. Increase epsilon (0.1 → 0.2) - wider tolerance

## Practical Insights for Stock Prediction

### When SVM Excels

**Scenario 1: High Volatility**
- Stock: TSLA (Tesla)
- Daily swings: ±5-10%
- SVM's ε-tube ignores small noise
- Captures underlying trend

**Scenario 2: Small Dataset**
- Only 6 months of data (~126 samples)
- Trees would overfit
- SVM generalizes better with <500 samples

**Scenario 3: Ensemble Diversification**
- Tree models (RF, GB, XGB) all agree
- SVM provides different perspective
- If SVM prediction differs → investigate

### When SVM Struggles

**Scenario 1: Large Datasets**
- 10+ years daily data (2500+ samples)
- O(n²) training becomes slow (>60 seconds)
- XGBoost trains faster and better

**Scenario 2: High-Dimensional Features**
- Adding 50+ technical indicators
- Kernel matrix becomes huge
- Memory issues, slow predictions

**Scenario 3: Trend Breakouts**
- Price breaks historical range
- SVM can't extrapolate
- All models fail here (need external signals)

## Feature Engineering for SVM

### Recommended Additions

**1. Percentage Changes (log-returns):**
```python
returns = np.log(prices / prices.shift(1))
# More stationary than raw prices
```

**2. Volatility Features:**
```python
rolling_std = returns.rolling(5).std()
# Captures market regime changes
```

**3. Momentum Indicators:**
```python
rsi = calculate_rsi(prices, period=14)
macd = calculate_macd(prices)
```

**Why helps SVM:**
- Kernel can combine: price + volatility + momentum
- Captures multi-dimensional patterns
- Example: "High price + low volatility → breakout coming"

### Scaling Importance

SVM is **highly sensitive** to feature scales:

**Bad (unscaled):**
```
lag_1: 150.23  (range: 100-200)
volume: 1,500,000  (range: 1M-3M)
```
Euclidean distance dominated by volume!

**Good (scaled):**
```
lag_1: 0.52  (range: -2 to +2)
volume: 0.31  (range: -2 to +2)
```
Equal contribution to kernel distance.

**Always use `StandardScaler` before SVM:**
```python
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## End of Advanced Model Theory