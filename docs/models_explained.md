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

End.