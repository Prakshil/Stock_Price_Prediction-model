# Stock Price Prediction — Beginner Data Mining Project

This repo contains a very simple, beginner-friendly approach to predict a stock’s next-day closing price using classic data mining techniques (no deep learning required).

- `web_stock_price_predictor.py` — a Streamlit app that:
  - downloads daily data with yfinance
  - builds basic lag features (previous N closes)
  - trains a Linear Regression model
  - evaluates accuracy on a time-based test split
  - shows a naive next-day prediction
- `stock_price.ipynb` — optional notebook to explore data; you can keep it simple with the same lagged regression approach.

## Quick start

1. Create a Python environment (recommended) and install dependencies:
   - yfinance
   - pandas
   - numpy
   - scikit-learn
   - matplotlib
   - streamlit

   Note: TensorFlow is not required. The app uses scikit-learn LinearRegression for clarity and ease.

2. Launch the Streamlit app from this folder:

```powershell
# From the stock_price_model folder
streamlit run .\web_stock_price_predictor.py
```

3. In the app sidebar:
   - Choose ticker (e.g., GOOG, AAPL)
   - Choose history period (e.g., 1y)
   - Set number of lag days (e.g., 5)
   - Choose test size percentage (e.g., 20%)

## Notes
- The app automatically picks `Adj Close` if available, otherwise falls back to `Close`.
- The next-day prediction is naive and for demonstration/learning, not investment advice.

## Method (beginner “data mining” style)

We use a univariate time series transformed into a supervised learning dataset:

1. Feature engineering: create lag features of the closing price: Close(t-1), Close(t-2), …, Close(t-N)
2. Split by time: use the most recent X% as the test set
3. Fit a Linear Regression model on the training set
4. Evaluate on the test set with MAE, RMSE, MAPE and plot Actual vs Predicted
5. Predict the next day using the most recent lag values

This keeps the project approachable for beginners while demonstrating classic steps: data collection, feature creation, model training, evaluation, and simple prediction.
